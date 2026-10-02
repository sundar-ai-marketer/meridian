# Copyright 2026 Sundar Ramesh Kumar.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# NOTICE: This file is new in this fork; see NOTICE at the repository root.

"""Arithmetic, pairing, unsupported-scope and sampling-gate contract tests."""

import dataclasses
import json
import types

from absl.testing import absltest
from absl.testing import parameterized
import arviz as az
from meridian import backend
from meridian import constants as c
from meridian.analysis import analyzer
from meridian.analysis import budget_decision
from meridian.data import input_data
import numpy as np
import xarray as xr

mock = absltest.mock
_POLICY = budget_decision.DownsidePolicy(
    loss_tolerance=0, max_loss_probability=0.1, max_expected_downside=0.1
)


def _outcomes(values):
  return xr.DataArray(
      np.asarray(values, dtype=float), dims=(c.CHAIN, c.DRAW, c.CHANNEL),
      coords={c.CHAIN: [0, 1], c.DRAW: [0, 1, 2], c.CHANNEL: ['a', 'b']},
      attrs={
          'posterior_id': 'same-fit', 'outcome_unit': 'conversions',
          'selected_geos': ('g0', 'g1'),
          'selected_times': ('2024-01-08', '2024-01-15', '2024-01-22'),
          'execution_semantics': 'scaled_historical_flight',
      },
  )


def _input_data():
  geos = ['g0', 'g1']
  times = ['2024-01-08', '2024-01-15', '2024-01-22']
  history = ['2024-01-01'] + times
  return input_data.InputData(
      kpi=xr.DataArray(np.arange(6).reshape(2, 3) + 10,
                       dims=(c.GEO, c.TIME),
                       coords={c.GEO: geos, c.TIME: times}, name=c.KPI),
      kpi_type=c.NON_REVENUE,
      population=xr.DataArray([100, 200], dims=c.GEO,
                              coords={c.GEO: geos}, name=c.POPULATION),
      media=xr.DataArray(np.ones((2, 4, 2)),
                         dims=(c.GEO, c.MEDIA_TIME, c.MEDIA_CHANNEL),
                         coords={c.GEO: geos, c.MEDIA_TIME: history,
                                 c.MEDIA_CHANNEL: ['a', 'b']}, name=c.MEDIA),
      media_spend=xr.DataArray(np.ones((2, 3, 2)),
                               dims=(c.GEO, c.TIME, c.MEDIA_CHANNEL),
                               coords={c.GEO: geos, c.TIME: times,
                                       c.MEDIA_CHANNEL: ['a', 'b']},
                               name=c.MEDIA_SPEND),
  )


def _fitted(data=None, *, n_chains=2, n_draws=3, second_channel_gain=1.0):
  data = data or _input_data()
  fitted = mock.Mock(spec=analyzer.Analyzer)
  fitted.model_context = types.SimpleNamespace(
      input_data=data, n_media_channels=2, n_rf_channels=0,
      media_tensors=types.SimpleNamespace(media=backend.to_tensor(
          data.media.values, dtype=backend.float_dtype)),
  )
  fitted.inference_data = az.from_dict(
      posterior={'beta': np.random.default_rng(123).normal(
          size=(n_chains, n_draws))},
      sample_stats={'diverging': np.zeros((n_chains, n_draws), dtype=bool)},
  )

  def spend(**kwargs):
    return data.media_spend.sel({c.GEO: kwargs['selected_geos'],
                                c.TIME: kwargs['selected_times']}).sum(
                                    (c.GEO, c.TIME)).rename(
                                        {c.MEDIA_CHANNEL: c.CHANNEL})

  def incremental(**kwargs):
    # Isolate the actual caller's scaling contract with exact arithmetic. A
    # separate E2E producer exercises the real Analyzer and fitted posterior.
    media = np.asarray(kwargs['new_data'].media)
    ratios = media[0, 0] / data.media.values[0, 0]
    coefficients = np.empty((n_chains, n_draws, 2))
    coefficients[..., 0] = 10
    coefficients[..., 1] = 10 + second_channel_gain
    return coefficients * ratios

  fitted.get_aggregated_spend.side_effect = spend
  fitted.incremental_outcome.side_effect = incremental
  return fitted


def _audit(specifications, **kwargs):
  args = dict(
      baseline={'a': 6, 'b': 6}, candidates={'move': {'a': 0, 'b': 12}},
      policy=_POLICY, outcome_unit='conversions', use_kpi=True,
  )
  args.update(kwargs)
  return budget_decision.audit_budget_decisions(specifications, **args)


class PairedOutcomeTest(parameterized.TestCase):

  def test_baseline_is_exactly_zero_every_draw(self):
    baseline = _outcomes(np.arange(12).reshape(2, 3, 2))
    delta = budget_decision.paired_outcome_delta(baseline, baseline)
    np.testing.assert_array_equal(delta, np.zeros((2, 3)))
    summary = budget_decision.summarize_paired_outcomes(
        baseline, baseline, policy=_POLICY
    )
    self.assertEqual(summary['mean_gain'], 0)
    self.assertEqual(summary['expected_downside'], 0)
    self.assertEqual(summary['probability_loss_beyond_tolerance'], 0)

  def test_sum_preserves_correlated_channels_before_quantiles(self):
    baseline = _outcomes(np.zeros((2, 3, 2)))
    candidate = _outcomes([[[100, -99], [-100, 101], [50, -49]],
                           [[-50, 51], [20, -19], [-20, 21]]])
    summary = budget_decision.summarize_paired_outcomes(
        baseline, candidate, policy=_POLICY
    )
    self.assertEqual(summary['central_interval']['lower'], 1)
    self.assertEqual(summary['central_interval']['upper'], 1)
    self.assertEqual(summary['mean_gain'], 1)

  def test_channel_permutation_aligns_by_labels(self):
    baseline = _outcomes(np.arange(12).reshape(2, 3, 2))
    candidate = (baseline + 2).sel({c.CHANNEL: ['b', 'a']})
    candidate.attrs = baseline.attrs.copy()
    np.testing.assert_array_equal(
        budget_decision.paired_outcome_delta(baseline, candidate),
        np.full((2, 3), 4),
    )

  @parameterized.parameters('posterior_id', 'outcome_unit', 'selected_geos',
                            'selected_times', 'execution_semantics')
  def test_mismatched_metadata_fails(self, key):
    baseline = _outcomes(np.zeros((2, 3, 2)))
    candidate = baseline.copy(deep=True)
    candidate.attrs[key] = 'different'
    with self.assertRaisesRegex(ValueError, 'does not match'):
      budget_decision.paired_outcome_delta(baseline, candidate)

  @parameterized.parameters('chain', 'draw', 'channel')
  def test_misaligned_coordinates_fail(self, dim):
    baseline = _outcomes(np.zeros((2, 3, 2)))
    candidate = baseline.assign_coords({dim: np.arange(baseline.sizes[dim]) + 10})
    with self.assertRaises(ValueError):
      budget_decision.paired_outcome_delta(baseline, candidate)

  @parameterized.parameters(np.nan, np.inf, -np.inf)
  def test_nonfinite_outcomes_fail(self, value):
    baseline = _outcomes(np.zeros((2, 3, 2)))
    candidate = baseline.copy(deep=True)
    candidate.values[0, 0, 0] = value
    with self.assertRaisesRegex(ValueError, 'non-finite'):
      budget_decision.paired_outcome_delta(baseline, candidate)

  def test_high_mean_with_downside_does_not_meet_policy(self):
    baseline = _outcomes(np.zeros((2, 3, 2)))
    values = np.zeros((2, 3, 2))
    values[..., 0] = [[100, 100, 100], [100, -50, -50]]
    summary = budget_decision.summarize_paired_outcomes(
        baseline, _outcomes(values), policy=_POLICY
    )
    self.assertGreater(summary['mean_gain'], 0)
    self.assertAlmostEqual(summary['expected_downside'], 100 / 6)
    self.assertAlmostEqual(summary['probability_loss_beyond_tolerance'], 2 / 6)
    self.assertFalse(summary['downside_policy_met_in_draws'])


class BudgetDecisionAuditTest(parameterized.TestCase):

  def test_strict_sampling_failure_remains_exploratory(self):
    fitted = _fitted()
    report = _audit({'weak': fitted})
    self.assertFalse(report.sampling_ready)
    self.assertEqual(report.fits['weak']['quantities_label'], 'exploratory')
    self.assertFalse(report.fits['weak']['sampling_quality']['sampling_passed'])
    self.assertEqual(report.fits['weak']['candidates']['move']['mean_gain'], 1)
    self.assertEqual(report.fits['weak']['candidates']['baseline']['mean_gain'], 0)
    self.assertIsNone(report.to_dict()['recommendation'])
    decoded = json.loads(report.to_json())
    self.assertEqual(decoded['scope']['total_budget'], 12)
    decoded['fits'].clear()
    self.assertIn('weak', report.fits)

  def test_independent_spec_shapes_and_crossing_prior_no_unanimous_gain(self):
    positive = _fitted(n_chains=2, n_draws=4)
    negative = _fitted(n_chains=4, n_draws=3, second_channel_gain=-1)
    report = _audit({'prior-a': positive, 'prior-b': negative})
    envelope = report.sensitivity['move']['mean_gain_envelope']
    self.assertEqual(envelope, {'minimum': -1, 'maximum': 1})
    self.assertFalse(report.sensitivity['move']['unanimous_positive_mean_gain'])
    self.assertFalse(report.sensitivity['move'][
        'downside_policy_met_in_every_specification_draws'])
    for fit in report.fits.values():
      self.assertEqual(fit['candidates']['baseline']['central_interval']['lower'], 0)

  def test_passed_sampling_screen_does_not_select_an_allocation(self):
    report = _audit({'iid': _fitted(n_chains=4, n_draws=2000)})
    self.assertTrue(report.sampling_ready)
    self.assertEqual(report.fits['iid']['quantities_label'], 'conditional_posterior')
    self.assertIsNone(report.to_dict()['recommendation'])

  def test_scaled_flight_includes_history_and_preserves_selected_window(self):
    fitted = _fitted()
    report = _audit(
        {'fit': fitted}, selected_geos=['g0'], selected_times=['2024-01-15'],
        baseline={'a': 1, 'b': 1}, candidates={'move': {'a': 0, 'b': 2}},
    )
    for call in fitted.incremental_outcome.call_args_list:
      self.assertEqual(call.kwargs['selected_geos'], ['g0'])
      self.assertEqual(call.kwargs['selected_times'], ['2024-01-15'])
      self.assertFalse(call.kwargs['include_non_paid_channels'])
    media = np.asarray(fitted.incremental_outcome.call_args.kwargs['new_data'].media)
    self.assertEqual(media.shape, (2, 4, 2))
    np.testing.assert_array_equal(media[..., 0], 0)
    np.testing.assert_array_equal(media[..., 1], 2)
    self.assertEqual(report.scope['scaled_media_times_including_history'][0],
                     '2024-01-01')

  def test_channel_order_can_differ_across_specs(self):
    data = _input_data()
    permuted = dataclasses.replace(
        data, media=data.media.sel({c.MEDIA_CHANNEL: ['b', 'a']}),
        media_spend=data.media_spend.sel({c.MEDIA_CHANNEL: ['b', 'a']}),
    )
    # The coefficients below are positional, so fix the mock's outcome labels
    # to reproduce the same channel-level effects under this permutation.
    second = _fitted(permuted, second_channel_gain=-1)
    original = second.incremental_outcome.side_effect
    second.incremental_outcome.side_effect = lambda **kw: original(**kw) * (1.1, 10/9)
    report = _audit({'a': _fitted(data), 'b': second})
    self.assertEqual(report.sensitivity['move']['mean_gain_envelope'],
                     {'minimum': 1, 'maximum': 1})

  def test_different_data_or_history_fails_before_evaluation(self):
    data = _input_data()
    other = dataclasses.replace(data, media=data.media * 2)
    first = _fitted(data)
    with self.assertRaisesRegex(ValueError, 'different input data/history'):
      _audit({'a': first, 'b': _fitted(other)})
    first.incremental_outcome.assert_not_called()

  def test_different_currency_rejected(self):
    data = _input_data()
    with self.assertRaisesRegex(ValueError, 'different currency'):
      _audit({'usd': _fitted(dataclasses.replace(data, currency_code='USD')),
              'inr': _fitted(dataclasses.replace(data, currency_code='INR'))})

  def test_nonfinite_total_budget_rejected(self):
    with np.errstate(over='ignore'):
      with self.assertRaisesRegex(ValueError, 'total allocation budget'):
        _audit({'fit': _fitted()}, baseline={'a': 1e308, 'b': 1e308})

  def test_missing_posterior_visible_and_blocks_readiness(self):
    fitted = _fitted()
    fitted.inference_data = az.InferenceData()
    report = _audit({'empty': fitted})
    self.assertFalse(report.sampling_ready)
    self.assertEqual(report.fits['empty']['outcome_status'], 'unavailable')
    self.assertIsNone(report.sensitivity['move']['mean_gain_envelope'])
    fitted.incremental_outcome.assert_not_called()

  def test_rf_rejected(self):
    fitted = _fitted()
    fitted.model_context.n_rf_channels = 1
    with self.assertRaisesRegex(ValueError, 'Reach/frequency'):
      _audit({'rf': fitted})

  def test_positive_spend_with_zero_reference_rejected(self):
    data = _input_data()
    spend = data.media_spend.copy(deep=True)
    spend.loc[{c.MEDIA_CHANNEL: 'a'}] = 0
    with self.assertRaisesRegex(ValueError, 'nonzero historical'):
      _audit({'fit': _fitted(dataclasses.replace(data, media_spend=spend))})

  @parameterized.parameters(
      {'a': -1, 'b': 13}, {'a': np.nan, 'b': 12},
      {'a': np.inf, 'b': 12}, {'a': 0, 'c': 12}, {'a': 0, 'b': 13},
  )
  def test_invalid_allocations_fail(self, **allocation):
    with self.assertRaises(ValueError):
      _audit({'fit': _fitted()}, candidates={'bad': allocation})

  @parameterized.parameters(
      {'selected_times': ['2099-01-01']}, {'selected_geos': ['unknown']},
      {'selected_times': []}, {'selected_geos': ['g0', 'g0']},
      {'outcome_unit': ''}, {'use_kpi': False}, {'batch_size': 0},
      {'confidence_level': 1}, {'lower_quantile_probability': np.nan},
      {'candidates': {'baseline': {'a': 6, 'b': 6}}},
  )
  def test_invalid_scope_fails(self, **kwargs):
    with self.assertRaises(ValueError):
      _audit({'fit': _fitted()}, **kwargs)

  @parameterized.parameters(
      (-1, .1, 1), (np.nan, .1, 1), (1, np.inf, 1),
      (1, 1.1, 1), (1, .1, -1), (1, .1, np.inf),
  )
  def test_invalid_business_policy_fails(self, tolerance, probability, downside):
    with self.assertRaises(ValueError):
      budget_decision.DownsidePolicy(tolerance, probability, downside)


if __name__ == '__main__':
  absltest.main()
