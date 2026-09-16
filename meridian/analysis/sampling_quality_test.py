# Copyright 2026 Sundar Ramesh Kumar.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NOTICE: This file is new in this fork and does not exist in the
# original google/meridian source.

"""Focused behavior and failure tests for strict sampling quality."""

from __future__ import annotations

import json
from unittest import mock

from absl.testing import absltest
import arviz as az
from meridian.analysis import sampling_quality
import numpy as np
import xarray as xr


class _PriorBroadcast:

  def __init__(self, deterministic_names=()):
    self._deterministic_names = set(deterministic_names)

  def has_deterministic_param(self, name):
    return name in self._deterministic_names


class _ModelContext:

  def __init__(self, deterministic_names=(), baseline_geo_idx=0):
    self.prior_broadcast = _PriorBroadcast(deterministic_names)
    self.baseline_geo_idx = baseline_geo_idx


def _idata(posterior, *, diverging=None, coords=None, dims=None):
  kwargs = {'posterior': posterior}
  if diverging is not None:
    kwargs['sample_stats'] = {'diverging': diverging}
  if coords is not None:
    kwargs['coords'] = coords
  if dims is not None:
    kwargs['dims'] = dims
  return az.from_dict(**kwargs)


def _issue_codes(report):
  return {issue['code'] for issue in report.blocking_issues}


class SamplingQualityTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(17)

  def _healthy(self, n_chains=4):
    draws = self.rng.normal(size=(n_chains, 2000))
    return _idata(
        {'theta': draws},
        diverging=np.zeros((n_chains, 2000), dtype=bool),
    )

  def test_healthy_draws_pass_and_json_is_standard(self):
    report = sampling_quality.assess_sampling_quality(self._healthy())

    self.assertTrue(report.sampling_passed)
    payload = json.loads(report.to_json())
    self.assertTrue(payload['sampling_passed'])
    self.assertEqual(payload['status'], 'pass')
    self.assertEqual(payload['thresholds']['bulk_ess_gte'], 400)
    self.assertEmpty(payload['blocking_issues'])

  def test_default_ess_scales_with_chain_count(self):
    report = sampling_quality.assess_sampling_quality(self._healthy(n_chains=5))

    self.assertEqual(report.thresholds['bulk_ess_gte'], 500)
    self.assertEqual(report.thresholds['tail_ess_gte'], 500)
    self.assertEqual(
        report.thresholds['minimum_ess_rule'], 'max(400, 100 * n_chains)'
    )

  def test_explicit_ess_threshold_is_reported_as_explicit(self):
    report = sampling_quality.assess_sampling_quality(
        self._healthy(), min_ess=600
    )

    self.assertEqual(report.thresholds['bulk_ess_gte'], 600)
    self.assertEqual(
        report.thresholds['minimum_ess_rule'], 'explicit min_ess=600'
    )

  def test_summary_keeps_a_precise_custom_rhat_threshold(self):
    report = sampling_quality.assess_sampling_quality(
        self._healthy(), rhat_threshold=1.005
    )

    self.assertIn('R-hat < 1.005', report.format_summary())

  def test_bad_coordinate_is_actionable_and_fails_every_required_metric(self):
    draws = np.empty((4, 2000, 2))
    draws[:, :, 0] = self.rng.normal(size=(4, 2000))
    for chain in range(4):
      draws[chain, :, 1] = chain + self.rng.normal(scale=0.01, size=2000)
    report = sampling_quality.assess_sampling_quality(
        _idata(
            {'beta_m': draws},
            diverging=np.zeros((4, 2000), dtype=bool),
            coords={'media_channel': ['Channel0', 'Channel1']},
            dims={'beta_m': ['media_channel']},
        )
    )

    self.assertFalse(report.sampling_passed)
    bad = next(
        cell
        for cell in report.parameters
        if cell['coordinates'] == {'media_channel': 'Channel1'}
    )
    self.assertEqual(bad['id'], 'beta_m[media_channel="Channel1"]')
    self.assertEqual(bad['status'], 'fail')
    self.assertGreaterEqual(
        bad['rank_normalized_rhat'],
        sampling_quality.STRICT_RANK_NORMALIZED_RHAT_THRESHOLD,
    )
    self.assertLess(bad['bulk_ess'], report.thresholds['bulk_ess_gte'])
    self.assertLess(bad['tail_ess'], report.thresholds['tail_ess_gte'])

  def test_rhat_threshold_is_strict_at_exact_boundary(self):
    data = self._healthy()
    rhat = xr.Dataset({'theta': xr.DataArray(1.01)})
    ess = xr.Dataset({'theta': xr.DataArray(1000.0)})
    with (
        mock.patch.object(sampling_quality.az, 'rhat', return_value=rhat),
        mock.patch.object(sampling_quality.az, 'ess', side_effect=[ess, ess]),
    ):
      report = sampling_quality.assess_sampling_quality(data)

    self.assertEqual(report.status, 'fail')
    self.assertFalse(report.sampling_passed)
    self.assertIn(
        'rank_normalized_rhat_not_below_threshold', _issue_codes(report)
    )

  def test_misaligned_metric_coordinates_are_unavailable(self):
    data = _idata(
        {'beta_m': self.rng.normal(size=(4, 2000, 2))},
        diverging=np.zeros((4, 2000), dtype=bool),
        coords={'media_channel': ['Channel0', 'Channel1']},
        dims={'beta_m': ['media_channel']},
    )
    reversed_coords = {'media_channel': ['Channel1', 'Channel0']}
    rhat = xr.Dataset(
        {
            'beta_m': xr.DataArray(
                [1.0, 1.0], dims=['media_channel'], coords=reversed_coords
            )
        }
    )
    ess = xr.Dataset(
        {
            'beta_m': xr.DataArray(
                [1000.0, 1000.0],
                dims=['media_channel'],
                coords=reversed_coords,
            )
        }
    )
    with (
        mock.patch.object(sampling_quality.az, 'rhat', return_value=rhat),
        mock.patch.object(sampling_quality.az, 'ess', side_effect=[ess, ess]),
    ):
      report = sampling_quality.assess_sampling_quality(data)

    self.assertEqual(report.status, 'unavailable')
    self.assertIn('rank_normalized_rhat_unavailable', _issue_codes(report))

  def test_metadata_confirmed_deterministic_constant_is_excluded(self):
    report = sampling_quality.assess_sampling_quality(
        _idata(
            {
                'theta': self.rng.normal(size=(4, 2000)),
                'slope_m': np.ones((4, 2000)),
            },
            diverging=np.zeros((4, 2000), dtype=bool),
        ),
        model_context=_ModelContext(deterministic_names={'slope_m'}),
    )

    self.assertTrue(report.sampling_passed)
    slope = next(
        cell for cell in report.parameters if cell['parameter'] == 'slope_m'
    )
    self.assertEqual(slope['status'], 'excluded')
    self.assertIn(
        'has_deterministic_param', slope['deterministic_metadata']['source']
    )

  def test_tau_g_baseline_is_explicitly_excluded_but_other_geo_is_checked(self):
    tau_g = np.empty((4, 2000, 2))
    tau_g[:, :, 0] = 0.0
    tau_g[:, :, 1] = self.rng.normal(size=(4, 2000))
    report = sampling_quality.assess_sampling_quality(
        _idata(
            {'tau_g': tau_g},
            diverging=np.zeros((4, 2000), dtype=bool),
            coords={'geo': ['baseline', 'other']},
            dims={'tau_g': ['geo']},
        ),
        model_context=_ModelContext(baseline_geo_idx=0),
    )

    self.assertTrue(report.sampling_passed)
    baseline, other = report.parameters
    self.assertEqual(baseline['status'], 'excluded')
    self.assertEqual(other['status'], 'pass')
    self.assertIn(
        'baseline_geo_idx', baseline['deterministic_metadata']['source']
    )

  def test_deterministic_tau_g_excluding_baseline_is_metadata_confirmed(self):
    report = sampling_quality.assess_sampling_quality(
        _idata(
            {'tau_g': np.zeros((4, 2000, 2))},
            diverging=np.zeros((4, 2000), dtype=bool),
            coords={'geo': ['baseline', 'other']},
            dims={'tau_g': ['geo']},
        ),
        model_context=_ModelContext(
            deterministic_names={'tau_g_excl_baseline'}, baseline_geo_idx=0
        ),
    )

    self.assertEqual(report.status, 'unavailable')
    baseline, other = report.parameters
    self.assertEqual(baseline['status'], 'excluded')
    self.assertEqual(other['status'], 'excluded')
    self.assertIn(
        'tau_g_excl_baseline', other['deterministic_metadata']['source']
    )

  def test_unverified_constant_is_unavailable_not_silently_dropped(self):
    report = sampling_quality.assess_sampling_quality(
        _idata(
            {'theta': np.ones((4, 2000))},
            diverging=np.zeros((4, 2000), dtype=bool),
        )
    )

    self.assertEqual(report.status, 'unavailable')
    self.assertFalse(report.sampling_passed)
    self.assertIn('constant_draws_metadata_unavailable', _issue_codes(report))

  def test_nonfinite_draws_and_missing_divergence_flags_are_unavailable(self):
    draws = self.rng.normal(size=(4, 2000))
    draws[0, 0] = np.nan
    report = sampling_quality.assess_sampling_quality(_idata({'theta': draws}))

    self.assertEqual(report.status, 'unavailable')
    self.assertIn('nonfinite_posterior_draws', _issue_codes(report))
    self.assertIn('divergence_not_reported', _issue_codes(report))
    payload = json.loads(report.to_json())
    self.assertIsNone(payload['parameters'][0]['rank_normalized_rhat'])

  def test_any_post_warmup_divergence_fails_the_gate(self):
    diverging = np.zeros((4, 2000), dtype=bool)
    diverging[2, 11] = True
    report = sampling_quality.assess_sampling_quality(
        _idata({'theta': self.rng.normal(size=(4, 2000))}, diverging=diverging)
    )

    self.assertEqual(report.status, 'fail')
    self.assertEqual(report.divergences['n_divergences'], 1)
    self.assertIn('divergent_transitions_detected', _issue_codes(report))

  def test_insufficient_chains_is_unavailable(self):
    report = sampling_quality.assess_sampling_quality(
        _idata(
            {'theta': self.rng.normal(size=(1, 20))},
            diverging=np.zeros((1, 20), dtype=bool),
        )
    )

    self.assertEqual(report.status, 'unavailable')
    self.assertIn('insufficient_chains', _issue_codes(report))


if __name__ == '__main__':
  absltest.main()
