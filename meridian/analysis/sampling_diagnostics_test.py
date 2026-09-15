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

"""Tests for MCMC sampling diagnostics (divergences and ESS)."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import arviz as az
from meridian import constants as c
from meridian.analysis import sampling_diagnostics
from meridian.analysis.review import configs as review_configs
from meridian.common import errors
from meridian.data import test_utils as data_test_utils
from meridian.model import model
from meridian.model import spec
import numpy as np


_N_GEOS = 4
_N_TIMES = 40
_N_CHAINS = 2
_N_KEEP = 80


def _build_model(n_geos: int = _N_GEOS) -> model.Meridian:
  data = data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
      n_geos=n_geos,
      n_times=_N_TIMES,
      n_media_times=_N_TIMES + 2,
      n_media_channels=2,
      n_controls=1,
  )
  return model.Meridian(
      input_data=data, model_spec=spec.ModelSpec(max_lag=2)
  )


class SamplingDiagnosticsTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.mmm = _build_model()
    cls.mmm.sample_prior(50, seed=0)
    cls.mmm.sample_posterior(
        n_chains=_N_CHAINS, n_adapt=60, n_burnin=60, n_keep=_N_KEEP, seed=0
    )
    cls.diag = sampling_diagnostics.SamplingDiagnostics(cls.mmm)

  # -----------------------------------------------------------------------
  # Construction / error handling.
  # -----------------------------------------------------------------------

  def test_raises_without_posterior(self):
    mmm = _build_model()
    mmm.sample_prior(20, seed=0)
    with self.assertRaisesRegex(errors.NotFittedModelError, 'no posterior'):
      sampling_diagnostics.SamplingDiagnostics(mmm)

  def test_draw_bookkeeping(self):
    self.assertEqual(self.diag.n_chains, _N_CHAINS)
    self.assertEqual(self.diag.n_draws, _N_KEEP)
    self.assertEqual(self.diag.total_draws, _N_CHAINS * _N_KEEP)

  # -----------------------------------------------------------------------
  # Divergences: the real posterior sample, which does report them.
  # -----------------------------------------------------------------------

  def test_divergences_are_reported_by_this_sampler(self):
    # posterior_sampler.py always writes `sample_stats['diverging']` for the
    # TFP NUTS sampler this model uses, so a real fit should report them.
    self.assertTrue(self.diag.divergences_reported)
    self.assertIsNotNone(self.diag.n_divergences)
    self.assertIsNotNone(self.diag.divergence_rate)

  def test_n_divergences_matches_raw_sample_stats(self):
    raw = np.asarray(
        self.mmm.inference_data.sample_stats[c.DIVERGING].values
    )
    self.assertEqual(self.diag.n_divergences, int(raw.sum()))

  def test_divergence_rate_matches_count_over_total_draws(self):
    self.assertAlmostEqual(
        self.diag.divergence_rate,
        self.diag.n_divergences / self.diag.total_draws,
        places=9,
    )

  def test_divergence_summary_shape_and_columns(self):
    frame = self.diag.divergence_summary()
    self.assertLen(frame, _N_CHAINS)
    self.assertCountEqual(
        frame.columns,
        [
            c.CHAIN,
            'n_divergences',
            'n_draws',
            'divergence_rate',
            'reported',
            'serious',
        ],
    )
    self.assertTrue(np.all(frame['reported']))
    self.assertTrue(np.all(frame['n_draws'] == _N_KEEP))

  def test_divergence_summary_counts_sum_to_total(self):
    frame = self.diag.divergence_summary()
    self.assertEqual(frame['n_divergences'].sum(), self.diag.n_divergences)

  def test_divergence_rate_is_serious_flag(self):
    with mock.patch.object(
        type(self.diag),
        'divergence_rate',
        new_callable=mock.PropertyMock,
        return_value=0.5,  # Far above DIVERGENCE_RATE_SERIOUS_THRESHOLD.
    ):
      self.assertTrue(self.diag.divergence_rate_is_serious)
    with mock.patch.object(
        type(self.diag),
        'divergence_rate',
        new_callable=mock.PropertyMock,
        return_value=0.0,
    ):
      self.assertFalse(self.diag.divergence_rate_is_serious)

  # -----------------------------------------------------------------------
  # Divergences: graceful degradation when not reported.
  # -----------------------------------------------------------------------

  def test_degrades_gracefully_without_sample_stats_group(self):
    fake_idata = mock.Mock()
    fake_idata.groups.return_value = [c.POSTERIOR]
    mmm = mock.Mock()
    mmm.inference_data = fake_idata
    mmm.model_context = self.mmm.model_context
    fake_idata.posterior = self.mmm.inference_data.posterior

    diag = sampling_diagnostics.SamplingDiagnostics(mmm)
    self.assertFalse(diag.divergences_reported)
    self.assertIsNone(diag.n_divergences)
    self.assertIsNone(diag.divergence_rate)
    self.assertIsNone(diag.divergence_rate_is_serious)

    frame = diag.divergence_summary()
    self.assertLen(frame, _N_CHAINS)
    self.assertTrue(np.all(~frame['reported']))
    self.assertTrue(np.all(frame['n_divergences'].isna()))

  def test_degrades_gracefully_without_diverging_key(self):
    # sample_stats group present, but without the 'diverging' variable --
    # e.g. a different sampler's trace metrics.
    real_sample_stats = self.mmm.inference_data.sample_stats
    stripped = real_sample_stats.drop_vars([c.DIVERGING])

    fake_idata = mock.Mock()
    fake_idata.groups.return_value = [c.POSTERIOR, 'sample_stats']
    fake_idata.posterior = self.mmm.inference_data.posterior
    fake_idata.sample_stats = stripped
    mmm = mock.Mock()
    mmm.inference_data = fake_idata
    mmm.model_context = self.mmm.model_context

    diag = sampling_diagnostics.SamplingDiagnostics(mmm)
    self.assertFalse(diag.divergences_reported)
    self.assertIsNone(diag.n_divergences)

  # -----------------------------------------------------------------------
  # r-hat.
  # -----------------------------------------------------------------------

  def test_rhat_max_matches_manual_nanmax_over_analyzer_get_rhat(self):
    rhat = self.diag._analyzer.get_rhat()  # pylint: disable=protected-access
    finite_maxes = []
    for value in rhat.values():
      arr = np.asarray(value, dtype=float).ravel()
      finite = arr[np.isfinite(arr)]
      if finite.size:
        finite_maxes.append(float(np.max(finite)))
    expected = max(finite_maxes) if finite_maxes else float('nan')
    if np.isnan(expected):
      self.assertTrue(np.isnan(self.diag.rhat_max))
    else:
      self.assertAlmostEqual(self.diag.rhat_max, expected, places=9)

  def test_rhat_threshold_ordering_is_internally_consistent(self):
    # Upstream's threshold (1.2) is looser than the modern one (1.01), so
    # clearing the modern bar must imply clearing the upstream one.
    self.assertLess(
        sampling_diagnostics.MODERN_RHAT_THRESHOLD,
        sampling_diagnostics.UPSTREAM_RHAT_THRESHOLD,
    )
    if self.diag.rhat_max_ok_modern:
      self.assertTrue(self.diag.rhat_max_ok_upstream)

  def test_upstream_threshold_matches_convergence_check_config(self):
    self.assertEqual(
        sampling_diagnostics.UPSTREAM_RHAT_THRESHOLD,
        review_configs.ConvergenceConfig().convergence_threshold,
    )

  # -----------------------------------------------------------------------
  # ESS.
  # -----------------------------------------------------------------------

  def test_min_bulk_ess_matches_manual_arviz_computation(self):
    ess = az.ess(self.mmm.inference_data.posterior, method='bulk')
    finite_mins = []
    for data_array in ess.data_vars.values():
      arr = np.asarray(data_array.values, dtype=float).ravel()
      finite = arr[np.isfinite(arr)]
      if finite.size:
        finite_mins.append(float(np.min(finite)))
    expected = min(finite_mins) if finite_mins else float('nan')
    self.assertAlmostEqual(self.diag.min_bulk_ess, expected, places=6)

  def test_min_bulk_ess_per_draw_is_consistent(self):
    self.assertAlmostEqual(
        self.diag.min_bulk_ess_per_draw,
        self.diag.min_bulk_ess / self.diag.total_draws,
        places=9,
    )

  def test_ess_ok_matches_thresholds(self):
    expected = (
        self.diag.min_bulk_ess >= sampling_diagnostics.MIN_BULK_ESS
        and self.diag.min_tail_ess >= sampling_diagnostics.MIN_TAIL_ESS
    )
    self.assertEqual(self.diag.ess_ok, expected)

  def test_tiny_sample_flags_low_ess(self):
    # 160 total post-warmup draws is well under the 400-draw ESS heuristic,
    # so this tiny test fixture should not read as ESS-healthy.
    self.assertLess(self.diag.total_draws, sampling_diagnostics.MIN_BULK_ESS)
    self.assertFalse(self.diag.ess_ok)

  # -----------------------------------------------------------------------
  # summary().
  # -----------------------------------------------------------------------

  def test_summary_columns(self):
    frame = self.diag.summary()
    self.assertCountEqual(
        frame.columns,
        [
            'parameter',
            'rhat',
            'bulk_ess',
            'tail_ess',
            'bulk_ess_per_draw',
            'tail_ess_per_draw',
            'rhat_ok_modern',
            'rhat_ok_upstream',
            'ess_ok',
        ],
    )
    self.assertNotEmpty(frame)

  def test_summary_is_sorted_worst_bulk_ess_first(self):
    frame = self.diag.summary()
    finite = frame['bulk_ess'].dropna().to_numpy()
    self.assertTrue(np.all(np.diff(finite) >= 0))
    # NaN rows (deterministic parameters), if any, sort to the end.
    if frame['bulk_ess'].isna().any():
      self.assertTrue(frame['bulk_ess'].iloc[-1] != frame['bulk_ess'].iloc[-1])

  def test_summary_min_bulk_ess_matches_property(self):
    frame = self.diag.summary()
    self.assertAlmostEqual(
        frame['bulk_ess'].min(), self.diag.min_bulk_ess, places=6
    )

  def test_summary_ess_ok_flag_matches_thresholds(self):
    frame = self.diag.summary()
    expected = (frame['bulk_ess'] >= sampling_diagnostics.MIN_BULK_ESS) & (
        frame['tail_ess'] >= sampling_diagnostics.MIN_TAIL_ESS
    )
    np.testing.assert_array_equal(
        frame['ess_ok'].to_numpy(), expected.to_numpy()
    )

  def test_summary_rhat_flags_match_thresholds(self):
    frame = self.diag.summary()
    np.testing.assert_array_equal(
        frame['rhat_ok_modern'].to_numpy(),
        (frame['rhat'] < sampling_diagnostics.MODERN_RHAT_THRESHOLD).to_numpy(),
    )
    np.testing.assert_array_equal(
        frame['rhat_ok_upstream'].to_numpy(),
        (
            frame['rhat'] < sampling_diagnostics.UPSTREAM_RHAT_THRESHOLD
        ).to_numpy(),
    )

  # -----------------------------------------------------------------------
  # verdict.
  # -----------------------------------------------------------------------

  def test_verdict_is_nonempty_string(self):
    self.assertIsInstance(self.diag.verdict, str)
    self.assertNotEmpty(self.diag.verdict)

  def test_verdict_mentions_low_ess_for_tiny_fixture(self):
    self.assertIn('ESS', self.diag.verdict)

  def test_verdict_reports_not_reported_divergences(self):
    fake_idata = mock.Mock()
    fake_idata.groups.return_value = [c.POSTERIOR]
    fake_idata.posterior = self.mmm.inference_data.posterior
    mmm = mock.Mock()
    mmm.inference_data = fake_idata
    mmm.model_context = self.mmm.model_context

    diag = sampling_diagnostics.SamplingDiagnostics(mmm)
    self.assertIn('not reported', diag.verdict)

  @parameterized.named_parameters(
      dict(
          testcase_name='serious',
          rate=0.5,
          expected='systematic geometry problem',
      ),
      dict(testcase_name='no_divergences', rate=0.0, expected='No divergent'),
  )
  def test_verdict_reflects_divergence_severity(self, rate, expected):
    n_divergences = round(rate * self.diag.total_draws)
    with mock.patch.object(
        type(self.diag),
        'divergence_rate',
        new_callable=mock.PropertyMock,
        return_value=rate,
    ), mock.patch.object(
        type(self.diag),
        'n_divergences',
        new_callable=mock.PropertyMock,
        return_value=n_divergences,
    ), mock.patch.object(
        type(self.diag),
        'divergence_rate_is_serious',
        new_callable=mock.PropertyMock,
        return_value=rate
        > sampling_diagnostics.DIVERGENCE_RATE_SERIOUS_THRESHOLD,
    ):
      self.assertIn(expected, self.diag.verdict)

  # -----------------------------------------------------------------------
  # `_nan_reduce` helper: deterministic-parameter handling.
  # -----------------------------------------------------------------------

  def test_nan_reduce_all_nan_returns_nan(self):
    result = sampling_diagnostics._nan_reduce(
        np.array([np.nan, np.nan]), np.max
    )
    self.assertTrue(np.isnan(result))

  def test_nan_reduce_ignores_nan_cells(self):
    result = sampling_diagnostics._nan_reduce(
        np.array([1.0, np.nan, 3.0]), np.max
    )
    self.assertEqual(result, 3.0)

  def test_nan_reduce_min_for_ess_style_reduction(self):
    result = sampling_diagnostics._nan_reduce(
        np.array([10.0, np.nan, 3.0]), np.min
    )
    self.assertEqual(result, 3.0)


if __name__ == '__main__':
  absltest.main()
