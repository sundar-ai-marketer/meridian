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

"""Tests for geo allocation reliability (google/meridian#1668)."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from meridian import constants as c
from meridian.analysis import analyzer as analyzer_module
from meridian.analysis import geo_diagnostics
from meridian.common import errors
from meridian.data import test_utils as data_test_utils
from meridian.model import model
from meridian.model import spec
import numpy as np


_N_GEOS = 4
_N_TIMES = 40


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


class GeoAllocationReliabilityTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.mmm = _build_model()
    cls.mmm.sample_prior(50, seed=0)
    cls.mmm.sample_posterior(
        n_chains=2, n_adapt=60, n_burnin=60, n_keep=80, seed=0
    )
    cls.diag = geo_diagnostics.GeoAllocationReliability(cls.mmm)

  def test_raises_without_posterior(self):
    mmm = _build_model()
    mmm.sample_prior(20, seed=0)
    with self.assertRaisesRegex(errors.NotFittedModelError, 'no posterior'):
      geo_diagnostics.GeoAllocationReliability(mmm)

  def test_raises_for_national_model(self):
    with self.assertRaisesRegex(ValueError, 'national model'):
      with mock.patch.object(
          type(self.mmm.model_context),
          'is_national',
          new_callable=mock.PropertyMock,
          return_value=True,
      ):
        geo_diagnostics.GeoAllocationReliability(self.mmm)

  def test_reliability_data_shape(self):
    data = self.diag.reliability_data
    self.assertEqual(data.sizes[c.GEO], _N_GEOS)
    self.assertEqual(data.sizes[c.CHANNEL], 2)
    self.assertCountEqual(
        list(data.data_vars),
        ['mean', 'sd', 'cv', 'prob_positive', 'ci_excludes_zero'],
    )

  def test_cv_matches_sd_over_abs_mean(self):
    data = self.diag.reliability_data
    expected = data['sd'].values / np.abs(data['mean'].values)
    np.testing.assert_allclose(data['cv'].values, expected, rtol=1e-9)

  def test_sd_is_non_negative(self):
    self.assertTrue(np.all(self.diag.reliability_data['sd'].values >= 0))

  def test_summary_is_sorted_worst_first(self):
    frame = self.diag.summary()
    self.assertLen(frame, _N_GEOS * 2)
    cvs = frame['cv'].to_numpy()
    self.assertTrue(np.all(np.diff(cvs) <= 0))

  def test_summary_reliable_flag_matches_threshold(self):
    frame = self.diag.summary()
    expected = frame['cv'] <= geo_diagnostics.RELIABLE_CV_THRESHOLD
    np.testing.assert_array_equal(
        frame['reliable'].to_numpy(), expected.to_numpy()
    )

  def test_precision_loss_ratio_is_consistent(self):
    frame = self.diag.precision_loss()
    self.assertLen(frame, 2)
    np.testing.assert_allclose(
        frame['cv_ratio'].to_numpy(),
        frame['median_geo_cv'].to_numpy()
        / frame['aggregated_cv'].to_numpy(),
        rtol=1e-9,
    )

  def test_fraction_reliable_matches_summary(self):
    frame = self.diag.summary()
    self.assertAlmostEqual(
        self.diag.fraction_reliable, frame['reliable'].mean(), places=9
    )

  @parameterized.named_parameters(
      dict(testcase_name='all_reliable', cv=0.1, expected='defensible'),
      dict(testcase_name='none_reliable', cv=2.0, expected='posterior noise'),
  )
  def test_verdict_reflects_reliability(self, cv: float, expected: str):
    fake = np.full((_N_GEOS, 2), cv)
    with mock.patch.object(
        self.diag,
        '_by_geo',
        geo_diagnostics._Reduced(
            mean=np.ones_like(fake),
            sd=fake,
            cv=fake,
            prob_positive=np.ones_like(fake),
            ci_excludes_zero=np.ones_like(fake, dtype=bool),
        ),
    ):
      self.assertIn(expected, self.diag.verdict)

  def test_zero_signal_yields_nan_not_a_number(self):
    """A cell with no media execution has no coefficient of variation."""
    mean, sd, cv = geo_diagnostics._coefficient_of_variation(
        np.zeros((4, 3)), axis=(0,)
    )
    self.assertTrue(np.all(mean == 0))
    self.assertTrue(np.all(sd == 0))
    self.assertTrue(np.all(np.isnan(cv)))

  def test_floating_point_residue_is_not_treated_as_signal(self):
    """The real failure mode: a channel absent from one geo.

    Incremental outcome is deterministically zero there, so mean and standard
    deviation collapse to floating-point residue rather than exact zero. The
    old `mean != 0` test fired and computed noise over noise, producing a CV
    anywhere from 1e-16 to 1e+2 depending on rounding -- either extreme
    corrupting the ranking and the reliable/unreliable split.
    """
    rng = np.random.default_rng(0)
    draws = rng.normal(100.0, 20.0, size=(2, 50, 3, 2))
    draws[:, :, 1, 0] = rng.normal(0.0, 1e-16, size=(2, 50))

    _, _, cv = geo_diagnostics._coefficient_of_variation(draws, axis=(0, 1))

    self.assertTrue(np.isnan(cv[1, 0]))
    # Every other cell keeps a real, finite CV.
    others = np.delete(cv.ravel(), np.ravel_multi_index((1, 0), cv.shape))
    self.assertTrue(np.all(np.isfinite(others)))

  def test_no_signal_cells_are_excluded_from_fraction_reliable(self):
    diag = self.diag
    cv = np.array([[0.1, np.nan], [0.2, 0.9]])
    with mock.patch.object(
        diag,
        '_by_geo',
        geo_diagnostics._Reduced(
            mean=np.ones_like(cv),
            sd=np.zeros_like(cv),
            cv=cv,
            prob_positive=np.full_like(cv, 0.5),
            ci_excludes_zero=np.zeros_like(cv, dtype=bool),
        ),
    ):
      # Two of the three informative cells clear the threshold; the NaN is
      # neither counted as reliable nor as unreliable.
      self.assertAlmostEqual(diag.fraction_reliable, 2 / 3)

  def test_sign_precision_all_positive_draws(self):
    """Draws that never cross zero: prob_positive is 1, CI excludes zero."""
    draws = np.full((2, 50), 10.0) + np.random.default_rng(0).normal(
        0.0, 0.1, size=(2, 50)
    )
    prob_positive, ci_excludes_zero = geo_diagnostics._sign_precision(
        draws, axis=(0, 1)
    )
    self.assertEqual(prob_positive, 1.0)
    self.assertTrue(ci_excludes_zero)

  def test_sign_precision_distinguishes_straddling_zero_from_no_signal(self):
    """The module's core claim: CV alone conflates these two cases.

    A geo with literally zero execution and a geo whose effect sign is
    genuinely uncertain both drive CV very high (or NaN), but only the
    second has real posterior mass on both sides of zero.
    """
    rng = np.random.default_rng(0)
    no_execution = np.zeros((2, 200))
    straddling_zero = rng.normal(0.5, 20.0, size=(2, 200))

    _, _, no_exec_cv = geo_diagnostics._coefficient_of_variation(
        no_execution, axis=(0, 1)
    )
    _, _, straddle_cv = geo_diagnostics._coefficient_of_variation(
        straddling_zero, axis=(0, 1)
    )
    self.assertTrue(np.isnan(no_exec_cv))
    self.assertGreater(straddle_cv, geo_diagnostics.RELIABLE_CV_THRESHOLD)

    no_exec_prob_pos, no_exec_ci_excl = geo_diagnostics._sign_precision(
        no_execution, axis=(0, 1)
    )
    straddle_prob_pos, straddle_ci_excl = geo_diagnostics._sign_precision(
        straddling_zero, axis=(0, 1)
    )
    # No execution: every draw is exactly zero, so it is not "positive" by
    # a strict `> 0` test, and the credible interval collapses to a point at
    # zero -- it does not exclude zero.
    self.assertEqual(no_exec_prob_pos, 0.0)
    self.assertFalse(no_exec_ci_excl)
    # Straddling zero: a real, high-variance posterior puts mass on both
    # sides, so probability of a positive draw is well away from 0 or 1, and
    # the credible interval does not exclude zero either -- but for a very
    # different reason than the no-execution case.
    self.assertBetween(straddle_prob_pos, 0.2, 0.8)
    self.assertFalse(straddle_ci_excl)

  def test_summary_sign_uncertain_excludes_no_signal_and_reliable_cells(self):
    # One row per geo (_N_GEOS = 4), two channels: [no-signal, reliable],
    # [straddling-zero, reliable], [reliable, reliable], [reliable, reliable].
    mean = np.array([[0.0, 5.0], [0.1, 8.0], [5.0, 6.0], [7.0, 9.0]])
    sd = np.array([[0.0, 1.0], [5.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    cv = np.array(
        [[np.nan, 0.2], [50.0, 0.125], [0.2, 0.167], [0.143, 0.111]]
    )
    prob_positive = np.array(
        [[0.5, 0.99], [0.55, 0.99], [0.99, 0.99], [0.99, 0.99]]
    )
    ci_excludes_zero = np.array(
        [[False, True], [False, True], [True, True], [True, True]]
    )
    with mock.patch.object(
        self.diag,
        '_by_geo',
        geo_diagnostics._Reduced(
            mean=mean,
            sd=sd,
            cv=cv,
            prob_positive=prob_positive,
            ci_excludes_zero=ci_excludes_zero,
        ),
    ):
      frame = self.diag.summary()
    # (0, 0): no signal -- not sign-uncertain, regardless of the CI.
    no_signal_row = frame[
        (frame[c.GEO] == self.diag._geos[0]) & (~frame['has_signal'])
    ]
    self.assertFalse(no_signal_row['sign_uncertain'].all())
    # (1, 0): informative, CV = 50 (unreliable), CI does not exclude zero --
    # the straddling-zero case this diagnostic exists to surface.
    straddling_row = frame[np.isclose(frame['cv'], 50.0)]
    self.assertTrue(straddling_row['sign_uncertain'].all())
    # cv = 0.2 and cv = 0.125 both clear RELIABLE_CV_THRESHOLD, so neither is
    # sign-uncertain even though their CI happens to exclude zero.
    reliable_rows = frame[frame['reliable']]
    self.assertFalse(reliable_rows['sign_uncertain'].any())

  def test_plot_reliability_one_bar_per_geo(self):
    chart = self.diag.plot_reliability()
    spec_dict = chart.to_dict()
    data_key = spec_dict['data']['name']
    dataset = spec_dict['datasets'][data_key]
    self.assertLen(dataset, _N_GEOS)
    self.assertEqual(spec_dict['mark']['type'], 'bar')

  def test_plot_reliability_includes_geos_with_no_informative_channel(self):
    """A geo where every channel is no-signal still gets a bar, at zero."""
    cv = np.full((_N_GEOS, 2), np.nan)
    with mock.patch.object(
        self.diag,
        '_by_geo',
        geo_diagnostics._Reduced(
            mean=np.zeros_like(cv),
            sd=np.zeros_like(cv),
            cv=cv,
            prob_positive=np.full_like(cv, 0.5),
            ci_excludes_zero=np.zeros_like(cv, dtype=bool),
        ),
    ):
      chart = self.diag.plot_reliability()
    spec_dict = chart.to_dict()
    data_key = spec_dict['data']['name']
    dataset = spec_dict['datasets'][data_key]
    self.assertLen(dataset, _N_GEOS)
    self.assertTrue(all(row['fraction_reliable'] == 0.0 for row in dataset))


class ResolveUseKpiTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.mmm = _build_model()
    cls.mmm.sample_prior(20, seed=0)

  def test_resolves_like_the_private_method(self):
    analyzer = analyzer_module.Analyzer(
        model_context=self.mmm.model_context,
        inference_data=self.mmm.inference_data,
    )
    self.assertEqual(
        geo_diagnostics.resolve_use_kpi(analyzer, False),
        analyzer._use_kpi(False),  # pylint: disable=protected-access
    )

  def test_raises_actionable_error_if_private_method_is_gone(self):
    analyzer = analyzer_module.Analyzer(
        model_context=self.mmm.model_context,
        inference_data=self.mmm.inference_data,
    )
    with mock.patch.object(type(analyzer), '_use_kpi', new=None):
      with self.assertRaisesRegex(AttributeError, 'TRIAGE.md'):
        geo_diagnostics.resolve_use_kpi(analyzer, False)


if __name__ == '__main__':
  absltest.main()
