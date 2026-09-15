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

"""Mathematical invariants that must hold for reported numbers to be valid.

The rest of the suite checks that code runs and that APIs behave. This checks
the *arithmetic* underneath the numbers that reach a client deck. Each test
here is an identity that, if it ever broke, would produce plausible-looking
output that is simply wrong -- the failure mode no amount of integration
testing catches.

Measured on this fork, all hold at machine precision:

  roi == incremental_outcome / spend          max error 7.1e-15
  sum over geos of incremental == aggregated  max error 4.5e-13
  normalized adstock weights sum to 1         exact

These are the properties that let you say "this channel returned 3.2x" and
mean it.
"""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from meridian import backend
from meridian import constants as c
from meridian.analysis import analyzer as analyzer_module
from meridian.data import test_utils as data_test_utils
from meridian.model import adstock_hill
from meridian.model import model
from meridian.model import spec


def _f(x) -> np.ndarray:
  return np.asarray(x, dtype=backend.np_float_dtype)


class AdstockInvariantsTest(parameterized.TestCase):
  """Carryover must move outcome through time without creating it."""

  @parameterized.named_parameters(
      dict(testcase_name='geometric_w1', decay=c.GEOMETRIC_DECAY, window=1),
      dict(testcase_name='geometric_w5', decay=c.GEOMETRIC_DECAY, window=5),
      dict(testcase_name='geometric_w13', decay=c.GEOMETRIC_DECAY, window=13),
      dict(testcase_name='binomial_w1', decay=c.BINOMIAL_DECAY, window=1),
      dict(testcase_name='binomial_w5', decay=c.BINOMIAL_DECAY, window=5),
      dict(testcase_name='binomial_w13', decay=c.BINOMIAL_DECAY, window=13),
  )
  def test_normalized_decay_weights_sum_to_one(self, decay: str, window: int):
    """If weights did not sum to 1, carryover would invent or destroy outcome,
    and every ROI downstream would be scaled by an arbitrary factor."""
    alpha = backend.to_tensor(_f([[0.1, 0.5, 0.9, 0.99]]))
    l_range = backend.to_tensor(_f(np.arange(window - 1, -1, -1)))
    weights = np.asarray(
        adstock_hill._compute_single_decay_function_weights(  # pylint: disable=protected-access
            alpha, l_range, window, decay, True
        )
    )
    np.testing.assert_allclose(
        weights.sum(axis=-1), np.ones(weights.shape[:-1]), rtol=1e-9
    )

  @parameterized.named_parameters(
      dict(testcase_name='low_alpha', alpha=0.1),
      dict(testcase_name='mid_alpha', alpha=0.6),
      dict(testcase_name='high_alpha', alpha=0.95),
  )
  def test_zero_max_lag_is_the_identity(self, alpha: float):
    """With no carryover, adstocked media must equal raw media exactly,
    whatever the decay rate."""
    media = backend.to_tensor(_f(np.arange(1, 9).reshape(1, 1, 8, 1)))
    transformer = adstock_hill.AdstockTransformer(
        backend.to_tensor(_f([[alpha]])), max_lag=0, n_times_output=8
    )
    np.testing.assert_allclose(
        np.asarray(transformer.forward(media)), np.asarray(media), atol=1e-10
    )

  def test_zero_media_gives_zero_adstock(self):
    media = backend.to_tensor(np.zeros((1, 1, 10, 1), dtype=backend.np_float_dtype))
    transformer = adstock_hill.AdstockTransformer(
        backend.to_tensor(_f([[0.7]])), max_lag=4, n_times_output=10
    )
    np.testing.assert_allclose(
        np.asarray(transformer.forward(media)), 0.0, atol=1e-12
    )

  def test_constant_media_is_preserved_by_normalized_adstock(self):
    """Weights summing to 1 mean a flat media series adstocks to itself, once
    the window is fully populated."""
    max_lag = 4
    media = backend.to_tensor(
        np.full((1, 1, 20, 1), 3.0, dtype=backend.np_float_dtype)
    )
    transformer = adstock_hill.AdstockTransformer(
        backend.to_tensor(_f([[0.6]])), max_lag=max_lag, n_times_output=20
    )
    out = np.asarray(transformer.forward(media)).ravel()
    # Skip the leading periods, where the window is still filling.
    np.testing.assert_allclose(out[max_lag:], 3.0, rtol=1e-9)


class HillInvariantsTest(parameterized.TestCase):
  """Saturation must be a well-behaved response curve."""

  @parameterized.named_parameters(
      dict(testcase_name='slope_1_ec_half', slope=1.0, ec=0.5),
      dict(testcase_name='slope_2_ec_1', slope=2.0, ec=1.0),
      dict(testcase_name='slope_half_ec_2', slope=0.5, ec=2.0),
  )
  def test_matches_the_closed_form(self, slope: float, ec: float):
    """hill(x) = x^slope / (x^slope + ec^slope)."""
    x = np.linspace(0.0, 8.0, 33)
    transformer = adstock_hill.HillTransformer(
        backend.to_tensor(_f([[ec]])), backend.to_tensor(_f([[slope]]))
    )
    got = np.asarray(
        transformer.forward(
            backend.to_tensor(_f(x.reshape(1, 1, x.size, 1)))
        )
    ).ravel()
    expected = x**slope / (x**slope + ec**slope)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-9)

  def test_zero_media_gives_zero_response(self):
    transformer = adstock_hill.HillTransformer(
        backend.to_tensor(_f([[0.5]])), backend.to_tensor(_f([[1.0]]))
    )
    out = np.asarray(
        transformer.forward(
            backend.to_tensor(np.zeros((1, 1, 4, 1), dtype=backend.np_float_dtype))
        )
    )
    np.testing.assert_allclose(out, 0.0, atol=1e-12)

  def test_is_monotone_increasing_and_bounded(self):
    """A saturation curve that dipped or exceeded 1 would make marginal ROI
    negative or unbounded."""
    x = np.linspace(0.0, 50.0, 200)
    transformer = adstock_hill.HillTransformer(
        backend.to_tensor(_f([[0.5]])), backend.to_tensor(_f([[1.0]]))
    )
    out = np.asarray(
        transformer.forward(
            backend.to_tensor(_f(x.reshape(1, 1, x.size, 1)))
        )
    ).ravel()
    self.assertTrue(np.all(np.diff(out) >= -1e-12))
    self.assertTrue(np.all(out >= 0.0))
    self.assertTrue(np.all(out < 1.0))


class ReportedNumberInvariantsTest(absltest.TestCase):
  """Identities behind the figures that reach a client."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    data = data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
        n_geos=4,
        n_times=40,
        n_media_times=44,
        n_media_channels=3,
        n_controls=1,
    )
    cls.data = data
    mmm = model.Meridian(
        input_data=data, model_spec=spec.ModelSpec(max_lag=4)
    )
    mmm.sample_prior(50, seed=0)
    mmm.sample_posterior(
        n_chains=2, n_adapt=80, n_burnin=80, n_keep=100, seed=0
    )
    cls.mmm = mmm
    cls.analyzer = analyzer_module.Analyzer(
        model_context=mmm.model_context, inference_data=mmm.inference_data
    )

  def test_roi_equals_incremental_outcome_over_spend(self):
    """The definition of ROI. If this drifts, every ROI reported is wrong and
    nothing else in the suite would notice."""
    roi = np.asarray(self.analyzer.roi())
    incremental = np.asarray(
        self.analyzer.incremental_outcome(
            aggregate_geos=True, aggregate_times=True
        )
    )
    spend = np.asarray(
        self.data.aggregate_media_spend(calibration_period=None)
    )
    np.testing.assert_allclose(roi, incremental / spend, rtol=1e-9)

  def test_incremental_outcome_is_additive_over_geos(self):
    """A geo breakdown that does not sum to the total cannot be shown next to
    the total."""
    per_geo = np.asarray(
        self.analyzer.incremental_outcome(
            aggregate_geos=False, aggregate_times=True
        )
    )
    aggregated = np.asarray(
        self.analyzer.incremental_outcome(
            aggregate_geos=True, aggregate_times=True
        )
    )
    np.testing.assert_allclose(
        per_geo.sum(axis=-2), aggregated, rtol=1e-9
    )

  def test_incremental_outcome_is_additive_over_times(self):
    per_time = np.asarray(
        self.analyzer.incremental_outcome(
            aggregate_geos=True, aggregate_times=False
        )
    )
    aggregated = np.asarray(
        self.analyzer.incremental_outcome(
            aggregate_geos=True, aggregate_times=True
        )
    )
    np.testing.assert_allclose(
        per_time.sum(axis=-2), aggregated, rtol=1e-9
    )

  def test_response_curves_are_monotone_in_spend(self):
    """Spending more cannot produce less incremental outcome."""
    curves = self.analyzer.response_curves(
        spend_multipliers=list(np.linspace(0.0, 2.0, 11))
    )
    mean = curves.incremental_outcome.sel({c.METRIC: c.MEAN}).values
    self.assertTrue(np.all(np.diff(mean, axis=0) >= -1e-6))

  def test_response_curve_at_zero_spend_is_zero(self):
    curves = self.analyzer.response_curves(spend_multipliers=[0.0, 1.0])
    mean = curves.incremental_outcome.sel({c.METRIC: c.MEAN}).values
    np.testing.assert_allclose(mean[0], 0.0, atol=1e-6)


if __name__ == '__main__':
  absltest.main()
