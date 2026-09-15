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

"""Tests for prior predictive checks (google/meridian#647)."""

from absl.testing import absltest
from absl.testing import parameterized
from meridian import backend
from meridian import constants as c
from meridian.analysis import prior_predictive
from meridian.common import errors
from meridian.data import test_utils as data_test_utils
from meridian.model import model
from meridian.model import prior_distribution
from meridian.model import spec
import numpy as np


_N_GEOS = 3
_N_TIMES = 40
_N_MEDIA_TIMES = 42
_N_DRAWS = 200


def _build_model(
    prior: prior_distribution.PriorDistribution | None = None,
) -> model.Meridian:
  data = data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
      n_geos=_N_GEOS,
      n_times=_N_TIMES,
      n_media_times=_N_MEDIA_TIMES,
      n_media_channels=2,
      n_controls=1,
  )
  model_spec = (
      spec.ModelSpec(prior=prior, max_lag=2)
      if prior is not None
      else spec.ModelSpec(max_lag=2)
  )
  return model.Meridian(input_data=data, model_spec=model_spec)


class PriorPredictiveCheckTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.mmm = _build_model()
    cls.mmm.sample_prior(_N_DRAWS, seed=0)

  def test_raises_without_prior_draws(self):
    mmm = _build_model()
    with self.assertRaisesRegex(
        errors.NotFittedModelError, 'no prior draws'
    ):
      prior_predictive.PriorPredictiveCheck(mmm)

  @parameterized.named_parameters(
      dict(testcase_name='zero', confidence_level=0.0),
      dict(testcase_name='one', confidence_level=1.0),
      dict(testcase_name='above_one', confidence_level=1.5),
      dict(testcase_name='negative', confidence_level=-0.1),
  )
  def test_raises_on_invalid_confidence_level(self, confidence_level: float):
    with self.assertRaisesRegex(ValueError, '`confidence_level` must be'):
      prior_predictive.PriorPredictiveCheck(
          self.mmm, confidence_level=confidence_level
      )

  def test_data_shape_and_coords(self):
    check = prior_predictive.PriorPredictiveCheck(self.mmm)
    data = check.prior_predictive_data
    self.assertEqual(data.sizes[c.TIME], _N_TIMES)
    self.assertEqual(data.sizes[c.METRIC], 3)
    self.assertCountEqual(
        data.coords[c.METRIC].values.tolist(), ['mean', 'ci_lo', 'ci_hi']
    )
    self.assertCountEqual(list(data.data_vars), ['expected', 'actual'])

  def test_credible_interval_is_ordered(self):
    check = prior_predictive.PriorPredictiveCheck(self.mmm)
    expected = check.prior_predictive_data['expected']
    lo = expected.sel({c.METRIC: 'ci_lo'}).values
    hi = expected.sel({c.METRIC: 'ci_hi'}).values
    self.assertTrue(np.all(lo <= hi))

  def test_wider_confidence_level_widens_interval(self):
    narrow = prior_predictive.PriorPredictiveCheck(
        self.mmm, confidence_level=0.5
    ).prior_predictive_data['expected']
    wide = prior_predictive.PriorPredictiveCheck(
        self.mmm, confidence_level=0.99
    ).prior_predictive_data['expected']
    narrow_width = (
        narrow.sel({c.METRIC: 'ci_hi'}) - narrow.sel({c.METRIC: 'ci_lo'})
    ).values
    wide_width = (
        wide.sel({c.METRIC: 'ci_hi'}) - wide.sel({c.METRIC: 'ci_lo'})
    ).values
    self.assertTrue(np.all(wide_width >= narrow_width))

  def test_summary_fields_are_consistent(self):
    check = prior_predictive.PriorPredictiveCheck(self.mmm)
    summary = check.summary()
    self.assertEqual(summary.n_draws, _N_DRAWS)
    self.assertEqual(summary.confidence_level, c.DEFAULT_CONFIDENCE_LEVEL)
    self.assertBetween(summary.coverage, 0.0, 1.0)
    self.assertGreater(summary.total_actual, 0.0)
    self.assertAlmostEqual(
        summary.total_ratio,
        summary.total_prior_median / summary.total_actual,
        places=6,
    )
    self.assertIsInstance(summary.verdict, str)
    self.assertNotEmpty(summary.verdict)

  def test_summary_frame_round_trips(self):
    summary = prior_predictive.PriorPredictiveCheck(self.mmm).summary()
    frame = summary.to_frame()
    self.assertEqual(frame.loc['n_draws', 'value'], _N_DRAWS)
    self.assertEqual(frame.loc['coverage', 'value'], summary.coverage)

  def test_absurd_prior_is_flagged(self):
    """A wildly optimistic ROI prior must be caught before any fitting."""
    # Use the backend's own TFP substrate and float dtype: a JAX-substrate
    # distribution handed to a TensorFlow-backed model fails when the sampler
    # receives a `tf.int32` seed.
    absurd = prior_distribution.PriorDistribution(
        roi_m=backend.tfd.LogNormal(
            backend.np_float_dtype(8.0),
            backend.np_float_dtype(0.1),
            name=c.ROI_M,
        )
    )
    mmm = _build_model(absurd)
    mmm.sample_prior(_N_DRAWS, seed=0)
    summary = prior_predictive.PriorPredictiveCheck(mmm).summary()

    self.assertGreater(summary.total_ratio, 10.0)
    self.assertFalse(summary.total_actual_within_ci)
    self.assertIn('order of magnitude', summary.verdict)

  def test_plot_layers_band_and_series(self):
    """Two layers: the credible band, and both lines sharing one colour
    scale so they appear in a single legend."""
    check = prior_predictive.PriorPredictiveCheck(self.mmm)
    spec_dict = check.plot_prior_predictive().to_dict()
    self.assertLen(spec_dict['layer'], 2)
    self.assertEqual(spec_dict['layer'][0]['mark']['type'], 'area')
    self.assertEqual(spec_dict['layer'][1]['mark']['type'], 'line')

  def test_plot_respects_selected_times(self):
    check = prior_predictive.PriorPredictiveCheck(self.mmm)
    times = [
        str(t)
        for t in check.prior_predictive_data.coords[c.TIME].values[:5]
    ]
    spec_dict = check.plot_prior_predictive(selected_times=times).to_dict()
    # Two datasets: the band keeps one row per period, the melted line frame
    # has one row per period per series.
    sizes = sorted(len(v) for v in spec_dict['datasets'].values())
    self.assertEqual(sizes, [len(times), len(times) * 2])

  def test_plot_has_one_legend_naming_all_three_series(self):
    """Without a colour encoding the band and both lines render in one hue
    with no legend, so a reader cannot tell which is which."""
    chart = prior_predictive.PriorPredictiveCheck(self.mmm).plot_prior_predictive()
    spec_dict = chart.to_dict()
    color = spec_dict['layer'][0]['encoding']['color']
    self.assertIn('Observed', color['scale']['domain'])
    self.assertIn('Prior mean', color['scale']['domain'])
    self.assertLen(color['scale']['domain'], 3)
    # Colour must not be the only cue.
    self.assertIn('strokeDash', spec_dict['layer'][1]['encoding'])

  def test_plot_width_is_fixed_not_scaled_by_period_count(self):
    """`bar_chart_width` sizes by number of bars: 156 weekly periods would
    produce a ~9700px chart and break any report it is placed in."""
    chart = prior_predictive.PriorPredictiveCheck(self.mmm).plot_prior_predictive()
    self.assertEqual(chart.to_dict()['width'], c.VEGALITE_FACET_EXTRA_LARGE_WIDTH)

  def test_plot_rejects_unknown_selected_times_clearly(self):
    check = prior_predictive.PriorPredictiveCheck(self.mmm)
    with self.assertRaisesRegex(ValueError, 'not in the model time'):
      check.plot_prior_predictive(selected_times=['1999-01-01'])

  def test_use_kpi_changes_scale(self):
    """With revenue_per_kpi present, revenue and KPI totals must differ."""
    revenue = prior_predictive.PriorPredictiveCheck(
        self.mmm, use_kpi=False
    ).summary()
    kpi = prior_predictive.PriorPredictiveCheck(
        self.mmm, use_kpi=True
    ).summary()
    self.assertNotAlmostEqual(
        revenue.total_actual, kpi.total_actual, places=3
    )


if __name__ == '__main__':
  absltest.main()
