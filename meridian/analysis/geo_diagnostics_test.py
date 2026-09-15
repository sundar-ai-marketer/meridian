# Copyright 2026 Meridian fork contributors.
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
    self.assertCountEqual(list(data.data_vars), ['mean', 'sd', 'cv'])

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
            mean=np.ones_like(fake), sd=fake, cv=fake
        ),
    ):
      self.assertIn(expected, self.diag.verdict)

  def test_zero_mean_yields_infinite_cv(self):
    mean, sd, cv = geo_diagnostics._coefficient_of_variation(
        np.zeros((4, 3)), axis=(0,)
    )
    self.assertTrue(np.all(mean == 0))
    self.assertTrue(np.all(sd == 0))
    self.assertTrue(np.all(np.isinf(cv)))


if __name__ == '__main__':
  absltest.main()
