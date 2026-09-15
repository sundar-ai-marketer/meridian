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

"""Executable verification of every disposition recorded in TRIAGE.md.

TRIAGE.md states, for each issue open on the upstream tracker, whether it is
already fixed upstream, fixed here, or not a defect. A document says that once;
this module asserts it on every run.

Two kinds of test live here:

  * `FixedHereTest` guards the behaviour this fork added. If a rebase drops one
    of those patches, the matching test fails.
  * `FixedUpstreamTest` guards behaviour this fork depends on but did not
    write. These are the more valuable ones: upstream could regress any of them
    in a future release, and without a guard the regression would arrive
    silently in a routine rebase.

Tests needing a fitted model share one tiny model, so the whole module runs in
well under a minute.
"""

import inspect
import os
import pathlib
import tempfile
import warnings

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd
import tensorflow_probability.substrates.jax as tfp_jax
import xarray as xr

from meridian import constants as c
from meridian.analysis import analyzer as analyzer_module
from meridian.data import data_frame_input_data_builder as dfb
from meridian.data import input_data as input_data_lib
from meridian.data import test_utils as data_test_utils
from meridian.model import model
from meridian.model import prior_distribution
from meridian.model import spec


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CHANNELS = ['c0', 'c1']


def _source_files() -> list[pathlib.Path]:
  """Every non-test Python source file in the package."""
  return [
      p
      for p in (_REPO_ROOT / 'meridian').rglob('*.py')
      if not p.name.endswith('_test.py')
  ]


def _frame(
    n_times: int = 40,
    max_lag: int = 0,
    populate_everything: bool = False,
    freq: str = 'W-MON',
) -> pd.DataFrame:
  """Builds a small tidy frame, optionally with burn-in rows."""
  rng = np.random.default_rng(0)
  n_media_times = n_times + max_lag
  dates = pd.date_range('2023-01-02', periods=n_media_times, freq=freq)
  dates = dates.strftime('%Y-%m-%d').tolist()
  window = set(dates[max_lag:])

  rows = []
  for geo in ['a', 'b']:
    for date in dates:
      inside = populate_everything or date in window
      row = {
          'geo': geo,
          'date': date,
          'population': 1e6,
          'revenue': rng.random() * 100 + 50 if inside else np.nan,
          'ctrl': rng.random() if inside else np.nan,
      }
      for ch in _CHANNELS:
        row[f'{ch}_impressions'] = rng.random() * 10_000 + 1_000
        row[f'{ch}_spend'] = rng.random() * 1_000 + 100 if inside else np.nan
      rows.append(row)
  return pd.DataFrame(rows)


def _build_input_data(df: pd.DataFrame):
  return (
      dfb.DataFrameInputDataBuilder(kpi_type='revenue')
      .with_kpi(
          df.dropna(subset=['revenue']),
          kpi_col='revenue',
          time_col='date',
          geo_col='geo',
      )
      .with_population(df, population_col='population', geo_col='geo')
      .with_controls(
          df.dropna(subset=['ctrl']),
          control_cols=['ctrl'],
          time_col='date',
          geo_col='geo',
      )
      .with_media(
          df,
          media_cols=[f'{ch}_impressions' for ch in _CHANNELS],
          media_spend_cols=[f'{ch}_spend' for ch in _CHANNELS],
          media_channels=_CHANNELS,
          time_col='date',
          geo_col='geo',
      )
      .build()
  )


def _fit(data, prior_type: str = 'roi', max_lag: int = 2) -> model.Meridian:
  mmm = model.Meridian(
      input_data=data,
      model_spec=spec.ModelSpec(media_prior_type=prior_type, max_lag=max_lag),
  )
  mmm.sample_prior(50, seed=0)
  mmm.sample_posterior(
      n_chains=2, n_adapt=60, n_burnin=60, n_keep=80, seed=0
  )
  return mmm


class StaticSourceTest(parameterized.TestCase):
  """Source-level guards. No model required, so these are instant."""

  def test_issue_1427_no_numpy_newshape_kwarg(self):
    """`reshape(newshape=)` was removed in NumPy 2."""
    offenders = [
        str(p.relative_to(_REPO_ROOT))
        for p in _source_files()
        if 'newshape' in p.read_text()
    ]
    self.assertEmpty(offenders)

  def test_issue_1453_no_numpy_concat_alias(self):
    """`np.concat` is a NumPy 2 alias absent from older runtimes."""
    offenders = [
        str(p.relative_to(_REPO_ROOT))
        for p in _source_files()
        if 'np.concat(' in p.read_text()
    ]
    self.assertEmpty(offenders)

  def test_issue_1538_tensorflow_pinned_above_cve(self):
    """CVE-2026-2492 is addressed by requiring TensorFlow >= 2.21."""
    pyproject = (_REPO_ROOT / 'pyproject.toml').read_text()
    self.assertIn('tensorflow >= 2.21.0, < 2.22', pyproject)

  def test_issue_1589_ppp_check_converts_xarray_to_tensor(self):
    """The JAX backend cannot einsum an xarray, so it must be converted."""
    checks = (
        _REPO_ROOT / 'meridian' / 'analysis' / 'review' / 'checks.py'
    ).read_text()
    self.assertIn('backend.to_tensor(', checks)

  def test_issue_1709_reconstruction_batch_size_is_exposed(self):
    """Chunking the reconstruction pass is what bounds peak memory."""
    params = inspect.signature(model.Meridian.sample_posterior).parameters
    self.assertIn('reconstruction_batch_size', params)

  def test_issue_1502_media_time_documented_as_superset(self):
    """The docstring described the relationship backwards, which is how
    people ended up with no burn-in."""
    builder = (
        _REPO_ROOT / 'meridian' / 'data' / 'data_frame_input_data_builder.py'
    ).read_text()
    self.assertIn('superset', builder)
    self.assertNotIn('potentially shorter than time', builder)


class ValidationTest(parameterized.TestCase):
  """Input-validation guards. Fast: no sampling required."""

  def test_issue_1644_reversed_spend_channel_order_is_rejected(self):
    """The most dangerous upstream bug: same channels, different order,
    silently mis-attributing spend. Fixed upstream in 1.7.1 -- guard it."""
    data = data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
        n_geos=3, n_times=20, n_media_times=22, n_media_channels=2,
        n_controls=1,
    )
    reversed_spend = data.media_spend.copy(deep=True)
    reversed_spend = reversed_spend.assign_coords({
        c.MEDIA_CHANNEL: list(
            reversed(reversed_spend.coords[c.MEDIA_CHANNEL].values.tolist())
        )
    })
    with self.assertRaises(ValueError):
      input_data_lib.InputData(
          kpi=data.kpi,
          kpi_type=data.kpi_type,
          population=data.population,
          controls=data.controls,
          revenue_per_kpi=data.revenue_per_kpi,
          media=data.media,
          media_spend=reversed_spend,
      )

  def test_issue_1596_missing_column_is_named(self):
    df = pd.DataFrame(
        {'geo': ['a'], 'date': ['2024-01-01'], 'Sales': [1.0]}
    )
    builder = dfb.DataFrameInputDataBuilder(kpi_type=c.NON_REVENUE)
    with self.assertRaises(ValueError) as cm:
      builder.with_kpi(df, kpi_col='Sales', geo_col='geo')
    self.assertIn("missing required column(s): ['time']", str(cm.exception))

  def test_issue_1713_negative_kpi_error_explains_and_locates(self):
    data = data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
        n_geos=3, n_times=20, n_media_times=22, n_media_channels=2,
        n_controls=1,
    )
    kpi = data.kpi.copy(deep=True)
    kpi.values[1, 3] = -1.0
    with self.assertRaises(ValueError) as cm:
      input_data_lib.InputData(
          kpi=kpi,
          kpi_type=data.kpi_type,
          population=data.population,
          controls=data.controls,
          revenue_per_kpi=data.revenue_per_kpi,
          media=data.media,
          media_spend=data.media_spend,
      )
    message = str(cm.exception)
    self.assertIn('Found 1 negative value(s)', message)
    self.assertIn('ill-defined', message)

  @parameterized.named_parameters(
      dict(testcase_name='python_floats', loc=0.2, scale=0.9),
      dict(testcase_name='python_tuples', loc=(0.2, 0.3), scale=(0.9, 0.9)),
  )
  def test_issues_1364_1404_float32_priors_are_accepted(self, loc, scale):
    """The documented prior idiom yields float32; the 64-bit JAX default
    rejected it, which blocked both issues."""
    from meridian import backend  # pylint: disable=g-import-not-at-top

    dist = tfp_jax.distributions.LogNormal(loc, scale, name=c.ROI_M)
    prior = prior_distribution.PriorDistribution(roi_m=dist)
    expected = backend.standardize_dtype(backend.float_dtype)
    self.assertEqual(
        backend.standardize_dtype(prior.roi_m.dtype), expected
    )

  def test_spec_rejects_explicit_none_prior(self):
    with self.assertRaisesRegex(ValueError, 'must be a `PriorDistribution`'):
      spec.ModelSpec(prior=None)

  def test_issue_1502_fully_populated_frame_warns_about_burn_in(self):
    """The silent half of #1502: every column populated for every date yields
    no burn-in at all, and adstock is then zero-padded."""
    data = _build_input_data(
        _frame(n_times=40, max_lag=0, populate_everything=True)
    )
    self.assertEqual(
        len(data.media.coords[c.MEDIA_TIME]), len(data.kpi.coords[c.TIME])
    )
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter('always')
      model.Meridian(
          input_data=data, model_spec=spec.ModelSpec(max_lag=8)
      )
    self.assertTrue(
        any(
            'Insufficient media history for adstock' in str(w.message)
            for w in caught
        )
    )

  def test_issue_1502_burn_in_rows_are_preserved(self):
    """The documented pattern -- spend missing during burn-in -- must keep the
    longer media axis."""
    data = _build_input_data(_frame(n_times=40, max_lag=6))
    self.assertEqual(len(data.kpi.coords[c.TIME]), 40)
    self.assertEqual(len(data.media.coords[c.MEDIA_TIME]), 46)

  def test_issue_1466_non_finite_variable_does_not_abort_vif(self):
    from meridian.model.eda import constants as eda_constants  # pylint: disable=g-import-not-at-top
    from meridian.model.eda import eda_engine  # pylint: disable=g-import-not-at-top

    rng = np.random.default_rng(0)
    values = rng.normal(size=(30, 3))
    values[5, 2] = np.nan
    da = xr.DataArray(
        values,
        dims=['sample', eda_constants.VARIABLE],
        coords={eda_constants.VARIABLE: ['good_a', 'good_b', 'bad']},
    )
    vif = eda_engine._calculate_vif(da, eda_constants.VARIABLE, 1e-3)
    self.assertTrue(
        np.isfinite(vif.sel({eda_constants.VARIABLE: 'good_a'}).item())
    )
    self.assertTrue(np.isnan(vif.sel({eda_constants.VARIABLE: 'bad'}).item()))


class FittedModelTest(parameterized.TestCase):
  """Guards that need posterior draws. One shared fit keeps this cheap."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.data = _build_input_data(_frame(n_times=40, max_lag=4))
    cls.mmm = _fit(cls.data)

  def test_issue_643_response_curves_ignore_a_full_time_selection(self):
    """Selecting every date must equal selecting none."""
    analyzer = analyzer_module.Analyzer(
        model_context=self.mmm.model_context,
        inference_data=self.mmm.inference_data,
    )
    every_time = [
        str(t) for t in self.data.kpi.coords[c.TIME].values
    ]
    default = analyzer.response_curves().incremental_outcome.values
    explicit = analyzer.response_curves(
        selected_times=every_time
    ).incremental_outcome.values
    np.testing.assert_allclose(default, explicit, rtol=1e-6)

  def test_issue_1751_prior_type_survives_serde(self):
    from meridian.schema.serde import meridian_serde  # pylint: disable=g-import-not-at-top

    mmm = _fit(self.data, prior_type='contribution')
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, 'model.binpb')
      meridian_serde.save_meridian(mmm, path)
      loaded = meridian_serde.load_meridian(path)
    self.assertEqual(str(loaded.model_spec.media_prior_type), 'contribution')

  def test_serde_round_trip_preserves_roi_exactly(self):
    from meridian.schema.serde import meridian_serde  # pylint: disable=g-import-not-at-top

    before = np.asarray(
        analyzer_module.Analyzer(
            model_context=self.mmm.model_context,
            inference_data=self.mmm.inference_data,
        ).roi()
    )
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, 'model.binpb')
      meridian_serde.save_meridian(self.mmm, path)
      loaded = meridian_serde.load_meridian(path)
    after = np.asarray(
        analyzer_module.Analyzer(
            model_context=loaded.model_context,
            inference_data=loaded.inference_data,
        ).roi()
    )
    np.testing.assert_allclose(before, after, rtol=0, atol=0)


class MonthlyTimeAxisTest(absltest.TestCase):
  """#1675: non-uniform (calendar-monthly) time coordinates must serialize."""

  def test_issue_1675_monthly_coordinates_serialize(self):
    from meridian.schema.serde import meridian_serde  # pylint: disable=g-import-not-at-top

    data = _build_input_data(_frame(n_times=36, max_lag=0, freq='MS'))
    gaps = np.diff(
        pd.to_datetime(data.kpi.coords[c.TIME].values)
    ).astype('timedelta64[D]').astype(int)
    # The point of the test: the axis really is non-uniform.
    self.assertGreater(len(set(gaps.tolist())), 1)

    mmm = _fit(data)
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, 'monthly.binpb')
      meridian_serde.save_meridian(mmm, path)
      self.assertTrue(os.path.exists(path))


if __name__ == '__main__':
  absltest.main()
