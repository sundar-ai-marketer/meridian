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

"""Tests for the Meridian benchmark (google/meridian#1396)."""

import json
import math
import subprocess
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from meridian.benchmark import benchmark


_TINY = benchmark.BenchmarkConfig(
    n_geos=2,
    n_times=20,
    n_media_channels=2,
    n_controls=1,
    max_lag=2,
    n_prior_draws=20,
    n_chains=2,
    n_adapt=20,
    n_burnin=20,
    n_keep=20,
)


class BenchmarkConfigTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='no_lag', n_times=50, max_lag=0, expected=50),
      dict(testcase_name='default_lag', n_times=104, max_lag=8, expected=112),
      dict(testcase_name='long_lag', n_times=10, max_lag=26, expected=36),
  )
  def test_media_times_covers_adstock_window(
      self, n_times: int, max_lag: int, expected: int
  ):
    config = benchmark.BenchmarkConfig(n_times=n_times, max_lag=max_lag)
    self.assertEqual(config.n_media_times, expected)


class EnvironmentInfoTest(absltest.TestCase):

  def test_reports_versions_and_devices(self):
    info = benchmark.environment_info()
    for key in ['python', 'platform', 'machine', 'meridian', 'jax', 'numpy']:
      self.assertIn(key, info)
    # The resolved device list is the point of the record: several upstream
    # reports were environment differences, not library behaviour.
    self.assertIn('jax_devices', info)

  def test_survives_a_missing_optional_dependency(self):
    with mock.patch.dict('sys.modules', {'tensorflow': None}):
      info = benchmark.environment_info()
    self.assertIn('tensorflow', info)
    self.assertStartsWith(info['tensorflow'], 'unavailable')


class ColdImportTest(absltest.TestCase):

  def test_returns_nan_when_subprocess_fails(self):
    with mock.patch.object(
        subprocess, 'run', side_effect=subprocess.TimeoutExpired('x', 1)
    ):
      self.assertTrue(math.isnan(benchmark.cold_import_seconds()))

  def test_returns_nan_on_unparseable_output(self):
    completed = mock.Mock(stdout='not-a-number\n')
    with mock.patch.object(subprocess, 'run', return_value=completed):
      self.assertTrue(math.isnan(benchmark.cold_import_seconds()))

  def test_parses_last_line(self):
    completed = mock.Mock(stdout='some warning\n1.25\n')
    with mock.patch.object(subprocess, 'run', return_value=completed):
      self.assertAlmostEqual(benchmark.cold_import_seconds(), 1.25)


class RunBenchmarkTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    # Stub the subprocess import measurement; it is covered separately and
    # would add ~8s of interpreter startup to this test.
    with mock.patch.object(
        benchmark, 'cold_import_seconds', return_value=1.0
    ):
      cls.result = benchmark.run_benchmark(_TINY)

  def test_reports_all_stages(self):
    self.assertCountEqual(
        self.result.timings_seconds,
        [
            'cold_import',
            'build_input_data',
            'build_model',
            'sample_prior',
            'sample_posterior',
            'total',
        ],
    )

  def test_stage_timings_are_positive(self):
    for stage, seconds in self.result.timings_seconds.items():
      self.assertGreater(seconds, 0.0, msg=stage)

  def test_total_excludes_cold_import(self):
    """`cold_import` is a separate process and can be NaN, so it must not
    contaminate the total."""
    timings = self.result.timings_seconds
    expected = sum(
        v for k, v in timings.items() if k not in ('total', 'cold_import')
    )
    self.assertAlmostEqual(timings['total'], expected, places=6)

  def test_nan_cold_import_does_not_poison_total(self):
    with mock.patch.object(
        benchmark, 'cold_import_seconds', return_value=float('nan')
    ):
      result = benchmark.run_benchmark(_TINY)
    self.assertTrue(math.isnan(result.timings_seconds['cold_import']))
    self.assertFalse(math.isnan(result.timings_seconds['total']))

  def test_throughput_is_consistent_with_timings(self):
    total_draws = _TINY.n_chains * (
        _TINY.n_adapt + _TINY.n_burnin + _TINY.n_keep
    )
    expected = total_draws / self.result.timings_seconds['sample_posterior']
    self.assertAlmostEqual(
        self.result.throughput['mcmc_draws_per_sec'], expected, places=6
    )

  def test_peak_rss_is_plausible(self):
    # Anything under 1 MB means the unit conversion is wrong.
    self.assertGreater(self.result.peak_rss_bytes, 1_000_000)

  def test_json_round_trips(self):
    parsed = json.loads(self.result.to_json())
    self.assertEqual(parsed['config']['n_geos'], _TINY.n_geos)
    self.assertIn('timings_seconds', parsed)

  def test_table_mentions_every_stage(self):
    table = self.result.format_table()
    for stage in self.result.timings_seconds:
      self.assertIn(stage, table)
    self.assertIn('Peak RSS', table)


if __name__ == '__main__':
  absltest.main()
