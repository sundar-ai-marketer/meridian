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

"""Tests for the ROI recovery check."""

from absl.testing import absltest
from absl.testing import parameterized
from meridian.validation import recovery
import numpy as np


class RecoveryConfigTest(parameterized.TestCase):

  def test_rejects_mismatched_channel_lengths(self):
    with self.assertRaisesRegex(ValueError, '`true_roi` has 1 entries'):
      recovery.RecoveryConfig(true_roi=(1.0,), spend_scale=(1.0, 2.0))

  def test_rejects_unknown_response_shape(self):
    with self.assertRaisesRegex(ValueError, "must be 'linear' or 'concave'"):
      recovery.RecoveryConfig(response='sigmoid')  # pytype: disable=wrong-arg-types

  @parameterized.named_parameters(
      dict(testcase_name='zero', level=0.0),
      dict(testcase_name='one', level=1.0),
  )
  def test_rejects_invalid_confidence_level(self, level):
    with self.assertRaisesRegex(ValueError, 'confidence_level'):
      recovery.RecoveryConfig(confidence_level=level)

  def test_media_times_covers_carryover(self):
    config = recovery.RecoveryConfig(n_times=50, max_lag=6)
    self.assertEqual(config.n_media_times, 56)

  def test_channel_names_match_roi_count(self):
    config = recovery.RecoveryConfig(
        true_roi=(1.0, 2.0), spend_scale=(10.0, 20.0)
    )
    self.assertLen(config.channels, 2)


class SimulateTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='linear', response='linear'),
      dict(testcase_name='concave', response='concave'),
  )
  def test_true_roi_is_exact_without_carryover(self, response):
    """With no carryover nothing spills across the window edge, so the
    realised ROI must equal the configured ROI exactly -- for both response
    shapes, since the concave case is rescaled to preserve totals."""
    config = recovery.RecoveryConfig(
        n_geos=3, n_times=30, max_lag=0, response=response
    )
    _, true_roi = recovery.simulate(config)
    np.testing.assert_allclose(true_roi, config.true_roi, rtol=1e-12)

  def test_carryover_shifts_realised_roi_only_at_the_edges(self):
    """With carryover, burn-in spend contributes inside the window while the
    tail contributes outside it, so the realised ROI differs slightly from the
    configured value. It is the realised figure that Meridian reports, so that
    is what recovery is judged against."""
    config = recovery.RecoveryConfig(n_geos=3, n_times=60, max_lag=4)
    _, true_roi = recovery.simulate(config)
    np.testing.assert_allclose(true_roi, config.true_roi, rtol=0.05)
    self.assertFalse(np.allclose(true_roi, config.true_roi, rtol=1e-12))

  def test_frame_has_burn_in_rows_without_spend(self):
    config = recovery.RecoveryConfig(n_geos=2, n_times=20, max_lag=5)
    df, _ = recovery.simulate(config)
    self.assertLen(df, 2 * config.n_media_times)
    burn_in = df[df['revenue'].isna()]
    # Burn-in rows carry impressions but no spend, KPI or controls.
    self.assertLen(burn_in, 2 * config.max_lag)
    self.assertTrue(burn_in['channel_0_spend'].isna().all())
    self.assertTrue(burn_in['channel_0_impressions'].notna().all())

  def test_simulation_is_deterministic_for_a_seed(self):
    config = recovery.RecoveryConfig(n_geos=2, n_times=20, max_lag=0)
    first, _ = recovery.simulate(config)
    second, _ = recovery.simulate(config)
    np.testing.assert_array_equal(
        first['revenue'].to_numpy(), second['revenue'].to_numpy()
    )

  def test_concave_and_linear_differ(self):
    base = dict(n_geos=2, n_times=20, max_lag=0)
    linear, _ = recovery.simulate(
        recovery.RecoveryConfig(response='linear', **base)
    )
    concave, _ = recovery.simulate(
        recovery.RecoveryConfig(response='concave', **base)
    )
    self.assertFalse(
        np.allclose(
            linear['revenue'].to_numpy(), concave['revenue'].to_numpy()
        )
    )


class ResultTest(absltest.TestCase):

  def _result(self, medians, lows, highs, r_hat=1.01):
    config = recovery.RecoveryConfig()
    channels = tuple(
        recovery.ChannelRecovery(
            channel=name,
            true_roi=true,
            median=med,
            ci_low=lo,
            ci_high=hi,
        )
        for name, true, med, lo, hi in zip(
            config.channels, config.true_roi, medians, lows, highs
        )
    )
    return recovery.RecoveryResult(
        config=config, channels=channels, max_r_hat=r_hat
    )

  def test_passes_when_covered_ordered_and_converged(self):
    result = self._result([1.1, 2.1, 3.9], [0.8, 1.8, 3.5], [1.4, 2.4, 4.4])
    self.assertTrue(result.all_covered)
    self.assertTrue(result.ordering_recovered)
    self.assertTrue(result.converged)
    self.assertTrue(result.passed)

  def test_fails_when_an_interval_misses(self):
    result = self._result([1.1, 2.1, 6.9], [0.8, 1.8, 6.5], [1.4, 2.4, 7.4])
    self.assertFalse(result.all_covered)
    self.assertFalse(result.passed)

  def test_fails_when_ordering_is_wrong(self):
    result = self._result([4.1, 2.1, 1.0], [0.1, 0.1, 0.1], [9.0, 9.0, 9.0])
    self.assertTrue(result.all_covered)
    self.assertFalse(result.ordering_recovered)
    self.assertFalse(result.passed)

  def test_fails_when_not_converged(self):
    result = self._result(
        [1.1, 2.1, 3.9], [0.8, 1.8, 3.5], [1.4, 2.4, 4.4], r_hat=1.6
    )
    self.assertFalse(result.converged)
    self.assertFalse(result.passed)

  def test_relative_error_signs(self):
    result = self._result([1.5, 1.0, 4.0], [0.1, 0.1, 0.1], [9.0, 9.0, 9.0])
    self.assertAlmostEqual(result.channels[0].relative_error, 0.5)
    self.assertAlmostEqual(result.channels[1].relative_error, -0.5)
    self.assertAlmostEqual(result.channels[2].relative_error, 0.0)

  def test_frame_and_report_cover_every_channel(self):
    result = self._result([1.1, 2.1, 3.9], [0.8, 1.8, 3.5], [1.4, 2.4, 4.4])
    frame = result.to_frame()
    self.assertLen(frame, 3)
    report = result.format_report()
    for channel in result.config.channels:
      self.assertIn(channel, report)
    self.assertIn('VERDICT', report)


class EndToEndTest(absltest.TestCase):
  """One very small fit, purely to prove the pipeline runs."""

  def test_run_recovery_produces_a_result(self):
    config = recovery.RecoveryConfig(
        n_geos=2,
        n_times=25,
        max_lag=0,
        true_roi=(1.0, 3.0),
        spend_scale=(3_000.0, 2_000.0),
        n_chains=2,
        n_adapt=40,
        n_burnin=40,
        n_keep=40,
    )
    result = recovery.run_recovery(config)
    self.assertLen(result.channels, 2)
    self.assertGreater(result.max_r_hat, 0.0)
    for channel in result.channels:
      self.assertLessEqual(channel.ci_low, channel.median)
      self.assertLessEqual(channel.median, channel.ci_high)
    # Too few draws to assert recovery; only that the report is well formed.
    self.assertIn('VERDICT', result.format_report())


if __name__ == '__main__':
  absltest.main()
