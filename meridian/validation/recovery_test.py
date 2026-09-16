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

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
from meridian.validation import recovery
import numpy as np


class RecoveryConfigTest(parameterized.TestCase):

  def test_rejects_mismatched_channel_lengths(self):
    with self.assertRaisesRegex(ValueError, '`true_roi` has 1 entries'):
      recovery.RecoveryConfig(true_roi=(1.0,), spend_scale=(1.0, 2.0))

  @parameterized.named_parameters(
      dict(testcase_name='empty', value=()),
      dict(testcase_name='zero', value=(0.0,)),
      dict(testcase_name='negative', value=(-1.0,)),
      dict(testcase_name='nan', value=(float('nan'),)),
      dict(testcase_name='infinite', value=(float('inf'),)),
  )
  def test_rejects_invalid_true_roi(self, value):
    with self.assertRaisesRegex(ValueError, '`true_roi`'):
      recovery.RecoveryConfig(true_roi=value, spend_scale=value)

  @parameterized.named_parameters(
      dict(testcase_name='zero', value=(0.0,)),
      dict(testcase_name='negative', value=(-1.0,)),
      dict(testcase_name='nan', value=(float('nan'),)),
      dict(testcase_name='infinite', value=(float('inf'),)),
  )
  def test_rejects_invalid_spend_scale(self, value):
    with self.assertRaisesRegex(ValueError, '`spend_scale`'):
      recovery.RecoveryConfig(true_roi=(1.0,), spend_scale=value)

  @parameterized.named_parameters(
      dict(testcase_name='zero_geos', field='n_geos', value=0),
      dict(testcase_name='negative_geos', field='n_geos', value=-1),
      dict(testcase_name='zero_times', field='n_times', value=0),
      dict(testcase_name='negative_lag', field='max_lag', value=-1),
      dict(testcase_name='zero_chains', field='n_chains', value=0),
      dict(testcase_name='negative_adapt', field='n_adapt', value=-1),
      dict(testcase_name='negative_burnin', field='n_burnin', value=-1),
      dict(testcase_name='zero_keep', field='n_keep', value=0),
  )
  def test_rejects_invalid_sampling_dimensions(self, field, value):
    with self.assertRaisesRegex(ValueError, f'`{field}`'):
      recovery.RecoveryConfig(**{field: value})

  @parameterized.named_parameters(
      dict(testcase_name='negative_alpha', field='alpha', value=-0.1),
      dict(testcase_name='nan_alpha', field='alpha', value=float('nan')),
      dict(testcase_name='negative_noise', field='noise_fraction', value=-0.1),
      dict(
          testcase_name='infinite_noise',
          field='noise_fraction',
          value=float('inf'),
      ),
      dict(
          testcase_name='zero_prior_median',
          field='prior_roi_median',
          value=0.0,
      ),
      dict(
          testcase_name='zero_prior_sigma', field='prior_roi_sigma', value=0.0
      ),
  )
  def test_rejects_invalid_numeric_settings(self, field, value):
    with self.assertRaisesRegex(ValueError, f'`{field}`'):
      recovery.RecoveryConfig(**{field: value})

  def test_allows_meaningful_zero_settings(self):
    config = recovery.RecoveryConfig(
        max_lag=0, noise_fraction=0.0, n_adapt=0, n_burnin=0
    )
    self.assertEqual(config.max_lag, 0)
    self.assertEqual(config.noise_fraction, 0.0)

  def test_rejects_unknown_response_shape(self):
    with self.assertRaisesRegex(ValueError, "must be 'linear' or 'concave'"):
      recovery.RecoveryConfig(
          response='sigmoid'
      )  # pytype: disable=wrong-arg-types

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

  def test_replications_defaults_to_one(self):
    self.assertEqual(recovery.RecoveryConfig().replications, 1)

  @parameterized.named_parameters(
      dict(testcase_name='zero', replications=0),
      dict(testcase_name='negative', replications=-3),
  )
  def test_rejects_non_positive_replications(self, replications):
    with self.assertRaisesRegex(ValueError, '`replications`'):
      recovery.RecoveryConfig(replications=replications)


class SeedDerivationTest(parameterized.TestCase):

  def test_seeds_are_deterministic_for_the_same_base_seed(self):
    first = recovery.derive_replication_seeds(7, 5)
    second = recovery.derive_replication_seeds(7, 5)
    self.assertEqual(first, second)

  def test_seeds_differ_from_each_other(self):
    seeds = recovery.derive_replication_seeds(7, 10)
    self.assertLen(set(seeds), 10)

  def test_different_base_seeds_give_different_children(self):
    self.assertNotEqual(
        recovery.derive_replication_seeds(7, 5),
        recovery.derive_replication_seeds(8, 5),
    )

  def test_seeds_are_plain_positive_ints(self):
    for seed in recovery.derive_replication_seeds(7, 3):
      self.assertIsInstance(seed, int)
      self.assertGreaterEqual(seed, 0)


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
        np.allclose(linear['revenue'].to_numpy(), concave['revenue'].to_numpy())
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

  def test_fails_when_rhat_is_not_finite(self):
    for r_hat in (float('nan'), float('inf'), float('-inf')):
      with self.subTest(r_hat=r_hat):
        result = self._result(
            [1.1, 2.1, 3.9],
            [0.8, 1.8, 3.5],
            [1.4, 2.4, 4.4],
            r_hat=r_hat,
        )
        self.assertFalse(result.converged)
        self.assertFalse(result.passed)
        self.assertIn(
            'finite rank-normalized r_hat < 1.2 threshold not met',
            result.format_report(),
        )

  def test_report_labels_rhat_as_a_threshold(self):
    result = self._result([1.1, 2.1, 3.9], [0.8, 1.8, 3.5], [1.4, 2.4, 4.4])

    report = result.format_report()

    self.assertIn('finite rank-normalized r_hat < 1.2 threshold met', report)
    self.assertNotIn('converged', report.lower())

  def test_report_uses_configured_interval_label(self):
    for level, label in ((0.8, '80% CI'), (0.925, '92.5% CI')):
      with self.subTest(level=level):
        result = dataclasses.replace(
            self._result([1.1, 2.1, 3.9], [0.8, 1.8, 3.5], [1.4, 2.4, 4.4]),
            config=recovery.RecoveryConfig(confidence_level=level),
        )
        report = result.format_report()
        self.assertIn(label, report)
        self.assertNotIn('90% CI', report)

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


class MultiRecoveryResultTest(absltest.TestCase):
  """Aggregation logic, tested against synthetic per-replication results so
  it does not need a real fit."""

  def _fake_result(self, medians, covered_los, covered_his, r_hat=1.01):
    config = recovery.RecoveryConfig()
    channels = tuple(
        recovery.ChannelRecovery(
            channel=name, true_roi=true, median=med, ci_low=lo, ci_high=hi
        )
        for name, true, med, lo, hi in zip(
            config.channels,
            config.true_roi,
            medians,
            covered_los,
            covered_his,
        )
    )
    return recovery.RecoveryResult(
        config=config, channels=channels, max_r_hat=r_hat
    )

  def _multi_result(self, results, ranks=None):
    config = dataclasses.replace(
        recovery.RecoveryConfig(), replications=len(results)
    )
    ranks = ranks or tuple((0.5, 0.5, 0.5) for _ in results)
    return recovery.MultiRecoveryResult(
        config=config,
        seeds=tuple(range(len(results))),
        replications=tuple(results),
        ranks=ranks,
    )

  def test_coverage_is_fraction_covered(self):
    # channel_0 always covers; channel_2 never does.
    results = [
        self._fake_result([1.0, 2.0, 10.0], [0.5, 1.5, 9.0], [1.5, 2.5, 9.5])
        for _ in range(4)
    ]
    multi = self._multi_result(results)
    summaries = {s.channel: s for s in multi.channel_summaries()}
    self.assertEqual(summaries['channel_0'].coverage, 1.0)
    self.assertEqual(summaries['channel_2'].coverage, 0.0)

  def test_coverage_ci_widens_with_fewer_replications(self):
    two = self._multi_result(
        [
            self._fake_result([1.0, 2.0, 4.0], [0.5, 1.5, 3.5], [1.5, 2.5, 4.5])
            for _ in range(2)
        ]
    )
    twenty = self._multi_result(
        [
            self._fake_result([1.0, 2.0, 4.0], [0.5, 1.5, 3.5], [1.5, 2.5, 4.5])
            for _ in range(20)
        ]
    )
    two_summary = two.channel_summaries()[0]
    twenty_summary = twenty.channel_summaries()[0]
    two_width = two_summary.coverage_ci_high - two_summary.coverage_ci_low
    twenty_width = (
        twenty_summary.coverage_ci_high - twenty_summary.coverage_ci_low
    )
    self.assertGreater(two_width, twenty_width)

  def test_all_converged_requires_every_replication(self):
    results = [
        self._fake_result([1.0, 2.0, 4.0], [0.5, 1.5, 3.5], [1.5, 2.5, 4.5]),
        self._fake_result(
            [1.0, 2.0, 4.0], [0.5, 1.5, 3.5], [1.5, 2.5, 4.5], r_hat=1.9
        ),
    ]
    multi = self._multi_result(results)
    self.assertFalse(multi.all_converged)
    self.assertFalse(multi.passed)

  def test_report_and_frame_cover_every_channel(self):
    results = [
        self._fake_result([1.0, 2.0, 4.0], [0.5, 1.5, 3.5], [1.5, 2.5, 4.5])
        for _ in range(3)
    ]
    multi = self._multi_result(results)
    frame = multi.to_frame()
    self.assertLen(frame, 3)
    report = multi.format_report()
    for channel in multi.config.channels:
      self.assertIn(channel, report)
    self.assertIn('replications=3', report)
    self.assertIn('VERDICT', report)
    self.assertIn(
        'met the finite max rank-normalized r_hat < 1.2 threshold', report
    )
    self.assertNotIn('CONVERGED', report)
    custom_level = dataclasses.replace(
        multi, config=dataclasses.replace(multi.config, confidence_level=0.925)
    ).format_report()
    self.assertIn('cov(92.5%)', custom_level)
    self.assertIn('nominal 92.5% interval', custom_level)

  def test_rank_fractions_are_descriptive_not_sbc(self):
    results = [
        self._fake_result([1.0, 2.0, 4.0], [0.5, 1.5, 3.5], [1.5, 2.5, 4.5])
        for _ in range(3)
    ]
    multi = self._multi_result(
        results, ranks=((0.1, 0.2, 0.3), (0.5, 0.4, 0.2), (0.9, 0.6, 0.1))
    )
    summary = multi.channel_summaries()[0]
    self.assertAlmostEqual(summary.rank_fraction_median, 0.5)
    self.assertTrue(np.isnan(summary.rank_ks_threshold))

    report = multi.format_report()
    self.assertIn('rank fraction', report)
    self.assertIn('not SBC', report)
    self.assertNotIn('SBC KS', report)


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

  def test_run_recovery_replications_produces_a_result(self):
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
        replications=2,
    )
    progress_lines = []
    multi = recovery.run_recovery_replications(
        config, progress=progress_lines.append
    )
    self.assertEqual(multi.n, 2)
    self.assertLen(multi.seeds, 2)
    self.assertNotEqual(multi.seeds[0], multi.seeds[1])
    self.assertLen(multi.ranks, 2)
    for rep_ranks in multi.ranks:
      self.assertLen(rep_ranks, 2)
      for rank in rep_ranks:
        self.assertBetween(rank, 0.0, 1.0)
    for summary in multi.channel_summaries():
      self.assertBetween(summary.coverage, 0.0, 1.0)
      self.assertBetween(summary.coverage_ci_low, 0.0, summary.coverage)
      self.assertBetween(summary.coverage_ci_high, summary.coverage, 1.0)
    # A real, non-empty progress trail, including the runtime estimate.
    self.assertTrue(progress_lines)
    self.assertIn('replication 2/2', progress_lines[-1])
    self.assertIn('r_hat', progress_lines[-1])
    self.assertIn('1.2', progress_lines[-1])
    self.assertNotIn('converged', progress_lines[-1].lower())
    self.assertIn('VERDICT', multi.format_report())


if __name__ == '__main__':
  absltest.main()
