# Copyright 2026 Sundar Ramesh Kumar.
# Licensed under the Apache License, Version 2.0.
# NOTICE: This file is new in this fork and does not exist in the original
# google/meridian source.

"""Contracts for the fresh-process recovery-study orchestrator."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest import mock

from scripts import run_recovery_study


@dataclasses.dataclass(frozen=True)
class _FakeConfig:
  true_roi: tuple[float, ...] = (1.0,)
  spend_scale: tuple[float, ...] = (1.0,)
  n_geos: int = 2
  n_times: int = 4
  max_lag: int = 0
  alpha: float = 0.6
  response: str = 'concave'
  noise_fraction: float = 0.05
  confidence_level: float = 0.9
  n_chains: int = 2
  n_adapt: int = 2
  n_burnin: int = 2
  n_keep: int = 4
  prior_roi_median: float = 2.0
  prior_roi_sigma: float = 0.7
  seed: int = 7
  replications: int = 2

  @property
  def channels(self) -> list[str]:
    return [f'channel_{i}' for i in range(len(self.true_roi))]


@dataclasses.dataclass(frozen=True)
class _FakeChannel:
  channel: str
  true_roi: float
  median: float
  ci_low: float
  ci_high: float

  @property
  def covered(self) -> bool:
    return self.ci_low <= self.true_roi <= self.ci_high

  @property
  def relative_error(self) -> float:
    return (self.median - self.true_roi) / self.true_roi


@dataclasses.dataclass(frozen=True)
class _FakeResult:
  config: _FakeConfig
  channels: tuple[_FakeChannel, ...]
  max_r_hat: float

  @property
  def converged(self) -> bool:
    return bool(math.isfinite(self.max_r_hat) and self.max_r_hat < 1.2)

  @property
  def all_covered(self) -> bool:
    return all(channel.covered for channel in self.channels)

  @property
  def ordering_recovered(self) -> bool:
    return True


@dataclasses.dataclass(frozen=True)
class _FakeSummary:
  channel: str = 'channel_0'
  n: int = 2
  median_relative_error: float = 0.1
  iqr_low: float = -0.1
  iqr_high: float = 0.2
  coverage: float = 0.5
  coverage_ci_low: float = 0.1
  coverage_ci_high: float = 0.9
  median_ci_width: float = 1.0
  rank_ks_statistic: float = 0.4
  rank_ks_threshold: float = float('nan')


class _FakeMulti:

  def __init__(self, config, seeds, replications, ranks):
    self.config = config
    self.seeds = seeds
    self.replications = replications
    self.ranks = ranks

  @property
  def n(self):
    return len(self.replications)

  @property
  def all_converged(self):
    return all(result.converged for result in self.replications)

  def channel_summaries(self):
    return (_FakeSummary(n=self.n),)

  def format_report(self):
    return 'fake aggregate report'


def _fake_recovery_module():
  module = types.SimpleNamespace(
      __file__=__file__,
      RecoveryConfig=_FakeConfig,
      ChannelRecovery=_FakeChannel,
      RecoveryResult=_FakeResult,
      MultiRecoveryResult=_FakeMulti,
      derive_replication_seeds=lambda seed, n: tuple(
          seed + 100 + i for i in range(n)
      ),
      backend=types.SimpleNamespace(
          computation_backend=lambda: types.SimpleNamespace(name='FAKE'),
          computation_precision=lambda: types.SimpleNamespace(name='FLOAT64'),
      ),
  )
  return module


def _fake_process_factory(calls, *, max_r_hat=1.05, write_result=True):

  def fake_process(command, **kwargs):
    calls.append((command, kwargs))
    index = int(command[4])
    seed = int(command[5])
    result_path = Path(command[6])
    config = json.loads(command[7])
    if write_result:
      payload = {
          'dataclass': 'meridian.validation.recovery.RecoveryResult',
          'replication_index': index,
          'seed': seed,
          'elapsed_seconds': 0.25,
          'backend': 'FAKE',
          'precision': 'FLOAT64',
          'config': config,
          'result': {
              'config': config,
              'channels': [
                  {
                      'channel': 'channel_0',
                      'true_roi': 1.0,
                      'median': 1.1,
                      'ci_low': 0.5,
                      'ci_high': 1.5,
                  }
              ],
              'max_r_hat': max_r_hat,
          },
          'ranks': [0.4],
          'draws_shape': [8, 1],
      }
      result_path.write_text(
          json.dumps(run_recovery_study._json_safe(payload), allow_nan=False)
          + '\n'
      )
    return subprocess.CompletedProcess(
        command,
        0,
        stdout=f'child log from {run_recovery_study.REPO_ROOT}\n',
        stderr='',
    )

  return fake_process


class RecoveryStudyContractTest(unittest.TestCase):

  def setUp(self):
    self.recovery = _fake_recovery_module()
    self.config = _FakeConfig()
    self.patches = mock.patch.multiple(
        run_recovery_study,
        _git_head=mock.Mock(return_value='a' * 40),
        _package_versions=mock.Mock(return_value={'fake-package': '1.0'}),
    )
    self.patches.start()
    self.addCleanup(self.patches.stop)

  def test_json_serialization_maps_nonfinite_values_to_null(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      path = Path(temp_dir) / 'values.json'
      run_recovery_study._write_json(
          path,
          {
              'nan': float('nan'),
              'positive_inf': float('inf'),
              'nested': [float('-inf')],
          },
      )
      text = path.read_text()
      parsed = json.loads(text)
    self.assertIsNone(parsed['nan'])
    self.assertIsNone(parsed['positive_inf'])
    self.assertIsNone(parsed['nested'][0])
    self.assertNotIn('NaN', text)

  def test_cli_exposes_reproduction_shape_and_sampling_options(self):
    args = run_recovery_study.parse_args(
        [
            '--replications',
            '10',
            '--seed',
            '7',
            '--output-dir',
            '/tmp/study',
            '--n-geos',
            '5',
            '--n-times',
            '104',
            '--max-lag',
            '0',
            '--response',
            'concave',
            '--n-chains',
            '4',
            '--n-adapt',
            '1000',
            '--n-burnin',
            '500',
            '--n-keep',
            '1000',
        ]
    )
    self.assertEqual(args.replications, 10)
    self.assertEqual(args.seed, 7)
    self.assertEqual(args.n_geos, 5)
    self.assertEqual(args.n_times, 104)
    self.assertEqual(args.max_lag, 0)
    self.assertEqual(args.response, 'concave')
    self.assertEqual(args.n_chains, 4)
    self.assertEqual(args.n_adapt, 1000)
    self.assertEqual(args.n_burnin, 500)
    self.assertEqual(args.n_keep, 1000)

  def test_each_seed_runs_in_fresh_process_and_bundle_is_safe_json(self):
    calls = []
    fake_process = _fake_process_factory(calls)
    with tempfile.TemporaryDirectory() as temp_dir:
      output_dir = Path(temp_dir) / 'study'
      aggregate = run_recovery_study.run_study(
          self.config,
          output_dir,
          recovery_module=self.recovery,
          python_executable='fake-python',
          run_process=fake_process,
      )
      self.assertEqual(len(calls), 2)
      self.assertEqual(aggregate['seeds'], [107, 108])
      self.assertTrue(aggregate['r_hat_screening']['all_replications_pass'])
      self.assertIn(
          'screening check only', aggregate['r_hat_screening']['description']
      )
      self.assertIsNone(aggregate['channel_summaries'][0]['rank_ks_threshold'])
      self.assertIn('not simulation-based calibration', aggregate['rank_note'])

      for index, (command, kwargs) in enumerate(calls, 1):
        self.assertEqual(command[0], 'fake-python')
        self.assertEqual(command[1], '-u')
        self.assertEqual(
            command[2], str(Path(run_recovery_study.__file__).resolve())
        )
        self.assertEqual(command[3], '--child')
        self.assertEqual(int(command[4]), index)
        self.assertEqual(int(command[5]), 106 + index)
        child_config = json.loads(command[7])
        self.assertEqual(child_config['replications'], 1)
        self.assertEqual(child_config['seed'], 106 + index)
        self.assertEqual(kwargs['cwd'], run_recovery_study.REPO_ROOT)
        self.assertEqual(kwargs['env']['MLFLOW_DISABLE_AGENT_HINT'], '1')

      metadata = json.loads((output_dir / 'metadata.json').read_text())
      metadata_text = json.dumps(metadata)
      self.assertNotIn(str(run_recovery_study.REPO_ROOT), metadata_text)
      self.assertNotIn(str(output_dir), metadata_text)
      self.assertEqual(metadata['derived_seeds'], [107, 108])
      self.assertTrue(metadata['fresh_subprocess_per_replication'])
      self.assertEqual(metadata['package_versions'], {'fake-package': '1.0'})
      self.assertEqual(metadata['status'], 'complete')

      log_text = (output_dir / 'replication-001.stdout.log').read_text()
      self.assertNotIn(str(run_recovery_study.REPO_ROOT), log_text)
      self.assertIn('$REPO_ROOT', log_text)
      manifest = json.loads((output_dir / 'manifest.json').read_text())
      self.assertIn('aggregate.json', manifest['sha256'])
      self.assertTrue((output_dir / 'replication-001.json').is_file())
      self.assertTrue((output_dir / 'replication-002.json').is_file())

  def test_nonfinite_rhat_is_null_and_screen_fails_without_abort(self):
    calls = []
    fake_process = _fake_process_factory(calls, max_r_hat=float('nan'))
    with tempfile.TemporaryDirectory() as temp_dir:
      aggregate = run_recovery_study.run_study(
          self.config,
          Path(temp_dir),
          recovery_module=self.recovery,
          run_process=fake_process,
      )
      payload = json.loads(
          (Path(temp_dir) / 'replication-001.json').read_text()
      )
      aggregate_json = json.loads(
          (Path(temp_dir) / 'aggregate.json').read_text()
      )
    self.assertIsNone(payload['result']['max_r_hat'])
    self.assertFalse(aggregate['r_hat_screening']['all_replications_pass'])
    self.assertIsNone(aggregate_json['replications'][0]['max_r_hat'])

  def test_child_failure_is_propagated_and_partial_metadata_is_retained(self):
    calls = []

    def failed_process(command, **kwargs):
      calls.append((command, kwargs))
      return subprocess.CompletedProcess(command, 17, stdout='', stderr='boom')

    with tempfile.TemporaryDirectory() as temp_dir:
      output_dir = Path(temp_dir)
      with self.assertRaisesRegex(RuntimeError, r'replication 1.*exit 17'):
        run_recovery_study.run_study(
            self.config,
            output_dir,
            recovery_module=self.recovery,
            run_process=failed_process,
        )
      metadata = json.loads((output_dir / 'metadata.json').read_text())
      self.assertEqual(metadata['status'], 'failed')
      self.assertIn('exit 17', metadata['failure'])
      self.assertEqual(
          (output_dir / 'replication-001.stderr.log').read_text(), 'boom'
      )
      self.assertEqual(len(calls), 1)
      self.assertFalse((output_dir / 'aggregate.json').exists())
      self.assertFalse((output_dir / 'manifest.json').exists())

  def test_existing_study_artifacts_are_preserved_and_refused(self):
    calls = []
    fake_process = _fake_process_factory(calls)
    with tempfile.TemporaryDirectory() as temp_dir:
      output_dir = Path(temp_dir)
      sentinel = output_dir / 'aggregate.json'
      sentinel.write_text('{"old": true}\n')
      with self.assertRaisesRegex(RuntimeError, r'choose a new --output-dir'):
        run_recovery_study.run_study(
            self.config,
            output_dir,
            recovery_module=self.recovery,
            run_process=fake_process,
        )
      self.assertEqual(sentinel.read_text(), '{"old": true}\n')
      self.assertEqual(calls, [])


if __name__ == '__main__':
  unittest.main(verbosity=2)
