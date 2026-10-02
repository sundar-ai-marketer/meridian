# Copyright 2026 Sundar Ramesh Kumar.
# Licensed under the Apache License, Version 2.0.
# NOTICE: This file is new in this fork; see NOTICE at the repository root.

"""Failure propagation and cancellation contracts for the full-suite runner."""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_tests


class TestRunnerContractTest(unittest.TestCase):

  def run_with_codes(self, codes, extra_args=()):
    results = [subprocess.CompletedProcess([], code) for code in codes]
    with mock.patch.object(
        run_tests.subprocess, 'run', side_effect=results
    ) as run:
      with contextlib.redirect_stdout(io.StringIO()):
        status = run_tests.main(['--workers', '1', *extra_args])
    return status, run

  def test_failed_phase_does_not_hide_later_results(self):
    status, run = self.run_with_codes([1] + [0] * 10)
    self.assertEqual(status, 1)
    self.assertEqual(run.call_count, 11)
    self.assertTrue(
        all(
            call.kwargs['cwd'] == run_tests.REPO_ROOT
            for call in run.call_args_list
        )
    )

  def test_empty_collection_is_failure(self):
    status, _ = self.run_with_codes([0] * 10 + [5])
    self.assertEqual(status, 1)

  def test_cancellation_does_not_start_more_fits(self):
    status, run = self.run_with_codes([0, -2])
    self.assertEqual(status, 130)
    self.assertEqual(run.call_count, 2)

  def test_collection_error_does_not_hide_other_phases(self):
    status, run = self.run_with_codes([2] + [0] * 10)
    self.assertEqual(status, 1)
    self.assertEqual(run.call_count, 11)

  def test_all_phases_pass(self):
    status, _ = self.run_with_codes([0] * 11)
    self.assertEqual(status, 0)

  def test_output_report_records_scope_and_phase_failures(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      report_path = Path(temp_dir) / 'suite.json'
      with mock.patch.object(
          run_tests, 'provenance', return_value={'script_sha256': 'abc123'}
      ), mock.patch.dict('os.environ', {'MERIDIAN_BACKEND': 'tensorflow'}):
        status, run = self.run_with_codes(
            [1] + [0] * 10, ['--output', str(report_path)]
        )

      report = json.loads(report_path.read_text(encoding='utf-8'))
      self.assertEqual(status, 1)
      self.assertEqual(run.call_count, 11)
      self.assertEqual(report['provenance']['script_sha256'], 'abc123')
      self.assertEqual(report['scope']['backend_requested'], 'tensorflow')
      self.assertTrue(report['all_phases_run'])
      self.assertEqual(report['suite_status'], 'failed')
      self.assertEqual(report['phase_results'][0]['exit_code'], 1)
      self.assertEqual(report['phase_results'][0]['status'], 'failed')
      self.assertEqual(report['phases_attempted'], 11)

  def test_interrupted_output_report_does_not_claim_full_suite(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      report_path = Path(temp_dir) / 'partial.json'
      with mock.patch.object(
          run_tests, 'provenance', return_value={'script_sha256': 'abc123'}
      ):
        status, run = self.run_with_codes(
            [0, -2], ['--output', str(report_path)]
        )

      report = json.loads(report_path.read_text(encoding='utf-8'))
      self.assertEqual(status, 130)
      self.assertEqual(run.call_count, 2)
      self.assertFalse(report['all_phases_run'])
      self.assertEqual(report['suite_status'], 'interrupted')
      self.assertEqual(report['phase_results'][-1]['status'], 'interrupted')
      self.assertEqual(report['phases_attempted'], 2)

  def test_keyboard_interrupt_writes_partial_report(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      report_path = Path(temp_dir) / 'interrupted.json'
      with mock.patch.object(
          run_tests, 'provenance', return_value={'script_sha256': 'abc123'}
      ), mock.patch.object(
          run_tests.subprocess, 'run', side_effect=KeyboardInterrupt
      ), contextlib.redirect_stdout(io.StringIO()):
        status = run_tests.main(
            ['--workers', '1', '--output', str(report_path)]
        )

      report = json.loads(report_path.read_text(encoding='utf-8'))
      self.assertEqual(status, 130)
      self.assertFalse(report['all_phases_run'])
      self.assertEqual(report['suite_status'], 'interrupted')
      self.assertEqual(report['phase_results'][0]['exit_code'], 130)

  def test_invalid_worker_counts_fail_before_running(self):
    for value in ('0', '-1', 'auto', '1.5'):
      with self.subTest(value=value):
        with mock.patch.object(run_tests.subprocess, 'run') as run:
          with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
              run_tests.main(['--workers', value])
          self.assertEqual(error.exception.code, 2)
          run.assert_not_called()

  def test_help_works_when_invoked_outside_the_checkout(self):
    script_path = Path(run_tests.__file__).resolve()
    with tempfile.TemporaryDirectory() as temp_dir:
      result = subprocess.run(
          [sys.executable, str(script_path), '--help'],
          cwd=temp_dir,
          capture_output=True,
          check=False,
          text=True,
      )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn('--output', result.stdout)

  def test_scripts_is_covered_by_a_named_phase(self):
    names = [name for name, _ in run_tests.test_phases(2)]
    self.assertIn('scripts', names)

  def test_scripts_phase_ignores_the_standalone_scripts(self):
    phases = dict(run_tests.test_phases(2))
    for path in run_tests.STANDALONE_SCRIPTS:
      self.assertIn(f'--ignore={path}', phases['scripts'])

  def test_scripts_phase_does_not_rerun_its_own_fit_suite(self):
    phases = dict(run_tests.test_phases(2))
    for suite in run_tests.FIT_SUITES:
      if suite.startswith('scripts/'):
        self.assertIn(f'--ignore={suite}', phases['scripts'])

  def test_every_fit_suite_gets_its_own_serial_phase(self):
    phases = dict(run_tests.test_phases(2))
    for suite in run_tests.FIT_SUITES:
      self.assertEqual(phases[suite], ['-q', suite])


if __name__ == '__main__':
  unittest.main()
