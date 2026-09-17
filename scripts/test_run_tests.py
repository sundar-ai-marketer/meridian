# Copyright 2026 Sundar Ramesh Kumar.
# Licensed under the Apache License, Version 2.0.
# NOTICE: This file is new in this fork; see NOTICE at the repository root.

"""Failure propagation and cancellation contracts for the full-suite runner."""

import contextlib
import io
import subprocess
import unittest
from unittest import mock

from scripts import run_tests


class TestRunnerContractTest(unittest.TestCase):

  def run_with_codes(self, codes):
    results = [subprocess.CompletedProcess([], code) for code in codes]
    with mock.patch.object(
        run_tests.subprocess, 'run', side_effect=results
    ) as run:
      with contextlib.redirect_stdout(io.StringIO()):
        status = run_tests.main(['--workers', '1'])
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

  def test_invalid_worker_counts_fail_before_running(self):
    for value in ('0', '-1', 'auto', '1.5'):
      with self.subTest(value=value):
        with mock.patch.object(run_tests.subprocess, 'run') as run:
          with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
              run_tests.main(['--workers', value])
          self.assertEqual(error.exception.code, 2)
          run.assert_not_called()

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
