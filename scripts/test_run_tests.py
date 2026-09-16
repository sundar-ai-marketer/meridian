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
    status, run = self.run_with_codes([1] + [0] * 8)
    self.assertEqual(status, 1)
    self.assertEqual(run.call_count, 9)
    self.assertTrue(
        all(
            call.kwargs['cwd'] == run_tests.REPO_ROOT
            for call in run.call_args_list
        )
    )

  def test_empty_collection_is_failure(self):
    status, _ = self.run_with_codes([0] * 8 + [5])
    self.assertEqual(status, 1)

  def test_cancellation_does_not_start_more_fits(self):
    status, run = self.run_with_codes([0, -2])
    self.assertEqual(status, 130)
    self.assertEqual(run.call_count, 2)

  def test_collection_error_does_not_hide_other_phases(self):
    status, run = self.run_with_codes([2] + [0] * 8)
    self.assertEqual(status, 1)
    self.assertEqual(run.call_count, 9)

  def test_all_phases_pass(self):
    status, _ = self.run_with_codes([0] * 9)
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


if __name__ == '__main__':
  unittest.main()
