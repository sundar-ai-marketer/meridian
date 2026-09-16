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

"""Quickstart contracts that are inexpensive enough for the normal suite."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import pathlib
import tempfile
from unittest import mock

from absl.testing import absltest
import arviz as az
from meridian import constants as c
from meridian.analysis import sampling_quality
import numpy as np
import pandas as pd

_QUICKSTART_PATH = (
    pathlib.Path(__file__).parents[2] / 'examples' / 'quickstart.py'
)
_SPEC = importlib.util.spec_from_file_location(
    'meridian_quickstart', _QUICKSTART_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
quickstart = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(quickstart)


class _Serde:

  def __init__(self):
    self.saved = []

  def save_meridian(self, meridian, path):
    del meridian
    pathlib.Path(path).write_bytes(b'fitted sample evidence')
    self.saved.append(path)


class QuickstartTest(absltest.TestCase):

  def test_build_input_data_uses_eight_observed_media_history_periods(self):
    source = pd.read_csv(quickstart.SAMPLE_CSV)
    data = quickstart.build_quickstart_input_data(source)
    source_times = np.sort(source['time'].unique())

    self.assertLen(data.kpi.coords[c.TIME], 148)
    self.assertLen(data.media.coords[c.MEDIA_TIME], 156)
    self.assertLen(data.media_spend.coords[c.TIME], 148)
    np.testing.assert_array_equal(
        data.kpi.coords[c.TIME].values,
        source_times[quickstart.MEDIA_HISTORY_PERIODS :],
    )
    np.testing.assert_array_equal(
        data.media.coords[c.MEDIA_TIME].values, source_times
    )
    np.testing.assert_array_equal(
        data.media_spend.coords[c.TIME].values,
        source_times[quickstart.MEDIA_HISTORY_PERIODS :],
    )

  def test_parser_rejects_invalid_sampling_mode_and_probability(self):
    parser = quickstart.build_parser()
    for argv, expected in (
        (['--sampling-mode', 'relaxed'], 'invalid choice'),
        (['--target-accept-prob', '1'], 'open interval'),
        (['--n-chains', '0'], 'positive integer'),
        (['--n-burnin', '-1'], 'non-negative integer'),
        (['--n-keep', '0'], 'positive integer'),
    ):
      stderr = io.StringIO()
      with (
          contextlib.redirect_stderr(stderr),
          self.assertRaises(SystemExit) as error,
      ):
        parser.parse_args(argv)
      self.assertEqual(error.exception.code, 2)
      self.assertIn(expected, stderr.getvalue())

  def test_strict_failure_keeps_sampling_evidence_and_blocks_optimizer(self):
    report = sampling_quality.assess_sampling_quality(
        az.from_dict(
            posterior={'theta': np.ones((4, 20))},
            sample_stats={'diverging': np.zeros((4, 20), dtype=bool)},
        )
    )
    self.assertFalse(report.sampling_passed)
    serde = _Serde()
    optimizer = mock.Mock()
    with tempfile.TemporaryDirectory() as temp_dir:
      out_dir = pathlib.Path(temp_dir)
      model_path, report_path = quickstart.save_sampling_evidence(
          object(), report, out_dir=out_dir, meridian_serde=serde
      )
      stdout = io.StringIO()
      with contextlib.redirect_stdout(stdout):
        if quickstart.allow_decision_outputs(report, 'strict'):
          optimizer()

      self.assertEqual(serde.saved, [str(model_path)])
      self.assertTrue(model_path.is_file())
      self.assertTrue(report_path.is_file())
      self.assertFalse(json.loads(report_path.read_text())['sampling_passed'])
      optimizer.assert_not_called()
      self.assertIn('STRICT SAMPLING GATE FAILED', stdout.getvalue())
      self.assertIn('skipped', stdout.getvalue())

  def test_strict_rerun_refuses_stale_decision_html_before_sampling(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      out_dir = pathlib.Path(temp_dir)
      summary = out_dir / 'summary.html'
      summary.write_text('prior decision output', encoding='utf-8')
      stderr = io.StringIO()
      with (
          contextlib.redirect_stderr(stderr),
          self.assertRaises(SystemExit) as error,
      ):
        quickstart.main(['--strict', '--output-dir', str(out_dir)])

      self.assertEqual(error.exception.code, 2)
      self.assertEqual(
          summary.read_text(encoding='utf-8'), 'prior decision output'
      )
      self.assertFalse((out_dir / 'model.binpb').exists())
      self.assertIn('refuses an output directory', stderr.getvalue())

  def test_exploratory_status_is_explicit_in_console_and_html_note(self):
    report = sampling_quality.assess_sampling_quality(
        az.from_dict(
            posterior={'theta': np.ones((4, 20))},
            sample_stats={'diverging': np.zeros((4, 20), dtype=bool)},
        )
    )
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
      allowed = quickstart.allow_decision_outputs(report, 'exploratory')

    self.assertTrue(allowed)
    self.assertIn('EXPLORATORY DEMO', stdout.getvalue())
    self.assertIn(
        'not decision-grade', quickstart.report_note_for('exploratory')
    )
    self.assertIn(
        'Sampling checks passed', quickstart.report_note_for('strict')
    )

  def test_explicit_sampling_arguments_are_validated(self):
    args = quickstart.build_parser().parse_args(
        [
            '--strict',
            '--knots',
            '13',
            '--n-adapt',
            '2000',
            '--n-chains',
            '4',
            '--n-burnin',
            '500',
            '--n-keep',
            '4000',
            '--target-accept-prob',
            '0.95',
        ]
    )

    self.assertEqual(args.sampling_mode, 'strict')
    self.assertEqual(args.knots, 13)
    self.assertEqual(args.n_adapt, 2000)
    self.assertEqual(args.n_chains, 4)
    self.assertEqual(args.n_burnin, 500)
    self.assertEqual(args.n_keep, 4000)
    self.assertAlmostEqual(args.target_accept_prob, 0.95)


if __name__ == '__main__':
  absltest.main()
