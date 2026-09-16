# Copyright 2026 Sundar Ramesh Kumar.
# Licensed under the Apache License, Version 2.0.
# NOTICE: This file is new in this fork; see NOTICE at the repository root.

"""Offline asset completeness, integrity, and path-boundary regressions."""

import base64
from pathlib import Path
from unittest import mock

from absl.testing import absltest
from meridian.templates import formatter
from meridian.templates import report_assets


class ReportAssetsTest(absltest.TestCase):

  def test_all_packaged_assets_match_recorded_hashes(self):
    for name in report_assets._manifest():
      with self.subTest(name=name):
        content = report_assets._read_asset(name)
        self.assertTrue(content)
        encoded = report_assets.data_uri(name).split(',', 1)[1]
        self.assertEqual(base64.b64decode(encoded), content)

  def test_unknown_and_parent_paths_are_refused(self):
    for name in ('../formatter.py', '/etc/passwd', 'missing.js'):
      with self.subTest(name=name):
        with self.assertRaises(ValueError):
          report_assets.data_uri(name)

  def test_corrupted_asset_prevents_report_generation(self):
    with mock.patch.object(Path, 'read_bytes', return_value=b'corrupted'):
      with self.assertRaisesRegex(RuntimeError, 'integrity check'):
        report_assets._read_asset('vega-6.4.0.min.js')

  def test_report_carries_assets_and_their_license_text(self):
    html = formatter.create_summary_html(
        formatter.create_template_env(), 'Offline', []
    )
    self.assertEqual(html.count('src="data:text/javascript;base64,'), 3)
    self.assertIn('data:font/otf;base64,', html)
    self.assertNotIn('src="https://', html)
    self.assertNotIn('href="https://fonts.', html)
    self.assertIn('University of Washington Interactive Data Lab', html)
    self.assertIn('Apache License', html)


if __name__ == '__main__':
  absltest.main()
