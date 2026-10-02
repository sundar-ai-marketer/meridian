#!/usr/bin/env python3
# Copyright 2026 Sundar Ramesh Kumar.
# Licensed under the Apache License, Version 2.0.
# NOTICE: This file is new in this fork; see NOTICE at the repository root.

"""Package existing native-report browser evidence without executing a browser.

The supporting bundle retains raw measured results, the saved orchestration
script and its log unchanged. One representative JAX optimization report used
its already accepted check; the orchestrator did not repeat that browser run.
This producer records collection provenance, not a new execution of the checks.
"""

import argparse
import datetime
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))
from scripts.evidence import provenance, sha256, write_evidence  # pylint: disable=wrong-import-position


def collect(bundle: Path, output: Path) -> dict:
  """Hash the retained inputs and preserve the raw measured results exactly."""
  raw = bundle / 'raw-results.json'
  results = json.loads(raw.read_text(encoding='utf-8'))
  if results.get('status') != 'PASS' or not results.get('reports'):
    raise ValueError('Expected the successful existing browser result bundle.')
  for report in results['reports']:
    if report.get('status') != 'PASS' or report.get('network') != 'disabled':
      raise ValueError('Each retained report must identify its offline result.')
  files = {
      'raw_results': raw,
      'saved_orchestrator': bundle / 'orchestrator.py',
      'orchestration_log': bundle / 'run.log',
      'frozen_browser_check': _ROOT / 'scripts/test_report_browser.py',
  }
  evidence = {}
  for name, path in files.items():
    resolved = path.resolve()
    evidence[name] = {
        'path': str(resolved.relative_to(_ROOT)),
        'sha256': sha256(resolved),
    }
  receipt = {
      'date': datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
      'provenance': provenance(Path(__file__)),
      'scope': {
          'operation': 'Collect existing browser results; do not rerun checks.',
          'browser_execution_by_this_producer': False,
          'accepted_check_reused': 'JAX geo_nonrevenue_media optimization report',
          'coverage': 'Offline Chromium; saved synthetic integration-only fits.',
          'native_formats': 'Browser SVG and editable Vega-Lite JSON.',
          'limitations': (
              'Not Office chart objects, field performance, universal browser '
              'support, or full accessibility certification.'
          ),
      },
      'retained_evidence': evidence,
      'results': results,
  }
  write_evidence(output, receipt)
  return receipt


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--bundle', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  collect(args.bundle, args.output)


if __name__ == '__main__':
  main()
