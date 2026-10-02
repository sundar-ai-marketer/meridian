#!/usr/bin/env python3
# Copyright 2026 Sundar Ramesh Kumar.
# Licensed under the Apache License, Version 2.0.
# NOTICE: This file is new in this fork; see NOTICE at the repository root.

"""Run the repository's Python test suite with bounded worker memory.

Covers every test file under ``meridian/``, ``scenarioplanner/``, and
``scripts/`` — see ``test_phases()`` for the handful of ``scripts/`` files
that are deliberately excluded because CI runs them directly as their own
steps. ``scripts/test_phase_coverage.py`` enforces that nothing under the
three roots is silently missed by every phase below.

Each phase gets fresh processes so numerical compilation caches are released.
Fit-heavy suites run serially to avoid retaining duplicate model fits. A failed
phase does not hide results from later phases; the final exit status is nonzero
if any phase fails. Run from any directory with the desired environment's Python.
"""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from scripts.evidence import provenance, write_evidence

FIT_SUITES = (
    'meridian/benchmark',
    'meridian/validation',
    'meridian/math_invariants_test.py',
    'meridian/upstream_issues_test.py',
    # Runs a handful of real (small) MCMC fits to compare GPU/CPU numerics;
    # measured at ~50s, in line with the other suites above -- worth its own
    # serial phase so parallel workers don't hold multiple fits in memory at
    # once, same reasoning as the meridian suites.
    'scripts/test_gpu_validation.py',
)
# These scripts/test_*.py files are intentionally not part of the phase
# partition below, and scripts/test_phase_coverage.py knows to exclude them
# too. Each is run directly as its own CI step, not as a pytest phase:
# test_end_to_end.py and test_report_browser.py take CLI arguments (an
# --output-dir, a report path) the phases don't supply; test_mlflow_integration
# .py stands up a real local MLflow server; test_proto_build.py runs earlier,
# in the build-distributions job, before the full Meridian install these
# phases assume. None of the first three even defines a unittest.TestCase --
# pytest collects zero tests from them today -- and test_proto_build.py's
# TestCase would just duplicate the build-distributions job's own step.
STANDALONE_SCRIPTS = (
    'scripts/test_end_to_end.py',
    'scripts/test_report_browser.py',
    'scripts/test_mlflow_integration.py',
    'scripts/test_proto_build.py',
)


def test_phases(workers: int) -> list[tuple[str, list[str]]]:
  """Partition meridian, scenarioplanner, and scripts without exclusions.

  Every test file under the three roots is collected by exactly one phase
  below, except for the CI-owns-it-directly files listed in
  ``STANDALONE_SCRIPTS`` (see the comment there for why each is excluded).
  """
  parallel = ['-q', '-n', str(workers), '--dist=worksteal']
  return [
      ('analyzer', parallel + ['meridian/analysis/analyzer_test.py']),
      ('optimizer', parallel + ['meridian/analysis/optimizer_test.py']),
      (
          'analysis',
          parallel
          + [
              'meridian/analysis',
              '--ignore=meridian/analysis/analyzer_test.py',
              '--ignore=meridian/analysis/optimizer_test.py',
          ],
      ),
      ('model', parallel + ['meridian/model']),
      (
          'remaining',
          parallel
          + [
              'meridian',
              'scenarioplanner',
              '--ignore=meridian/analysis',
              '--ignore=meridian/model',
              *(f'--ignore={suite}' for suite in FIT_SUITES),
          ],
      ),
      (
          'scripts',
          parallel
          + [
              'scripts',
              *(
                  f'--ignore={suite}'
                  for suite in FIT_SUITES
                  if suite.startswith('scripts/')
              ),
              *(f'--ignore={path}' for path in STANDALONE_SCRIPTS),
          ],
      ),
      *((suite, ['-q', suite]) for suite in FIT_SUITES),
  ]


def positive_integer(value: str) -> int:
  try:
    result = int(value)
  except ValueError as error:
    raise argparse.ArgumentTypeError(
        'workers must be a positive integer'
    ) from error
  if result < 1:
    raise argparse.ArgumentTypeError('workers must be a positive integer')
  return result


def _phase_status(returncode: int) -> str:
  if returncode == 0:
    return 'passed'
  if returncode < 0:
    return 'interrupted'
  return 'failed'


def _write_report(
    path: Path,
    *,
    workers: int,
    phases: list[tuple[str, list[str]]],
    phase_results: list[dict[str, object]],
    started: float,
    exit_code: int,
    all_phases_run: bool,
) -> None:
  """Writes what ran and its outcome, including partial interrupted suites."""
  output_path = path if path.is_absolute() else REPO_ROOT / path
  statuses = [str(result['status']) for result in phase_results]
  if not all_phases_run:
    suite_status = 'interrupted'
  elif any(status == 'failed' for status in statuses):
    suite_status = 'failed'
  else:
    suite_status = 'passed'
  requested_backend = os.environ.get('MERIDIAN_BACKEND') or 'project default'
  write_evidence(
      output_path,
      {
          'date': datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
          'provenance': provenance(Path(__file__).resolve()),
          'scope': {
              'backend_requested': requested_backend,
              'workers': workers,
              'phase_count': len(phases),
              'phase_names': [name for name, _ in phases],
          },
          'suite_status': suite_status,
          'all_phases_run': all_phases_run,
          'phases_attempted': len(phase_results),
          'exit_code': exit_code,
          'elapsed_seconds': round(time.monotonic() - started, 3),
          'phase_results': phase_results,
      },
  )


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--workers',
      type=positive_integer,
      default=2,
      help='Parallel workers per non-fit phase (default: 2; use 1 for less RAM).',
  )
  parser.add_argument(
      '--output',
      type=Path,
      help='Write machine-readable suite evidence to this JSON path.',
  )
  args = parser.parse_args(argv)
  phases = test_phases(args.workers)
  phase_results: list[dict[str, object]] = []
  started = time.monotonic()
  all_phases_run = True
  exit_code = 0
  for name, pytest_args in phases:
    print(f'\n=== {name} ===', flush=True)
    phase_started = time.monotonic()
    try:
      result = subprocess.run(
          [sys.executable, '-m', 'pytest', *pytest_args],
          cwd=REPO_ROOT,
          check=False,
      )
    except KeyboardInterrupt:
      phase_results.append(
          {
              'name': name,
              'exit_code': 130,
              'status': 'interrupted',
              'elapsed_seconds': round(time.monotonic() - phase_started, 3),
          }
      )
      print(
          f'Interrupted during {name}; remaining phases were not run.',
          flush=True,
      )
      all_phases_run = False
      exit_code = 130
      break
    phase_results.append(
        {
            'name': name,
            'exit_code': result.returncode,
            'status': _phase_status(result.returncode),
            'elapsed_seconds': round(time.monotonic() - phase_started, 3),
        }
    )
    if result.returncode < 0:
      # Respect cancellation rather than starting more expensive fits.
      print(
          f'Interrupted during {name}; remaining phases were not run.',
          flush=True,
      )
      all_phases_run = False
      exit_code = 130
      break
  if all_phases_run:
    exit_code = int(
        any(result['exit_code'] != 0 for result in phase_results)
    )
  suite_status = 'interrupted' if not all_phases_run else (
      'failed' if exit_code else 'passed'
  )
  print(
      f'\nSuite {suite_status} after {time.monotonic() - started:.1f}s:',
      flush=True,
  )
  for result in phase_results:
    print(
        f"  {result['name']}: {result['status']} "
        f"(exit {result['exit_code']}, {result['elapsed_seconds']:.1f}s)",
        flush=True,
    )
  if args.output:
    _write_report(
        args.output,
        workers=args.workers,
        phases=phases,
        phase_results=phase_results,
        started=started,
        exit_code=exit_code,
        all_phases_run=all_phases_run,
    )
  return exit_code


if __name__ == '__main__':
  try:
    raise SystemExit(main())
  except KeyboardInterrupt:
    raise SystemExit(130) from None
