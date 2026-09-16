#!/usr/bin/env python3
# Copyright 2026 Sundar Ramesh Kumar.
# Licensed under the Apache License, Version 2.0.
# NOTICE: This file is new in this fork; see NOTICE at the repository root.

"""Run the complete scientific suite with bounded worker memory.

Each phase gets fresh processes so numerical compilation caches are released.
Fit-heavy suites run serially to avoid retaining duplicate model fits. A failed
phase does not hide results from later phases; the final exit status is nonzero
if any phase fails. Run from any directory with the desired environment's Python.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
FIT_SUITES = (
    'meridian/benchmark',
    'meridian/validation',
    'meridian/math_invariants_test.py',
    'meridian/upstream_issues_test.py',
)


def test_phases(workers: int) -> list[tuple[str, list[str]]]:
  """Partition the two packages without excluding any test cases."""
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


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--workers',
      type=positive_integer,
      default=2,
      help='Parallel workers per non-fit phase (default: 2; use 1 for less RAM).',
  )
  args = parser.parse_args(argv)
  results = []
  started = time.monotonic()
  for name, pytest_args in test_phases(args.workers):
    print(f'\n=== {name} ===', flush=True)
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', *pytest_args],
        cwd=REPO_ROOT,
        check=False,
    )
    results.append((name, result.returncode))
    if result.returncode < 0:
      # Respect cancellation rather than starting more expensive fits.
      print(
          f'Interrupted during {name}; remaining phases were not run.',
          flush=True,
      )
      return 130
  print(f'\nSuite completed in {time.monotonic() - started:.1f}s:', flush=True)
  for name, status in results:
    print(
        f'  {name}: {"PASS" if status == 0 else f"FAIL (exit {status})"}',
        flush=True,
    )
  return int(any(status != 0 for _, status in results))


if __name__ == '__main__':
  try:
    raise SystemExit(main())
  except KeyboardInterrupt:
    raise SystemExit(130) from None
