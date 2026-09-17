# Copyright 2026 Sundar Ramesh Kumar.
# Licensed under the Apache License, Version 2.0.
# NOTICE: This file is new in this fork; see NOTICE at the repository root.

"""Guardrail: every real test file is collected by exactly one run_tests.py phase.

Before this test existed, two independent test-selection mechanisms existed --
``run_tests.py``'s ``test_phases()`` and a hand-maintained list of module names
embedded in ``.github/workflows/ci.yml`` -- and neither one knew about the
other or about what test files actually exist on disk. A new
``scripts/test_*.py`` file could be silently uncovered by both. This test
makes that failure mode loud: it fails and names the exact file.

Approach and measured runtime: a naive version of this test would run
``pytest --collect-only`` once per phase (11 phases today), each in its own
subprocess so it can import cleanly. Each subprocess re-imports this
package, which pulls in JAX and TensorFlow; measured standalone, that import
plus collection over all of ``scripts/`` alone already takes ~7s, so 11
separate subprocesses would likely land close to or over the ~90s budget a
guardrail test needs to stay under to not get disabled. Instead, this test
calls ``pytest --collect-only`` exactly **once**, over the union of the three
roots, and gets the ground-truth set of files pytest actually collects
something from. Measured: this single combined call takes about 9s. Which
*phase* each of those files belongs to is then computed by matching each
phase's own target/``--ignore`` arguments against that same file list, in
pure Python -- no second pytest invocation, and in particular no in-process
``pytest.main()`` call, which would be unsafe here: this very file is itself
collected and run by pytest as part of the "scripts" phase (possibly inside
an xdist worker), and pytest does not support being invoked reentrantly from
inside its own already-running process. A fresh subprocess for the one
combined collection avoids that entirely, matching how ``run_tests.py``
itself always shells out to a new "python -m pytest" process per phase.

Total measured runtime of this whole test file: see the docstring on
`PhaseCoverageTest` -- comfortably under a second beyond the ~9s collection
call, i.e. well inside the ~90s budget.
"""

from __future__ import annotations

import re
import subprocess
import sys
import unittest

from scripts import run_tests

REPO_ROOT = run_tests.REPO_ROOT
ROOTS = ('meridian', 'scenarioplanner', 'scripts')

# Matches the file part of a `path/to/file.py::TestClass::test_method[id]`
# collection nodeid, as printed by `pytest --collect-only -q`. Deliberately
# requires the literal "::" pytest uses to separate the file from the test
# id, so it never matches an unrelated `path.py:123:` warning line (single
# colon) that also shows up in collect-only output.
_NODEID_FILE = re.compile(r'^([^\s:]+\.py)::')


def _is_fixture_stem(stem: str) -> bool:
  """True if `stem` (a filename without ".py") names a fixture, not a test.

  This repo names test files two ways -- `*_test.py` and `test_*.py` -- and
  a handful of files match one of those globs but hold shared fixtures other
  tests import, not tests of their own: meridian/analysis/test_utils.py,
  meridian/data/test_utils.py, meridian/backend/test_utils.py,
  meridian/model/model_test_data.py, meridian/schema/test_data.py,
  meridian/schema/serde/test_data.py, and
  scenarioplanner/converters/test_data.py. All seven share one shape: the
  stem is exactly "test_utils", exactly "test_data", or ends in
  "_test_data". That shape -- not the file list above -- is the rule this
  function encodes, so a new fixture file following either convention is
  excluded automatically. It does not over-exclude: a genuine test file for
  one of these fixtures, e.g. meridian/model/model_test_data_test.py, has a
  stem ending in "_test", not "_test_data", so it is left alone.
  """
  return stem in ('test_utils', 'test_data') or stem.endswith('_test_data')


def _on_disk_test_files() -> set[str]:
  """Every real test module under the three roots, as repo-relative paths."""
  files = set()
  for root in ROOTS:
    for pattern in ('*_test.py', 'test_*.py'):
      for path in (REPO_ROOT / root).rglob(pattern):
        if _is_fixture_stem(path.stem):
          continue
        files.add(path.relative_to(REPO_ROOT).as_posix())
  return files


def _collect_files(pytest_args: list[str]) -> set[str]:
  """Runs `pytest --collect-only -q` in a fresh subprocess; returns files."""
  result = subprocess.run(
      [sys.executable, '-m', 'pytest', '--collect-only', '-q', *pytest_args],
      cwd=REPO_ROOT,
      capture_output=True,
      text=True,
      check=False,
  )
  # Exit code 5 is pytest's "no tests collected" -- not treated specially
  # here; an empty result is checked for explicitly by the caller instead of
  # being silently accepted or silently treated as an error.
  if result.returncode not in (0, 5):
    raise AssertionError(
        'pytest --collect-only failed unexpectedly for args '
        f'{pytest_args!r} (exit {result.returncode}).\n'
        f'--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}'
    )
  files = set()
  for line in result.stdout.splitlines():
    match = _NODEID_FILE.match(line)
    if match:
      files.add(match.group(1))
  return files


def _parse_targets_and_ignores(
    pytest_args: list[str],
) -> tuple[list[str], list[str]]:
  """Splits one phase's pytest_args into path targets and --ignore paths.

  Every phase in run_tests.test_phases() uses only bare file/directory
  targets and `--ignore=<path>` flags to select tests -- no globs, no `-k`
  expressions -- so this simple split is a faithful model of what pytest
  would actually collect for that phase, without having to invoke pytest a
  second time.
  """
  targets = []
  ignores = []
  skip_next = False
  for arg in pytest_args:
    if skip_next:
      skip_next = False
      continue
    if arg in ('-q',):
      continue
    if arg == '-n':
      skip_next = True  # Consume the worker-count value that follows.
      continue
    if arg.startswith('--dist'):
      continue
    if arg.startswith('--ignore='):
      ignores.append(arg[len('--ignore=') :])
      continue
    targets.append(arg)
  return targets, ignores


def _under_any(file: str, paths: list[str]) -> bool:
  return any(file == path or file.startswith(path + '/') for path in paths)


def _files_for_phase(pytest_args: list[str], universe: set[str]) -> set[str]:
  """Which files of `universe` this phase's targets/ignores would select."""
  targets, ignores = _parse_targets_and_ignores(pytest_args)
  return {
      f
      for f in universe
      if _under_any(f, targets) and not _under_any(f, ignores)
  }


class PhaseCoverageTest(unittest.TestCase):
  """Asserts run_tests.test_phases() partitions every real test file.

  Runtime is dominated by exactly one `pytest --collect-only` subprocess
  call over the union of meridian/, scenarioplanner/, and scripts/ (measured
  ~9s standalone; see the module docstring for why this test does not shell
  out to pytest once per phase).
  """

  @classmethod
  def setUpClass(cls):
    cls.expected_on_disk = _on_disk_test_files()
    # scripts/test_end_to_end.py, scripts/test_report_browser.py,
    # scripts/test_mlflow_integration.py, and scripts/test_proto_build.py are
    # deliberately not part of the phase partition -- see
    # run_tests.STANDALONE_SCRIPTS for why each one is run as its own CI step
    # instead. They are real, non-fixture test-shaped files, so
    # `_on_disk_test_files()` finds them; this is where that's accounted for,
    # against the single source of truth in run_tests.py rather than a second
    # hardcoded list here.
    cls.expected_in_partition = cls.expected_on_disk - set(
        run_tests.STANDALONE_SCRIPTS
    )
    cls.phases = run_tests.test_phases(workers=2)
    ignore_standalone = [
        f'--ignore={path}' for path in run_tests.STANDALONE_SCRIPTS
    ]
    cls.collected_union = _collect_files([*ROOTS, *ignore_standalone])

  def test_standalone_scripts_are_real_files_not_typos(self):
    # If one of these paths were misspelled, it would silently vanish from
    # `expected_in_partition` above without ever being flagged -- so pin it
    # down directly here instead.
    missing = [
        path
        for path in run_tests.STANDALONE_SCRIPTS
        if not (REPO_ROOT / path).is_file()
    ]
    self.assertFalse(
        missing, f'run_tests.STANDALONE_SCRIPTS names paths that do not '
        f'exist on disk: {sorted(missing)}'
    )

  def test_collection_found_every_expected_file(self):
    # A file that exists on disk, matches the test-file glob, and isn't a
    # declared fixture or standalone script, but that pytest's own collector
    # does not surface at all (e.g. a real import error) needs to be a loud
    # failure in its own right, distinct from "no phase claims it".
    missing_from_collection = self.expected_in_partition - self.collected_union
    self.assertFalse(
        missing_from_collection,
        'These test files exist on disk but pytest --collect-only did not '
        f'find any tests in them at all: {sorted(missing_from_collection)}',
    )

  def test_no_phase_collects_zero_files(self):
    empty_phases = [
        name
        for name, args in self.phases
        if not _files_for_phase(args, self.collected_union)
    ]
    self.assertFalse(
        empty_phases,
        'These phases collect zero test files, which must be stated '
        f'explicitly rather than passed over silently: {empty_phases}',
    )

  def test_no_file_is_collected_by_two_phases(self):
    file_to_phases: dict[str, list[str]] = {}
    for name, args in self.phases:
      for f in _files_for_phase(args, self.collected_union):
        file_to_phases.setdefault(f, []).append(name)
    duplicates = {
        f: names for f, names in file_to_phases.items() if len(names) > 1
    }
    self.assertFalse(
        duplicates,
        f'These test files are claimed by more than one phase: {duplicates}',
    )

  def test_every_expected_file_is_claimed_by_some_phase(self):
    """The reason this test exists: name every file no phase collects."""
    collected_by_any_phase: set[str] = set()
    for _, args in self.phases:
      collected_by_any_phase |= _files_for_phase(args, self.collected_union)

    uncovered = self.expected_in_partition - collected_by_any_phase
    self.assertFalse(
        uncovered,
        'These test files are not collected by any phase in '
        'run_tests.test_phases(): '
        f'{sorted(uncovered)}. Add each one to a phase in '
        'scripts/run_tests.py, or, if CI legitimately runs it another way, '
        'add it to run_tests.STANDALONE_SCRIPTS with a stated reason.',
    )

    # The inverse gap should be empty by construction (every phase's files
    # come from `self.collected_union`, and duplicates are checked above),
    # but assert it anyway: a phase could otherwise claim a file pytest
    # never actually collects, which would hide a broken --ignore path.
    phantom = collected_by_any_phase - self.expected_in_partition
    self.assertFalse(
        phantom,
        'These files are claimed by a phase but are not in the expected '
        f'on-disk test set: {sorted(phantom)}',
    )


if __name__ == '__main__':
  unittest.main()
