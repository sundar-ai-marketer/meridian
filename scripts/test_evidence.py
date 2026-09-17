# Copyright 2026 Sundar Ramesh Kumar.
# Licensed under the Apache License, Version 2.0.
# NOTICE: This file is new in this fork; see NOTICE at the repository root.

"""Contracts for the shared evidence helpers.

These replaced four drifted copies. The tests pin the behaviours the copies
disagreed on, because those are the ones that will drift again: booleans must
not serialise as integers, non-finite floats must not emit invalid JSON, and
`git_head` must report this repository rather than whichever one the caller
happens to be standing in.
"""

import dataclasses
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from scripts import evidence


@dataclasses.dataclass(frozen=True)
class _Sample:
  name: str
  score: float
  passed: bool


class JsonSafeTest(unittest.TestCase):

  def test_booleans_stay_booleans(self):
    # bool subclasses int; an int-first ordering writes 1 and 0, which turns a
    # recorded "did this converge" into something the reader has to decode.
    result = evidence.json_safe({'yes': True, 'no': False})
    self.assertIs(result['yes'], True)
    self.assertIs(result['no'], False)
    self.assertEqual(json.dumps(result), '{"yes": true, "no": false}')

  def test_non_finite_floats_become_null(self):
    result = evidence.json_safe(
        {'nan': float('nan'), 'inf': float('inf'), 'ninf': float('-inf')}
    )
    self.assertEqual(result, {'nan': None, 'inf': None, 'ninf': None})
    # json.dumps would otherwise emit NaN/Infinity, which is not valid JSON.
    json.dumps(result, allow_nan=False)

  def test_finite_floats_are_preserved(self):
    self.assertEqual(evidence.json_safe(1.5), 1.5)
    self.assertEqual(evidence.json_safe(0.0), 0.0)

  def test_dataclasses_are_expanded(self):
    result = evidence.json_safe(_Sample(name='a', score=1.5, passed=True))
    self.assertEqual(result, {'name': 'a', 'score': 1.5, 'passed': True})
    self.assertIs(result['passed'], True)

  def test_a_dataclass_type_is_not_mistaken_for_an_instance(self):
    self.assertIs(evidence.json_safe(_Sample), _Sample)

  def test_mappings_and_sequences_recurse(self):
    result = evidence.json_safe({'a': [{'b': (True, float('nan'))}]})
    self.assertEqual(result, {'a': [{'b': [True, None]}]})

  def test_sets_become_lists(self):
    self.assertEqual(sorted(evidence.json_safe({3, 1, 2})), [1, 2, 3])

  def test_non_string_mapping_keys_become_strings(self):
    self.assertEqual(evidence.json_safe({1: 'a'}), {'1': 'a'})
    json.dumps(evidence.json_safe({1: 'a'}))

  def test_numpy_scalars_are_converted(self):
    np = self._numpy()
    result = evidence.json_safe({
        'f': np.float64(1.5),
        'i': np.int64(3),
        'b': np.bool_(True),
        'nan': np.float64('nan'),
    })
    self.assertEqual(result['f'], 1.5)
    self.assertEqual(result['i'], 3)
    self.assertIs(result['b'], True)
    self.assertIsNone(result['nan'])
    json.dumps(result, allow_nan=False)

  def test_a_numpy_array_is_handled_without_raising(self):
    np = self._numpy()
    json.dumps(evidence.json_safe({'a': np.array([1.0, 2.0]).tolist()}))

  def _numpy(self):
    try:
      import numpy  # pylint: disable=g-import-not-at-top

      return numpy
    except ImportError:
      self.skipTest('numpy not installed')


class Sha256Test(unittest.TestCase):

  def test_digests_a_file(self):
    with tempfile.TemporaryDirectory() as directory:
      path = pathlib.Path(directory) / 'f.txt'
      path.write_bytes(b'meridian')
      # Digest of b'meridian', checked against hashlib directly.
      import hashlib  # pylint: disable=g-import-not-at-top

      self.assertEqual(
          evidence.sha256(path), hashlib.sha256(b'meridian').hexdigest()
      )

  def test_a_file_larger_than_one_block_digests_correctly(self):
    # The implementation streams in 1 MiB blocks; a multi-block file proves the
    # loop accumulates rather than digesting only the first block.
    import hashlib  # pylint: disable=g-import-not-at-top

    payload = os.urandom(3 * 1024 * 1024)
    with tempfile.TemporaryDirectory() as directory:
      path = pathlib.Path(directory) / 'big.bin'
      path.write_bytes(payload)
      self.assertEqual(
          evidence.sha256(path), hashlib.sha256(payload).hexdigest()
      )

  def test_a_missing_file_raises_rather_than_recording_a_null_hash(self):
    # Provenance must fail loudly: a null hash beside a published result is
    # indistinguishable from one nobody computed.
    with self.assertRaises(OSError):
      evidence.sha256(pathlib.Path('/nonexistent/file'))


class GitHeadTest(unittest.TestCase):

  def test_it_reads_this_repository_not_the_working_directory(self):
    # The bug this replaced: without cwd, running a script from elsewhere
    # recorded a different repository's HEAD into this repository's evidence.
    captured = {}

    def fake_run(_command, **kwargs):
      captured.update(kwargs)
      return mock.Mock(stdout='a' * 40 + '\n')

    with mock.patch.object(evidence.subprocess, 'run', side_effect=fake_run):
      self.assertEqual(evidence.git_head(), 'a' * 40)
    self.assertEqual(captured['cwd'], evidence.REPO_ROOT)

  def test_a_failure_returns_none(self):
    with mock.patch.object(
        evidence.subprocess, 'run', side_effect=OSError('no git')
    ):
      self.assertIsNone(evidence.git_head())

  def test_empty_output_returns_none(self):
    with mock.patch.object(
        evidence.subprocess, 'run', return_value=mock.Mock(stdout='  \n')
    ):
      self.assertIsNone(evidence.git_head())


class PackageVersionsTest(unittest.TestCase):

  def test_absent_packages_are_recorded_as_none_not_omitted(self):
    # An omitted key is indistinguishable from a forgotten one; None says the
    # package was looked for and was not installed.
    versions = evidence.package_versions(['definitely-not-installed-xyz'])
    self.assertEqual(versions, {'definitely-not-installed-xyz': None})

  def test_the_default_list_covers_what_decides_reproducibility(self):
    for name in ('numpy', 'jax', 'tensorflow', 'arviz'):
      self.assertIn(name, evidence.DEFAULT_PACKAGES)

  def test_an_installed_package_reports_a_version(self):
    versions = evidence.package_versions(['numpy'])
    self.assertIsNotNone(versions['numpy'])

  def test_result_is_json_serializable(self):
    json.dumps(evidence.json_safe(evidence.package_versions()))


class RepoRootTest(unittest.TestCase):

  def test_repo_root_is_the_checkout_containing_this_file(self):
    self.assertTrue((evidence.REPO_ROOT / 'scripts').is_dir())
    self.assertTrue((evidence.REPO_ROOT / 'meridian').is_dir())


if __name__ == '__main__':
  unittest.main()
