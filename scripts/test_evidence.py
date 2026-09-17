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
import datetime
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

  def test_none_becomes_json_null_not_the_string_none(self):
    """`None` means "not available" in this evidence and must stay null.

    Regression: the catch-all `str(value)` fallback was added without a
    `None` branch in front of it, so `json_safe(None)` returned the string
    `'None'`. Every producer writes optional fields -- an unavailable R-hat,
    a package that is not installed -- and a quoted "None" reads as a present
    value rather than a missing one. `scripts/test_run_recovery_study.py`
    caught it downstream; this catches it at the source.
    """
    self.assertIsNone(evidence.json_safe(None))
    self.assertEqual(json.dumps(evidence.json_safe(None)), "null")
    self.assertEqual(
        json.loads(
            json.dumps(
                evidence.json_safe(
                    {"max_r_hat": None, "values": [1, None, float("nan")]}
                )
            )
        ),
        {"max_r_hat": None, "values": [1, None, None]},
    )


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
    # `dataclasses.is_dataclass` is True for the class itself, not just
    # instances, so the `not isinstance(value, type)` guard keeps
    # `dataclasses.asdict` (which requires an instance) from being called on
    # it. A bare class is not JSON-serialisable either, so it now falls all
    # the way through to the same str() fallback as any other unrecognised
    # type, rather than being handed back raw and unserialisable.
    result = evidence.json_safe(_Sample)
    self.assertEqual(result, str(_Sample))
    json.dumps(result)

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

  def test_a_multi_element_ndarray_is_converted_not_returned_raw(self):
    # The confirmed defect: `.item()` raises ValueError on a multi-element
    # array, `_as_numpy_scalar` swallows it and returns None, and json_safe
    # used to fall through to `return value`, handing back the raw ndarray.
    np = self._numpy()
    result = evidence.json_safe(np.array([1.0, 2.0, 3.0]))
    self.assertEqual(result, [1.0, 2.0, 3.0])
    self.assertIsInstance(result, list)
    json.dumps(result)

  def test_a_non_finite_value_nested_in_an_ndarray_becomes_null(self):
    np = self._numpy()
    result = evidence.json_safe(np.array([1.0, float('nan'), float('inf')]))
    self.assertEqual(result, [1.0, None, None])
    json.dumps(result, allow_nan=False)

  def test_a_multidimensional_ndarray_is_converted(self):
    np = self._numpy()
    result = evidence.json_safe(np.array([[1, 2], [3, 4]]))
    self.assertEqual(result, [[1, 2], [3, 4]])
    json.dumps(result)

  def test_a_path_becomes_a_string(self):
    result = evidence.json_safe(pathlib.Path('/tmp/evidence.json'))
    self.assertEqual(result, '/tmp/evidence.json')
    self.assertIsInstance(result, str)

  def test_a_pure_posix_path_becomes_a_string(self):
    result = evidence.json_safe(pathlib.PurePosixPath('a/b'))
    self.assertEqual(result, 'a/b')

  def test_a_datetime_becomes_an_isoformat_string(self):
    value = datetime.datetime(2026, 9, 17, 12, 30, 0)
    self.assertEqual(evidence.json_safe(value), value.isoformat())

  def test_a_date_becomes_an_isoformat_string(self):
    value = datetime.date(2026, 9, 17)
    self.assertEqual(evidence.json_safe(value), '2026-09-17')

  def test_a_time_becomes_an_isoformat_string(self):
    value = datetime.time(12, 30, 0)
    self.assertEqual(evidence.json_safe(value), value.isoformat())

  def test_a_timedelta_becomes_total_seconds(self):
    value = datetime.timedelta(minutes=1, seconds=30)
    self.assertEqual(evidence.json_safe(value), 90.0)

  def test_an_unrecognised_type_falls_back_to_str(self):
    # This is the only remaining silent-conversion path. It is deliberate: an
    # evidence file must never lose a value outright, but a second such
    # fallback elsewhere would hide the same kind of drift this one exists to
    # surface, so this test pins that there is exactly one.
    class _Unrecognised:

      def __str__(self):
        return 'unrecognised-marker'

    result = evidence.json_safe(_Unrecognised())
    self.assertEqual(result, 'unrecognised-marker')
    json.dumps(result)

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

  # These two assert the *git* path alone, so the environment fallback has to
  # be out of the picture. Without clearing it they pass locally and fail on
  # any GitHub Actions runner, where GITHUB_SHA is always set.
  # GitHeadFallbackTest covers the fallback itself.
  def _without_revision_env(self):
    return mock.patch.dict(
        os.environ,
        {
            key: value
            for key, value in os.environ.items()
            if key not in ('MERIDIAN_GIT_HEAD', 'GITHUB_SHA')
        },
        clear=True,
    )

  def test_a_failure_returns_none(self):
    with self._without_revision_env(), mock.patch.object(
        evidence.subprocess, 'run', side_effect=OSError('no git')
    ):
      self.assertIsNone(evidence.git_head())

  def test_empty_output_returns_none(self):
    with self._without_revision_env(), mock.patch.object(
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


class ProvenanceTest(unittest.TestCase):

  def test_returns_all_four_non_null_subfields(self):
    result = evidence.provenance(pathlib.Path(__file__).resolve())
    for key in ('git_head', 'architecture', 'package_versions', 'script_sha256'):
      self.assertIn(key, result)
      self.assertIsNotNone(result[key])
    self.assertIsInstance(result['package_versions'], dict)
    json.dumps(evidence.json_safe(result))

  def test_architecture_names_system_and_machine(self):
    with mock.patch.object(evidence.platform, 'system', return_value='Linux'):
      with mock.patch.object(evidence.platform, 'machine', return_value='x86_64'):
        result = evidence.provenance(pathlib.Path(__file__).resolve())
    self.assertEqual(result['architecture'], 'Linux-x86_64')

  def test_script_sha256_matches_the_given_file(self):
    result = evidence.provenance(pathlib.Path(__file__).resolve())
    self.assertEqual(
        result['script_sha256'], evidence.sha256(pathlib.Path(__file__).resolve())
    )


class RepoRootTest(unittest.TestCase):

  def test_repo_root_is_the_checkout_containing_this_file(self):
    self.assertTrue((evidence.REPO_ROOT / 'scripts').is_dir())
    self.assertTrue((evidence.REPO_ROOT / 'meridian').is_dir())


class GitHeadFallbackTest(unittest.TestCase):
  """A producer that cannot run git must still be able to say its revision.

  The runtime container is the case: `.dockerignore` excludes `.git` and the
  image does not install git, so the reachability probe running inside it has
  no way to discover HEAD. Its evidence then records a null revision, which
  `scripts/test_evidence_provenance.py` rejects -- so the weekly container
  rescan would commit a file its own CI refuses. The workflow passes the
  revision in through the environment instead.
  """

  def _with_broken_git(self):
    return mock.patch.object(
        evidence.subprocess, "run", side_effect=FileNotFoundError("no git")
    )

  def test_git_is_preferred_when_it_answers(self):
    completed = mock.Mock(stdout="  abc123\n  ", returncode=0)
    with mock.patch.object(
        evidence.subprocess, "run", return_value=completed
    ), mock.patch.dict(os.environ, {"MERIDIAN_GIT_HEAD": "from-env"}):
      self.assertEqual(evidence.git_head(), "abc123")

  def test_explicit_env_var_is_used_when_git_is_absent(self):
    with self._with_broken_git(), mock.patch.dict(
        os.environ, {"MERIDIAN_GIT_HEAD": "deadbeef"}, clear=False
    ):
      self.assertEqual(evidence.git_head(), "deadbeef")

  def test_github_sha_is_the_second_choice(self):
    env = {k: v for k, v in os.environ.items() if k != "MERIDIAN_GIT_HEAD"}
    env["GITHUB_SHA"] = "runner-sha"
    with self._with_broken_git(), mock.patch.dict(
        os.environ, env, clear=True
    ):
      self.assertEqual(evidence.git_head(), "runner-sha")

  def test_explicit_var_wins_over_github_sha(self):
    with self._with_broken_git(), mock.patch.dict(
        os.environ,
        {"MERIDIAN_GIT_HEAD": "explicit", "GITHUB_SHA": "runner-sha"},
        clear=False,
    ):
      self.assertEqual(evidence.git_head(), "explicit")

  def test_none_when_nothing_knows_the_revision(self):
    # Better than a fabricated value: the provenance gate then says so.
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("MERIDIAN_GIT_HEAD", "GITHUB_SHA")
    }
    with self._with_broken_git(), mock.patch.dict(os.environ, env, clear=True):
      self.assertIsNone(evidence.git_head())

  def test_a_blank_env_var_does_not_count_as_a_revision(self):
    with self._with_broken_git(), mock.patch.dict(
        os.environ, {"MERIDIAN_GIT_HEAD": "   ", "GITHUB_SHA": ""}, clear=False
    ):
      self.assertIsNone(evidence.git_head())


if __name__ == '__main__':
  unittest.main()
