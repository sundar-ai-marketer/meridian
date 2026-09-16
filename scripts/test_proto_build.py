#!/usr/bin/env python3
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

"""Contract checks for the sibling MMM proto package build command.

This is intentionally a standard-library ``unittest`` script. It loads the
small ``proto/setup.py`` command directly and mocks subprocess calls, so it
does not import Meridian, TensorFlow, JAX, or make network calls.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SETUP_PATH = _REPO_ROOT / 'proto' / 'setup.py'
_MODULE_SPEC = importlib.util.spec_from_file_location(
    'meridian_proto_setup_under_test', _SETUP_PATH
)
if _MODULE_SPEC is None or _MODULE_SPEC.loader is None:
  raise ImportError(f'Unable to load proto build module from {_SETUP_PATH}')
_PROTO_SETUP = importlib.util.module_from_spec(_MODULE_SPEC)
_MODULE_SPEC.loader.exec_module(_PROTO_SETUP)


def _new_builder():
  """Creates a ProtoBuild instance without reading the checkout config."""
  builder = object.__new__(_PROTO_SETUP.ProtoBuild)
  builder._root = Path('proto')
  builder._deps = {}
  builder._srcs = []
  return builder


class ProtoBuildContractTest(unittest.TestCase):

  def test_protoc_version_failure_is_propagated(self):
    failure = subprocess.CalledProcessError(
        127,
        ['python', '-m', 'grpc_tools.protoc', '--version'],
        stderr='grpc_tools.protoc is unavailable',
    )
    with mock.patch.object(_PROTO_SETUP.subprocess, 'run', side_effect=failure):
      with self.assertRaisesRegex(subprocess.CalledProcessError, '127'):
        _new_builder()._check_protoc_version()

  def test_git_failure_is_propagated_and_path_stays_one_argument(self):
    failure = subprocess.CalledProcessError(
        128,
        ['git', 'clone'],
        stderr='repository unavailable',
    )
    builder = _new_builder()
    builder._deps = {
        'googleapis': (
            'https://github.com/googleapis/googleapis.git',
            'master',
        )
    }
    with tempfile.TemporaryDirectory(prefix='proto build ') as temp_dir:
      root = Path(temp_dir) / 'dependency root with spaces'
      with mock.patch.object(
          _PROTO_SETUP.subprocess, 'run', side_effect=failure
      ) as run:
        with self.assertRaisesRegex(subprocess.CalledProcessError, '128'):
          builder._pull_deps(root)

      command = run.call_args.args[0]
      self.assertEqual(
          command,
          [
              'git',
              'clone',
              '--quiet',
              '--depth=1',
              '--branch',
              'master',
              'https://github.com/googleapis/googleapis.git',
              str(root / 'googleapis'),
          ],
      )

  def test_pinned_git_revision_is_fetched_and_verified(self):
    revision = '0123456789abcdef0123456789abcdef01234567'
    builder = _new_builder()
    builder._deps = {
        'googleapis': ('https://example.invalid/googleapis.git', revision),
    }
    completed = subprocess.CompletedProcess([], 0, stdout='', stderr='')
    verified = subprocess.CompletedProcess(
        [], 0, stdout=f'{revision}\n', stderr=''
    )
    with tempfile.TemporaryDirectory(prefix='proto pinned build ') as temp_dir:
      root = Path(temp_dir) / 'dependency root with spaces'
      with mock.patch.object(
          _PROTO_SETUP.subprocess,
          'run',
          side_effect=[completed, completed, completed, completed, verified],
      ) as run:
        self.assertEqual(builder._pull_deps(root), 0)

      self.assertEqual(
          [call.args[0] for call in run.call_args_list],
          [
              ['git', 'init', '--quiet', str(root / 'googleapis')],
              [
                  'git',
                  '-C',
                  str(root / 'googleapis'),
                  'remote',
                  'add',
                  'origin',
                  'https://example.invalid/googleapis.git',
              ],
              [
                  'git',
                  '-C',
                  str(root / 'googleapis'),
                  'fetch',
                  '--quiet',
                  '--depth=1',
                  '--no-tags',
                  'origin',
                  revision,
              ],
              [
                  'git',
                  '-C',
                  str(root / 'googleapis'),
                  'checkout',
                  '--quiet',
                  '--detach',
                  'FETCH_HEAD',
              ],
              [
                  'git',
                  '-C',
                  str(root / 'googleapis'),
                  'rev-parse',
                  '--verify',
                  'HEAD',
              ],
          ],
      )

  def test_pinned_git_revision_mismatch_fails(self):
    revision = '0123456789abcdef0123456789abcdef01234567'
    builder = _new_builder()
    builder._deps = {
        'googleapis': ('https://example.invalid/googleapis.git', revision),
    }
    completed = subprocess.CompletedProcess([], 0, stdout='', stderr='')
    mismatch = subprocess.CompletedProcess(
        [], 0, stdout='fedcba9876543210fedcba9876543210fedcba98\n', stderr=''
    )
    with tempfile.TemporaryDirectory(prefix='proto pinned build ') as temp_dir:
      with mock.patch.object(
          _PROTO_SETUP.subprocess,
          'run',
          side_effect=[completed, completed, completed, completed, mismatch],
      ):
        with self.assertRaisesRegex(RuntimeError, 'expected'):
          builder._pull_deps(Path(temp_dir) / 'deps')

  def test_protoc_failure_is_propagated_and_paths_stay_one_argument(self):
    failure = subprocess.CalledProcessError(
        1,
        ['python', '-m', 'grpc_tools.protoc'],
        stderr='invalid schema',
    )
    builder = _new_builder()
    source = Path('proto sources') / 'nested dir' / 'example.proto'
    include = Path('include roots') / 'googleapis'
    builder._srcs = [source]
    with mock.patch.object(
        _PROTO_SETUP.subprocess, 'run', side_effect=failure
    ) as run:
      with self.assertRaisesRegex(subprocess.CalledProcessError, '1'):
        builder._compile_proto_in_place([include])

    command = run.call_args.args[0]
    self.assertEqual(
        command,
        [
            _PROTO_SETUP.sys.executable,
            '-m',
            'grpc_tools.protoc',
            f'-I{include}',
            '--python_out=.',
            str(source),
        ],
    )

  def test_too_old_protoc_fails_before_dependency_pull(self):
    builder = _new_builder()
    builder._check_protoc_version = mock.Mock(return_value=26)
    builder._pull_deps = mock.Mock()

    with self.assertRaisesRegex(
        RuntimeError, r'requires protoc major version 27 or newer; found 26'
    ):
      builder.run()

    builder._pull_deps.assert_not_called()

  def test_success_runs_commands_and_preserves_existing_artifact(self):
    builder = _new_builder()
    with tempfile.TemporaryDirectory(prefix='proto artifact ') as temp_dir:
      artifact = Path(temp_dir) / 'mmm/v1/mmm_pb2.py'
      artifact.parent.mkdir(parents=True)
      original = b'generated schema artifact\n'
      artifact.write_bytes(original)

      builder._deps = {
          'googleapis': ('https://example.invalid/googleapis.git', 'main'),
      }
      source = Path(temp_dir) / 'source with spaces' / 'example.proto'
      source.parent.mkdir(parents=True)
      source.write_text('syntax = "proto3";\n')
      builder._srcs = [source]
      completed = subprocess.CompletedProcess([], 0, stdout='', stderr='')
      with mock.patch.object(
          _PROTO_SETUP.subprocess, 'run', return_value=completed
      ) as run:
        self.assertEqual(builder._pull_deps(Path(temp_dir) / 'deps'), 0)
        self.assertEqual(builder._compile_proto_in_place([Path(temp_dir)]), 0)

      self.assertEqual(artifact.read_bytes(), original)
      self.assertEqual(run.call_count, 2)
      self.assertEqual(
          run.call_args_list[0].args[0][-1],
          str(Path(temp_dir) / 'deps/googleapis'),
      )
      self.assertEqual(run.call_args_list[1].args[0][-1], str(source))


if __name__ == '__main__':
  unittest.main(verbosity=2)
