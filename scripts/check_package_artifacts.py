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
# NOTICE: This file is new in this fork; see NOTICE at the repository root.

"""Check that built distributions retain runtime payloads and exclude tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import tarfile
from zipfile import ZipFile

_TEST_MODULE = re.compile(r'(?:^|/)[^/]+_test\.py$')


def _archive_names(path: Path) -> list[str]:
  if path.suffix == '.whl':
    with ZipFile(path) as archive:
      return archive.namelist()
  if path.name.endswith('.tar.gz'):
    with tarfile.open(path, 'r:gz') as archive:
      return archive.getnames()
  raise ValueError(f'Unsupported distribution archive: {path}')


def _one_artifact(directory: Path, pattern: str) -> Path:
  matches = sorted(directory.glob(pattern))
  if len(matches) != 1:
    raise ValueError(
        f'Expected one {pattern} in {directory}, found {len(matches)}'
    )
  return matches[0]


def _check_wheel(path: Path, required: tuple[str, ...]) -> list[str]:
  names = set(_archive_names(path))
  errors = [f'{path}: missing {name}' for name in required if name not in names]
  leaked_tests = sorted(name for name in names if _TEST_MODULE.search(name))
  errors.extend(f'{path}: contains test module {name}' for name in leaked_tests)
  errors.extend(_check_asset_manifest(path, names, is_wheel=True))
  for required_license in ('LICENSE', 'NOTICE'):
    if not any(
        name.endswith(f'.dist-info/licenses/{required_license}')
        for name in names
    ):
      errors.append(f'{path}: missing dist-info license {required_license}')
  return errors


def _check_sdist(
    path: Path,
    required: tuple[str, ...],
    test_sources: tuple[str, ...],
) -> list[str]:
  names = set(_archive_names(path))
  errors = [
      f'{path}: missing {name}'
      for name in required
      if not any(entry.endswith(name) for entry in names)
  ]
  errors.extend(
      f'{path}: missing source test {name}'
      for name in test_sources
      if not any(entry.endswith(name) for entry in names)
  )
  errors.extend(_check_asset_manifest(path, names, is_wheel=False))
  return errors


def _check_asset_manifest(
    path: Path, names: set[str], *, is_wheel: bool
) -> list[str]:
  """Ensure every declared report asset ships in the built artifact."""
  manifest_name = 'meridian/templates/assets/manifest.json'
  manifest_path = next(
      (name for name in names if name.endswith(manifest_name)), None
  )
  if manifest_path is None:
    return []  # The caller reports the missing manifest as a required file.

  try:
    if is_wheel:
      with ZipFile(path) as archive:
        manifest = json.loads(archive.read(manifest_path))
    else:
      with tarfile.open(path, 'r:gz') as archive:
        extracted = archive.extractfile(manifest_path)
        if extracted is None:
          return [f'{path}: cannot read report asset manifest']
        manifest = json.loads(extracted.read())
  except (OSError, json.JSONDecodeError, KeyError) as exc:
    return [f'{path}: cannot read report asset manifest: {exc}']

  return [
      f'{path}: manifest asset is missing: {asset_name}'
      for asset_name in manifest
      if f'meridian/templates/assets/{asset_name}' not in names
  ]


def check_distributions(core_dir: Path, proto_dir: Path) -> list[str]:
  """Return contract violations for the core and schema distributions."""
  core_wheel = _one_artifact(core_dir, 'meridian_mmm_fork-*.whl')
  core_sdist = _one_artifact(core_dir, 'meridian_mmm_fork-*.tar.gz')
  proto_wheel = _one_artifact(proto_dir, 'mmm_proto_schema-*.whl')
  proto_sdist = _one_artifact(proto_dir, 'mmm_proto_schema-*.tar.gz')

  errors = _check_wheel(
      core_wheel,
      (
          'meridian/analysis/budget_decision.py',
          'meridian/templates/formatter.py',
          'meridian/templates/report_assets.py',
          'meridian/templates/style.css',
          'meridian/templates/assets/manifest.json',
      ),
  )
  errors.extend(
      _check_sdist(
          core_sdist,
          (
              'meridian/analysis/budget_decision.py',
              'LICENSE',
              'NOTICE',
              'meridian/templates/style.scss',
              'meridian/templates/assets/manifest.json',
          ),
          (
              'meridian/analysis/budget_decision_test.py',
              'meridian/math_invariants_test.py',
              'meridian/templates/formatter_test.py',
          ),
      )
  )
  errors.extend(
      _check_wheel(
          proto_wheel,
          (
              'mmm/v1/model/meridian/meridian_model.proto',
              'mmm/v1/model/meridian/meridian_model_pb2.py',
              'LICENSE',
              'NOTICE',
          ),
      )
  )
  errors.extend(
      _check_sdist(
          proto_sdist,
          (
              'mmm/v1/model/meridian/meridian_model.proto',
              'mmm/v1/model/meridian/meridian_model_pb2.py',
          ),
          (),
      )
  )
  return errors


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--core-dir', type=Path, default=Path('dist'))
  parser.add_argument('--proto-dir', type=Path, default=Path('proto/dist'))
  args = parser.parse_args(argv)

  try:
    errors = check_distributions(args.core_dir, args.proto_dir)
  except (OSError, ValueError) as exc:
    print(f'distribution check failed: {exc}', file=sys.stderr)
    return 1
  if errors:
    print('\n'.join(errors), file=sys.stderr)
    return 1
  print('Distribution contents satisfy runtime, source, and license contracts.')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
