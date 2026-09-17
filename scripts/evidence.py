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

"""Shared helpers for the scripts that write dated evidence files.

Every validation script in this directory writes JSON that some claim in
AUDIT.md rests on, and each had grown its own copy of the same four helpers.
The copies had drifted in ways that mattered rather than in ways that did not:
one `sha256` read whole files into memory while another streamed them, and one
`git_head` ran `git` in the process's working directory, so it reported
whichever repository the caller happened to be standing in rather than this
one. Four copies also meant four places to fix anything found wrong in one.

This is the union of the best of them. `json_safe` in particular has to handle
every shape the callers produce -- dataclasses, mappings, numpy scalars,
non-finite floats -- because an evidence file that fails to serialise loses the
run that produced it.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import importlib.metadata as package_metadata
import json
import math
import pathlib
import platform
import subprocess
from collections.abc import Iterable, Mapping
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# The packages whose versions decide whether a numerical result reproduces.
DEFAULT_PACKAGES = (
    'google-meridian',
    'meridian-mmm-fork',
    'jax',
    'jaxlib',
    'tensorflow',
    'tensorflow-probability',
    'tfp-nightly',
    'numpy',
    'arviz',
    'protobuf',
)


def json_safe(value: Any) -> Any:
  """Converts a result structure into something `json.dumps` accepts.

  Non-finite floats become null rather than the `NaN` and `Infinity` tokens
  that `json.dumps` emits by default, because those are not valid JSON and a
  strict reader rejects the whole file.

  `bool` is tested before the integer branch deliberately: `bool` subclasses
  `int`, so an int-first ordering silently writes `1` and `0`, which turns a
  recorded "did this converge" into something the reader has to decode.
  """
  # `None` is valid JSON and means "not available" throughout this evidence --
  # an unavailable R-hat, an absent package version. It must reach `null`, not
  # the catch-all stringifier at the end, which would write the string "None"
  # and turn a missing measurement into a present-looking one.
  if value is None:
    return None
  if isinstance(value, bool):
    return value
  if dataclasses.is_dataclass(value) and not isinstance(value, type):
    return json_safe(dataclasses.asdict(value))
  if isinstance(value, Mapping):
    return {str(key): json_safe(item) for key, item in value.items()}
  if isinstance(value, (list, tuple, set, frozenset)):
    return [json_safe(item) for item in value]

  # numpy is optional here: the container-side scripts run on a bare
  # interpreter, so this module must import without it. A multi-element
  # ndarray is detected by module and class name for the same reason: an
  # `isinstance` check would require importing numpy unconditionally.
  module = type(value).__module__
  if (
      module
      and module.startswith('numpy')
      and type(value).__name__ == 'ndarray'
  ):
    return json_safe(value.tolist())

  if isinstance(value, pathlib.PurePath):
    return str(value)
  if isinstance(value, datetime.timedelta):
    return value.total_seconds()
  if isinstance(value, (datetime.date, datetime.time)):
    # `datetime.datetime` subclasses `datetime.date`, so this also covers it.
    return value.isoformat()

  # numpy is optional here: the container-side scripts run on a bare
  # interpreter, so this module must import without it.
  numpy_scalar = _as_numpy_scalar(value)
  if numpy_scalar is not None:
    return json_safe(numpy_scalar)

  if isinstance(value, float):
    return value if math.isfinite(value) else None
  if isinstance(value, int):
    return value

  # Anything still unrecognised: make the drift visible rather than handing
  # `json.dumps` a value it will reject. This must stay the only remaining
  # fallback -- a second silent stringification branch would hide exactly the
  # kind of drift this one exists to surface.
  return str(value)


def _as_numpy_scalar(value: Any) -> Any:
  """Returns a plain Python scalar for a numpy value, else None."""
  module = type(value).__module__
  if not module or not module.startswith('numpy'):
    return None
  item = getattr(value, 'item', None)
  if item is None:
    return None
  try:
    return item()
  except (TypeError, ValueError):
    return None


def sha256(path: pathlib.Path) -> str:
  """Digests a file in blocks, so a large artifact does not have to fit in RAM.

  Raises rather than returning None on an unreadable file. This value is
  provenance: a null hash silently recorded next to a result is worse than a
  run that stops and says the file it was meant to fingerprint is missing.
  """
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b''):
      digest.update(block)
  return digest.hexdigest()


def git_head() -> str | None:
  """Returns this repository's HEAD, regardless of the caller's directory."""
  try:
    result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
  except (OSError, subprocess.CalledProcessError):
    return None
  return result.stdout.strip() or None


def package_versions(names: Iterable[str] = DEFAULT_PACKAGES) -> dict[str, str | None]:
  """Records installed versions, with None for packages that are absent."""
  versions: dict[str, str | None] = {}
  for name in names:
    try:
      versions[name] = package_metadata.version(name)
    except package_metadata.PackageNotFoundError:
      versions[name] = None
  return versions


def provenance(script_path: pathlib.Path) -> dict[str, Any]:
  """Everything needed to tell whether an evidence file still applies."""
  return {
      'git_head': git_head(),
      'architecture': f'{platform.system()}-{platform.machine()}',
      'package_versions': package_versions(),
      'script_sha256': sha256(script_path),
  }


def write_evidence(path: pathlib.Path, payload: Any) -> None:
  """Writes one evidence file, in the one format every producer uses.

  Every producer had its own copy of this four-line tail, which is how the
  drifted-helper problem starts: `sort_keys` on one path and not another makes
  two runs of the same measurement diff against each other for no reason, and
  a missing `json_safe` turns a numpy scalar into a crash at write time rather
  than an error where the value was built.

  Args:
    path: Destination. Parent directories are created.
    payload: Any structure `json_safe` can convert.
  """
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
      json.dumps(json_safe(payload), indent=2, sort_keys=True) + '\n',
      encoding='utf-8',
  )
