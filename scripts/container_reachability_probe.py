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

"""Records which OS packages the container's default workflow actually loads.

Run this INSIDE the runtime container:

    docker run --rm --entrypoint python meridian:ci \
        scripts/container_reachability_probe.py --output /tmp/probe.json

Why this exists. A Trivy report lists advisories for every package dpkg
installed, whether or not anything ever loads it. That inventory is the right
thing to publish, but on its own it cannot distinguish `libc6`, which every
Python process maps, from `login`, which nothing in a non-interactive analysis
container ever executes. Triage needs that distinction, and it needs it as
measurement rather than assertion.

The measurement is deliberately narrow. It imports the analysis stack, runs a
small amount of real numerical work so lazily-loaded backends actually open
their shared objects, and then reads `/proc/self/maps` to see what the kernel
mapped into this process. Each mapped file is attributed to its dpkg package.

What this does NOT establish:

  - That an unmapped package is unreachable. A user who runs `docker run -it
    meridian bash`, or any command other than the default, reaches more than
    this probe does. The output describes the default analysis workflow only.
  - That a mapped package is exploitable. Loading `libc6` is not a finding.
  - Anything about code paths reached only later in a long fit that this short
    probe does not execute.

The output is evidence for a reachability argument, not a substitute for one.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import platform
import subprocess
import sys
from collections.abc import Iterable, Sequence


def _installed_packages() -> list[dict[str, str]]:
  """Returns every dpkg-installed package with its version and architecture."""
  result = subprocess.run(
      [
          "dpkg-query",
          "--show",
          "--showformat=${Package}\\t${Version}\\t${Architecture}\\n",
      ],
      capture_output=True,
      text=True,
      check=False,
  )
  if result.returncode != 0:
    return []
  packages = []
  for line in result.stdout.splitlines():
    parts = line.split("\t")
    if len(parts) == 3:
      packages.append({"name": parts[0], "version": parts[1], "arch": parts[2]})
  return sorted(packages, key=lambda p: p["name"])


def _exercise_analysis_stack() -> dict[str, object]:
  """Imports and runs enough of Meridian that lazy backends really load.

  Returns a record of what ran, so the evidence file states which code path the
  mapped-object list corresponds to rather than leaving it implied.
  """
  performed: list[str] = []

  import meridian  # pylint: disable=g-import-not-at-top

  performed.append(f"import meridian ({meridian.__version__})")

  import scenarioplanner  # pylint: disable=g-import-not-at-top

  performed.append(f"import scenarioplanner ({scenarioplanner.__name__})")

  import numpy as np  # pylint: disable=g-import-not-at-top

  performed.append(f"import numpy ({np.__version__})")

  backend_env = os.environ.get("MERIDIAN_BACKEND", "(unset, library default)")

  # Real array work, not just an import: the numerical backends open their
  # accelerator and BLAS shared objects on first use, not at import time.
  from meridian import backend as meridian_backend  # pylint: disable=g-import-not-at-top

  matrix = meridian_backend.to_tensor(np.eye(8, dtype=np.float32))
  squared = matrix * matrix
  # `reduce_sum` is module-level for both backends but is not in `__all__`, so
  # fall back rather than couple this probe to a name that is not public.
  reduce_sum = getattr(meridian_backend, "reduce_sum", None)
  if reduce_sum is not None:
    _ = reduce_sum(squared)
    performed.append("meridian.backend tensor round trip and reduction")
  else:
    _ = np.asarray(squared).sum()
    performed.append("meridian.backend tensor round trip, numpy reduction")

  # Report rendering pulls in the template and stylesheet machinery, which is
  # where file and compression libraries get touched.
  from meridian.analysis import summarizer  # pylint: disable=g-import-not-at-top

  performed.append(f"import meridian.analysis.summarizer ({summarizer.__name__})")

  return {"backend_env": backend_env, "steps_performed": performed}


def _mapped_files() -> list[str]:
  """Returns absolute paths of files this process currently has mapped."""
  paths: set[str] = set()
  maps = pathlib.Path("/proc/self/maps")
  if not maps.exists():
    return []
  for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
    # Format: address perms offset dev inode  pathname
    fields = line.split(maxsplit=5)
    if len(fields) < 6:
      continue
    path = fields[5].strip()
    if not path.startswith("/"):
      continue
    if path.endswith(" (deleted)"):
      path = path[: -len(" (deleted)")]
    paths.add(path)
  return sorted(paths)


def _packages_for_paths(
    paths: Iterable[str],
) -> tuple[dict[str, list[str]], list[str]]:
  """Attributes each mapped path to a dpkg package.

  Returns (package name -> sorted paths, paths owned by no dpkg package). The
  unowned list is expected to be large and is not a problem: everything
  installed by pip into the virtualenv lands there.
  """
  path_list = [p for p in paths if pathlib.Path(p).exists()]
  if not path_list:
    return {}, []

  owned: dict[str, list[str]] = {}
  unowned: list[str] = []

  result = subprocess.run(
      ["dpkg-query", "--search", *path_list],
      capture_output=True,
      text=True,
      check=False,
  )
  resolved: dict[str, str] = {}
  for line in result.stdout.splitlines():
    if ": " not in line:
      continue
    packages, _, path = line.partition(": ")
    # dpkg reports "pkg1, pkg2: /path" when several packages ship a diversion.
    first = packages.split(",")[0].strip()
    # Strip an architecture qualifier such as "libc6:arm64".
    resolved[path.strip()] = first.split(":")[0]

  for path in path_list:
    package = resolved.get(path)
    if package:
      owned.setdefault(package, []).append(path)
    else:
      unowned.append(path)

  return {k: sorted(v) for k, v in sorted(owned.items())}, sorted(unowned)


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--output",
      type=pathlib.Path,
      required=True,
      help="Path to write the JSON evidence file.",
  )
  args = parser.parse_args(argv)

  workflow = _exercise_analysis_stack()
  mapped = _mapped_files()
  loaded_packages, unowned = _packages_for_paths(mapped)
  installed = _installed_packages()

  loaded_names = sorted(loaded_packages)
  installed_names = [p["name"] for p in installed]
  not_loaded = sorted(set(installed_names) - set(loaded_names))

  evidence = {
      "date": datetime.date.today().isoformat(),
      "probe": "scripts/container_reachability_probe.py",
      "platform": {
          "machine": platform.machine(),
          "python": platform.python_version(),
          "libc": " ".join(platform.libc_ver()).strip(),
      },
      "workflow": workflow,
      "installed_package_count": len(installed),
      "installed_packages": installed,
      "loaded_os_packages": loaded_names,
      "loaded_os_package_files": loaded_packages,
      "not_loaded_os_packages": not_loaded,
      "mapped_file_count": len(mapped),
      "mapped_files_without_dpkg_owner": len(unowned),
      "limitations": [
          "Describes the default non-interactive analysis workflow only.",
          "A package absent from loaded_os_packages is not proven unreachable; "
          "any other container command may load it.",
          "Presence in loaded_os_packages is not a finding and does not imply "
          "an exploitable path.",
          "Short probe: code paths reached only during a long fit are not "
          "covered.",
      ],
  }

  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )

  print(f"installed OS packages: {len(installed)}")
  print(f"loaded by the default workflow: {len(loaded_names)}")
  print(f"present but not loaded: {len(not_loaded)}")
  print(f"wrote {args.output}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
