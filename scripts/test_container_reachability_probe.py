# Copyright 2026 Sundar Ramesh Kumar.
# Licensed under the Apache License, Version 2.0.
# NOTICE: This file is new in this fork; see NOTICE at the repository root.

"""Parsing contracts for the container reachability probe.

The probe's whole value is the list of OS packages a real analysis process
loaded. That list comes from parsing `/proc/self/maps` and attributing each
mapped file to a dpkg package, and both steps have formats that are easy to
mis-handle: map lines with spaces in the path, deleted mappings, anonymous
regions, and dpkg's architecture-qualified and diverted package names. These
tests drive those cases with synthetic input, so they run anywhere rather than
only inside the container.
"""

import json
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

from scripts import container_reachability_probe as probe


_MAPS = """\
55a4c2a00000-55a4c2a21000 r--p 00000000 fd:01 100  /usr/bin/python3.11
55a4c2a21000-55a4c2a22000 rw-p 00021000 fd:01 100  /usr/bin/python3.11
7f2b4c000000-7f2b4c021000 rw-p 00000000 00:00 0
7f2b4c100000-7f2b4c200000 r-xp 00000000 fd:01 200  /lib/x86_64-linux-gnu/libc.so.6
7f2b4c300000-7f2b4c400000 r-xp 00000000 fd:01 300  /tmp/deleted.so (deleted)
7ffd1a000000-7ffd1a021000 rw-p 00000000 00:00 0                          [stack]
ffffffffff600000-ffffffffff601000 --xp 00000000 00:00 0                  [vsyscall]
"""


class MappedFilesTest(unittest.TestCase):

  def _parse(self, text):
    with tempfile.TemporaryDirectory() as directory:
      maps = pathlib.Path(directory) / "maps"
      maps.write_text(text, encoding="utf-8")
      with mock.patch.object(probe.pathlib, "Path", return_value=maps):
        return probe._mapped_files()

  def test_only_absolute_file_paths_are_returned(self):
    paths = self._parse(_MAPS)
    self.assertIn("/usr/bin/python3.11", paths)
    self.assertIn("/lib/x86_64-linux-gnu/libc.so.6", paths)
    # Anonymous regions and kernel pseudo-mappings are not files.
    self.assertNotIn("[stack]", paths)
    self.assertNotIn("[vsyscall]", paths)

  def test_repeated_mappings_of_one_file_collapse(self):
    # A shared object is mapped several times with different permissions;
    # counting them separately would inflate nothing useful.
    paths = self._parse(_MAPS)
    self.assertEqual(paths.count("/usr/bin/python3.11"), 1)

  def test_the_deleted_suffix_is_stripped(self):
    paths = self._parse(_MAPS)
    self.assertIn("/tmp/deleted.so", paths)
    self.assertNotIn("/tmp/deleted.so (deleted)", paths)

  def test_a_path_containing_spaces_survives(self):
    text = (
        "55a4c2a00000-55a4c2a21000 r--p 00000000 fd:01 100  "
        "/opt/my app/lib.so\n"
    )
    self.assertEqual(self._parse(text), ["/opt/my app/lib.so"])

  def test_results_are_sorted(self):
    paths = self._parse(_MAPS)
    self.assertEqual(paths, sorted(paths))

  def test_a_missing_maps_file_returns_empty_rather_than_raising(self):
    # The probe is importable off-Linux; it must degrade rather than explode.
    with tempfile.TemporaryDirectory() as directory:
      missing = pathlib.Path(directory) / "absent"
      with mock.patch.object(probe.pathlib, "Path", return_value=missing):
        self.assertEqual(probe._mapped_files(), [])

  def test_malformed_lines_are_skipped(self):
    self.assertEqual(self._parse("garbage\n\n55a4-55a5 r--p\n"), [])


class PackageAttributionTest(unittest.TestCase):

  def _resolve(self, paths, stdout):
    completed = subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")
    with mock.patch.object(
        probe.subprocess, "run", return_value=completed
    ), mock.patch.object(probe.pathlib.Path, "exists", return_value=True):
      return probe._packages_for_paths(paths)

  def test_paths_are_attributed_to_their_package(self):
    owned, unowned = self._resolve(
        ["/lib/libc.so.6", "/usr/bin/perl"],
        "libc6: /lib/libc.so.6\nperl-base: /usr/bin/perl\n",
    )
    self.assertEqual(sorted(owned), ["libc6", "perl-base"])
    self.assertEqual(unowned, [])

  def test_an_architecture_qualifier_is_stripped_from_the_package_name(self):
    # dpkg reports "libc6:arm64"; the scan report names the package "libc6",
    # so leaving the qualifier on would break every join between them.
    owned, _ = self._resolve(
        ["/lib/libc.so.6"], "libc6:arm64: /lib/libc.so.6\n"
    )
    self.assertEqual(list(owned), ["libc6"])

  def test_a_diverted_path_takes_the_first_named_package(self):
    owned, _ = self._resolve(
        ["/usr/bin/x"], "pkg-a, pkg-b: /usr/bin/x\n"
    )
    self.assertEqual(list(owned), ["pkg-a"])

  def test_paths_dpkg_does_not_own_are_reported_separately(self):
    # Everything pip installs into the virtualenv lands here; it is expected,
    # not an error, but it must not be silently attributed to an OS package.
    owned, unowned = self._resolve(
        ["/lib/libc.so.6", "/opt/venv/lib/site-packages/x.so"],
        "libc6: /lib/libc.so.6\n",
    )
    self.assertEqual(list(owned), ["libc6"])
    self.assertEqual(unowned, ["/opt/venv/lib/site-packages/x.so"])

  def test_several_files_of_one_package_group_under_it(self):
    owned, _ = self._resolve(
        ["/lib/libc.so.6", "/lib/libc-2.41.so"],
        "libc6: /lib/libc.so.6\nlibc6: /lib/libc-2.41.so\n",
    )
    self.assertEqual(list(owned), ["libc6"])
    self.assertEqual(len(owned["libc6"]), 2)

  def test_no_existing_paths_short_circuits(self):
    with mock.patch.object(probe.pathlib.Path, "exists", return_value=False):
      self.assertEqual(probe._packages_for_paths(["/nope"]), ({}, []))

  def test_an_empty_input_returns_empty(self):
    self.assertEqual(probe._packages_for_paths([]), ({}, []))


class InstalledPackagesTest(unittest.TestCase):

  def test_packages_are_parsed_and_sorted(self):
    stdout = "perl-base\t5.40\tarm64\nlibc6\t2.41\tarm64\n"
    completed = subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")
    with mock.patch.object(probe.subprocess, "run", return_value=completed):
      packages = probe._installed_packages()
    self.assertEqual([p["name"] for p in packages], ["libc6", "perl-base"])
    self.assertEqual(packages[0]["version"], "2.41")

  def test_a_failed_dpkg_query_returns_empty_rather_than_raising(self):
    completed = subprocess.CompletedProcess([], 1, stdout="", stderr="no dpkg")
    with mock.patch.object(probe.subprocess, "run", return_value=completed):
      self.assertEqual(probe._installed_packages(), [])

  def test_malformed_rows_are_skipped(self):
    completed = subprocess.CompletedProcess(
        [], 0, stdout="libc6\t2.41\tarm64\nbroken-row\n", stderr=""
    )
    with mock.patch.object(probe.subprocess, "run", return_value=completed):
      packages = probe._installed_packages()
    self.assertEqual([p["name"] for p in packages], ["libc6"])


class ProvenanceWiringTest(unittest.TestCase):
  """Exercises the provenance-merging code path without a real container.

  `main()` otherwise requires `/proc/self/maps`, `dpkg-query`, and importing
  `scenarioplanner` -- none of which are available on this host -- so the
  workflow-exercising and system-inspection steps are mocked and only the
  evidence-assembly and JSON-writing code is exercised for real.
  """

  def test_written_evidence_carries_a_complete_provenance_block(self):
    with tempfile.TemporaryDirectory() as directory:
      out = pathlib.Path(directory) / "probe.json"
      with mock.patch.object(
          probe, "_exercise_analysis_stack",
          return_value={"backend_env": "test", "steps_performed": []},
      ), mock.patch.object(
          probe, "_mapped_files", return_value=["/lib/libc.so.6"]
      ), mock.patch.object(
          probe, "_packages_for_paths",
          return_value=({"libc6": ["/lib/libc.so.6"]}, []),
      ), mock.patch.object(
          probe, "_installed_packages",
          return_value=[{"name": "libc6", "version": "2.41", "arch": "arm64"}],
      ):
        status = probe.main(["--output", str(out)])

      self.assertEqual(status, 0)
      written = json.loads(out.read_text(encoding="utf-8"))

    self.assertIn("provenance", written)
    for key in ("git_head", "architecture", "package_versions", "script_sha256"):
      self.assertIn(key, written["provenance"])
    # The pre-existing top-level "platform" key must survive unchanged; only
    # "provenance" was added.
    self.assertIn("platform", written)
    self.assertIn("loaded_os_packages", written)


if __name__ == "__main__":
  unittest.main()
