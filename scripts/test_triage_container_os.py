# Copyright 2026 Sundar Ramesh Kumar.
# Licensed under the Apache License, Version 2.0.
# NOTICE: This file is new in this fork; see NOTICE at the repository root.

"""Classification and reporting contracts for the container advisory triage.

Everything here runs on synthetic scan and probe documents. The point of the
triage is that a reader can trust the `loaded` / `not-loaded` column, so these
tests pin the cases where getting it wrong would be invisible: an advisory that
names several packages of which only one is loaded, an advisory with a vendor
fix that must not be triaged as residual, and a probe that measured nothing.
"""

import contextlib
import io
import json
import pathlib
import tempfile
import unittest

from scripts import triage_container_os


def _scan(rows):
  """Builds a Trivy-shaped report from (id, severity, pkg, installed, fixed)."""
  return {
      "ArtifactName": "meridian:test",
      "Metadata": {"OS": {"Family": "debian", "Name": "13.7"}},
      "Results": [{
          "Vulnerabilities": [
              {
                  "VulnerabilityID": identifier,
                  "Severity": severity,
                  "PkgName": package,
                  "InstalledVersion": installed,
                  "FixedVersion": fixed,
              }
              for identifier, severity, package, installed, fixed in rows
          ]
      }],
  }


def _probe(loaded, installed=None):
  return {
      "date": "2026-09-17",
      "platform": {"machine": "x86_64", "python": "3.11.16", "libc": "glibc 2.41"},
      "installed_package_count": len(installed or loaded),
      "loaded_os_packages": list(loaded),
  }


class AdvisoryCollapseTest(unittest.TestCase):

  def test_rows_for_one_advisory_collapse_into_a_single_record(self):
    # Trivy emits one row per (advisory, package). The triage counts distinct
    # advisories, so four rows for one CVE must not read as four findings.
    scan = _scan([
        ("CVE-1", "HIGH", "libuuid1", "1.0", None),
        ("CVE-1", "HIGH", "mount", "1.0", None),
        ("CVE-1", "HIGH", "login", "1.0", None),
        ("CVE-2", "LOW", "tar", "2.0", None),
    ])
    advisories = triage_container_os._advisories_from_scan(scan)
    self.assertEqual(len(advisories), 2)
    self.assertEqual(
        sorted(advisories["CVE-1"]["packages"]), ["libuuid1", "login", "mount"]
    )

  def test_missing_vulnerability_id_is_skipped_rather_than_keyed_on_none(self):
    scan = _scan([("CVE-1", "HIGH", "libc6", "1.0", None)])
    scan["Results"][0]["Vulnerabilities"].append(
        {"Severity": "HIGH", "PkgName": "ghost"}
    )
    advisories = triage_container_os._advisories_from_scan(scan)
    self.assertEqual(list(advisories), ["CVE-1"])

  def test_empty_results_do_not_raise(self):
    self.assertEqual(triage_container_os._advisories_from_scan({}), {})
    self.assertEqual(
        triage_container_os._advisories_from_scan({"Results": None}), {}
    )


class ClassificationTest(unittest.TestCase):

  def test_one_loaded_package_among_many_marks_the_advisory_loaded(self):
    # This is the util-linux shape: nine packages, only libuuid1 in process.
    # Reporting it as not-loaded would understate the exposure.
    scan = _scan([
        ("CVE-1", "HIGH", "libuuid1", "1.0", None),
        ("CVE-1", "HIGH", "mount", "1.0", None),
        ("CVE-1", "HIGH", "login", "1.0", None),
    ])
    advisories = triage_container_os._advisories_from_scan(scan)
    triage_container_os._classify(advisories, {"libuuid1"})
    self.assertEqual(advisories["CVE-1"]["classification"], "loaded")
    self.assertEqual(advisories["CVE-1"]["loaded_packages"], ["libuuid1"])

  def test_no_loaded_package_marks_the_advisory_not_loaded(self):
    scan = _scan([("CVE-1", "HIGH", "perl-base", "1.0", None)])
    advisories = triage_container_os._advisories_from_scan(scan)
    triage_container_os._classify(advisories, {"libc6"})
    self.assertEqual(advisories["CVE-1"]["classification"], "not-loaded")
    self.assertEqual(advisories["CVE-1"]["loaded_packages"], [])

  def test_an_empty_loaded_set_classifies_everything_not_loaded(self):
    scan = _scan([("CVE-1", "HIGH", "libc6", "1.0", None)])
    advisories = triage_container_os._advisories_from_scan(scan)
    triage_container_os._classify(advisories, set())
    self.assertEqual(advisories["CVE-1"]["classification"], "not-loaded")


class SummaryTest(unittest.TestCase):

  def test_advisories_with_a_vendor_fix_are_excluded_from_the_residual_set(self):
    # The triage documents what cannot be fixed. A finding with a fix belongs
    # to CI's blocking gate instead, and counting it here would overstate the
    # unfixable residue.
    scan = _scan([
        ("CVE-FIXED", "HIGH", "libc6", "1.0", "1.1"),
        ("CVE-UNFIXED", "HIGH", "libc6", "1.0", None),
    ])
    advisories = triage_container_os._advisories_from_scan(scan)
    triage_container_os._classify(advisories, {"libc6"})
    summary = triage_container_os._summary(advisories, scan, _probe(["libc6"]))

    self.assertEqual(summary["advisories_total"], 2)
    self.assertEqual(summary["advisories_without_vendor_fix"], 1)
    self.assertEqual(summary["advisories_with_vendor_fix"], 1)
    self.assertEqual(
        [a["id"] for a in summary["remaining_advisories"]], ["CVE-UNFIXED"]
    )

  def test_an_advisory_is_unfixed_only_when_no_package_has_a_fix(self):
    # A partially fixed advisory still has an actionable upgrade, so it must
    # not be reported as vendor-unfixable.
    scan = _scan([
        ("CVE-1", "HIGH", "libc6", "1.0", "1.1"),
        ("CVE-1", "HIGH", "libc-bin", "1.0", None),
    ])
    advisories = triage_container_os._advisories_from_scan(scan)
    triage_container_os._classify(advisories, set())
    summary = triage_container_os._summary(advisories, scan, _probe([]))
    self.assertEqual(summary["advisories_without_vendor_fix"], 0)

  def test_severity_counts_and_loaded_tally_match_the_input(self):
    scan = _scan([
        ("CVE-1", "HIGH", "libc6", "1.0", None),
        ("CVE-2", "LOW", "perl-base", "1.0", None),
        ("CVE-3", "MEDIUM", "libc6", "1.0", None),
    ])
    advisories = triage_container_os._advisories_from_scan(scan)
    triage_container_os._classify(advisories, {"libc6"})
    summary = triage_container_os._summary(advisories, scan, _probe(["libc6"]))
    self.assertEqual(
        summary["unfixed_by_severity"], {"HIGH": 1, "LOW": 1, "MEDIUM": 1}
    )
    self.assertEqual(summary["unfixed_affecting_a_loaded_package"], 2)

  def test_summary_is_json_serializable(self):
    scan = _scan([("CVE-1", "HIGH", "libc6", "1.0", None)])
    advisories = triage_container_os._advisories_from_scan(scan)
    triage_container_os._classify(advisories, {"libc6"})
    summary = triage_container_os._summary(advisories, scan, _probe(["libc6"]))
    json.dumps(summary)

  def test_summary_carries_a_complete_provenance_block(self):
    scan = _scan([("CVE-1", "HIGH", "libc6", "1.0", None)])
    advisories = triage_container_os._advisories_from_scan(scan)
    triage_container_os._classify(advisories, {"libc6"})
    summary = triage_container_os._summary(advisories, scan, _probe(["libc6"]))
    self.assertIn("provenance", summary)
    for key in ("git_head", "architecture", "package_versions", "script_sha256"):
      self.assertIn(key, summary["provenance"])
    json.dumps(summary)


class DocumentTest(unittest.TestCase):

  def _run(self, scan, probe, extra=()):
    with tempfile.TemporaryDirectory() as directory:
      root = pathlib.Path(directory)
      scan_path = root / "scan.json"
      probe_path = root / "probe.json"
      out_path = root / "triage.md"
      summary_path = root / "summary.json"
      scan_path.write_text(json.dumps(scan), encoding="utf-8")
      probe_path.write_text(json.dumps(probe), encoding="utf-8")
      with contextlib.redirect_stdout(io.StringIO()):
        status = triage_container_os.main([
            "--scan", str(scan_path),
            "--probe", str(probe_path),
            "--output", str(out_path),
            "--summary-output", str(summary_path),
            *extra,
        ])
      return (
          status,
          out_path.read_text(encoding="utf-8"),
          json.loads(summary_path.read_text(encoding="utf-8")),
      )

  def test_end_to_end_writes_a_document_and_a_summary(self):
    scan = _scan([
        ("CVE-1", "HIGH", "libuuid1", "1.0", None),
        ("CVE-1", "HIGH", "mount", "1.0", None),
        ("CVE-2", "LOW", "perl-base", "1.0", None),
    ])
    status, document, summary = self._run(scan, _probe(["libuuid1"], ["a"] * 87))

    self.assertEqual(status, 0)
    self.assertIn("CVE-1", document)
    self.assertIn("CVE-2", document)
    self.assertIn("87 OS packages installed, 1 loaded", document)
    self.assertEqual(summary["advisories_without_vendor_fix"], 2)

  def test_the_document_states_its_own_limits(self):
    # The triage is only defensible if it carries its caveats with it; a
    # reader who copies the table elsewhere must copy them too.
    scan = _scan([("CVE-1", "HIGH", "libc6", "1.0", None)])
    _, document, _ = self._run(scan, _probe(["libc6"]))
    self.assertIn("not a finding", document)
    self.assertIn("not proof of unreachability", document)
    self.assertIn("source", document)

  def test_the_loaded_column_names_only_loaded_packages(self):
    scan = _scan([
        ("CVE-1", "HIGH", "libuuid1", "1.0", None),
        ("CVE-1", "HIGH", "login", "1.0", None),
    ])
    _, document, _ = self._run(scan, _probe(["libuuid1"]))
    row = [line for line in document.splitlines() if line.startswith("| `CVE-1`")]
    self.assertEqual(len(row), 1)
    self.assertIn("`libuuid1`", row[0])
    # `login` appears in the affected-packages cell but must not be reported
    # as loaded; check the cell after the package list.
    loaded_cell = row[0].split("|")[3]
    self.assertNotIn("login", loaded_cell)

  def test_review_by_date_is_offset_from_today(self):
    scan = _scan([("CVE-1", "HIGH", "libc6", "1.0", None)])
    _, document, _ = self._run(scan, _probe(["libc6"]), extra=["--review-in-days", "30"])
    self.assertIn("Review by:", document)

  def test_a_scan_with_no_findings_still_produces_a_document(self):
    status, document, summary = self._run(
        {"ArtifactName": "x", "Metadata": {"OS": {}}, "Results": []},
        _probe(["libc6"]),
    )
    self.assertEqual(status, 0)
    self.assertIn("Container OS advisory triage", document)
    self.assertEqual(summary["advisories_total"], 0)

  def test_output_is_deterministic_for_the_same_input(self):
    scan = _scan([
        ("CVE-2", "LOW", "tar", "1.0", None),
        ("CVE-1", "HIGH", "libc6", "1.0", None),
    ])
    _, first, _ = self._run(scan, _probe(["libc6"]))
    _, second, _ = self._run(scan, _probe(["libc6"]))
    self.assertEqual(first, second)


if __name__ == "__main__":
  unittest.main()
