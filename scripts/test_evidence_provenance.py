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

"""Coherence gate over every JSON evidence file in `docs/validation/`.

This exists because the evidence convention used to be incoherent: a file
named `container-os-latest.json` and a file named `container-os-2026-09-16.json`
looked like a rolling alias and its dated snapshot, but carried two entirely
different schemas -- one was a Trivy-advisory inventory, the other a triage
summary. Nothing caught that but a human reading both files side by side.

The convention this test enforces:

  `<family>-<YYYY-MM-DD>.json`  a dated, never-overwritten evidence file.
  `<family>-latest.json`        a rolling alias, always the newest dated
                                 file in `<family>`, byte-identical to it
                                 (or identical modulo a field this file
                                 names explicitly -- see
                                 `_ALIAS_EXEMPT_TOP_LEVEL_FIELDS`).

Family grouping rule (stated here, not encoded as a per-file list): strip a
trailing `-latest.json` or `-YYYY-MM-DD.json` from the filename; what is left
is the family name. Two files are in the same family iff that stripped name
is equal. This is a regex over the filename, so a brand-new evidence family
-- some future script writing `foo-latest.json` next to `foo-2026-10-01.json`
-- is grouped and checked automatically, with no edit to this file.

What is deliberately NOT checked here: whether an evidence file is "stale"
relative to the source it describes, by comparing git log dates. The
MCMC-derived files (recovery, coverage-grid, prior-sensitivity) take minutes
to regenerate; gating on a source-file touch date would block an unrelated
commit such as a docstring edit to `meridian/validation/recovery.py`. A
staleness check, if wanted, belongs scoped to the container-os family alone,
since `container-rescan.yml` already refreshes it weekly and a family whose
own automation re-measures it on a schedule is the one case where "this
hasn't been refreshed in N weeks" is actionable rather than noise. No such
check is added here; scoping it correctly is a separate decision.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import re
import unittest

_VALIDATION_DIR = (
    pathlib.Path(__file__).resolve().parents[1] / "docs" / "validation"
)

# See the module docstring: family = filename with a trailing
# `-<date>.json` or `-latest.json` removed.
_DATED_RE = re.compile(r"^(?P<family>.+)-(?P<date>\d{4}-\d{2}-\d{2})\.json$")
_ALIAS_RE = re.compile(r"^(?P<family>.+)-latest\.json$")

_PROVENANCE_SUBFIELDS = (
    "git_head",
    "architecture",
    "package_versions",
    "script_sha256",
)

# Top-level keys an alias is allowed to differ on from the dated file it
# mirrors, without that being treated as drift. Empty today: every current
# alias is produced by copying the dated file verbatim (see
# .github/workflows/container-rescan.yml), so nothing needs an exemption.
# If a producer ever legitimately needs the alias and the dated copy to
# differ (e.g. a `generated_at` timestamp on the alias only), name the field
# here -- do not loosen the comparison to a fuzzy diff.
_ALIAS_EXEMPT_TOP_LEVEL_FIELDS: frozenset[str] = frozenset()

# Evidence committed before `scripts/evidence.py` grew `provenance()` and the
# producers were wired to call it. These files record measurements that were
# genuinely taken, and the provenance they lack cannot be reconstructed: the
# git HEAD and package versions of the run are not recoverable after the fact,
# and inventing them would be worse than their absence. Three of them
# (the container families) cannot be regenerated on a macOS host at all --
# they need Docker and Trivy.
#
# So they are grandfathered by name rather than by a date cutoff, which keeps
# the list finite and shrinking: re-running any producer writes provenance,
# and the file's name changes with its date, so the new file is covered by the
# gate automatically and the old entry can be deleted from this list.
#
# Nothing may be added here. A new evidence file comes from a producer that
# already calls `provenance()`.
_PRE_PROVENANCE_EVIDENCE: frozenset[str] = frozenset({
    "container-os-inventory-2026-09-16.json",
    "container-reachability-2026-09-17.json",
    "coverage-grid-2026-09-17.json",
    "os-triage-summary-2026-09-17.json",
    "prior-predictive-audit-2026-09-17.json",
    "prior-sensitivity-2026-09-17.json",
    "recovery-2026-09-16.json",
    "strict-quickstart-2026-09-16.json",
    "strict-sampling-2026-09-16.json",
})


def _json_files(validation_dir: pathlib.Path) -> list[pathlib.Path]:
  return sorted(validation_dir.glob("*.json"))


def _load(path: pathlib.Path):
  with path.open(encoding="utf-8") as handle:
    return json.load(handle)


def _is_alias(path: pathlib.Path) -> bool:
  return _ALIAS_RE.match(path.name) is not None


def _family_of(path: pathlib.Path) -> str | None:
  """Returns the family name for a dated or alias file, else None."""
  alias_match = _ALIAS_RE.match(path.name)
  if alias_match:
    return alias_match.group("family")
  dated_match = _DATED_RE.match(path.name)
  if dated_match:
    return dated_match.group("family")
  return None


def _newest_dated_sibling(
    alias_path: pathlib.Path, all_files: list[pathlib.Path]
) -> pathlib.Path | None:
  """Returns the highest-dated file in `alias_path`'s family, else None."""
  family = _family_of(alias_path)
  candidates = []
  for other in all_files:
    if other == alias_path or _is_alias(other):
      continue
    dated_match = _DATED_RE.match(other.name)
    if dated_match and dated_match.group("family") == family:
      candidates.append((dated_match.group("date"), other))
  if not candidates:
    return None
  candidates.sort(key=lambda pair: pair[0])
  return candidates[-1][1]


class AllFilesParseAndDateTest(unittest.TestCase):
  """Every evidence file must be readable JSON with a valid ISO date."""

  def test_every_json_file_parses_and_has_a_valid_iso_date(self):
    files = _json_files(_VALIDATION_DIR)
    self.assertTrue(files, f"no JSON evidence files found in {_VALIDATION_DIR}")
    for path in files:
      with self.subTest(file=path.name):
        try:
          data = _load(path)
        except (OSError, json.JSONDecodeError) as error:
          self.fail(f"{path.name} is not valid JSON: {error}")
        self.assertIsInstance(
            data, dict, f"{path.name} does not decode to a JSON object"
        )
        self.assertIn(
            "date", data, f"{path.name} has no top-level 'date' field"
        )
        try:
          datetime.date.fromisoformat(str(data["date"]))
        except ValueError as error:
          self.fail(
              f"{path.name} has a 'date' field that is not a valid ISO "
              f"date ({data['date']!r}): {error}"
          )


class ProvenanceCompletenessTest(unittest.TestCase):
  """Every non-alias evidence file must carry a complete provenance block."""

  def test_every_non_alias_file_carries_complete_provenance(self):
    files = _json_files(_VALIDATION_DIR)
    non_alias = [path for path in files if not _is_alias(path)]
    self.assertTrue(
        non_alias, f"no non-alias JSON evidence files found in {_VALIDATION_DIR}"
    )
    covered = [
        path
        for path in non_alias
        if path.name not in _PRE_PROVENANCE_EVIDENCE
    ]
    self.assertTrue(
        covered,
        "every evidence file is grandfathered, so this gate is checking "
        "nothing -- the exemption list has outlived its purpose",
    )
    for path in covered:
      with self.subTest(file=path.name):
        data = _load(path)
        self.assertIn(
            "provenance",
            data,
            f"{path.name} is not an alias and has no top-level "
            "'provenance' object",
        )
        provenance = data["provenance"]
        self.assertIsInstance(
            provenance,
            dict,
            f"{path.name}'s 'provenance' field is not an object",
        )
        for subfield in _PROVENANCE_SUBFIELDS:
          self.assertIn(
              subfield,
              provenance,
              f"{path.name}'s provenance block is missing '{subfield}'",
          )
          value = provenance.get(subfield)
          self.assertTrue(
              value,
              f"{path.name}'s provenance.{subfield} is empty ({value!r})",
          )


class GrandfatherListTest(unittest.TestCase):
  """The exemption list may not outlive the files it exempts."""

  def test_every_exempt_file_still_exists(self):
    present = {path.name for path in _json_files(_VALIDATION_DIR)}
    stale = sorted(_PRE_PROVENANCE_EVIDENCE - present)
    self.assertFalse(
        stale,
        f"_PRE_PROVENANCE_EVIDENCE exempts files that are no longer in "
        f"{_VALIDATION_DIR}: {stale}. They were regenerated or removed, so "
        "delete these entries -- a stale exemption silently widens the gate.",
    )

  def test_no_exempt_file_already_carries_provenance(self):
    # If one does, it was regenerated and the exemption is now hiding a file
    # the gate could be checking.
    redundant = []
    for path in _json_files(_VALIDATION_DIR):
      if path.name in _PRE_PROVENANCE_EVIDENCE and _load(path).get(
          "provenance"
      ):
        redundant.append(path.name)
    self.assertFalse(
        sorted(redundant),
        f"these files now carry provenance and no longer need exempting: "
        f"{sorted(redundant)}. Remove them from _PRE_PROVENANCE_EVIDENCE.",
    )


class AliasCoherenceTest(unittest.TestCase):
  """Every `<family>-latest.json` must mirror its family's newest dated file."""

  def test_every_alias_matches_its_newest_dated_sibling(self):
    files = _json_files(_VALIDATION_DIR)
    aliases = [path for path in files if _is_alias(path)]
    self.assertTrue(
        aliases, f"no alias (*-latest.json) files found in {_VALIDATION_DIR}"
    )
    for alias_path in aliases:
      with self.subTest(file=alias_path.name):
        family = _family_of(alias_path)
        sibling = _newest_dated_sibling(alias_path, files)
        self.assertIsNotNone(
            sibling,
            f"{alias_path.name} is an alias for family '{family}' but no "
            f"dated '{family}-YYYY-MM-DD.json' file exists in "
            f"{_VALIDATION_DIR} -- an alias with no family member is a "
            "failure, not a silent pass.",
        )
        alias_data = _load(alias_path)
        sibling_data = _load(sibling)
        alias_compare = {
            key: value
            for key, value in alias_data.items()
            if key not in _ALIAS_EXEMPT_TOP_LEVEL_FIELDS
        }
        sibling_compare = {
            key: value
            for key, value in sibling_data.items()
            if key not in _ALIAS_EXEMPT_TOP_LEVEL_FIELDS
        }
        self.assertEqual(
            alias_compare,
            sibling_compare,
            f"{alias_path.name} has drifted from its newest dated sibling "
            f"{sibling.name}: the two are not identical (modulo "
            f"{sorted(_ALIAS_EXEMPT_TOP_LEVEL_FIELDS) or 'no exempt fields'}).",
        )


if __name__ == "__main__":
  unittest.main()
