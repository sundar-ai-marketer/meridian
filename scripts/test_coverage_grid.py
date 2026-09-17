# Copyright 2026 Sundar Ramesh Kumar.
# Licensed under the Apache License, Version 2.0.
# NOTICE: This file is new in this fork; see NOTICE at the repository root.

"""Grid, dry-run and reporting contracts for the coverage study.

A full run is fifty model fits, so the grid definition and the plumbing around
it have to be right before anyone spends two hours on them. These tests check
the grid is a real one-factor-at-a-time design, that `--dry-run` fits nothing,
and that a cell's results are recorded with the convergence state that tells a
reader whether to believe them.
"""

import contextlib
import dataclasses
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock

from scripts import coverage_grid


@dataclasses.dataclass(frozen=True)
class _FakeSummary:
  channel: str
  n: int = 10
  coverage: float = 0.9
  coverage_ci_low: float = 0.6
  coverage_ci_high: float = 0.98
  median_relative_error: float = 0.05
  median_ci_width: float = 1.0


@dataclasses.dataclass(frozen=True)
class _FakeReplication:
  converged: bool = True
  max_r_hat: float = 1.01


class _FakeResult:

  def __init__(self, channels, replications):
    self._channels = channels
    self.replications = replications

  def channel_summaries(self):
    return tuple(_FakeSummary(channel=name) for name in self._channels)


class GridDefinitionTest(unittest.TestCase):

  def test_the_grid_has_a_baseline_and_varies_one_factor_at_a_time(self):
    # A coverage drop is only attributable if exactly one thing changed.
    by_label = {row[0]: row[1:] for row in coverage_grid._GRID}
    baseline = by_label["baseline"]
    for label, cell in by_label.items():
      if label == "baseline":
        continue
      differences = sum(1 for a, b in zip(baseline, cell) if a != b)
      self.assertEqual(differences, 1, f"{label} varies {differences} factors")

  def test_the_grid_exercises_the_misspecified_response(self):
    responses = {row[4] for row in coverage_grid._GRID}
    self.assertIn("linear", responses)
    self.assertIn("concave", responses)

  def test_grid_labels_are_unique_and_shapes_are_valid(self):
    labels = [row[0] for row in coverage_grid._GRID]
    self.assertEqual(len(labels), len(set(labels)))
    for label, geos, times, noise, response in coverage_grid._GRID:
      self.assertGreaterEqual(geos, 1, label)
      self.assertGreaterEqual(times, 1, label)
      self.assertGreaterEqual(noise, 0.0, label)
      self.assertIn(response, ("linear", "concave"), label)


class DryRunTest(unittest.TestCase):

  def test_dry_run_fits_nothing_and_prints_the_plan(self):
    with mock.patch.object(
        coverage_grid, "__name__", coverage_grid.__name__
    ), mock.patch(
        "meridian.validation.recovery.run_recovery_replications"
    ) as run:
      buffer = io.StringIO()
      with contextlib.redirect_stdout(buffer):
        status = coverage_grid.main(["--dry-run"])
    self.assertEqual(status, 0)
    run.assert_not_called()
    output = buffer.getvalue()
    self.assertIn("nothing was fitted", output)
    for label, *_ in coverage_grid._GRID:
      self.assertIn(label, output)

  def test_dry_run_reports_the_fit_count(self):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
      coverage_grid.main(["--dry-run", "--replications", "4"])
    self.assertIn(f"{4 * len(coverage_grid._GRID)}", buffer.getvalue())


class CellReportingTest(unittest.TestCase):

  def _run(self, replications, extra=()):
    result = _FakeResult(["channel_0", "channel_1", "channel_2"], replications)
    with tempfile.TemporaryDirectory() as directory:
      out = pathlib.Path(directory) / "grid.json"
      with mock.patch(
          "meridian.validation.recovery.run_recovery_replications",
          return_value=result,
      ):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
          status = coverage_grid.main(
              ["--replications", "3", "--output", str(out), *extra]
          )
      return status, buffer.getvalue(), json.loads(out.read_text("utf-8"))

  def test_every_cell_is_recorded_with_its_channels(self):
    status, _, evidence = self._run([_FakeReplication()] * 3)
    self.assertEqual(status, 0)
    self.assertEqual(len(evidence["cells"]), len(coverage_grid._GRID))
    self.assertEqual(len(evidence["cells"][0]["channels"]), 3)
    self.assertEqual(
        [cell["label"] for cell in evidence["cells"]],
        [row[0] for row in coverage_grid._GRID],
    )

  def test_a_coverage_estimate_carries_its_wilson_interval(self):
    # Coverage off ten trials is noisy; reporting the point estimate alone
    # invites reading 0.9 as precise.
    _, _, evidence = self._run([_FakeReplication()] * 3)
    channel = evidence["cells"][0]["channels"][0]
    self.assertIn("coverage_ci_low", channel)
    self.assertIn("coverage_ci_high", channel)
    self.assertLessEqual(channel["coverage_ci_low"], channel["coverage"])
    self.assertGreaterEqual(channel["coverage_ci_high"], channel["coverage"])

  def test_non_converged_replications_are_counted_and_warned_about(self):
    replications = [
        _FakeReplication(converged=True, max_r_hat=1.01),
        _FakeReplication(converged=False, max_r_hat=1.9),
        _FakeReplication(converged=True, max_r_hat=1.02),
    ]
    _, output, evidence = self._run(replications)
    self.assertIn("WARNING", output)
    cell = evidence["cells"][0]
    self.assertEqual(cell["replications_converged"], 2)
    self.assertEqual(cell["replications_total"], 3)
    self.assertAlmostEqual(cell["max_r_hat"], 1.9)

  def test_all_converged_produces_no_warning(self):
    _, output, _ = self._run([_FakeReplication()] * 3)
    self.assertNotIn("WARNING", output)

  def test_evidence_records_its_limitations(self):
    _, _, evidence = self._run([_FakeReplication()] * 3)
    joined = " ".join(evidence["limitations"])
    self.assertIn("not Bayesian calibration", joined)
    self.assertIn("noisy", joined)

  def test_quick_mode_is_flagged_in_the_evidence(self):
    # A short-chain run must never be mistaken for the real thing later.
    _, _, evidence = self._run([_FakeReplication()] * 3, extra=["--quick"])
    self.assertTrue(evidence["quick_mode"])

  def test_non_finite_r_hat_does_not_break_serialization(self):
    replications = [_FakeReplication(converged=False, max_r_hat=float("nan"))]
    _, _, evidence = self._run(replications)
    self.assertIsNone(evidence["cells"][0]["max_r_hat"])


if __name__ == "__main__":
  unittest.main()
