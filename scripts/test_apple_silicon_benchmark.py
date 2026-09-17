# Copyright 2026 Sundar Ramesh Kumar.
# Licensed under the Apache License, Version 2.0.
# NOTICE: This file is new in this fork; see NOTICE at the repository root.

"""Aggregation and comparison contracts for the precision benchmark.

The measuring itself needs minutes of sampling, so these tests drive the parts
that decide what the evidence file claims: whether a failed repetition is
noticed, whether a median is taken over the runs that actually succeeded, and
whether a difference is reported as separable from run-to-run noise. The last
one is the consequential case -- a comparison that calls every difference real
would turn laptop jitter into a recommendation.
"""

import contextlib
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock

from scripts import apple_silicon_benchmark as asb
from scripts import evidence


def _run(posterior, draws_per_sec, peak_gb=2.0, ok=True):
  """One synthetic `meridian.benchmark` payload."""
  if not ok:
    return {"ok": False, "returncode": 1, "stderr_tail": "boom"}
  return {
      "ok": True,
      "timings_seconds": {
          "sample_posterior": posterior,
          "sample_prior": 1.5,
          "build_model": 1.0,
          "total": posterior + 2.5,
      },
      "throughput": {
          "mcmc_draws_per_sec": draws_per_sec,
          "seconds_per_1k_mcmc_draws": 1000.0 / draws_per_sec,
      },
      "peak_rss_bytes": int(peak_gb * 2**30),
      "environment": {"meridian_precision": "FLOAT64"},
      "config": {"n_geos": 10},
  }


class AggregateTest(unittest.TestCase):

  def test_median_and_range_come_from_the_runs(self):
    summary = asb._aggregate(
        [_run(10.0, 40.0), _run(12.0, 50.0), _run(11.0, 45.0)]
    )
    metric = summary["metrics"]["timings_seconds.sample_posterior"]
    self.assertEqual(metric["median"], 11.0)
    self.assertEqual(metric["min"], 10.0)
    self.assertEqual(metric["max"], 12.0)
    self.assertEqual(summary["repetitions"], 3)
    self.assertEqual(summary["repetitions_usable"], 3)

  def test_a_failed_repetition_is_counted_but_not_averaged_in(self):
    summary = asb._aggregate([_run(10.0, 40.0), _run(0, 0, ok=False)])
    self.assertEqual(summary["repetitions"], 2)
    self.assertEqual(summary["repetitions_usable"], 1)
    metric = summary["metrics"]["timings_seconds.sample_posterior"]
    self.assertEqual(metric["median"], 10.0)

  def test_all_repetitions_failing_yields_no_metrics(self):
    # Reporting a median over nothing would be reporting a number that was
    # never measured.
    summary = asb._aggregate([_run(0, 0, ok=False), _run(0, 0, ok=False)])
    self.assertEqual(summary["repetitions_usable"], 0)
    self.assertNotIn("metrics", summary)


class CompareTest(unittest.TestCase):

  def _by_precision(self, f64_runs, f32_runs):
    return {
        "float64": asb._aggregate(f64_runs),
        "float32": asb._aggregate(f32_runs),
    }

  def test_a_clear_difference_reports_non_overlapping_ranges(self):
    by_precision = self._by_precision(
        [_run(12.7, 47.2), _run(12.8, 47.0), _run(13.0, 46.0)],
        [_run(13.9, 43.0), _run(14.0, 42.8), _run(14.0, 43.1)],
    )
    metrics = asb._compare(by_precision)["float32"]["metrics"]
    posterior = metrics["timings_seconds.sample_posterior"]
    self.assertGreater(posterior["percent_change"], 0.0)
    self.assertFalse(posterior["ranges_overlap"])

  def test_overlapping_ranges_are_flagged_as_such(self):
    # Same underlying speed, different jitter: the percent change is nonzero
    # but the ranges overlap, and only the second fact makes it dismissible.
    by_precision = self._by_precision(
        [_run(12.0, 45.0), _run(13.0, 45.0)],
        [_run(12.4, 45.0), _run(12.9, 45.0)],
    )
    metrics = asb._compare(by_precision)["float32"]["metrics"]
    self.assertTrue(
        metrics["timings_seconds.sample_posterior"]["ranges_overlap"]
    )

  def test_percent_change_is_signed_relative_to_float64(self):
    by_precision = self._by_precision([_run(10.0, 40.0)], [_run(5.0, 80.0)])
    metrics = asb._compare(by_precision)["float32"]["metrics"]
    self.assertAlmostEqual(
        metrics["timings_seconds.sample_posterior"]["percent_change"],
        -50.0,
        places=6,
    )
    self.assertAlmostEqual(
        metrics["throughput.mcmc_draws_per_sec"]["percent_change"],
        100.0,
        places=6,
    )

  def test_without_a_baseline_nothing_is_comparable(self):
    by_precision = {
        "float64": asb._aggregate([_run(0, 0, ok=False)]),
        "float32": asb._aggregate([_run(10.0, 40.0)]),
    }
    result = asb._compare(by_precision)
    self.assertFalse(result["comparable"])
    self.assertIn("baseline", result["reason"])


class CommandLineTest(unittest.TestCase):

  def _fake(self, times):
    def run_benchmark(precision, _extra, _scratch):
      return _run(times[precision], 1000.0 / times[precision])

    return run_benchmark

  def test_it_writes_evidence_with_provenance(self):
    with tempfile.TemporaryDirectory() as raw:
      output = pathlib.Path(raw) / "out.json"
      with mock.patch.object(
          asb,
          "_run_benchmark",
          side_effect=self._fake({"float64": 12.0, "float32": 14.0}),
      ):
        with contextlib.redirect_stdout(io.StringIO()):
          status = asb.main(["--reps", "2", "--output", str(output)])
      self.assertEqual(status, 0)
      payload = json.loads(output.read_text(encoding="utf-8"))

    for field in ("git_head", "architecture", "package_versions",
                  "script_sha256"):
      self.assertIn(field, payload["provenance"])
    self.assertIn("float64", payload["by_precision"])
    self.assertIn("limitations", payload)
    json.dumps(evidence.json_safe(payload))

  def test_a_failed_repetition_makes_the_run_exit_non_zero(self):
    # A partially-measured comparison must not read as a successful run.
    with mock.patch.object(
        asb, "_run_benchmark", return_value=_run(0, 0, ok=False)
    ):
      with contextlib.redirect_stdout(io.StringIO()):
        status = asb.main(["--reps", "1"])
    self.assertEqual(status, 1)

  def test_a_single_precision_can_be_measured_alone(self):
    with mock.patch.object(
        asb,
        "_run_benchmark",
        side_effect=self._fake({"float64": 12.0}),
    ) as run_benchmark:
      with contextlib.redirect_stdout(io.StringIO()):
        status = asb.main(["--reps", "1", "--precision", "float64"])
    self.assertEqual(status, 0)
    self.assertEqual(run_benchmark.call_count, 1)

  def test_unrecognised_arguments_pass_through_to_the_benchmark(self):
    seen = {}

    def run_benchmark(precision, extra, _scratch):
      seen[precision] = list(extra)
      return _run(12.0, 80.0)

    with mock.patch.object(asb, "_run_benchmark", side_effect=run_benchmark):
      with contextlib.redirect_stdout(io.StringIO()):
        asb.main(["--reps", "1", "--precision", "float64", "--n-geos", "4"])
    self.assertEqual(seen["float64"], ["--n-geos", "4"])


if __name__ == "__main__":
  unittest.main()
