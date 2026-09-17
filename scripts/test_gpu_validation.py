# Copyright 2026 Sundar Ramesh Kumar.
# Licensed under the Apache License, Version 2.0.
# NOTICE: This file is new in this fork; see NOTICE at the repository root.

"""Comparison and device-detection contracts for the GPU validation script.

The fitting itself needs hardware, so these tests drive the parts that decide
what the evidence file says: whether an accelerator was really seen, how a
child process is pinned to a device, and whether two posteriors are judged to
agree. Getting the last one wrong is the dangerous case -- a comparison that
passes whatever the numbers are would report a GPU as validated when it is not.
"""

import contextlib
import io
import json
import math
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from scripts import evidence
from scripts import gpu_validation


def _leg(means, mcses, channels=None):
  names = channels or [f"channel_{i}" for i in range(len(means))]
  return {
      "ok": True,
      "channels": [
          {
              "channel": name,
              "posterior_mean": mean,
              "mcse": mcse,
              "posterior_sd": 1.0,
              "ess_bulk": 100.0,
              "true_roi": 1.0,
              "median": mean,
          }
          for name, mean, mcse in zip(names, means, mcses)
      ],
  }


class AcceleratorDetectionTest(unittest.TestCase):

  def test_a_cpu_only_jax_backend_is_not_an_accelerator(self):
    report = {
        "jax": {"available": True, "default_backend": "cpu"},
        "tensorflow": {"available": True, "gpus": []},
    }
    self.assertFalse(gpu_validation._has_accelerator(report))

  def test_a_cuda_jax_backend_counts(self):
    report = {
        "jax": {"available": True, "default_backend": "cuda"},
        "tensorflow": {"available": False},
    }
    self.assertTrue(gpu_validation._has_accelerator(report))

  def test_a_tensorflow_gpu_counts_even_when_jax_is_on_cpu(self):
    report = {
        "jax": {"available": True, "default_backend": "cpu"},
        "tensorflow": {"available": True, "gpus": ["/physical_device:GPU:0"]},
    }
    self.assertTrue(gpu_validation._has_accelerator(report))

  def test_a_failed_import_is_not_an_accelerator(self):
    report = {
        "jax": {"available": False, "error": "no jax"},
        "tensorflow": {"available": False, "error": "no tf"},
    }
    self.assertFalse(gpu_validation._has_accelerator(report))


class ChildEnvironmentTest(unittest.TestCase):

  def test_cpu_leg_is_pinned_away_from_every_accelerator(self):
    env = gpu_validation._child_environment("cpu")
    self.assertEqual(env["JAX_PLATFORMS"], "cpu")
    self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")

  def test_gpu_leg_clears_an_inherited_cpu_pin(self):
    # Without this the whole comparison silently runs CPU against CPU and
    # reports agreement that means nothing.
    with mock.patch.dict(
        os.environ, {"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""}
    ):
      env = gpu_validation._child_environment("gpu")
    self.assertNotIn("JAX_PLATFORMS", env)
    self.assertNotIn("CUDA_VISIBLE_DEVICES", env)


class ComparisonTest(unittest.TestCase):

  def test_identical_posteriors_agree(self):
    result = gpu_validation._compare(_leg([2.0], [0.05]), _leg([2.0], [0.05]))
    self.assertTrue(result["comparable"])
    self.assertEqual(result["max_abs_z"], 0.0)
    self.assertIn("consistent", result["verdict"])

  def test_a_difference_inside_monte_carlo_error_still_agrees(self):
    # Two correct samplers differ; the test must tolerate that or it will fail
    # on working hardware.
    result = gpu_validation._compare(_leg([2.00], [0.05]), _leg([2.05], [0.05]))
    self.assertLess(result["max_abs_z"], 3.0)
    self.assertIn("consistent", result["verdict"])

  def test_a_difference_far_beyond_monte_carlo_error_is_flagged(self):
    result = gpu_validation._compare(_leg([2.0], [0.01]), _leg([3.0], [0.01]))
    self.assertGreater(result["max_abs_z"], 3.0)
    self.assertIn("investigate", result["verdict"])

  def test_the_worst_channel_decides_the_verdict(self):
    cpu = _leg([2.0, 5.0], [0.01, 0.01])
    gpu = _leg([2.0, 9.0], [0.01, 0.01])
    result = gpu_validation._compare(cpu, gpu)
    self.assertGreater(result["max_abs_z"], 3.0)

  def test_z_uses_the_combined_error_of_both_legs(self):
    cpu = _leg([2.0], [0.3])
    gpu = _leg([2.5], [0.4])
    result = gpu_validation._compare(cpu, gpu)
    expected = 0.5 / math.sqrt(0.3**2 + 0.4**2)
    self.assertAlmostEqual(result["channels"][0]["z"], -expected, places=6)

  def test_a_failed_leg_is_not_reported_as_agreement(self):
    failed = {"ok": False, "returncode": 1, "stderr_tail": "boom"}
    result = gpu_validation._compare(_leg([2.0], [0.05]), failed)
    self.assertFalse(result["comparable"])
    self.assertIn("did not complete", result["reason"])

  def test_zero_mcse_does_not_raise(self):
    result = gpu_validation._compare(_leg([2.0], [0.0]), _leg([2.0], [0.0]))
    self.assertTrue(math.isnan(result["channels"][0]["z"]))

  def test_comparison_is_json_serializable(self):
    result = gpu_validation._compare(_leg([2.0], [0.0]), _leg([2.0], [0.0]))
    json.dumps(evidence.json_safe(result))


class NoYardstickTest(unittest.TestCase):
  """A comparison with nothing to measure against is not an agreement.

  This is the dangerous direction. If every channel's Monte Carlo error is
  undefined -- a degenerate `az.ess`, a stuck chain, a leg that produced
  summaries but no usable spread -- then there is no scale on which to judge
  the difference, and the running maximum of |z| never leaves its initial
  zero. Reporting that as "consistent" would turn a missing measurement into
  a pass, and the process would exit 0 with it.
  """

  def test_undefined_error_on_every_channel_is_not_comparable(self):
    cpu = _leg([1.0, 1.0], [float("nan"), float("nan")])
    gpu = _leg([100.0, 200.0], [float("nan"), float("nan")])
    result = gpu_validation._compare(cpu, gpu)
    self.assertFalse(result["comparable"])
    self.assertIn("Monte Carlo error", result["reason"])

  def test_zero_error_on_every_channel_is_not_comparable(self):
    result = gpu_validation._compare(_leg([2.0], [0.0]), _leg([9.0], [0.0]))
    self.assertFalse(result["comparable"])

  def test_a_usable_channel_still_decides_the_verdict(self):
    # One channel has a real error and disagrees wildly; the NaN channel must
    # not dilute that into a pass.
    cpu = _leg([1.0, 1.0], [float("nan"), 0.01])
    gpu = _leg([1.0, 2.0], [float("nan"), 0.01])
    result = gpu_validation._compare(cpu, gpu)
    self.assertTrue(result["comparable"])
    self.assertEqual(result["channels_compared"], 1)
    self.assertEqual(result["channels_total"], 2)
    self.assertGreater(result["max_abs_z"], 3.0)
    self.assertIn("investigate", result["verdict"])

  def test_partial_comparability_is_stated_in_the_verdict(self):
    cpu = _leg([1.0, 1.0], [float("nan"), 0.05])
    gpu = _leg([1.0, 1.0], [float("nan"), 0.05])
    result = gpu_validation._compare(cpu, gpu)
    self.assertTrue(result["comparable"])
    self.assertIn("1 of 2", result["verdict"])

  def test_mismatched_channels_are_not_paired_by_position(self):
    # Pairing by position would compare one channel's ROI against another's
    # and attribute the difference to the device.
    cpu = _leg([2.0, 3.0], [0.05, 0.05], channels=["tv", "search"])
    gpu = _leg([2.0, 3.0], [0.05, 0.05], channels=["search", "tv"])
    result = gpu_validation._compare(cpu, gpu)
    self.assertFalse(result["comparable"])
    self.assertIn("different channels", result["reason"])


class OperationProbeTest(unittest.TestCase):
  """The capability probe, run for real on this host's default device."""

  def test_every_required_operation_runs_on_this_host(self):
    # On a CPU all of these lower, so this doubles as a check that each probe
    # in the battery is itself well-formed rather than failing for its own
    # reasons.
    result = gpu_validation._probe_operations()
    self.assertEqual(result["unsupported"], [])
    probed = set(result["operations"])
    declared = {name for name, _ in gpu_validation._REQUIRED_OPERATIONS}
    self.assertEqual(probed, declared)
    for name, detail in result["operations"].items():
      self.assertTrue(detail["ok"], f"{name}: {detail}")
      self.assertTrue(detail["needed_for"])

  def test_the_probe_result_is_json_serializable(self):
    json.dumps(evidence.json_safe(gpu_validation._probe_operations()))


class ExitStatusTest(unittest.TestCase):
  """Exit status has to separate the outcomes a caller acts on differently."""

  _CUDA = {
      "jax": {"available": True, "default_backend": "cuda"},
      "tensorflow": {"available": False},
  }

  def _run(self, run_child):
    with mock.patch.object(
        gpu_validation, "_device_report", return_value=self._CUDA
    ), mock.patch.object(gpu_validation, "_run_child", side_effect=run_child):
      buffer = io.StringIO()
      with contextlib.redirect_stdout(buffer):
        status = gpu_validation.main([])
    return status, buffer.getvalue()

  def test_a_crashed_gpu_leg_exits_non_zero(self):
    # The failure the script exists to catch. Exiting 0 here would make a
    # crashed GPU leg indistinguishable from a passing run.
    def run_child(device, *_args, mode="fit", **_kwargs):
      if mode == "probe-ops":
        return {"ok": False, "returncode": 1, "stderr_tail": "unreadable"}
      if device == "cpu":
        return _leg([2.0], [0.05])
      return {"ok": False, "returncode": -6, "stderr_tail": "Aborted"}

    status, _ = self._run(run_child)
    self.assertEqual(status, 2)

  def test_an_accelerator_that_cannot_run_the_model_exits_three(self):
    def run_child(device, *_args, mode="fit", **_kwargs):
      del device
      if mode == "probe-ops":
        return {
            "ok": True,
            "x64_enabled": True,
            "dtype": "float64",
            "operations": {
                "dot_general": {
                    "ok": False,
                    "needed_for": "every tensor contraction",
                    "error": "XlaRuntimeError",
                    "message": "failed to legalize operation 'mhlo.dot_general'",
                }
            },
            "unsupported": ["dot_general"],
        }
      raise AssertionError("no fit should be attempted once an op is missing")

    status, output = self._run(run_child)
    self.assertEqual(status, 3)
    self.assertIn("dot_general", output)

  def test_an_unreadable_probe_does_not_block_the_fits(self):
    # The probe saves time; it must not become a gate that fails closed.
    attempted = []

    def run_child(device, *_args, mode="fit", **_kwargs):
      if mode == "probe-ops":
        return {"ok": True, "unexpected": "shape"}
      attempted.append(device)
      return _leg([2.0], [0.05])

    status, output = self._run(run_child)
    self.assertEqual(status, 0)
    self.assertEqual(sorted(attempted), ["cpu", "gpu"])
    self.assertIn("inconclusive", output)

  def test_capability_only_records_without_fitting(self):
    def run_child(device, *_args, mode="fit", **_kwargs):
      del device
      if mode == "probe-ops":
        return {
            "ok": True,
            "operations": {
                name: {"ok": True, "needed_for": need, "value": 1.0}
                for name, need in gpu_validation._REQUIRED_OPERATIONS
            },
            "unsupported": [],
        }
      raise AssertionError("--capability-only must not fit")

    with mock.patch.object(
        gpu_validation, "_device_report", return_value=self._CUDA
    ), mock.patch.object(gpu_validation, "_run_child", side_effect=run_child):
      with contextlib.redirect_stdout(io.StringIO()):
        status = gpu_validation.main(["--capability-only"])
    self.assertEqual(status, 0)


class MemoryProbeTest(unittest.TestCase):
  """Each attempt is isolated, and what is known is written down as it goes."""

  def test_each_attempt_runs_in_its_own_child_process(self):
    calls = []

    def run_child(device, seed, n_geos, n_times, output, **_kwargs):
      del device, seed, n_times, output
      calls.append(n_geos)
      return {"ok": n_geos < 8}

    with mock.patch.object(
        gpu_validation, "_run_child", side_effect=run_child
    ):
      result = gpu_validation._memory_probe(
          1, 2, 16, pathlib.Path(tempfile.gettempdir())
      )

    self.assertEqual(calls, [2, 4, 8])
    self.assertEqual([a["n_geos"] for a in result["attempts"]], [2, 4, 8])
    self.assertFalse(result["attempts"][-1]["ok"])

  def test_a_driver_level_abort_is_recorded_rather_than_lost(self):
    # A negative return code is a signal, not an exception; an in-process loop
    # would simply die here and write nothing.
    def run_child(*_args, **_kwargs):
      return {"ok": False, "returncode": -9, "stderr_tail": "Killed"}

    with mock.patch.object(
        gpu_validation, "_run_child", side_effect=run_child
    ):
      result = gpu_validation._memory_probe(
          1, 4, 64, pathlib.Path(tempfile.gettempdir())
      )
    self.assertEqual(result["attempts"][0]["returncode"], -9)

  def test_progress_is_persisted_after_every_attempt(self):
    seen = []

    def run_child(device, seed, n_geos, *_args, **_kwargs):
      del device, seed
      return {"ok": n_geos < 8}

    with mock.patch.object(
        gpu_validation, "_run_child", side_effect=run_child
    ):
      gpu_validation._memory_probe(
          1, 2, 16, pathlib.Path(tempfile.gettempdir()),
          persist=lambda attempts: seen.append(len(attempts)),
      )
    self.assertEqual(seen, [1, 2, 3])


class FitPathTest(unittest.TestCase):
  """The fit itself, run for real on whatever device this host has.

  Without this the only untested code in the script is the part that runs on
  the user's expensive GPU. A quick fit is cheap enough to keep in CI and
  proves the summaries the comparison consumes are actually produced.
  """

  def test_a_quick_fit_returns_the_fields_the_comparison_needs(self):
    result = gpu_validation._fit_once(seed=11, n_geos=2, n_times=24, quick=True)

    self.assertIn("channels", result)
    self.assertEqual(len(result["channels"]), 3)
    for channel in result["channels"]:
      for key in ("posterior_mean", "posterior_sd", "ess_bulk", "mcse"):
        self.assertIn(key, channel)
      self.assertTrue(math.isfinite(channel["posterior_mean"]))
      self.assertGreater(channel["posterior_sd"], 0.0)

    self.assertGreater(result["elapsed_seconds"], 0.0)
    self.assertIn("devices", result)
    json.dumps(evidence.json_safe(result))

  def test_mcse_is_derived_from_the_effective_sample_size(self):
    # The comparison divides by this; if it were the plain standard error the
    # test would be far too strict and would fail on correct hardware.
    result = gpu_validation._fit_once(seed=11, n_geos=2, n_times=24, quick=True)
    channel = result["channels"][0]
    if math.isfinite(channel["ess_bulk"]) and channel["ess_bulk"] > 0:
      expected = channel["posterior_sd"] / math.sqrt(channel["ess_bulk"])
      self.assertAlmostEqual(channel["mcse"], expected, places=9)

  def test_two_quick_fits_of_the_same_seed_agree_with_themselves(self):
    # Same device, same seed: the comparison must report agreement, which is
    # the sanity check that the z statistic is not simply always large.
    first = gpu_validation._fit_once(seed=11, n_geos=2, n_times=24, quick=True)
    second = gpu_validation._fit_once(seed=11, n_geos=2, n_times=24, quick=True)
    first["ok"] = second["ok"] = True
    comparison = gpu_validation._compare(first, second)
    self.assertTrue(comparison["comparable"])
    self.assertLess(comparison["max_abs_z"], 3.0)


class CommandLineTest(unittest.TestCase):

  def test_without_an_accelerator_it_records_and_exits_cleanly(self):
    # It must not fabricate a pass by comparing a device with itself.
    report = {
        "jax": {"available": True, "default_backend": "cpu"},
        "tensorflow": {"available": True, "gpus": []},
    }
    with mock.patch.object(
        gpu_validation, "_device_report", return_value=report
    ), mock.patch.object(gpu_validation, "_run_child") as run_child:
      buffer = io.StringIO()
      with contextlib.redirect_stdout(buffer):
        status = gpu_validation.main([])
    self.assertEqual(status, 0)
    run_child.assert_not_called()
    self.assertIn("No GPU is visible", buffer.getvalue())

  def test_a_disagreeing_comparison_exits_non_zero(self):
    report = {
        "jax": {"available": True, "default_backend": "cuda"},
        "tensorflow": {"available": False},
    }
    legs = {"cpu": _leg([2.0], [0.01]), "gpu": _leg([3.0], [0.01])}
    with mock.patch.object(
        gpu_validation, "_device_report", return_value=report
    ), mock.patch.object(
        gpu_validation,
        "_run_child",
        side_effect=lambda device, *a, **k: legs[device],
    ):
      with contextlib.redirect_stdout(io.StringIO()):
        status = gpu_validation.main([])
    self.assertEqual(status, 1)

  def test_an_agreeing_comparison_exits_zero(self):
    report = {
        "jax": {"available": True, "default_backend": "cuda"},
        "tensorflow": {"available": False},
    }
    legs = {"cpu": _leg([2.0], [0.05]), "gpu": _leg([2.01], [0.05])}
    with mock.patch.object(
        gpu_validation, "_device_report", return_value=report
    ), mock.patch.object(
        gpu_validation,
        "_run_child",
        side_effect=lambda device, *a, **k: legs[device],
    ):
      with contextlib.redirect_stdout(io.StringIO()):
        status = gpu_validation.main([])
    self.assertEqual(status, 0)


if __name__ == "__main__":
  unittest.main()
