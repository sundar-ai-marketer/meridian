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
import unittest
from unittest import mock

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


class JsonSafeTest(unittest.TestCase):

  def test_booleans_survive_as_booleans(self):
    # bool subclasses int; an int-first branch silently writes 1 and 0, which
    # turns "no GPU was visible" into something a reader has to decode.
    result = gpu_validation._json_safe({"a": True, "b": False})
    self.assertIs(result["a"], True)
    self.assertIs(result["b"], False)
    self.assertEqual(json.dumps(result), '{"a": true, "b": false}')

  def test_non_finite_floats_become_null_rather_than_invalid_json(self):
    result = gpu_validation._json_safe(
        {"nan": float("nan"), "inf": float("inf")}
    )
    self.assertIsNone(result["nan"])
    self.assertIsNone(result["inf"])
    json.dumps(result)

  def test_nested_structures_are_converted_throughout(self):
    result = gpu_validation._json_safe({"a": [{"b": (True, 1.5)}]})
    self.assertEqual(result, {"a": [{"b": [True, 1.5]}]})


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
    json.dumps(gpu_validation._json_safe(result))


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
    json.dumps(gpu_validation._json_safe(result))

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
