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

"""Checks that a GPU fit agrees with a CPU fit, and records the evidence.

Run on a machine with a GPU:

    python scripts/gpu_validation.py --output docs/validation/gpu-<date>.json

AUDIT.md states that GPU and CUDA environments are untested. This closes that
by measurement rather than assertion, but the measurement has to be the right
one, and the obvious one is wrong.

**Bit-identical results are not the test.** Meridian compiles through XLA, and
XLA fuses and reorders floating-point operations differently on a GPU than on a
CPU. Those differences are tiny per operation, but an MCMC trajectory is
chaotic: a difference in the last bits of one leapfrog step moves the next
proposal, and after a few hundred steps the two chains have explored the space
along different paths. Identical seeds do not produce identical draws across
devices and should not be expected to.

**Agreement within Monte Carlo error is the test.** Two correct samplers
targeting the same posterior produce estimates that differ by sampling noise.
That noise is quantifiable: the Monte Carlo standard error of a posterior mean
is the posterior standard deviation divided by the square root of the effective
sample size. This script fits the same synthetic dataset on CPU and on GPU,
then for each channel's ROI computes

    z = (mean_cpu - mean_gpu) / sqrt(mcse_cpu^2 + mcse_gpu^2)

and reports it. Under correct behaviour on both devices, `z` behaves like a
standard normal draw. A |z| above roughly 3 on any channel is worth
investigating; a |z| of 0.5 is what agreement looks like.

The two fits run in separate subprocesses because device selection is set by
environment variables read at import time, and both backends cache the choice
process-wide.

What this does NOT establish:

  - Anything about a larger model than the one it fits. Memory behaviour is the
    usual GPU failure and it is size-dependent; `--memory-probe` grows the geo
    count until something fails and records where, but that bound is specific
    to the card it ran on.
  - That the GPU is faster. Timings are recorded, but a small synthetic fit is
    often slower on a GPU than on a CPU because transfer and compilation
    dominate. Read the timing as a fact about this fit, not as a benchmark.
  - Anything about multi-GPU, mixed precision, or a cloud scheduler.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

# Runs both as `python scripts/x.py` and as `python -m unittest scripts.test_x`,
# so anchor the repo root before importing the shared helpers.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from scripts.evidence import (  # pylint: disable=g-import-not-at-top,g-bad-import-order
    git_head,
    json_safe,
    package_versions,
    sha256,
)

_CHILD_TIMEOUT_SECONDS = 60 * 60


def _device_report() -> dict[str, Any]:
  """Describes the accelerators this process can actually see."""
  report: dict[str, Any] = {
      "jax": {"available": False},
      "tensorflow": {"available": False},
  }
  try:
    import jax  # pylint: disable=g-import-not-at-top

    devices = jax.devices()
    report["jax"] = {
        "available": True,
        "default_backend": jax.default_backend(),
        "devices": [str(d) for d in devices],
        "device_kinds": sorted({d.device_kind for d in devices}),
        "x64_enabled": bool(jax.config.read("jax_enable_x64")),
    }
  except Exception as error:  # pylint: disable=broad-except
    report["jax"] = {"available": False, "error": str(error)[:300]}

  try:
    import tensorflow as tf  # pylint: disable=g-import-not-at-top

    gpus = tf.config.list_physical_devices("GPU")
    report["tensorflow"] = {
        "available": True,
        "gpus": [d.name for d in gpus],
        "built_with_cuda": bool(tf.test.is_built_with_cuda()),
    }
  except Exception as error:  # pylint: disable=broad-except
    report["tensorflow"] = {"available": False, "error": str(error)[:300]}
  return report


def _has_accelerator(report: dict[str, Any]) -> bool:
  jax_report = report.get("jax") or {}
  if jax_report.get("available") and jax_report.get("default_backend") not in (
      None,
      "cpu",
  ):
    return True
  tf_report = report.get("tensorflow") or {}
  return bool(tf_report.get("available") and tf_report.get("gpus"))


def _fit_once(
    seed: int, n_geos: int, n_times: int, quick: bool = False
) -> dict[str, Any]:
  """Fits the synthetic recovery dataset and returns ROI posterior summaries.

  `quick` shortens the chains so the path can be exercised cheaply -- by the
  test suite, and by anyone wanting to confirm the script works on their card
  before spending an hour of GPU time on the real thing. A quick fit's Monte
  Carlo error is large, so the comparison it feeds is not evidence.
  """
  import arviz as az  # pylint: disable=g-import-not-at-top
  from meridian.analysis import analyzer as analyzer_module  # pylint: disable=g-import-not-at-top
  from meridian.validation import recovery  # pylint: disable=g-import-not-at-top

  config = recovery.RecoveryConfig(
      n_geos=n_geos,
      n_times=n_times,
      n_chains=2 if quick else 4,
      n_adapt=50 if quick else 500,
      n_burnin=50 if quick else 500,
      n_keep=50 if quick else 500,
      seed=seed,
  )
  started = time.time()
  frame, realised_roi = recovery.simulate(config)
  model = recovery._build_model(frame, config)  # pylint: disable=protected-access
  model.sample_prior(20 if quick else 100, seed=config.seed)
  model.sample_posterior(
      n_chains=config.n_chains,
      n_adapt=config.n_adapt,
      n_burnin=config.n_burnin,
      n_keep=config.n_keep,
      seed=config.seed,
  )
  elapsed = time.time() - started

  roi = np.asarray(
      analyzer_module.Analyzer(
          model_context=model.model_context,
          inference_data=model.inference_data,
      ).roi()
  )
  # (chain, draw, channel) -> per-channel mean, sd, and effective sample size.
  chains, draws = roi.shape[0], roi.shape[1]
  flat = roi.reshape(-1, roi.shape[-1])

  channels = []
  for index, name in enumerate(config.channels):
    # arviz reads a bare (chain, draw) array directly; the effective sample
    # size is what turns a posterior spread into a Monte Carlo standard error.
    per_chain = roi[..., index].reshape(chains, draws)
    try:
      ess = float(np.asarray(az.ess(per_chain)).ravel()[0])
    except Exception:  # pylint: disable=broad-except
      ess = float("nan")
    mean = float(flat[:, index].mean())
    sd = float(flat[:, index].std(ddof=1))
    mcse = sd / np.sqrt(ess) if np.isfinite(ess) and ess > 0 else float("nan")
    channels.append({
        "channel": name,
        "true_roi": float(realised_roi[index]),
        "posterior_mean": mean,
        "posterior_sd": sd,
        "ess_bulk": ess,
        "mcse": float(mcse),
        "median": float(np.median(flat[:, index])),
    })

  r_hat = az.rhat(  # pytype: disable=attribute-error
      model.inference_data.posterior, method="rank"
  )
  per_variable = []
  for name in r_hat.data_vars:  # pytype: disable=attribute-error
    values = np.asarray(r_hat[name].values, dtype=float)
    finite = values[~np.isnan(values)]
    if finite.size:
      per_variable.append(float(finite.max()))

  return {
      "config": dataclasses.asdict(config),
      "elapsed_seconds": elapsed,
      "channels": channels,
      "max_r_hat": max(per_variable) if per_variable else float("nan"),
      "devices": _device_report(),
      "package_versions": package_versions(),
  }


def _memory_probe(seed: int, start_geos: int, max_geos: int) -> dict[str, Any]:
  """Grows the model until it fails, and records where."""
  attempts = []
  geos = start_geos
  while geos <= max_geos:
    try:
      started = time.time()
      _fit_once(seed=seed, n_geos=geos, n_times=104)
      attempts.append({
          "n_geos": geos,
          "ok": True,
          "elapsed_seconds": time.time() - started,
      })
    except Exception as error:  # pylint: disable=broad-except
      attempts.append({
          "n_geos": geos,
          "ok": False,
          "error": type(error).__name__,
          "message": str(error)[:400],
      })
      break
    geos *= 2
  return {"attempts": attempts}


def _child_environment(device: str) -> dict[str, str]:
  """Environment that pins a child process to CPU or to the accelerator."""
  env = dict(os.environ)
  if device == "cpu":
    env["JAX_PLATFORMS"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
  else:
    env.pop("JAX_PLATFORMS", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
  return env


def _run_child(
    device: str,
    seed: int,
    n_geos: int,
    n_times: int,
    output: pathlib.Path,
    quick: bool = False,
) -> dict[str, Any]:
  command = [
      sys.executable,
      str(pathlib.Path(__file__).resolve()),
      "--child",
      "--device",
      device,
      "--seed",
      str(seed),
      "--geos",
      str(n_geos),
      "--times",
      str(n_times),
      "--output",
      str(output),
  ]
  if quick:
    command.append("--quick")
  result = subprocess.run(
      command,
      env=_child_environment(device),
      capture_output=True,
      text=True,
      timeout=_CHILD_TIMEOUT_SECONDS,
      check=False,
  )
  if result.returncode != 0 or not output.exists():
    return {
        "device": device,
        "ok": False,
        "returncode": result.returncode,
        # Keep the tail: the traceback matters, the import chatter does not.
        "stderr_tail": result.stderr[-2000:],
    }
  payload = json.loads(output.read_text(encoding="utf-8"))
  payload["device"] = device
  payload["ok"] = True
  return payload


def _compare(cpu: dict[str, Any], gpu: dict[str, Any]) -> dict[str, Any]:
  """Compares two fits by Monte Carlo error rather than by equality."""
  if not (cpu.get("ok") and gpu.get("ok")):
    return {"comparable": False, "reason": "at least one fit did not complete"}

  by_channel = []
  worst = 0.0
  for cpu_channel, gpu_channel in zip(cpu["channels"], gpu["channels"]):
    mcse_cpu = cpu_channel["mcse"]
    mcse_gpu = gpu_channel["mcse"]
    combined = float(np.sqrt(mcse_cpu**2 + mcse_gpu**2))
    difference = cpu_channel["posterior_mean"] - gpu_channel["posterior_mean"]
    z = difference / combined if combined > 0 else float("nan")
    if np.isfinite(z):
      worst = max(worst, abs(z))
    by_channel.append({
        "channel": cpu_channel["channel"],
        "cpu_posterior_mean": cpu_channel["posterior_mean"],
        "gpu_posterior_mean": gpu_channel["posterior_mean"],
        "difference": float(difference),
        "combined_mcse": combined,
        "z": float(z),
        "relative_difference_pct": float(
            100.0 * difference / cpu_channel["posterior_mean"]
        )
        if cpu_channel["posterior_mean"]
        else float("nan"),
    })

  return {
      "comparable": True,
      "channels": by_channel,
      "max_abs_z": worst,
      "verdict": (
          "consistent within Monte Carlo error"
          if worst < 3.0
          else "a channel differs by more than 3 Monte Carlo standard errors; "
          "investigate before trusting GPU results"
      ),
      "interpretation": (
          "z is the CPU-versus-GPU difference in posterior mean ROI divided by "
          "the combined Monte Carlo standard error. Two correct samplers give "
          "z values that behave like standard normal draws. Identical draws "
          "are not expected: XLA reorders floating-point work per device and "
          "an MCMC trajectory amplifies that difference."
      ),
  }


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=pathlib.Path, default=None)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--geos", type=int, default=5)
  parser.add_argument("--times", type=int, default=104)
  parser.add_argument(
      "--memory-probe",
      action="store_true",
      help="Also grow the geo count until the fit fails, and record where.",
  )
  parser.add_argument("--max-geos", type=int, default=160)
  parser.add_argument(
      "--quick",
      action="store_true",
      help=(
          "Short chains. Confirms the script runs on your card; the resulting"
          " comparison is not evidence."
      ),
  )
  parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
  parser.add_argument("--device", default="cpu", help=argparse.SUPPRESS)
  args = parser.parse_args(argv)

  if args.child:
    if args.output is None:
      parser.error("--child requires --output")
    payload = _fit_once(
        seed=args.seed,
        n_geos=args.geos,
        n_times=args.times,
        quick=args.quick,
    )
    args.output.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0

  devices = _device_report()
  accelerator = _has_accelerator(devices)
  print("devices visible to this process:")
  print(json.dumps(json_safe(devices), indent=2))
  if not accelerator:
    print(
        "\nNo GPU is visible. Recording the environment and stopping: running "
        "both legs on the same CPU would compare a device with itself and "
        "prove nothing."
    )

  scratch = pathlib.Path(
      os.environ.get("TMPDIR", "/tmp")
  ) / f"meridian-gpu-validation-{os.getpid()}"
  scratch.mkdir(parents=True, exist_ok=True)

  legs: dict[str, Any] = {}
  if accelerator:
    for device in ("cpu", "gpu"):
      print(f"\nfitting on {device} ...")
      legs[device] = _run_child(
          device,
          args.seed,
          args.geos,
          args.times,
          scratch / f"{device}.json",
          quick=args.quick,
      )
      status = "ok" if legs[device].get("ok") else "FAILED"
      print(f"  {device}: {status}")

  comparison = (
      _compare(legs["cpu"], legs["gpu"])
      if {"cpu", "gpu"} <= legs.keys()
      else {"comparable": False, "reason": "no accelerator visible"}
  )

  evidence: dict[str, Any] = {
      "date": datetime.date.today().isoformat(),
      "script": "scripts/gpu_validation.py",
      "script_sha256": sha256(pathlib.Path(__file__).resolve()),
      "git_head": git_head(),
      "host": {
          "platform": platform.platform(),
          "machine": platform.machine(),
          "python": platform.python_version(),
      },
      "quick_mode": bool(args.quick),
      "accelerator_visible": accelerator,
      "devices": devices,
      "legs": legs,
      "comparison": comparison,
      "limitations": [
          "Bit-identical results across devices are not expected and are not "
          "tested; XLA reorders floating-point work per device.",
          "Agreement is judged against Monte Carlo error on one synthetic "
          "dataset at one size.",
          "Timings describe this fit on this hardware and are not a benchmark; "
          "a small fit is often slower on a GPU than on a CPU.",
          "Nothing here covers multi-GPU, mixed precision, or cloud "
          "schedulers.",
      ],
  }

  if args.memory_probe and accelerator:
    print("\nmemory probe: growing the geo count until the fit fails ...")
    evidence["memory_probe"] = _memory_probe(args.seed, args.geos, args.max_geos)

  if comparison.get("comparable"):
    print(f"\nmax |z| across channels: {comparison['max_abs_z']:.2f}")
    print(comparison["verdict"])

  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_safe(evidence), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {args.output}")

  if comparison.get("comparable") and comparison["max_abs_z"] >= 3.0:
    return 1
  return 0


if __name__ == "__main__":
  sys.exit(main())
