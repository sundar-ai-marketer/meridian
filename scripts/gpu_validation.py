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

"""Measures what an accelerator can do for Meridian, and records the evidence.

Run on a machine with a GPU:

    python scripts/gpu_validation.py --output docs/validation/gpu-<date>.json

To answer only "can this machine's accelerator run the model at all", which
takes seconds rather than an hour:

    python scripts/gpu_validation.py --capability-only --output <path>

The script answers two questions in order.

**First: can the accelerator run the operations the model needs?** A device
registering with JAX says nothing about which XLA operations its plugin
lowers, and the model needs a specific set -- tensor contraction, Cholesky
factorisation, a scan, a gradient, and the rest of `_REQUIRED_OPERATIONS`. The
probe runs each one on the device at the configured precision and names both
the failure and the part of the model that needed it. When an operation is
missing the script records that and stops, because the fits cannot succeed.

**Second: does a GPU fit agree with a CPU fit?** The right comparison here is
agreement within Monte Carlo error, not bit-identical draws. Meridian compiles
through XLA, and XLA fuses and reorders floating-point operations differently
on a GPU than on a CPU. Those differences are tiny per operation, but an MCMC
trajectory is chaotic: a difference in the last bits of one leapfrog step
moves the next proposal, and after a few hundred steps the two chains have
explored the space along different paths. Identical seeds do not produce
identical draws across devices.

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

Every leg runs in its own subprocess. Device selection is set by environment
variables read at import time and cached process-wide, and a plugin that
cannot lower an operation -- or a device that runs out of memory -- may
terminate the process instead of raising, which would otherwise take the
collected evidence with it.

Exit codes:

  0  evidence recorded; any comparison performed agreed
  1  a comparison was performed and a channel exceeded 3 Monte Carlo errors
  2  an accelerator was visible but a fit leg did not complete
  3  an accelerator was visible but cannot run the model's operations

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
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
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
    provenance,
    sha256,
    write_evidence,
)

_CHILD_TIMEOUT_SECONDS = 60 * 60
# An operation probe either answers in seconds or the plugin is wedged.
_PROBE_TIMEOUT_SECONDS = 5 * 60

# The XLA operations the sampler cannot do without, each named with the part of
# the model that needs it, so a failure reads as a consequence rather than an
# opcode. A device that registers with JAX has not thereby shown it can run
# any of these: an accelerator plugin lowers a subset of XLA, and the subset is
# what decides whether a fit is possible.
_REQUIRED_OPERATIONS: tuple[tuple[str, str], ...] = (
    ("dot_general", "every tensor contraction: media transforms and the linear predictor"),
    ("cholesky", "multivariate normal draws in the hierarchical geo prior"),
    ("grad", "the gradient each leapfrog step of HMC needs"),
    ("scan", "the sampler's trajectory loop"),
    ("while_loop", "NUTS tree building"),
    ("random_normal", "draw generation"),
    ("cumsum", "adstock accumulation"),
    ("erf", "normal CDFs in the likelihood"),
    ("gammaln", "gamma and negative-binomial log densities"),
    ("sort", "posterior quantiles and rank-normalised R-hat"),
)


def _probe_operations() -> dict[str, Any]:
  """Runs each required operation on the default device and records the outcome.

  Runs in a child process: a plugin that cannot lower an operation may abort
  the process at the driver level rather than raise, and an abort must not take
  the surrounding evidence with it.
  """
  # Importing the backend applies Meridian's own precision choice, which is
  # what the probe has to run at to be about Meridian. Without the package
  # installed, read the same environment variable the backend reads.
  try:
    from meridian import backend as _backend  # pylint: disable=g-import-not-at-top,unused-import
  except Exception:  # pylint: disable=broad-except
    import jax as _jax_for_config  # pylint: disable=g-import-not-at-top

    _jax_for_config.config.update(
        "jax_enable_x64",
        os.environ.get("MERIDIAN_ENABLE_JAX_X64", "true").lower()
        in ("1", "true"),
    )

  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top

  x64 = bool(jax.config.read("jax_enable_x64"))
  dtype = jnp.float64 if x64 else jnp.float32

  # Every probe builds its own inputs inside its own body, including the PRNG
  # key. A device whose plugin cannot lower anything at all fails on the first
  # operation, and shared setup outside the guarded loop would let that
  # failure escape as a crash -- reported as "inconclusive", which is the one
  # answer this probe exists to avoid.
  probes = {
      "dot_general": lambda: float(
          jnp.einsum(
              "gtc,gtc->g",
              jnp.ones((3, 8, 2), dtype),
              jnp.ones((3, 8, 2), dtype),
          ).sum()
      ),
      "cholesky": lambda: float(
          jnp.linalg.cholesky(jnp.eye(16, dtype=dtype) * 2.0).sum()
      ),
      "grad": lambda: float(
          jax.grad(lambda v: jnp.sum(jnp.log1p(jnp.exp(v))))(jnp.ones(8, dtype))[0]
      ),
      "scan": lambda: float(
          jax.jit(
              lambda v: jax.lax.scan(
                  lambda c, x: (c + x, c), jnp.zeros((), dtype), v
              )[0]
          )(jnp.arange(64, dtype=dtype))
      ),
      "while_loop": lambda: int(
          jax.lax.while_loop(lambda i: i < 32, lambda i: i + 1, 0)
      ),
      "random_normal": lambda: float(
          jax.random.normal(jax.random.PRNGKey(0), (512,), dtype).std()
      ),
      "cumsum": lambda: float(jnp.cumsum(jnp.ones(128, dtype))[-1]),
      "erf": lambda: float(jax.scipy.special.erf(jnp.asarray(0.5, dtype))),
      "gammaln": lambda: float(jax.scipy.special.gammaln(jnp.asarray(3.5, dtype))),
      "sort": lambda: float(
          jnp.sort(jax.random.normal(jax.random.PRNGKey(0), (256,), dtype))[0]
      ),
  }

  results: dict[str, Any] = {}
  for name, need in _REQUIRED_OPERATIONS:
    try:
      value = probes[name]()
      results[name] = {"ok": True, "needed_for": need, "value": value}
    except Exception as error:  # pylint: disable=broad-except
      # The first line carries the lowering diagnostic. Keep the note line
      # too when there is one: for a plugin that cannot read the compiler's
      # output at all, the bytecode version it choked on is the whole story.
      lines = [
          line.strip() for line in str(error).strip().splitlines() if line.strip()
      ]
      detail = lines[0] if lines else ""
      note = next((line for line in lines[1:] if "bytecode version" in line), "")
      results[name] = {
          "ok": False,
          "needed_for": need,
          "error": type(error).__name__,
          "message": (f"{detail} ({note})" if note else detail)[:400],
      }

  try:
    resolved_dtype = str(jnp.zeros((), dtype).dtype)
  except Exception:  # pylint: disable=broad-except
    # Even materialising a zero scalar is a device operation.
    resolved_dtype = "float64" if x64 else "float32"

  return {
      "x64_enabled": x64,
      "dtype": resolved_dtype,
      "operations": results,
      "unsupported": sorted(n for n, r in results.items() if not r["ok"]),
      "devices": _device_report(),
  }


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
  from meridian.analysis import sampling_diagnostics  # pylint: disable=g-import-not-at-top
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

  max_r_hat = sampling_diagnostics.max_rank_normalized_rhat(
      model.inference_data.posterior
  )

  return {
      "config": dataclasses.asdict(config),
      "elapsed_seconds": elapsed,
      "channels": channels,
      "max_r_hat": max_r_hat,
      "devices": _device_report(),
      "package_versions": package_versions(),
  }


def _memory_probe(
    seed: int,
    start_geos: int,
    max_geos: int,
    scratch: pathlib.Path,
    persist: Callable[[list[dict[str, Any]]], None] | None = None,
) -> dict[str, Any]:
  """Grows the model until it fails, and records where.

  Each attempt runs in its own child process, and `persist` is called with
  everything known so far after each one. Running out of memory on an
  accelerator commonly aborts the process instead of raising a Python
  exception, so an in-process loop would lose the failure and every attempt
  that preceded it -- which is the whole result. Isolating the attempt turns an
  abort into a recorded return code.
  """
  attempts: list[dict[str, Any]] = []
  geos = start_geos
  while geos <= max_geos:
    started = time.time()
    leg = _run_child("gpu", seed, geos, 104, scratch / f"memory-{geos}.json")
    elapsed = time.time() - started
    if leg.get("ok"):
      attempts.append(
          {"n_geos": geos, "ok": True, "elapsed_seconds": elapsed}
      )
    else:
      attempts.append({
          "n_geos": geos,
          "ok": False,
          "elapsed_seconds": elapsed,
          "returncode": leg.get("returncode"),
          "stderr_tail": leg.get("stderr_tail"),
      })
      if persist is not None:
        persist(attempts)
      break
    if persist is not None:
      persist(attempts)
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
    mode: str = "fit",
) -> dict[str, Any]:
  """Runs one leg in a fresh process pinned to `device`.

  `mode` selects what the child does: `fit` samples the synthetic dataset,
  `probe-ops` runs the required-operation battery. Both need their own process
  because device selection is read at import time and cached process-wide, and
  because either can be terminated by the driver rather than raise.
  """
  command = [
      sys.executable,
      str(pathlib.Path(__file__).resolve()),
      "--child",
      "--mode",
      mode,
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
      timeout=(
          _PROBE_TIMEOUT_SECONDS
          if mode == "probe-ops"
          else _CHILD_TIMEOUT_SECONDS
      ),
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

  cpu_names = [channel["channel"] for channel in cpu["channels"]]
  gpu_names = [channel["channel"] for channel in gpu["channels"]]
  if cpu_names != gpu_names:
    # Pairing by position would compare one channel's ROI against another's
    # and call the difference a device discrepancy.
    return {
        "comparable": False,
        "reason": "the two fits reported different channels",
        "cpu_channels": cpu_names,
        "gpu_channels": gpu_names,
    }

  by_channel = []
  worst = 0.0
  compared = 0
  for cpu_channel, gpu_channel in zip(cpu["channels"], gpu["channels"]):
    mcse_cpu = cpu_channel["mcse"]
    mcse_gpu = gpu_channel["mcse"]
    combined = float(np.sqrt(mcse_cpu**2 + mcse_gpu**2))
    difference = cpu_channel["posterior_mean"] - gpu_channel["posterior_mean"]
    z = difference / combined if combined > 0 else float("nan")
    if np.isfinite(z):
      compared += 1
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

  if not compared:
    # Every channel's Monte Carlo error was zero or undefined, so there is no
    # yardstick to measure the difference against. `worst` would still be its
    # initial 0.0 here, and returning that as agreement would report the
    # absence of a measurement as a pass.
    return {
        "comparable": False,
        "reason": (
            "Monte Carlo error was unavailable for every channel, so the two "
            "fits cannot be compared"
        ),
        "channels": by_channel,
        "channels_total": len(by_channel),
    }

  return {
      "comparable": True,
      "channels": by_channel,
      "channels_compared": compared,
      "channels_total": len(by_channel),
      "max_abs_z": worst,
      "verdict": (
          (
              "consistent within Monte Carlo error"
              if compared == len(by_channel)
              else f"consistent within Monte Carlo error on the {compared} of "
              f"{len(by_channel)} channels that had a usable Monte Carlo "
              "error; the rest were not compared"
          )
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
  parser.add_argument(
      "--capability-only",
      action="store_true",
      help=(
          "Report which required operations the visible accelerator can run,"
          " and stop without fitting."
      ),
  )
  parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
  parser.add_argument("--mode", default="fit", help=argparse.SUPPRESS)
  parser.add_argument("--device", default="cpu", help=argparse.SUPPRESS)
  args = parser.parse_args(argv)

  if args.child:
    if args.output is None:
      parser.error("--child requires --output")
    if args.mode == "probe-ops":
      payload = _probe_operations()
    else:
      payload = _fit_once(
          seed=args.seed,
          n_geos=args.geos,
          n_times=args.times,
          quick=args.quick,
      )
    write_evidence(args.output, payload)
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

  # A device that registers with JAX has not thereby shown it can run the
  # model. Establish that first: an hour of sampling that ends in an opaque
  # lowering error answers the same question far more slowly, and a plugin
  # that cannot lower one operation names itself in seconds.
  capability: dict[str, Any] = {"probed": False}
  if accelerator:
    print("\nchecking which required operations this accelerator can run ...")
    probe = _run_child(
        "gpu", args.seed, args.geos, args.times,
        scratch / "capability.json", mode="probe-ops",
    )
    if probe.get("ok") and isinstance(probe.get("operations"), dict):
      capability = {
          "probed": True,
          "x64_enabled": probe.get("x64_enabled"),
          "dtype": probe.get("dtype"),
          "operations": probe["operations"],
          "unsupported": probe.get("unsupported") or [],
      }
      if capability["unsupported"]:
        for name in capability["unsupported"]:
          detail = probe["operations"][name]
          print(
              f"  {name}: unsupported -- needed for {detail['needed_for']}"
          )
          print(f"    {detail.get('error')}: {detail.get('message')}")
      else:
        print("  all required operations ran on the accelerator")
    else:
      # Treat an unreadable probe as inconclusive and go on to the fits: the
      # fits are the measurement, and the probe exists only to save time.
      capability = {
          "probed": False,
          "reason": "the capability probe did not return a readable result",
          "returncode": probe.get("returncode"),
          "stderr_tail": probe.get("stderr_tail"),
      }
      print("  inconclusive; continuing to the fits")

  blocked = bool(capability.get("probed") and capability.get("unsupported"))

  legs: dict[str, Any] = {}
  if accelerator and not blocked and not args.capability_only:
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

  if {"cpu", "gpu"} <= legs.keys():
    comparison = _compare(legs["cpu"], legs["gpu"])
  elif blocked:
    comparison = {
        "comparable": False,
        "reason": (
            "the accelerator cannot run every operation the model needs: "
            + ", ".join(capability["unsupported"])
        ),
    }
  elif args.capability_only:
    comparison = {
        "comparable": False,
        "reason": "--capability-only was requested, so no fit was run",
    }
  else:
    comparison = {"comparable": False, "reason": "no accelerator visible"}

  evidence: dict[str, Any] = {
      "date": datetime.date.today().isoformat(),
      "script": "scripts/gpu_validation.py",
      # `script_sha256` and `git_head` are also kept at the top level: they
      # were there before `provenance()` existed and readers reference them.
      "script_sha256": sha256(pathlib.Path(__file__).resolve()),
      "git_head": git_head(),
      "provenance": provenance(pathlib.Path(__file__).resolve()),
      "host": {
          "platform": platform.platform(),
          "machine": platform.machine(),
          "python": platform.python_version(),
      },
      "quick_mode": bool(args.quick),
      "accelerator_visible": accelerator,
      "accelerator_can_run_model": (
          None if not capability.get("probed") else not blocked
      ),
      "capability": capability,
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
          "The capability probe covers the operations listed in "
          "_REQUIRED_OPERATIONS at the configured precision. An accelerator "
          "that passes it has shown it can run those operations, not that a "
          "full fit succeeds.",
      ],
  }

  def _write_evidence() -> None:
    if args.output is None:
      return
    write_evidence(args.output, evidence)

  if args.memory_probe and accelerator and not blocked:
    print("\nmemory probe: growing the geo count until the fit fails ...")

    def _persist(attempts: list[dict[str, Any]]) -> None:
      # Written after every attempt so a driver-level abort on the next one
      # still leaves the bound that was reached on disk.
      evidence["memory_probe"] = {"attempts": attempts}
      _write_evidence()

    evidence["memory_probe"] = _memory_probe(
        args.seed, args.geos, args.max_geos, scratch, persist=_persist
    )

  if comparison.get("comparable"):
    print(f"\nmax |z| across channels: {comparison['max_abs_z']:.2f}")
    print(comparison["verdict"])

  _write_evidence()
  if args.output is not None:
    print(f"\nwrote {args.output}")

  shutil.rmtree(scratch, ignore_errors=True)

  # Exit codes, so a caller can tell the outcomes apart:
  #   0  evidence recorded; any comparison performed agreed
  #   1  a comparison was performed and a channel exceeded 3 MCSE
  #   2  an accelerator was visible but a fit leg did not complete
  #   3  an accelerator was visible but cannot run the model's operations
  if comparison.get("comparable") and comparison["max_abs_z"] >= 3.0:
    return 1
  if blocked:
    return 3
  if legs and not all(leg.get("ok") for leg in legs.values()):
    # The failure this script exists to catch. Reporting it as a clean exit
    # would make a crashed GPU leg indistinguishable from a passing run.
    return 2
  return 0


if __name__ == "__main__":
  sys.exit(main())
