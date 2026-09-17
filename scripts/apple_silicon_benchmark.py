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

"""Measures the effect of Meridian's precision setting on this machine.

    python scripts/apple_silicon_benchmark.py --reps 3 \
        --output docs/validation/apple-silicon-benchmark-<date>.json

Apple Silicon has no GPU backend for this model -- `scripts/gpu_validation.py
--capability-only` records why -- so the CPU configuration is the whole of the
performance question on a Mac. The one setting that plausibly changes it is
`MERIDIAN_ENABLE_JAX_X64`, which selects float64 (the default) or float32.

Lower precision is widely assumed to be faster. Measuring it is cheap, so this
script measures it rather than assuming either way, and reports the median of
several repetitions with the observed spread, because a single timing on a
laptop is mostly noise.

It drives `python -m meridian.benchmark` in a fresh subprocess per repetition
rather than timing anything itself. The backend reads its precision from the
environment at import time, so a new process is the only way to change it, and
a cold process is also what a user's job actually pays for.

**What a throughput difference here does and does not mean.** Wall-clock time
for a fixed `n_keep` is what a user waits, and that is what this reports. It is
not a pure arithmetic-speed comparison: changing precision changes the
numerical trajectory NUTS explores, so the two configurations can do different
amounts of work to produce the same number of kept draws. Read the result as
"this configuration finishes this job in this long on this machine", which is
the decision-relevant quantity, and not as "float32 arithmetic is slower".
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from typing import Any

# Runs both as `python scripts/x.py` and as `python -m unittest scripts.test_x`,
# so anchor the repo root before importing the shared helpers.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from scripts.evidence import (  # pylint: disable=g-import-not-at-top,g-bad-import-order
    provenance,
    write_evidence,
)

_REP_TIMEOUT_SECONDS = 30 * 60

# The precision values the backend accepts, mapped to the env var it reads.
_PRECISIONS = {"float64": "true", "float32": "false"}

# Metrics worth aggregating, as (section, key) into the benchmark's JSON.
_METRICS: tuple[tuple[str, str], ...] = (
    ("timings_seconds", "sample_posterior"),
    ("timings_seconds", "sample_prior"),
    ("timings_seconds", "build_model"),
    ("timings_seconds", "total"),
    ("throughput", "mcmc_draws_per_sec"),
    ("throughput", "seconds_per_1k_mcmc_draws"),
)


def _run_benchmark(
    precision: str, extra_args: Sequence[str], scratch: pathlib.Path
) -> dict[str, Any]:
  """Runs one benchmark repetition in a fresh process at `precision`."""
  env = dict(os.environ)
  env["MERIDIAN_ENABLE_JAX_X64"] = _PRECISIONS[precision]
  output = scratch / f"{precision}-{os.getpid()}-{id(extra_args)}.json"
  result = subprocess.run(
      [
          sys.executable,
          "-m",
          "meridian.benchmark",
          "--json",
          str(output),
          *extra_args,
      ],
      cwd=_REPO_ROOT,
      env=env,
      capture_output=True,
      text=True,
      timeout=_REP_TIMEOUT_SECONDS,
      check=False,
  )
  if result.returncode != 0 or not output.exists():
    return {
        "ok": False,
        "returncode": result.returncode,
        "stderr_tail": result.stderr[-2000:],
    }
  payload = json.loads(output.read_text(encoding="utf-8"))
  payload["ok"] = True
  return payload


def _aggregate(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
  """Median, min and max per metric across repetitions."""
  usable = [run for run in runs if run.get("ok")]
  summary: dict[str, Any] = {
      "repetitions": len(runs),
      "repetitions_usable": len(usable),
  }
  if not usable:
    return summary

  metrics: dict[str, Any] = {}
  for section, key in _METRICS:
    values = [
        run[section][key]
        for run in usable
        if isinstance(run.get(section), dict) and key in run[section]
    ]
    if not values:
      continue
    metrics[f"{section}.{key}"] = {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "values": values,
    }
  summary["metrics"] = metrics

  peaks = [run["peak_rss_bytes"] for run in usable if run.get("peak_rss_bytes")]
  if peaks:
    summary["peak_rss_bytes"] = {
        "median": statistics.median(peaks),
        "min": min(peaks),
        "max": max(peaks),
    }
  # One environment block is enough; they agree except for the precision.
  summary["environment"] = usable[0].get("environment")
  summary["config"] = usable[0].get("config")
  return summary


def _compare(by_precision: dict[str, dict[str, Any]]) -> dict[str, Any]:
  """Compares each precision against float64, the shipped default."""
  baseline = by_precision.get("float64", {})
  base_metrics = baseline.get("metrics") or {}
  if not base_metrics:
    return {"comparable": False, "reason": "the float64 baseline did not run"}

  comparisons: dict[str, Any] = {}
  for precision, summary in by_precision.items():
    if precision == "float64":
      continue
    metrics = summary.get("metrics") or {}
    if not metrics:
      comparisons[precision] = {
          "comparable": False,
          "reason": "no usable repetition",
      }
      continue

    deltas = {}
    for name, base in base_metrics.items():
      other = metrics.get(name)
      if not other or not base["median"]:
        continue
      deltas[name] = {
          "float64_median": base["median"],
          f"{precision}_median": other["median"],
          "percent_change": 100.0 * (other["median"] / base["median"] - 1.0),
          # Non-overlapping ranges are what separate a real difference from
          # laptop noise; a reader should not have to work that out.
          "ranges_overlap": not (
              other["min"] > base["max"] or other["max"] < base["min"]
          ),
      }
    comparisons[precision] = {"comparable": True, "metrics": deltas}
  return comparisons


def _table(by_precision: dict[str, dict[str, Any]]) -> str:
  """A compact human-readable summary."""
  names = [p for p in _PRECISIONS if p in by_precision]
  lines = [f"{'metric':34s}" + "".join(f"{n:>22s}" for n in names)]
  lines.append("-" * len(lines[0]))
  first = by_precision[names[0]].get("metrics") or {}
  for name in first:
    row = f"{name:34s}"
    for precision in names:
      metric = (by_precision[precision].get("metrics") or {}).get(name)
      row += (
          f"{metric['median']:12.2f} [{metric['min']:.2f}-{metric['max']:.2f}]"
          if metric
          else f"{'--':>22s}"
      )
    lines.append(row)
  for precision in names:
    peak = by_precision[precision].get("peak_rss_bytes")
    if peak:
      lines.append(
          f"{'peak_rss_gb (' + precision + ')':34s}"
          f"{peak['median'] / 2**30:12.2f}"
      )
  return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=pathlib.Path, default=None)
  parser.add_argument(
      "--reps",
      type=int,
      default=3,
      help="Repetitions per precision (default: 3). One is not a measurement.",
  )
  parser.add_argument(
      "--precision",
      action="append",
      choices=sorted(_PRECISIONS),
      help="Restrict to one precision. Repeatable. Default: both.",
  )
  args, benchmark_args = parser.parse_known_args(argv)

  if args.reps < 1:
    parser.error("--reps must be at least 1")
  precisions = args.precision or list(_PRECISIONS)

  by_precision: dict[str, dict[str, Any]] = {}
  with tempfile.TemporaryDirectory(prefix="meridian-precision-") as raw:
    scratch = pathlib.Path(raw)
    for precision in precisions:
      runs = []
      for rep in range(args.reps):
        print(f"{precision}: repetition {rep + 1} of {args.reps} ...", flush=True)
        run = _run_benchmark(precision, benchmark_args, scratch)
        if not run.get("ok"):
          print(f"  FAILED (exit {run.get('returncode')})", flush=True)
        runs.append(run)
      by_precision[precision] = _aggregate(runs)

  evidence: dict[str, Any] = {
      "date": datetime.date.today().isoformat(),
      "script": "scripts/apple_silicon_benchmark.py",
      "provenance": provenance(pathlib.Path(__file__).resolve()),
      "host": {
          "platform": platform.platform(),
          "machine": platform.machine(),
          "processor": platform.processor(),
          "python": platform.python_version(),
      },
      "benchmark_args": list(benchmark_args),
      "by_precision": by_precision,
      "comparison": _compare(by_precision),
      "limitations": [
          "Wall-clock time for a fixed number of kept draws, on one machine, "
          "at one model size. Not a cross-machine benchmark.",
          "Changing precision changes the numerical trajectory NUTS explores, "
          "so a throughput difference is not purely an arithmetic-speed "
          "difference.",
          "Medians over a handful of repetitions on a laptop. Compare the "
          "reported ranges before treating a small difference as real.",
          "Says nothing about accuracy. A precision change can alter results, "
          "which this does not measure.",
      ],
  }

  print()
  print(_table(by_precision))

  if args.output is not None:
    write_evidence(args.output, evidence)
    print(f"\nwrote {args.output}")

  # Nonzero when a repetition failed: a partially-measured comparison should
  # not read as a successful run.
  for summary in by_precision.values():
    if summary["repetitions_usable"] != summary["repetitions"]:
      return 1
  return 0


if __name__ == "__main__":
  sys.exit(main())
