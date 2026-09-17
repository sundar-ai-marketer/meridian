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

"""Measures interval coverage across dataset shapes, not just one of them.

    # See the plan and the runtime estimate without fitting anything:
    python scripts/coverage_grid.py --dry-run

    # A real run. This takes hours; start it in the evening.
    python scripts/coverage_grid.py --replications 10 \
        --output docs/validation/coverage-grid-<date>.json

Why this exists. AUDIT.md's ten-seed study measures recovery at one fixed
setting: five geos, 104 periods, 5% noise, a concave response the model expects.
That is one point. It says nothing about whether the 90% interval still covers
at 40 periods, or at 20% noise, or when the true response is linear and the
model's saturation assumption is wrong. Those are the conditions a real dataset
actually lands in, and coverage is exactly the quantity that degrades there.

Each grid cell runs `replications` independent simulate-and-fit cycles and
reports empirical coverage with a Wilson interval, because coverage estimated
from ten trials is itself noisy: 9/10 is consistent with anything from roughly
60% to 99% true coverage. The Wilson bound is printed so nobody reads 90% off
ten trials as a precise number.

The `linear` response cells are deliberate misspecification. Meridian assumes a
concave saturating response; simulating a linear one and refitting sizes the
bias from that assumption being wrong. Expect worse coverage there. That is the
measurement, not a bug.

What this does NOT establish:

  - Calibration in the Bayesian sense. Fixed-truth frequentist coverage over
    repeated simulations is a different quantity from the posterior
    probability, and they are not guaranteed to match. See AUDIT.md.
  - Anything about a real dataset's collinearity or spend variation.
  - Anything for cells whose replications did not converge; those are reported
    with their R-hat rather than silently averaged in.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import pathlib
import sys
import time
from collections.abc import Sequence
from typing import Literal

import numpy as np

# Runs both as `python scripts/x.py` and as `python -m unittest scripts.test_x`,
# so anchor the repo root before importing the shared helpers.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from scripts.evidence import provenance, write_evidence  # pylint: disable=g-import-not-at-top,g-bad-import-order

# (label, n_geos, n_times, noise_fraction, response). Chosen to vary one thing
# at a time from the audit's baseline so a coverage drop is attributable.
_GRID: tuple[tuple[str, int, int, float, Literal["linear", "concave"]], ...] = (
    ("baseline", 5, 104, 0.05, "concave"),
    ("short history", 5, 40, 0.05, "concave"),
    ("few geos", 2, 104, 0.05, "concave"),
    ("high noise", 5, 104, 0.20, "concave"),
    ("misspecified response", 5, 104, 0.05, "linear"),
)


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=pathlib.Path, default=None)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument(
      "--replications",
      type=int,
      default=10,
      help="Replications per grid cell. Coverage from fewer than ~10 is noise.",
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Print the grid and a runtime estimate, fit nothing.",
  )
  parser.add_argument(
      "--quick",
      action="store_true",
      help="Short chains and 2 replications. Proves the script runs; not evidence.",
  )
  args = parser.parse_args(argv)

  from meridian.validation import recovery  # pylint: disable=g-import-not-at-top

  short = args.quick
  replications = 2 if short else args.replications
  total_fits = replications * len(_GRID)

  print(f"grid cells: {len(_GRID)}  replications each: {replications}")
  print(f"total simulate-and-fit cycles: {total_fits}")
  print(
      "at roughly 2 minutes per fit that is about "
      f"{total_fits * 2 / 60:.1f} hours on a laptop CPU.\n"
  )
  for label, geos, times, noise, response in _GRID:
    print(
        f"  {label:22s} geos={geos:<3d} times={times:<4d} "
        f"noise={noise:<5.2f} response={response}"
    )
  if args.dry_run:
    print("\n--dry-run: nothing was fitted.")
    return 0
  if short:
    print("\n--quick: chains are too short and replications too few to be evidence.")
  print()

  cells = []
  started_all = time.time()
  for label, geos, times, noise, response in _GRID:
    config = recovery.RecoveryConfig(
        n_geos=geos,
        n_times=times,
        noise_fraction=noise,
        response=response,
        seed=args.seed,
        replications=replications,
        n_chains=2 if short else 4,
        n_adapt=100 if short else 1_000,
        n_burnin=100 if short else 500,
        n_keep=100 if short else 500,
    )
    print(f"--- {label} ---")
    started = time.time()
    result = recovery.run_recovery_replications(config)
    elapsed = time.time() - started

    channels = []
    for summary in result.channel_summaries():
      channels.append({
          "channel": summary.channel,
          "n": summary.n,
          "coverage": summary.coverage,
          "coverage_ci_low": summary.coverage_ci_low,
          "coverage_ci_high": summary.coverage_ci_high,
          "median_relative_error": summary.median_relative_error,
          "median_ci_width": summary.median_ci_width,
      })
      print(
          f"    {summary.channel:12s} coverage {summary.coverage:5.2f} "
          f"[{summary.coverage_ci_low:.2f}, {summary.coverage_ci_high:.2f}] "
          f"median rel. error {summary.median_relative_error:+.3f}"
      )

    converged = [r.converged for r in result.replications]
    max_r_hat = max(
        (r.max_r_hat for r in result.replications if np.isfinite(r.max_r_hat)),
        default=float("nan"),
    )
    if not all(converged):
      print(
          f"    WARNING: {converged.count(False)}/{len(converged)} replications "
          f"did not meet the R-hat screen (max R-hat {max_r_hat:.3f}); "
          "read this cell's coverage with that in mind."
      )

    cells.append({
        "label": label,
        "config": dataclasses.asdict(config),
        "nominal_interval": config.confidence_level,
        "channels": channels,
        "replications_converged": sum(converged),
        "replications_total": len(converged),
        "max_r_hat": max_r_hat,
        "elapsed_seconds": elapsed,
    })
    print(f"    ({elapsed / 60:.1f} min)\n")

  evidence = {
      "date": datetime.date.today().isoformat(),
      "script": "scripts/coverage_grid.py",
      "provenance": provenance(pathlib.Path(__file__).resolve()),
      "quick_mode": bool(short),
      "replications_per_cell": replications,
      "base_seed": args.seed,
      "total_elapsed_seconds": time.time() - started_all,
      "cells": cells,
      "interpretation": (
          "Empirical coverage of the nominal interval, per channel, per "
          "dataset shape, with a Wilson interval on the coverage estimate "
          "itself. The 'misspecified response' cell simulates a linear "
          "response the model does not assume; worse coverage there sizes "
          "that assumption's cost."
      ),
      "limitations": [
          "Fixed-truth frequentist coverage is not Bayesian calibration.",
          "Coverage from ten replications is noisy; read the Wilson interval.",
          "Cells with non-converged replications are reported with their "
          "R-hat rather than excluded.",
          "Synthetic data only; says nothing about a real dataset's "
          "collinearity.",
      ],
  }

  if args.output is not None:
    write_evidence(args.output, evidence)
    print(f"wrote {args.output}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
