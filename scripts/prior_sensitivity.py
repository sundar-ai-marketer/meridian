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

"""Measures how much of a reported ROI comes from the prior rather than the data.

    python scripts/prior_sensitivity.py \
        --output docs/validation/prior-sensitivity-<date>.json

Why this exists. ROI in a marketing mix model is weakly identified: spend moves
slowly, channels move together, and the data often cannot separate them. When
that happens the prior does the separating, quietly. A single fit reports a
number and an interval and gives no hint how much of either was assumed.

This holds the dataset fixed -- one simulation, one known true ROI per channel
-- and refits it under several ROI priors. If the posterior barely moves, the
data is deciding. If the posterior tracks the prior, the reported ROI is
largely an assumption and the interval understates that.

The reported quantity is the spread of the posterior median across priors,
relative to the true ROI:

    prior_share = (max_median - min_median) / true_roi

Read it as the fraction of the true value that the choice of prior alone can
move the answer by. A channel where a defensible change of prior moves the
median by half the true ROI is not a channel to make a budget decision on
without saying so.

What this does NOT establish:

  - That any one prior is correct. It measures sensitivity, not accuracy.
  - Anything about a real dataset. It runs on the synthetic recovery fixture,
    where the truth is known; a real dataset's identifiability depends on its
    own spend variation and collinearity. Point `--geos`/`--times` at a shape
    like yours, but the honest version of this check is to run the same sweep
    on your own data.
  - Calibration. Sensitivity to the prior is not the same as the interval
    being wrong.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import pathlib
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

from scripts.evidence import provenance, write_evidence  # pylint: disable=g-import-not-at-top,g-bad-import-order

# (label, prior median ROI, prior log-scale sigma). The spread is deliberate:
# a sceptical analyst, the library's own documented idiom, and an optimistic
# one. All three are defensible before seeing the data, which is the point.
_PRIOR_GRID: tuple[tuple[str, float, float], ...] = (
    ("sceptical, tight", 0.8, 0.3),
    ("sceptical, loose", 0.8, 0.9),
    ("neutral", 2.0, 0.7),
    ("optimistic, tight", 5.0, 0.3),
    ("optimistic, loose", 5.0, 0.9),
)


def _fit_under_prior(recovery, frame, config, median: float, sigma: float):
  """Refits the SAME data under one ROI prior; returns per-channel summaries."""
  from meridian.analysis import analyzer as analyzer_module  # pylint: disable=g-import-not-at-top
  from meridian.analysis import sampling_diagnostics  # pylint: disable=g-import-not-at-top

  tuned = dataclasses.replace(
      config, prior_roi_median=median, prior_roi_sigma=sigma
  )
  model = recovery._build_model(frame, tuned)  # pylint: disable=protected-access
  model.sample_prior(100, seed=tuned.seed)
  model.sample_posterior(
      n_chains=tuned.n_chains,
      n_adapt=tuned.n_adapt,
      n_burnin=tuned.n_burnin,
      n_keep=tuned.n_keep,
      seed=tuned.seed,
  )
  roi = np.asarray(
      analyzer_module.Analyzer(
          model_context=model.model_context,
          inference_data=model.inference_data,
      ).roi()
  )
  draws = roi.reshape(-1, roi.shape[-1])
  tail = (1.0 - tuned.confidence_level) / 2.0

  max_r_hat = sampling_diagnostics.max_rank_normalized_rhat(
      model.inference_data.posterior
  )

  return {
      "channels": [
          {
              "channel": name,
              "median": float(np.median(draws[:, i])),
              "ci_low": float(np.quantile(draws[:, i], tail)),
              "ci_high": float(np.quantile(draws[:, i], 1.0 - tail)),
          }
          for i, name in enumerate(tuned.channels)
      ],
      "max_r_hat": max_r_hat,
  }


def summarize(
    fits: Sequence[dict[str, Any]],
    channels: Sequence[str],
    true_roi: Sequence[float],
) -> list[dict[str, Any]]:
  """Collapses the per-prior fits into one record per channel.

  `spread` is the range of the posterior median across priors, and
  `prior_share_of_true_roi` expresses it as a fraction of the channel's true
  value, which is what makes it comparable between channels of different size.
  """
  per_channel = []
  for index, name in enumerate(channels):
    medians = np.array([f["channels"][index]["median"] for f in fits])
    truth = float(true_roi[index])
    spread = float(medians.max() - medians.min())
    per_channel.append({
        "channel": name,
        "true_roi": truth,
        "median_by_prior": {
            f["label"]: f["channels"][index]["median"] for f in fits
        },
        "spread": spread,
        "prior_share_of_true_roi": spread / truth if truth else float("nan"),
        "every_interval_covers_truth": all(
            f["channels"][index]["ci_low"]
            <= truth
            <= f["channels"][index]["ci_high"]
            for f in fits
        ),
    })
  return per_channel


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=pathlib.Path, default=None)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--geos", type=int, default=5)
  parser.add_argument("--times", type=int, default=104)
  parser.add_argument(
      "--quick",
      action="store_true",
      help="Short chains. For checking the script runs, not for evidence.",
  )
  args = parser.parse_args(argv)

  from meridian.validation import recovery  # pylint: disable=g-import-not-at-top

  # Spelled out rather than unpacked from a dict so the dataclass keeps its
  # field types.
  short = args.quick
  config = recovery.RecoveryConfig(
      n_geos=args.geos,
      n_times=args.times,
      seed=args.seed,
      n_chains=2 if short else 4,
      n_adapt=100 if short else 500,
      n_burnin=100 if short else 500,
      n_keep=100 if short else 500,
  )

  # One dataset, held fixed across every prior. This is the whole design: any
  # movement in the posterior is attributable to the prior and nothing else.
  frame, true_roi = recovery.simulate(config)
  truth_line = ", ".join(
      f"{name}={float(value):.3f}"
      for name, value in zip(config.channels, true_roi)
  )
  print(f"true ROI: {truth_line}")
  if args.quick:
    print("running in --quick mode: chains are too short to be evidence\n")

  fits = []
  for label, median, sigma in _PRIOR_GRID:
    started = time.time()
    result = _fit_under_prior(recovery, frame, config, median, sigma)
    result.update({
        "label": label,
        "prior_roi_median": median,
        "prior_roi_sigma": sigma,
        "elapsed_seconds": time.time() - started,
    })
    fits.append(result)
    medians = ", ".join(
        f"{c['channel']}={c['median']:.2f}" for c in result["channels"]
    )
    print(
        f"  {label:20s} median ROI({median:.1f}, sd {sigma}): {medians}"
        f"   [r-hat {result['max_r_hat']:.3f}]"
    )

  per_channel = summarize(fits, config.channels, true_roi)

  print("\nprior-driven movement, as a fraction of true ROI:")
  for entry in per_channel:
    covered = "all priors' intervals cover the truth" if entry[
        "every_interval_covers_truth"
    ] else "at least one interval MISSES the truth"
    print(
        f"  {entry['channel']:12s} true {entry['true_roi']:.2f}  "
        f"spread {entry['spread']:.2f}  "
        f"= {100 * entry['prior_share_of_true_roi']:.0f}% of true ROI  ({covered})"
    )

  evidence = {
      "date": datetime.date.today().isoformat(),
      "script": "scripts/prior_sensitivity.py",
      "provenance": provenance(pathlib.Path(__file__).resolve()),
      "quick_mode": bool(args.quick),
      "config": dataclasses.asdict(config),
      "true_roi": [float(v) for v in true_roi],
      "priors": [
          {"label": label, "roi_median": median, "roi_sigma": sigma}
          for label, median, sigma in _PRIOR_GRID
      ],
      "fits": fits,
      "per_channel": per_channel,
      "interpretation": (
          "One dataset, refitted under five ROI priors. Movement in the "
          "posterior median is attributable to the prior alone. "
          "prior_share_of_true_roi is that movement as a fraction of the "
          "channel's true ROI."
      ),
      "limitations": [
          "Measures sensitivity, not accuracy: it does not say which prior "
          "is right.",
          "Synthetic data with known truth. A real dataset's identifiability "
          "depends on its own spend variation and collinearity.",
          "Sensitivity to the prior is not the same as a miscalibrated "
          "interval.",
          "Short chains when --quick is set; those runs are not evidence.",
      ],
  }

  if args.output is not None:
    write_evidence(args.output, evidence)
    print(f"\nwrote {args.output}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
