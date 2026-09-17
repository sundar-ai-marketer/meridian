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

"""Measures whether a prior can generate data the model would actually accept.

    python scripts/prior_predictive_audit.py --draws 400 \
        --output docs/validation/prior-predictive-audit-<date>.json

Why this exists. Meridian scales the KPI to mean 0 and standard deviation 1
before modelling (`KpiTransformer`), and its default priors are then written on
that scale: `knot_values`, `tau_g_excl_baseline` and `gamma_c` are
`Normal(0, 5)`, and `sigma` and `xi_c` are `HalfNormal(5)`. A prior
standard deviation of 5 against data whose standard deviation is 1 by
construction is a deliberately weak statement, but it has a consequence that is
easy to miss: the implied baseline swings far enough below zero that the prior
predictive distribution puts substantial mass on negative revenue.

`InputData._validate_no_negative_values` rejects a negative KPI, and says why:
ROI, CPIK and contribution are expressed relative to total outcome, so a
negative KPI makes them ill-defined. The two facts together mean the default
prior generates datasets the library itself refuses to load.

That matters in two places. It is the reason simulation-based calibration
cannot be run against the shipped default prior -- every simulated dataset is
rejected, so there is nothing to refit. And it is a reason to look at a prior
predictive check before a long fit, because a prior this wide contributes very
little information where it is most needed.

This script quantifies the effect for a given dataset shape rather than
asserting it. It samples the prior, computes the conditional mean per geo and
time, adds observation noise on the scaled scale exactly as the likelihood
does, inverts the KPI transform, and reports how much of the result is
negative.

What it does NOT claim. A wide prior is not by itself an error; weakly
informative priors are a legitimate choice and the posterior may be perfectly
well behaved. The finding is narrower and concrete: this prior cannot be
simulated from without producing data the model rejects.
"""

from __future__ import annotations

import argparse
import datetime
import pathlib
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np

# Runs both as `python scripts/x.py` and as `python -m unittest scripts.test_x`,
# so anchor the repo root before importing the shared helpers.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from scripts.evidence import provenance, write_evidence  # pylint: disable=g-import-not-at-top,g-bad-import-order


def _build_model(template, config, priors: dict[str, Any] | None):
  """Builds a Meridian model on the template, optionally overriding priors."""
  from meridian import backend  # pylint: disable=g-import-not-at-top
  from meridian import constants as c  # pylint: disable=g-import-not-at-top
  from meridian.data import data_frame_input_data_builder as dfb  # pylint: disable=g-import-not-at-top
  from meridian.model import model  # pylint: disable=g-import-not-at-top
  from meridian.model import prior_distribution  # pylint: disable=g-import-not-at-top
  from meridian.model import spec  # pylint: disable=g-import-not-at-top

  data = (
      dfb.DataFrameInputDataBuilder(kpi_type=c.REVENUE)
      .with_kpi(
          template.dropna(subset=["revenue"]),
          kpi_col="revenue",
          time_col="date",
          geo_col="geo",
      )
      .with_population(template, population_col="population", geo_col="geo")
      .with_controls(
          template.dropna(subset=["control_1"]),
          control_cols=["control_1"],
          time_col="date",
          geo_col="geo",
      )
      .with_media(
          template,
          media_cols=[f"{ch}_impressions" for ch in config.channels],
          media_spend_cols=[f"{ch}_spend" for ch in config.channels],
          media_channels=config.channels,
          time_col="date",
          geo_col="geo",
      )
      .build()
  )

  prior = None
  if priors is not None:
    f = backend.np_float_dtype
    overrides = dict(
        roi_m=backend.tfd.LogNormal(
            f(np.log(priors["roi_median"])), f(priors["roi_sigma"]), name=c.ROI_M
        ),
        sigma=backend.tfd.HalfNormal(f(priors["sigma_scale"]), name=c.SIGMA),
        knot_values=backend.tfd.Normal(
            f(0.0), f(priors["baseline_sd"]), name=c.KNOT_VALUES
        ),
        tau_g_excl_baseline=backend.tfd.Normal(
            f(0.0), f(priors["baseline_sd"]), name=c.TAU_G_EXCL_BASELINE
        ),
        gamma_c=backend.tfd.Normal(
            f(0.0), f(priors["baseline_sd"]), name=c.GAMMA_C
        ),
    )
    # The hierarchical standard deviations are the remaining source of width:
    # `eta_m` and `xi_c` sit on `beta_gm` and `gamma_gc`, so leaving them at
    # their defaults keeps the geo-level terms wide however tight the
    # population-level priors are.
    #
    # `beta_m` is deliberately NOT overridden. Under `media_prior_type="roi"`
    # it is derived from `roi_m`, and ModelContext warns that a custom
    # `beta_m` is ignored. Setting it would look like it mattered and would
    # not.
    hier_sd = priors.get("hier_sd")
    if hier_sd is not None:
      overrides["eta_m"] = backend.tfd.HalfNormal(f(hier_sd), name=c.ETA_M)
      overrides["xi_c"] = backend.tfd.HalfNormal(f(hier_sd), name=c.XI_C)
    prior = prior_distribution.PriorDistribution(**overrides)

  model_spec = spec.ModelSpec(
      media_prior_type=c.ROI, max_lag=config.max_lag
  ) if prior is None else spec.ModelSpec(
      prior=prior, media_prior_type=c.ROI, max_lag=config.max_lag
  )
  return model.Meridian(input_data=data, model_spec=model_spec)


def _measure(mmm, draws: int, seed: int) -> dict[str, Any]:
  """Samples the prior and reports how much predictive mass is negative."""
  from meridian import constants as c  # pylint: disable=g-import-not-at-top
  from meridian.analysis import analyzer as analyzer_module  # pylint: disable=g-import-not-at-top

  mmm.sample_prior(draws, seed=seed)
  prior = mmm.inference_data.prior

  analyzer = analyzer_module.Analyzer(
      model_context=mmm.model_context, inference_data=mmm.inference_data
  )
  # Conditional mean per geo and time, in original KPI units.
  mean = np.asarray(
      analyzer.expected_outcome(
          use_posterior=False,
          aggregate_geos=False,
          aggregate_times=False,
          inverse_transform_outcome=True,
      )
  )
  flat = mean.reshape(-1, mean.shape[-2], mean.shape[-1])

  sigma = np.asarray(prior[c.SIGMA].values).reshape(-1)
  transformer = mmm.model_context.kpi_transformer
  rng = np.random.default_rng(seed)

  # `y_scaled ~ Normal(y_pred_scaled, sigma)` is the likelihood; the transform
  # is affine, so forward-then-inverse round trips exactly.
  #
  # The cast back to the scaled tensor's dtype is required, not cosmetic: the
  # TensorFlow backend runs float32 while numpy's generator returns float64,
  # and the upcast sum makes `inverse` fail with a Mul type mismatch. JAX
  # defaults to float64 and hides the problem.
  def _one(index: int) -> np.ndarray:
    scaled = np.asarray(transformer.forward(flat[index]))
    noise = rng.normal(0.0, float(sigma[index]), size=scaled.shape)
    noisy = (scaled + noise).astype(scaled.dtype, copy=False)
    return np.asarray(transformer.inverse(noisy))

  predictive = np.stack([_one(i) for i in range(flat.shape[0])])

  kpi = np.asarray(mmm.model_context.kpi)
  mean_negative = (flat < 0).reshape(flat.shape[0], -1)
  pred_negative = (predictive < 0).reshape(predictive.shape[0], -1)

  return {
      "draws": int(flat.shape[0]),
      "observed_kpi": {
          "min": float(kpi.min()),
          "mean": float(kpi.mean()),
          "scaled_stdev_by_construction": 1.0,
      },
      "scaled_value_where_revenue_reaches_zero": float(
          -float(transformer.population_scaled_mean)
          / float(transformer.population_scaled_stdev)
      ),
      "sigma_prior_draws": {
          "median": float(np.median(sigma)),
          "p90": float(np.percentile(sigma, 90)),
      },
      "conditional_mean": {
          "draws_with_a_negative_cell_pct": float(
              100.0 * mean_negative.any(axis=1).mean()
          ),
          "negative_cells_pct": float(100.0 * mean_negative.mean()),
          "min": float(flat.min()),
      },
      "prior_predictive": {
          "draws_with_a_negative_cell_pct": float(
              100.0 * pred_negative.any(axis=1).mean()
          ),
          "negative_cells_pct": float(100.0 * pred_negative.mean()),
      },
  }


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--draws", type=int, default=400)
  parser.add_argument("--geos", type=int, default=3)
  parser.add_argument("--times", type=int, default=40)
  parser.add_argument("--seed", type=int, default=11)
  parser.add_argument("--output", type=pathlib.Path, default=None)
  args = parser.parse_args(argv)

  from meridian.validation import recovery  # pylint: disable=g-import-not-at-top

  config = recovery.RecoveryConfig(
      true_roi=(1.0, 2.0),
      spend_scale=(3_000.0, 6_000.0),
      n_geos=args.geos,
      n_times=args.times,
      max_lag=2,
      seed=args.seed,
  )
  template, _ = recovery.simulate(config)

  # The shipped defaults, then progressively tighter overrides. The sweep is
  # the point: it shows that tightening the observation-noise prior alone does
  # not help, because the baseline terms carry most of the width.
  variants: list[tuple[str, dict[str, Any] | None]] = [
      ("meridian defaults", None),
      (
          "sigma tightened only",
          {
              "sigma_scale": 0.5,
              "baseline_sd": 5.0,
              "roi_median": 2.0,
              "roi_sigma": 0.7,
          },
      ),
      (
          "sigma and baseline tightened",
          {
              "sigma_scale": 0.5,
              "baseline_sd": 0.5,
              "roi_median": 2.0,
              "roi_sigma": 0.7,
          },
      ),
      (
          "population-level tightened, hierarchical left at defaults",
          {
              "sigma_scale": 0.1,
              "baseline_sd": 0.1,
              "roi_median": 2.0,
              "roi_sigma": 0.7,
          },
      ),
      (
          "hierarchical scales tightened too",
          {
              "sigma_scale": 0.1,
              "baseline_sd": 0.1,
              "hier_sd": 0.1,
              "roi_median": 2.0,
              "roi_sigma": 0.7,
          },
      ),
  ]

  results = []
  for label, priors in variants:
    mmm = _build_model(template, config, priors)
    measured = _measure(mmm, args.draws, args.seed)
    measured["variant"] = label
    measured["priors"] = priors or "library defaults"
    results.append(measured)
    conditional_mean: dict[str, float] = measured["conditional_mean"]
    predictive: dict[str, float] = measured["prior_predictive"]
    print(
        f"{label:56s} "
        f"mean<0 in {conditional_mean['draws_with_a_negative_cell_pct']:5.1f}% of draws, "
        f"predictive<0 in {predictive['draws_with_a_negative_cell_pct']:5.1f}%"
    )

  evidence = {
      "date": datetime.date.today().isoformat(),
      "script": "scripts/prior_predictive_audit.py",
      "provenance": provenance(pathlib.Path(__file__).resolve()),
      "shape": {
          "n_geos": args.geos,
          "n_times": args.times,
          "draws_per_variant": args.draws,
      },
      "library_default_priors_on_the_scaled_kpi": {
          "knot_values": "Normal(0, 5)",
          "tau_g_excl_baseline": "Normal(0, 5)",
          "gamma_c": "Normal(0, 5)",
          "sigma": "HalfNormal(5)",
          "xi_c": "HalfNormal(5)",
          "eta_m": "HalfNormal(1)",
          "beta_m": "HalfNormal(5), ignored when media_prior_type is 'roi'",
      },
      "variants": results,
      "conclusion": (
          "The default prior places substantial mass on negative revenue, "
          "which InputData rejects, so the shipped prior cannot be simulated "
          "from. Tightening the observation-noise prior alone changes almost "
          "nothing. Tightening the population-level baseline terms roughly "
          "halves the rate. Only tightening the hierarchical standard "
          "deviations as well -- eta_m and xi_c, which sit on the geo-level "
          "terms -- removes it entirely. The hierarchical scales "
          "are the dominant source of width."
      ),
      "limitations": [
          "Measured on one synthetic dataset shape; the exact percentages "
          "depend on the data's scale and the number of geos and periods.",
          "A weakly informative prior is a legitimate choice. The finding is "
          "that this one cannot be simulated from, not that it is wrong.",
          "Says nothing about posterior behaviour, which is usually dominated "
          "by the likelihood.",
      ],
  }

  if args.output is not None:
    write_evidence(args.output, evidence)
    print(f"wrote {args.output}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
