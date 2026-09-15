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

"""A complete Meridian run, start to finish, on the bundled sample data.

    python examples/quickstart.py            # ~3 minutes, small MCMC
    python examples/quickstart.py --full     # ~35 minutes, demo settings

Everything here is what a real analysis does, in order:

  1. shape a DataFrame into `InputData`
  2. set an ROI prior
  3. check that prior against the data BEFORE fitting  <- cheap, skip at your peril
  4. fit
  5. check convergence
  6. read ROI
  7. optimize a budget
  8. save the model

Step 3 is the one people leave out. It costs seconds and catches a prior that
cannot produce the observed outcome -- which a fit will not fix, and which you
would otherwise discover after waiting for the fit to finish.

The data is `meridian/data/simulated_data/csv/geo_media.csv`: 20 geos, 156
weeks, 4 channels, a non-revenue KPI with revenue per conversion. Swap in your
own DataFrame at the marked spot; the rest is unchanged.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SAMPLE_CSV = (
    REPO_ROOT / 'meridian' / 'data' / 'simulated_data' / 'csv' / 'geo_media.csv'
)
CHANNELS = ['Channel0', 'Channel1', 'Channel2', 'Channel3']


def banner(step: str, started: float) -> None:
  print(f'\n[{time.time() - started:6.1f}s] {step}', flush=True)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
      prog='python examples/quickstart.py',
      description='A complete Meridian run on the bundled sample data.',
  )
  parser.add_argument(
      '--full',
      action='store_true',
      help="Use the demo's MCMC settings. Much slower, properly converged.",
  )
  parser.add_argument(
      '--output-dir',
      default=str(REPO_ROOT / 'quickstart_output'),
      help='Where to write the summary report and saved model.',
  )
  args = parser.parse_args(argv)

  started = time.time()
  out_dir = pathlib.Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  from meridian import backend
  from meridian import constants as c
  from meridian.analysis import analyzer as analyzer_module
  from meridian.analysis import optimizer as optimizer_module
  from meridian.analysis import prior_predictive
  from meridian.analysis import summarizer as summarizer_module
  from meridian.data import data_frame_input_data_builder as dfb
  from meridian.model import model, prior_distribution, spec
  from meridian.schema.serde import meridian_serde

  banner('1. Loading data', started)
  # ---- Replace this DataFrame with your own -------------------------------
  df = pd.read_csv(SAMPLE_CSV)
  # -------------------------------------------------------------------------
  print(f'    {len(df):,} rows, {df["geo"].nunique()} geos,'
        f' {df["time"].nunique()} periods')

  banner('2. Building InputData', started)
  data = (
      dfb.DataFrameInputDataBuilder(kpi_type=c.NON_REVENUE)
      .with_kpi(df, kpi_col='conversions', time_col='time', geo_col='geo')
      .with_revenue_per_kpi(
          df,
          revenue_per_kpi_col='revenue_per_conversion',
          time_col='time',
          geo_col='geo',
      )
      .with_population(df, population_col='population', geo_col='geo')
      .with_controls(
          df,
          control_cols=[
              'competitor_activity_score_control',
              'sentiment_score_control',
          ],
          time_col='time',
          geo_col='geo',
      )
      .with_media(
          df,
          media_cols=[f'{ch}_impression' for ch in CHANNELS],
          media_spend_cols=[f'{ch}_spend' for ch in CHANNELS],
          media_channels=CHANNELS,
          time_col='time',
          geo_col='geo',
      )
      .build()
  )
  n_times = len(data.kpi.coords[c.TIME])
  n_media_times = len(data.media.coords[c.MEDIA_TIME])
  print(f'    n_times={n_times}  n_media_times={n_media_times}')
  if n_media_times == n_times:
    print('    NOTE: no adstock burn-in. Media history does not extend before')
    print('          the modelling window, so carryover for the first periods')
    print('          is computed against zero-padded history. To add burn-in,')
    print('          supply extra periods of impressions with KPI, controls')
    print('          and spend left blank. The model will warn about this too.')

  banner('3. Setting an ROI prior', started)
  # Priors must match the backend float dtype. Python floats give float32,
  # which the 64-bit JAX default would otherwise reject -- this fork widens
  # them automatically, but being explicit is clearer.
  roi_prior = backend.tfd.LogNormal(
      backend.np_float_dtype(0.2),
      backend.np_float_dtype(0.9),
      name=c.ROI_M,
  )
  model_spec = spec.ModelSpec(
      prior=prior_distribution.PriorDistribution(roi_m=roi_prior),
      media_prior_type=c.ROI,
      max_lag=8,
  )
  mmm = model.Meridian(input_data=data, model_spec=model_spec)

  banner('4. Prior predictive check (before fitting)', started)
  mmm.sample_prior(500, seed=1)
  check = prior_predictive.PriorPredictiveCheck(mmm)
  summary = check.summary()
  print(f'    interval coverage      {summary.coverage:.0%}')
  print(f'    prior/actual total     {summary.total_ratio:.2f}x')
  print(f'    observed total in CI   {summary.total_actual_within_ci}')
  print(f'    -> {summary.verdict}')

  banner('5. Fitting (the slow part)', started)
  if args.full:
    chains, adapt, burnin, keep = 7, 2000, 500, 1000
  else:
    chains, adapt, burnin, keep = 4, 400, 400, 500
    print('    Small MCMC for speed. Expect r_hat above 1.2 -- that is')
    print('    under-sampling, not a broken model. Use --full for a real fit.')
  fit_started = time.time()
  mmm.sample_posterior(
      n_chains=chains, n_adapt=adapt, n_burnin=burnin, n_keep=keep, seed=1
  )
  print(f'    fitted in {time.time() - fit_started:.0f}s')

  banner('6. Convergence', started)
  import arviz as az

  r_hat = az.rhat(mmm.inference_data.posterior)
  worst = 0.0
  for name in r_hat.data_vars:
    values = np.asarray(r_hat[name].values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size:
      worst = max(worst, float(finite.max()))
  verdict = 'converged' if worst < 1.2 else 'NOT converged (r_hat >= 1.2)'
  print(f'    max r_hat = {worst:.4f}  -> {verdict}')

  banner('7. ROI by channel', started)
  analyzer = analyzer_module.Analyzer(
      model_context=mmm.model_context, inference_data=mmm.inference_data
  )
  roi = np.asarray(analyzer.roi())
  draws = roi.reshape(-1, roi.shape[-1])
  print(f'    {"channel":<12}{"median":>9}{"90% interval":>22}')
  for i, channel in enumerate(CHANNELS):
    lo, hi = np.quantile(draws[:, i], [0.05, 0.95])
    print(f'    {channel:<12}{np.median(draws[:, i]):9.3f}'
          f'  [{lo:8.3f},{hi:8.3f}]')
  print('    Intervals cover parameter uncertainty only -- not error in the')
  print('    assumed saturation shape. See "Reading ROI intervals honestly".')

  banner('8. Budget optimization', started)
  results = optimizer_module.BudgetOptimizer(mmm).optimize()
  print(f'    current ROI    {float(results.nonoptimized_data.total_roi):.3f}')
  print(f'    optimized ROI  {float(results.optimized_data.total_roi):.3f}')

  banner('9. Saving outputs', started)
  summarizer_module.Summarizer(mmm).output_model_results_summary(
      'summary.html',
      str(out_dir),
      str(data.kpi.coords[c.TIME].values[0]),
      str(data.kpi.coords[c.TIME].values[-1]),
  )
  model_path = out_dir / 'model.binpb'
  meridian_serde.save_meridian(mmm, str(model_path))
  print(f'    {out_dir / "summary.html"}')
  print(f'    {model_path}')

  print(f'\nDone in {time.time() - started:.0f}s.')
  print('Reload the model later with:')
  print('    from meridian.schema.serde import meridian_serde')
  print(f'    mmm = meridian_serde.load_meridian("{model_path}")')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
