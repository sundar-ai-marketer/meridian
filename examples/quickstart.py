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

"""A complete Meridian example on the bundled sample data.

    python examples/quickstart.py
    python examples/quickstart.py --full --sampling-mode strict
    python examples/quickstart.py --full --knots 13 --n-keep 4000 \\
        --target-accept-prob 0.99 --sampling-mode strict

The default is an EXPLORATORY demo: it renders its outputs for inspection but
does not claim that they are decision-grade. ``--sampling-mode strict`` saves
the fitted model and machine-readable diagnostic evidence, then refuses ROI,
optimization, and HTML decision outputs unless all strict sampling checks pass.

The sample has 156 observed weeks. The first eight are retained as actual media
history only; KPI, revenue-per-KPI, controls, and spend use the following 148
weeks. No media history is fabricated and the model has matching eight-week
history for its ``max_lag=8`` adstock setting.

Strict mode refuses an output directory that already contains ``summary.html``.
This preserves a prior decision-output report rather than letting it appear
current beside evidence from a failed rerun; use a fresh output directory.
"""

# Imports from Meridian stay lazy so ``--help`` remains fast and the TensorFlow
# log setting is applied before either backend is imported.
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import argparse
import os
import pathlib
import time

import numpy as np
import pandas as pd

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SAMPLE_CSV = (
    REPO_ROOT / 'meridian' / 'data' / 'simulated_data' / 'csv' / 'geo_media.csv'
)
CHANNELS = ['Channel0', 'Channel1', 'Channel2', 'Channel3']
MEDIA_HISTORY_PERIODS = 8
STRICT_FAILURE_EXIT_CODE = 2
_DECISION_OUTPUT_FILENAMES = ('summary.html',)


def banner(step: str, started: float) -> None:
  print(f'\n[{time.time() - started:6.1f}s] {step}', flush=True)


def positive_integer(value: str) -> int:
  """Argparse validator for sampling settings that must be positive."""
  try:
    result = int(value)
  except ValueError as error:
    raise argparse.ArgumentTypeError('must be a positive integer') from error
  if result < 1:
    raise argparse.ArgumentTypeError('must be a positive integer')
  return result


def nonnegative_integer(value: str) -> int:
  """Argparse validator for a count that may be zero."""
  try:
    result = int(value)
  except ValueError as error:
    raise argparse.ArgumentTypeError(
        'must be a non-negative integer'
    ) from error
  if result < 0:
    raise argparse.ArgumentTypeError('must be a non-negative integer')
  return result


def open_probability(value: str) -> float:
  """Argparse validator for a target acceptance probability in (0, 1)."""
  try:
    result = float(value)
  except ValueError as error:
    raise argparse.ArgumentTypeError(
        'must be a number in the open interval (0, 1)'
    ) from error
  if not 0 < result < 1:
    raise argparse.ArgumentTypeError('must be in the open interval (0, 1)')
  return result


def build_parser() -> argparse.ArgumentParser:
  """Builds the compatibility-preserving command-line interface."""
  parser = argparse.ArgumentParser(
      prog='python examples/quickstart.py',
      description='A complete Meridian run on the bundled sample data.',
  )
  parser.add_argument(
      '--full',
      action='store_true',
      help=(
          "Use the demo's longer MCMC defaults; sampling checks still decide "
          'status.'
      ),
  )
  parser.add_argument(
      '--sampling-mode',
      choices=('exploratory', 'strict'),
      default='exploratory',
      help=(
          'Whether to render an explicitly exploratory demo (default) or '
          'block decision outputs until strict sampling checks pass.'
      ),
  )
  parser.add_argument(
      '--strict',
      dest='sampling_mode',
      action='store_const',
      const='strict',
      help='Short alias for --sampling-mode strict.',
  )
  parser.add_argument(
      '--knots',
      type=positive_integer,
      default=None,
      help=(
          'Optional explicit number of time-effect knots. The default remains '
          'ModelSpec knots=None; choosing a value changes the model and must '
          'be reviewed as such.'
      ),
  )
  parser.add_argument(
      '--n-adapt',
      type=positive_integer,
      default=None,
      help='Override the demo adaptation draws per chain.',
  )
  parser.add_argument(
      '--n-chains',
      type=positive_integer,
      default=None,
      help='Override the number of MCMC chains.',
  )
  parser.add_argument(
      '--n-burnin',
      type=nonnegative_integer,
      default=None,
      help='Override the post-adaptation burn-in draws per chain.',
  )
  parser.add_argument(
      '--n-keep',
      type=positive_integer,
      default=None,
      help='Override the demo retained post-warmup draws per chain.',
  )
  parser.add_argument(
      '--target-accept-prob',
      type=open_probability,
      default=None,
      help=(
          'Optional NUTS dual-averaging target acceptance probability in '
          '(0, 1), for example 0.95. This is an explicit sampler change.'
      ),
  )
  parser.add_argument(
      '--output-dir',
      default=str(REPO_ROOT / 'quickstart_output'),
      help=(
          'Where to write sampling evidence and any permitted decision '
          'outputs.'
      ),
  )
  parser.add_argument(
      '--sampling-quality-json',
      default=None,
      help=(
          'Optional sampling-quality JSON path. Defaults to '
          'OUTPUT_DIR/sampling-quality.json.'
      ),
  )
  return parser


def build_quickstart_input_data(
    df: pd.DataFrame, *, history_periods: int = MEDIA_HISTORY_PERIODS
):
  """Builds input data with observed pre-window media history.

  Media executions keep every source period. Leading media spend is set missing
  only in the builder input so its ``stack()`` operation excludes it from the
  model window, while retaining the observed executions as adstock history.
  """
  from meridian import constants as c
  from meridian.data import data_frame_input_data_builder as dfb

  if history_periods < 1:
    raise ValueError('`history_periods` must be at least 1.')
  if 'time' not in df:
    raise ValueError('The source DataFrame must include a `time` column.')
  times = np.sort(df['time'].dropna().unique())
  if len(times) <= history_periods:
    raise ValueError(
        '`history_periods` must leave at least one analysis-window period.'
    )
  history_times = times[:history_periods]
  analysis_times = times[history_periods:]
  analysis_df = df.loc[df['time'].isin(analysis_times)].copy()
  media_df = df.copy()
  spend_columns = [f'{channel}_spend' for channel in CHANNELS]
  media_df.loc[media_df['time'].isin(history_times), spend_columns] = np.nan

  return (
      dfb.DataFrameInputDataBuilder(kpi_type=c.NON_REVENUE)
      .with_kpi(
          analysis_df, kpi_col='conversions', time_col='time', geo_col='geo'
      )
      .with_revenue_per_kpi(
          analysis_df,
          revenue_per_kpi_col='revenue_per_conversion',
          time_col='time',
          geo_col='geo',
      )
      .with_population(analysis_df, population_col='population', geo_col='geo')
      .with_controls(
          analysis_df,
          control_cols=[
              'competitor_activity_score_control',
              'sentiment_score_control',
          ],
          time_col='time',
          geo_col='geo',
      )
      .with_media(
          media_df,
          media_cols=[f'{channel}_impression' for channel in CHANNELS],
          media_spend_cols=spend_columns,
          media_channels=CHANNELS,
          time_col='time',
          geo_col='geo',
      )
      .build()
  )


def prepare_output_dir(out_dir: pathlib.Path, sampling_mode: str) -> None:
  """Creates a safe output directory before any expensive sampling work.

  Strict runs refuse a directory containing a prior generated decision report.
  The file is left untouched so a failed rerun cannot make it look current.
  """
  if sampling_mode == 'strict':
    stale_outputs = [
        filename
        for filename in _DECISION_OUTPUT_FILENAMES
        if (out_dir / filename).exists()
    ]
    if stale_outputs:
      rendered = ', '.join(stale_outputs)
      raise ValueError(
          'Strict mode refuses an output directory containing prior decision '
          f'output ({rendered}). Choose a fresh --output-dir; existing files '
          'were left untouched.'
      )
  out_dir.mkdir(parents=True, exist_ok=True)


def report_sampling_quality(mmm):
  """Prints and returns the strict, per-cell sampling-quality report."""
  from meridian.analysis import sampling_quality

  report = sampling_quality.assess_sampling_quality(
      mmm.inference_data, model_context=mmm.model_context
  )
  print(f'    {report.format_summary().replace(chr(10), chr(10) + "    ")}')
  return report


def save_sampling_evidence(
    mmm,
    report,
    *,
    out_dir: pathlib.Path,
    meridian_serde,
    quality_json_path: str | None = None,
) -> tuple[pathlib.Path, pathlib.Path]:
  """Persists raw fitted samples and diagnostics before any strict gate.

  These files are diagnostic evidence, including when strict checks fail; they
  are intentionally saved before ROI, optimization, or report rendering.
  """
  model_path = out_dir / 'model.binpb'
  report_path = (
      pathlib.Path(quality_json_path)
      if quality_json_path is not None
      else out_dir / 'sampling-quality.json'
  )
  report_path.parent.mkdir(parents=True, exist_ok=True)
  meridian_serde.save_meridian(mmm, str(model_path))
  report_path.write_text(report.to_json(indent=2) + '\n', encoding='utf-8')
  return model_path, report_path


def allow_decision_outputs(report, sampling_mode: str) -> bool:
  """Prints sampling status and returns whether decision outputs may run."""
  if sampling_mode == 'strict':
    if report.sampling_passed:
      print(
          '    Strict sampling checks passed. Causal validity and model '
          'adequacy still require review.'
      )
      return True
    print('    STRICT SAMPLING GATE FAILED: no decision outputs will be made.')
    print(
        '    The fitted model and sampling-quality JSON were saved as evidence.'
    )
    print(
        '    ROI estimates, budget optimization, and HTML summary were skipped.'
    )
    return False

  print('    EXPLORATORY DEMO — not decision-grade.')
  print(
      '    Any ROI, optimization, and HTML output below is for inspection '
      'only. '
      'Use --sampling-mode strict to gate decision outputs.'
  )
  return True


def report_note_for(sampling_mode: str) -> str:
  """Returns the visible status note carried into any generated HTML summary."""
  if sampling_mode == 'strict':
    return (
        'Sampling checks passed; causal validity and model adequacy still '
        'require review.'
    )
  return (
      'EXPLORATORY DEMO — not decision-grade. This report is for inspection '
      'only; strict sampling checks were not used to authorize decisions.'
  )


def write_decision_outputs(
    mmm,
    data,
    *,
    out_dir: pathlib.Path,
    started: float,
    analyzer_module,
    optimizer_module,
    summarizer_module,
    report_note: str,
) -> None:
  """Produces ROI, optimization, and HTML only after the sampling gate."""
  from meridian import constants as c

  banner('8. ROI by channel', started)
  analyzer = analyzer_module.Analyzer(
      model_context=mmm.model_context, inference_data=mmm.inference_data
  )
  roi = np.asarray(analyzer.roi())
  draws = roi.reshape(-1, roi.shape[-1])
  print(f'    {"channel":<12}{"median":>9}{"90% interval":>22}')
  for index, channel in enumerate(CHANNELS):
    low, high = np.quantile(draws[:, index], [0.05, 0.95])
    print(
        f'    {channel:<12}{np.median(draws[:, index]):9.3f}'
        f'  [{low:8.3f},{high:8.3f}]'
    )
  print('    Intervals cover parameter uncertainty only — not error in the')
  print('    assumed saturation shape. See "Reading ROI intervals honestly".')

  banner('9. Budget optimization', started)
  results = optimizer_module.BudgetOptimizer(mmm).optimize()
  print(f'    current ROI    {float(results.nonoptimized_data.total_roi):.3f}')
  print(f'    optimized ROI  {float(results.optimized_data.total_roi):.3f}')

  banner('10. Rendering permitted HTML output', started)
  summarizer_module.Summarizer(
      mmm, report_note=report_note
  ).output_model_results_summary(
      'summary.html',
      str(out_dir),
      str(data.kpi.coords[c.TIME].values[0]),
      str(data.kpi.coords[c.TIME].values[-1]),
  )
  print(f'    {out_dir / "summary.html"}')


def main(argv: list[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  started = time.time()
  out_dir = pathlib.Path(args.output_dir)
  try:
    prepare_output_dir(out_dir, args.sampling_mode)
  except ValueError as error:
    parser.error(str(error))

  from meridian import backend
  from meridian import constants as c
  from meridian.analysis import analyzer as analyzer_module
  from meridian.analysis import optimizer as optimizer_module
  from meridian.analysis import prior_predictive
  from meridian.analysis import summarizer as summarizer_module
  from meridian.model import model, prior_distribution, spec
  from meridian.schema.serde import meridian_serde

  banner('1. Loading data', started)
  # ---- Replace this DataFrame with your own -------------------------------
  df = pd.read_csv(SAMPLE_CSV)
  # -------------------------------------------------------------------------
  print(
      f'    {len(df):,} rows, {df["geo"].nunique()} geos, '
      f'{df["time"].nunique()} source periods'
  )

  banner('2. Building InputData with observed media history', started)
  data = build_quickstart_input_data(df)
  n_times = len(data.kpi.coords[c.TIME])
  n_media_times = len(data.media.coords[c.MEDIA_TIME])
  first_analysis_time = data.kpi.coords[c.TIME].values[0]
  print(f'    n_times={n_times}  n_media_times={n_media_times}')
  print(
      f'    {MEDIA_HISTORY_PERIODS} observed periods ending before '
      f'{first_analysis_time} are media history only; KPI, revenue, controls, '
      'and spend use the following analysis window.'
  )

  banner('3. Setting an ROI prior', started)
  roi_prior = backend.tfd.LogNormal(
      backend.np_float_dtype(0.2),
      backend.np_float_dtype(0.9),
      name=c.ROI_M,
  )
  spec_args = {
      'prior': prior_distribution.PriorDistribution(roi_m=roi_prior),
      'media_prior_type': c.ROI,
      'max_lag': MEDIA_HISTORY_PERIODS,
  }
  if args.knots is not None:
    spec_args['knots'] = args.knots
    print(
        f'    Using explicit --knots {args.knots}; this changes time effects.'
    )
  else:
    print('    Using the existing ModelSpec default knots=None.')
  mmm = model.Meridian(input_data=data, model_spec=spec.ModelSpec(**spec_args))

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
    default_chains, default_adapt, default_burnin, default_keep = (
        7,
        2000,
        500,
        1000,
    )
  else:
    default_chains, default_adapt, default_burnin, default_keep = (
        4,
        400,
        400,
        500,
    )
    print('    Small MCMC for speed. It is an exploratory sampling demo, not')
    print('    evidence that strict sampling checks will pass.')
  chains = args.n_chains or default_chains
  adapt = args.n_adapt or default_adapt
  burnin = args.n_burnin if args.n_burnin is not None else default_burnin
  keep = args.n_keep or default_keep
  sampling_args = {
      'n_chains': chains,
      'n_adapt': adapt,
      'n_burnin': burnin,
      'n_keep': keep,
      'seed': 1,
  }
  if args.target_accept_prob is not None:
    sampling_args['dual_averaging_kwargs'] = {
        'target_accept_prob': args.target_accept_prob
    }
    print(
        '    Using explicit '
        f'--target-accept-prob {args.target_accept_prob:.3f}.'
    )
  print(
      f'    chains={chains}  n_adapt={adapt}  n_burnin={burnin}  '
      f'n_keep={keep}'
  )
  fit_started = time.time()
  mmm.sample_posterior(**sampling_args)
  print(f'    fitted in {time.time() - fit_started:.0f}s')

  banner('6. Strict sampling diagnostics', started)
  sampling_report = report_sampling_quality(mmm)

  banner('7. Saving fitted-sample evidence', started)
  model_path, report_path = save_sampling_evidence(
      mmm,
      sampling_report,
      out_dir=out_dir,
      meridian_serde=meridian_serde,
      quality_json_path=args.sampling_quality_json,
  )
  print(f'    {model_path}')
  print(f'    {report_path}')
  if not allow_decision_outputs(sampling_report, args.sampling_mode):
    print(f'\nStopped after diagnostics in {time.time() - started:.0f}s.')
    return STRICT_FAILURE_EXIT_CODE

  write_decision_outputs(
      mmm,
      data,
      out_dir=out_dir,
      started=started,
      analyzer_module=analyzer_module,
      optimizer_module=optimizer_module,
      summarizer_module=summarizer_module,
      report_note=report_note_for(args.sampling_mode),
  )
  print(f'\nDone in {time.time() - started:.0f}s.')
  print('Reload the fitted-sample evidence later with:')
  print('    from meridian.schema.serde import meridian_serde')
  print(f'    mmm = meridian_serde.load_meridian("{model_path}")')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
