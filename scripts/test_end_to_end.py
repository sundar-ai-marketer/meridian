#!/usr/bin/env python3
# Copyright 2026 Sundar Ramesh Kumar.
# Licensed under the Apache License, Version 2.0.
# NOTICE: This file is new in this fork; see NOTICE at the repository root.

"""Exercise the real model-to-report workflow; small draws test plumbing only.

Run with either MERIDIAN_BACKEND=jax or MERIDIAN_BACKEND=tensorflow.
This checks executable integration, not statistical convergence or ROI accuracy.
The default case retains the original output paths. ``--all-cases`` exercises
national, geographic, media, reach/frequency, and mixed-treatment inputs in
separate processes to bound compilation memory.
"""

import argparse
import dataclasses
import datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

from meridian import backend
from meridian import constants as c
from meridian.analysis import (
    analyzer,
    budget_decision,
    optimizer,
    sampling_quality,
    summarizer,
    tensors,
)
from meridian.data import test_utils
from meridian.model import model, spec
from meridian.schema.serde import distribution, function_registry, hyperparameters, meridian_serde

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))
from scripts.evidence import provenance, sha256, write_evidence  # pylint: disable=g-import-not-at-top,g-bad-import-order

_CASES = {
    'geo_nonrevenue_media': {
        'n_geos': 2,
        'n_media_channels': 2,
    },
    'national_revenue_media': {
        'n_geos': 1,
        'n_media_channels': 2,
    },
    'national_revenue_rf': {
        'n_geos': 1,
        'n_rf_channels': 2,
    },
    'geo_revenue_mixed': {
        'n_geos': 2,
        'n_media_channels': 1,
        'n_rf_channels': 1,
        'n_organic_media_channels': 1,
        'n_non_media_channels': 1,
    },
}
_DEFAULT_CASE = 'geo_nonrevenue_media'
_REPORT_NOTE = (
    'INTEGRATION SMOKE — not decision-grade. Short chains test executable '
    'workflow and arithmetic only; they do not establish convergence, '
    'causal validity, or ROI recovery. See sampling-quality.json.'
)


def _check_arithmetic(analysis, data) -> dict:
  """Check paid-channel identities on the actual fitted posterior draws."""
  rtol = (
      1e-9
      if backend.standardize_dtype(backend.float_dtype) == 'float64'
      else 1e-5
  )
  roi = np.asarray(analysis.roi())
  paid_channels = len(data.get_all_paid_channels())
  if roi.shape[-1] != paid_channels or not np.all(np.isfinite(roi)):
    raise AssertionError(
        'The fitted model must produce finite paid-channel ROI.'
    )
  incremental = np.asarray(
      analysis.incremental_outcome(include_non_paid_channels=False)
  )
  spend = np.asarray(data.get_total_spend())
  spend = spend.sum(axis=tuple(range(spend.ndim - 1)))
  np.testing.assert_allclose(roi, incremental / spend, rtol=rtol)
  for dimension in ('geos', 'times'):
    granular = np.asarray(
        analysis.incremental_outcome(
            include_non_paid_channels=False,
            **{f'aggregate_{dimension}': False},
        )
    )
    np.testing.assert_allclose(
        granular.sum(axis=-2), incremental, rtol=rtol, atol=rtol
    )
  curves = analysis.response_curves(spend_multipliers=[0.0, 0.5, 1.0, 1.5])
  mean = curves.incremental_outcome.sel({c.METRIC: c.MEAN}).values
  if not np.all(np.isfinite(mean)):
    raise AssertionError('Response curves must be finite.')
  np.testing.assert_allclose(mean[0], 0.0, atol=rtol)
  if np.any(np.diff(mean, axis=0) < -rtol):
    raise AssertionError('Paid response curves must be monotone in spend.')
  return {
      'roi_shape': list(roi.shape),
      'paid_channels': paid_channels,
      'roi_identity_max_abs_delta': float(
          np.max(np.abs(roi - incremental / spend))
      ),
      'incremental_geo_time_additivity': 'PASS',
      'response_zero_and_monotonicity': 'PASS',
  }


def _check_optimization(optimization) -> dict:
  """Check fixed-budget accounting and the explicitly requested spend bounds."""
  for dataset in (optimization.nonoptimized_data, optimization.optimized_data):
    if not np.isfinite(float(dataset.total_roi)):
      raise AssertionError('Budget optimization produced non-finite ROI.')
    if not np.all(np.isfinite(dataset.spend)) or np.any(dataset.spend < 0):
      raise AssertionError('Budget optimization produced invalid spend.')
    np.testing.assert_allclose(
        float(dataset.total_roi),
        float(dataset.total_incremental_outcome) / float(dataset.budget),
    )
    np.testing.assert_allclose(
        float(dataset.budget), float(dataset.spend.sum())
    )
  baseline = np.asarray(optimization.nonoptimized_data.spend)
  allocation = np.asarray(optimization.optimized_data.spend)
  # The optimizer rounds per-channel budgets to its discrete grid. Compare the
  # optimized allocation to that rounded baseline, not unrounded input spend.
  step = 10 ** (-optimization.optimization_grid.round_factor)
  budget_delta = float(allocation.sum() - baseline.sum())
  if abs(budget_delta) > step * len(baseline):
    raise AssertionError(
        'Fixed-budget optimization changed the rounded budget.'
    )
  lower, upper = optimizer.get_optimization_bounds(
      n_channels=len(baseline),
      spend=baseline,
      round_factor=optimization.optimization_grid.round_factor,
      spend_constraint_lower=0.3,
      spend_constraint_upper=0.3,
  )
  if np.any(allocation < lower) or np.any(allocation > upper):
    raise AssertionError('Optimization violated its channel spend bounds.')
  return {
      'current_roi': float(optimization.nonoptimized_data.total_roi),
      'optimized_roi': float(optimization.optimized_data.total_roi),
      'fixed_budget_max_rounding_delta': step * len(baseline),
      'fixed_budget_actual_delta': budget_delta,
      'channel_spend_bounds': 'PASS',
  }


def _check_report(path: Path) -> int:
  html = path.read_text(encoding='utf-8')
  style = html.split('<style>', 1)[1].split('</style>', 1)[0].strip()
  if len(style) < 1000 or 'MeridianCharts.mount(' not in html:
    raise AssertionError('Report must contain compiled CSS and chart specs.')
  if _REPORT_NOTE not in html:
    raise AssertionError(
        'Each saved report must identify the integration-only scope.'
    )
  return path.stat().st_size


def _check_decision(analysis, optimization, quality, output_dir: Path) -> dict:
  """Exercise paired gain evidence for the helper's supported media-only scope."""
  if analysis.model_context.input_data.reach is not None:
    return {'decision_audit': 'outside media-only scope (reach/frequency)'}
  baseline_data = optimization.nonoptimized_data
  optimized_data = optimization.optimized_data
  channels = baseline_data.channel.values
  baseline = dict(zip(channels, np.asarray(baseline_data.spend, dtype=float)))
  candidate = dict(zip(channels, np.asarray(optimized_data.spend, dtype=float)))
  report = budget_decision.audit_budget_decisions(
      specifications={'smoke': analysis},
      baseline=baseline,
      candidates={'optimized': candidate},
      outcome_unit='revenue',
      use_kpi=False,
      policy=budget_decision.DownsidePolicy(
          loss_tolerance=0.0,
          max_loss_probability=0.1,
          max_expected_downside=0.0,
      ),
      selected_times=optimization.optimization_grid.selected_times,
  )
  evidence = report.to_dict()
  baseline_result = evidence['fits']['smoke']['candidates']['baseline']
  for field in (
      'mean_gain',
      'median_gain',
      'expected_downside',
      'probability_loss_beyond_tolerance',
  ):
    if baseline_result[field] != 0.0:
      raise AssertionError(
          'Comparing a baseline to itself must give zero gain/loss.'
      )
  if report.sampling_ready != quality.sampling_passed:
    raise AssertionError(
        'Decision sampling screen must retain the smoke fit quality.'
    )
  if evidence['recommendation'] is not None:
    raise AssertionError(
        'The exploratory decision audit must not select a budget.'
    )
  gain = evidence['fits']['smoke']['candidates']['optimized']['mean_gain']
  outcome_scale = max(
      abs(float(optimized_data.total_incremental_outcome)),
      abs(float(baseline_data.total_incremental_outcome)),
      1.0,
  )
  # Subtracting rounded posterior means can lose precision relative to averaging
  # paired draw differences. Bound that cancellation by the underlying dtype.
  native_epsilon = float(np.finfo(backend.np_float_dtype).eps)
  optimizer_gain = float(optimized_data.total_incremental_outcome) - float(
      baseline_data.total_incremental_outcome
  )
  draws = []
  history = (
      analysis.get_aggregated_spend(
          selected_times=optimization.optimization_grid.selected_times,
          include_media=True,
          include_rf=False,
      )
      .sel({c.CHANNEL: channels})
      .values.astype(float)
  )
  for allocation in (baseline_data.spend, optimized_data.spend):
    ratio = np.asarray(allocation, dtype=float) / history
    draws.append(
        np.asarray(
            analysis.incremental_outcome(
                new_data=tensors.DataTensors(
                    media=backend.to_tensor(ratio, dtype=backend.float_dtype)
                    * analysis.model_context.media_tensors.media
                ),
                selected_geos=analysis.model_context.input_data.geo.values.tolist(),
                selected_times=(
                    optimization.optimization_grid.selected_times
                    or analysis.model_context.input_data.time.values.tolist()
                ),
                use_kpi=False,
                include_non_paid_channels=False,
            )
        )
    )
  reduction_terms = int(draws[0].size)
  # gamma_n bounds accumulated rounding in a reduction of n native values.
  # Compare it at the outcome scale, before subtracting near-equal means.
  gamma_n = (
      reduction_terms * native_epsilon / (1 - reduction_terms * native_epsilon)
  )
  roundoff_bound = max(1e-9, 2 * gamma_n * outcome_scale)
  direct_paired_gain = float((draws[1] - draws[0]).sum(axis=-1).mean())
  separately_averaged_gain = float(draws[1].mean(axis=(0, 1)).sum()) - float(
      draws[0].mean(axis=(0, 1)).sum()
  )
  np.testing.assert_allclose(
      gain, direct_paired_gain, rtol=0, atol=roundoff_bound
  )
  for reduced in (optimizer_gain, separately_averaged_gain):
    np.testing.assert_allclose(gain, reduced, rtol=1e-5, atol=roundoff_bound)
  (output_dir / 'budget-decision.json').write_text(
      report.to_json(indent=2) + '\n', encoding='utf-8'
  )
  return {
      'decision_audit': 'PASS (exploratory paired posterior evidence)',
      'decision_sampling_screen_passed': report.sampling_ready,
      'paired_mean_gain': gain,
      'direct_paired_mean_gain': direct_paired_gain,
      'direct_paired_gain_abs_delta': abs(gain - direct_paired_gain),
      'separately_averaged_mean_gain': separately_averaged_gain,
      'paired_optimizer_gain_abs_delta': abs(gain - optimizer_gain),
      'paired_optimizer_gain_roundoff_bound': roundoff_bound,
      'paired_reduction_terms': reduction_terms,
      'paired_outcome_scale': outcome_scale,
      'paired_native_epsilon': float(native_epsilon),
  }


def _fixture_data(case: str):
  """Construct the deterministic case without sampling."""
  if case not in _CASES:
    raise ValueError(f'Unknown smoke case: {case}')
  factory = (
      test_utils.sample_input_data_non_revenue_revenue_per_kpi
      if case == _DEFAULT_CASE
      else test_utils.sample_input_data_revenue
  )
  data = factory(
      n_times=20,
      n_media_times=22,
      n_controls=1,
      **_CASES[case],
  )
  if case == 'geo_revenue_mixed':
    # Helpers reuse seed 0 across fields, creating perfectly correlated paired
    # columns at equal channel counts. Independent streams keep the fixture
    # estimable without bypassing the model's actual EDA fitting guardrails.
    data = dataclasses.replace(
        data,
        organic_media=test_utils.random_organic_media_da(
            n_geos=2,
            n_times=20,
            n_media_times=22,
            n_organic_media_channels=1,
            seed=1,
        ),
        non_media_treatments=test_utils.random_non_media_treatments_da(
            media=data.media,
            n_geos=2,
            n_times=20,
            n_non_media_channels=1,
            seed=2,
        ),
    )
  return data


def _fit_case(case: str):
  """Fit the deterministic smoke fixture with the actual sampler."""
  mmm = model.Meridian(
      input_data=_fixture_data(case), model_spec=spec.ModelSpec(max_lag=2)
  )
  mmm.sample_prior(20, seed=0)
  mmm.sample_posterior(n_chains=2, n_adapt=20, n_burnin=20, n_keep=20, seed=0)
  return mmm


def _validate_saved_case(mmm, case: str) -> None:
  """Prevent a saved arbitrary fit from being relabeled as a smoke fixture."""
  if case not in _CASES:
    raise ValueError(f'Unknown smoke case: {case}')
  expected = dict.fromkeys(
      (
          'n_media_channels',
          'n_rf_channels',
          'n_organic_media_channels',
          'n_organic_rf_channels',
          'n_non_media_channels',
      ),
      0,
  )
  expected.update(n_times=20, n_media_times=22, n_controls=1, **_CASES[case])
  for name, value in expected.items():
    if getattr(mmm.model_context, name) != value:
      raise ValueError(f'Saved model does not match case {case}: {name}.')
  expected_kpi = c.NON_REVENUE if case == _DEFAULT_CASE else c.REVENUE
  if mmm.input_data.kpi_type != expected_kpi or mmm.model_spec.max_lag != 2:
    raise ValueError(
        f'Saved model does not match case {case}: KPI/specification.'
    )
  expected_spec = spec.ModelSpec(max_lag=2)
  if hyperparameters.HyperparametersSerde().serialize(
      mmm.model_spec
  ) != hyperparameters.HyperparametersSerde().serialize(
      expected_spec
  ) or distribution.DistributionSerde(
      function_registry.FunctionRegistry()
  ).serialize(
      mmm.model_spec.prior
  ) != distribution.DistributionSerde(
      function_registry.FunctionRegistry()
  ).serialize(
      expected_spec.prior
  ):
    raise ValueError('Saved smoke model must retain its default specification.')
  expected_data = _fixture_data(case)
  for field in dataclasses.fields(expected_data):
    expected_array = getattr(expected_data, field.name)
    actual_array = getattr(mmm.input_data, field.name)
    if hasattr(expected_array, 'equals'):
      matches = actual_array is not None and expected_array.equals(actual_array)
    else:
      matches = expected_array == actual_array
    if not matches:
      raise ValueError(
          f'Saved model does not match case {case}: input field {field.name}.'
      )
  posterior = mmm.inference_data.posterior
  if posterior.sizes.get(c.CHAIN) != 2 or posterior.sizes.get(c.DRAW) != 20:
    raise ValueError('Saved smoke model must contain two chains of 20 draws.')


def run(
    output_dir: Path, case: str = _DEFAULT_CASE, *, reuse_models=False
) -> dict:
  """Fit or reuse a validated fixture, then check the real downstream workflow."""
  started = time.perf_counter()
  output_dir.mkdir(parents=True, exist_ok=True)
  source_hash = None
  if reuse_models:
    source = output_dir / 'model.binpb'
    if not source.is_file():
      raise FileNotFoundError(f'No saved smoke model to reuse: {source}')
    source_hash = sha256(source)
    mmm = meridian_serde.load_meridian(str(source))
    _validate_saved_case(mmm, case)
  else:
    mmm = _fit_case(case)
  # Retain the actual fit even when a downstream integration assertion fails.
  meridian_serde.save_meridian(mmm, str(output_dir / 'model.binpb'))
  data = mmm.input_data
  quality = sampling_quality.assess_sampling_quality(
      mmm.inference_data, model_context=mmm.model_context
  )
  (output_dir / 'sampling-quality.json').write_text(
      quality.to_json(indent=2) + '\n', encoding='utf-8'
  )
  analysis = analyzer.Analyzer(
      model_context=mmm.model_context, inference_data=mmm.inference_data
  )
  roi = np.asarray(analysis.roi())
  arithmetic = _check_arithmetic(analysis, data)
  optimization = optimizer.BudgetOptimizer(mmm).optimize(
      spend_constraint_lower=0.3, spend_constraint_upper=0.3
  )
  accounting = _check_optimization(optimization)
  decision = _check_decision(analysis, optimization, quality, output_dir)

  model_path = output_dir / 'model.binpb'
  meridian_serde.save_meridian(mmm, str(model_path))
  restored = meridian_serde.load_meridian(str(model_path))
  restored_analysis = analyzer.Analyzer(
      model_context=restored.model_context,
      inference_data=restored.inference_data,
  )
  np.testing.assert_array_equal(roi, np.asarray(restored_analysis.roi()))
  if restored.model_spec.media_prior_type != mmm.model_spec.media_prior_type:
    raise AssertionError('Serialization changed the media prior type.')

  summarizer.Summarizer(
      restored, report_note=_REPORT_NOTE
  ).output_model_results_summary(
      'summary.html',
      str(output_dir),
      str(data.kpi.time.values[0]),
      str(data.kpi.time.values[-1]),
  )
  optimization.output_optimization_summary('optimization.html', str(output_dir))
  optimization_path = output_dir / 'optimization.html'
  html = optimization_path.read_text(encoding='utf-8')
  optimization_path.write_text(
      html.replace(
          '<cards role="main">',
          '<cards role="main"><div class="advisory-warning-banner" '
          f'role="note"><p>{_REPORT_NOTE}</p></div>',
          1,
      ),
      encoding='utf-8',
  )
  results = {
      'date': datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
      'status': 'PASS',
      'scope': 'integration smoke; not a convergence test',
      'case': case,
      'reuse_models': bool(reuse_models),
      'source_model_sha256': source_hash,
      'input_shape': {
          'n_times': 20,
          'n_media_times': 22,
          'n_controls': 1,
          'max_lag': 2,
          'kpi_type': data.kpi_type,
          **_CASES[case],
      },
      'backend': backend.computation_backend().name,
      'precision': backend.standardize_dtype(backend.float_dtype),
      'sampling_quality_status': quality.status,
      'serialization_roi_max_delta': 0.0,
      'report_bytes': _check_report(output_dir / 'summary.html'),
      'optimization_report_bytes': _check_report(optimization_path),
      'elapsed_seconds': time.perf_counter() - started,
      'provenance': provenance(Path(__file__)),
      **arithmetic,
      **accounting,
      **decision,
  }
  write_evidence(output_dir / 'results.json', results)
  return results


def run_matrix(output_dir: Path, *, reuse_models=False) -> dict:
  """Run each real fixture in a fresh process; retain reports and failure logs."""
  output_dir.mkdir(parents=True, exist_ok=True)
  results = []
  for case in _CASES:
    case_dir = output_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    print(f'Running integration case: {case}', flush=True)
    with (case_dir / 'run.log').open('w', encoding='utf-8') as log:
      subprocess.run(
          [
              sys.executable,
              str(Path(__file__).resolve()),
              '--case',
              case,
              '--output-dir',
              str(case_dir.resolve()),
          ]
          + (['--reuse-models'] if reuse_models else []),
          cwd=_REPO_ROOT,
          stdout=log,
          stderr=subprocess.STDOUT,
          check=True,
      )
    results.append(json.loads((case_dir / 'results.json').read_text('utf-8')))
  payload = {
      'date': datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
      'status': 'PASS',
      'scope': 'integration smoke matrix; not convergence or ROI recovery',
      'reuse_models': bool(reuse_models),
      'cases': results,
      'provenance': provenance(Path(__file__)),
  }
  write_evidence(output_dir / 'results.json', payload)
  return payload


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output-dir', type=Path)
  parser.add_argument(
      '--reuse-models',
      action='store_true',
      help='Validate saved case models and refresh downstream reports; no resampling.',
  )
  selection = parser.add_mutually_exclusive_group()
  selection.add_argument('--case', choices=tuple(_CASES), default=_DEFAULT_CASE)
  selection.add_argument('--all-cases', action='store_true')
  args = parser.parse_args()
  if args.reuse_models and args.output_dir is None:
    parser.error(
        '--reuse-models requires --output-dir containing saved models.'
    )
  runner = (
      lambda path: run_matrix(path, reuse_models=args.reuse_models)
      if args.all_cases
      else run(path, args.case, reuse_models=args.reuse_models)
  )
  if args.output_dir:
    results = runner(args.output_dir)
  else:
    with tempfile.TemporaryDirectory(prefix='meridian-e2e-') as tmp:
      results = runner(Path(tmp))
  print(json.dumps(results, indent=2))


if __name__ == '__main__':
  main()
