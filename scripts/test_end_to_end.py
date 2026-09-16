#!/usr/bin/env python3
# Copyright 2026 Sundar Ramesh Kumar.
# Licensed under the Apache License, Version 2.0.
# NOTICE: This file is new in this fork; see NOTICE at the repository root.

"""Exercise the real model-to-report workflow; small draws test plumbing only.

Run with either MERIDIAN_BACKEND=jax or MERIDIAN_BACKEND=tensorflow.
This checks executable integration, not statistical convergence or ROI accuracy.
"""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from meridian.analysis import analyzer, optimizer, summarizer
from meridian.data import test_utils
from meridian.model import model, spec
from meridian.schema.serde import meridian_serde


def run(output_dir: Path) -> dict:
  """Fit, analyze, optimize, serialize, reload, and render without mocks."""
  output_dir.mkdir(parents=True, exist_ok=True)
  data = test_utils.sample_input_data_non_revenue_revenue_per_kpi(
      n_geos=2,
      n_times=20,
      n_media_times=22,
      n_media_channels=2,
      n_controls=1,
  )
  mmm = model.Meridian(input_data=data, model_spec=spec.ModelSpec(max_lag=2))
  mmm.sample_prior(20, seed=0)
  mmm.sample_posterior(n_chains=2, n_adapt=20, n_burnin=20, n_keep=20, seed=0)
  analysis = analyzer.Analyzer(
      model_context=mmm.model_context, inference_data=mmm.inference_data
  )
  roi = np.asarray(analysis.roi())
  if roi.shape[-1] != 2 or not np.all(np.isfinite(roi)):
    raise AssertionError('The fitted model must produce finite channel ROI.')

  optimization = optimizer.BudgetOptimizer(mmm).optimize()
  for dataset in (optimization.nonoptimized_data, optimization.optimized_data):
    if not np.isfinite(float(dataset.total_roi)):
      raise AssertionError('Budget optimization produced non-finite ROI.')
    if not np.all(np.isfinite(dataset.spend)) or np.any(dataset.spend < 0):
      raise AssertionError('Budget optimization produced invalid spend.')

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

  summarizer.Summarizer(restored).output_model_results_summary(
      'summary.html',
      str(output_dir),
      str(data.kpi.time.values[0]),
      str(data.kpi.time.values[-1]),
  )
  html = (output_dir / 'summary.html').read_text()
  style = html.split('<style>', 1)[1].split('</style>', 1)[0].strip()
  if len(style) < 1000 or 'vegaEmbed(' not in html:
    raise AssertionError('Report must contain compiled CSS and chart specs.')
  results = {
      'status': 'PASS',
      'scope': 'integration smoke; not a convergence test',
      'roi_shape': list(roi.shape),
      'serialization_roi_max_delta': 0.0,
      'current_roi': float(optimization.nonoptimized_data.total_roi),
      'optimized_roi': float(optimization.optimized_data.total_roi),
      'report_bytes': (output_dir / 'summary.html').stat().st_size,
  }
  (output_dir / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
  return results


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output-dir', type=Path)
  args = parser.parse_args()
  if args.output_dir:
    results = run(args.output_dir)
  else:
    with tempfile.TemporaryDirectory(prefix='meridian-e2e-') as tmp:
      results = run(Path(tmp))
  print(json.dumps(results, indent=2))


if __name__ == '__main__':
  main()
