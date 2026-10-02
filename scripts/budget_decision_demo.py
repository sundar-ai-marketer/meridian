#!/usr/bin/env python3
# Copyright 2026 Sundar Ramesh Kumar.
# Licensed under the Apache License, Version 2.0.
# NOTICE: This file is new in this fork; see NOTICE at the repository root.

"""Write exact-arithmetic budget-comparison fixtures, without fitting a model.

This is executable contract evidence only. The posterior-shaped arrays below
are hand-constructed fixtures, not posterior samples, fitted-prior sensitivity,
or scientific validation. Real fitted integration lives in test_end_to_end.py.
"""

import argparse
import datetime
from pathlib import Path
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meridian.analysis import budget_decision  # pylint: disable=g-import-not-at-top
from scripts import evidence  # pylint: disable=g-import-not-at-top


def _fixture(values, specification):
  """Labels an unfitted arithmetic fixture for within-scenario pairing."""
  values = np.asarray(values, dtype=float)
  return xr.DataArray(
      values, dims=('chain', 'draw', 'channel'),
      coords={'chain': list(range(values.shape[0])),
              'draw': list(range(values.shape[1])), 'channel': ['a', 'b']},
      attrs={'posterior_id': specification, 'outcome_unit': 'synthetic_units',
             'selected_geos': ('synthetic_geo',),
             'selected_times': ('2024-01-01',),
             'execution_semantics': 'hand_constructed_unfitted_fixture'},
  )


def run_demo():
  """Assert expected arithmetic before returning the evidence payload."""
  policy = budget_decision.DownsidePolicy(
      loss_tolerance=0, max_loss_probability=0.1, max_expected_downside=0.1
  )
  correlated = np.array([
      [[100, -99], [-100, 101], [50, -49]],
      [[-50, 51], [20, -19], [-20, 21]],
  ], dtype=float)
  tail_risk = np.zeros_like(correlated)
  tail_risk[..., 0] = [[100, 100, 100], [100, -50, -50]]
  summaries = {}
  for name, values in (
      ('correlated_channels', correlated), ('positive_mean_high_downside', tail_risk)
  ):
    baseline = _fixture(np.zeros_like(values), name)
    candidate = _fixture(values, name)
    summaries[name] = budget_decision.summarize_paired_outcomes(
        baseline, candidate, policy=policy
    )
    np.testing.assert_array_equal(
        budget_decision.paired_outcome_delta(baseline, baseline), 0
    )
  if summaries['correlated_channels']['central_interval'] != {
      'probability': 0.9, 'lower': 1.0, 'upper': 1.0
  }:
    raise AssertionError('Channel correlation must be preserved before quantiles.')
  if summaries['positive_mean_high_downside']['downside_policy_met_in_draws']:
    raise AssertionError('High mean gain must not erase observed downside.')

  separate = {}
  for name, value, shape in (
      ('fixture_a', 1, (2, 4, 2)), ('fixture_b', -1, (4, 3, 2))
  ):
    values = np.zeros(shape)
    values[..., 0] = value
    separate[name] = budget_decision.summarize_paired_outcomes(
        _fixture(np.zeros(shape), name), _fixture(values, name), policy=policy
    )
  means = [summary['mean_gain'] for summary in separate.values()]
  if not min(means) == -1 or not max(means) == 1:
    raise AssertionError('Opposite specification fixtures must remain separate.')
  return {
      'evidence_kind': 'exact_arithmetic_contract_fixtures_only',
      'fitted_model': False,
      'scientific_validation': False,
      'sampling_ready': False,
      'decision_status': 'requires_fitted_model_and_business_review',
      'recommendation': None,
      'summaries': summaries,
      'separate_specification_fixtures': separate,
      'mean_gain_sensitivity_envelope': {'minimum': min(means), 'maximum': max(means)},
      'unanimous_positive_mean_gain': all(mean > 0 for mean in means),
      'limitations': [
          'Arrays are hand-constructed, unfitted, and not posterior samples.',
          'Fixture fractions demonstrate arithmetic only; they estimate no '
          'real campaign risk, prior sensitivity, or scientific reliability.',
          'No allocation is selected. Tail functional Monte Carlo uncertainty '
          'is not assessed by the report.',
      ],
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path)
  args = parser.parse_args()
  date = datetime.date.today().isoformat()
  output = args.output or (
      evidence.REPO_ROOT / 'docs' / 'validation' /
      f'budget-decision-contract-{date}.json'
  )
  payload = {
      'date': date, **run_demo(),
      'provenance': evidence.provenance(Path(__file__)),
  }
  evidence.write_evidence(output, payload)
  print(f'Wrote unfitted arithmetic contract evidence: {output}')


if __name__ == '__main__':
  main()
