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

"""Paired posterior evidence for a bounded set of equal-budget allocations.

This report evaluates user-supplied allocations, not a new optimizer. Each
candidate and the baseline use the SAME posterior chain/draw in each fit.
Channel contributions are summed before quantiles are computed. Fitted priors
or model specifications remain separate: their min/max is a sensitivity
envelope, never a pooled posterior or a probability over specifications.

The first contract supports paid media without reach/frequency. It uses the
optimizer's constant-cost scaled historical flight: spend in the selected
outcome window supplies the denominator, and the resulting channel ratio
scales ALL historical media, including pre-window adstock history. This is a
conditional expected-outcome comparison, not a forward intervention that
changes only future execution, realized outcome uncertainty, or profit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import dataclasses
import hashlib
import json
from typing import Any

from meridian import backend
from meridian import constants as c
from meridian.analysis import analyzer as analyzer_module
from meridian.analysis import sampling_quality
from meridian.analysis import tensors
import numpy as np
import xarray as xr

__all__ = [
    'BudgetDecisionReport',
    'DownsidePolicy',
    'audit_budget_decisions',
    'paired_outcome_delta',
    'summarize_paired_outcomes',
]

_BASELINE = 'baseline'
_EXECUTION = 'constant_cost_scaled_historical_flight_including_adstock_history'
_PAIR_METADATA = (
    'posterior_id', 'outcome_unit', 'selected_geos', 'selected_times',
    'execution_semantics',
)
_LIMITATIONS = (
    'Results condition on each fitted model and its causal assumptions; '
    'sampling checks do not establish identification or model adequacy.',
    'Outcome uncertainty concerns conditional expected incremental outcome; '
    'it excludes future observation noise and is not profit.',
    'Loss probabilities are empirical fractions of correlated posterior '
    'draws. Monte Carlo uncertainty for the decision functionals is not '
    'assessed, so policy flags do not authorize a budget decision.',
    'Specification envelopes are sensitivity ranges, not pooled posterior '
    'credible intervals or probabilities over specifications.',
    'Candidates are a bounded user-supplied set; this report makes no global '
    'optimality claim and selects no allocation.',
)


@dataclasses.dataclass(frozen=True)
class DownsidePolicy:
  """Explicit business tolerances in the declared expected-outcome unit.

  ``loss_tolerance`` defines the event delta < -loss_tolerance. Expected
  downside is mean(max(-delta, 0)), including losses below that tolerance.
  The probability and downside ceilings describe the supplied draws, not a
  guarantee or an automatic decision rule.
  """

  loss_tolerance: float
  max_loss_probability: float
  max_expected_downside: float

  def __post_init__(self):
    for name in ('loss_tolerance', 'max_expected_downside'):
      value = getattr(self, name)
      if not np.isfinite(value) or value < 0:
        raise ValueError(f'`{name}` must be finite and non-negative.')
    if (
        not np.isfinite(self.max_loss_probability)
        or not 0 <= self.max_loss_probability <= 1
    ):
      raise ValueError('`max_loss_probability` must be between 0 and 1.')


@dataclasses.dataclass(frozen=True)
class BudgetDecisionReport:
  """JSON-safe per-fit evidence with a sampling-only readiness screen."""

  scope: dict[str, Any]
  policy: DownsidePolicy
  fits: dict[str, Any]
  sensitivity: dict[str, Any]
  sampling_ready: bool

  def to_dict(self) -> dict[str, Any]:
    """Returns an independent standards-compliant JSON representation."""
    return copy.deepcopy({
        'schema_version': 1,
        'sampling_ready': self.sampling_ready,
        'decision_status': 'requires_model_and_business_review',
        'sampling_ready_meaning': (
            'Every supplied fit passed strict sampling checks and produced '
            'finite paired quantities. This is a sampling-only screen, '
            'conditional on the recorded scope and model assumptions.'
        ),
        'recommendation': None,
        'scope': self.scope,
        'policy': dataclasses.asdict(self.policy),
        'fits': self.fits,
        'sensitivity': self.sensitivity,
        'limitations': list(_LIMITATIONS),
    })

  def to_json(self, *, indent: int | None = None) -> str:
    """Serializes without nonstandard NaN/Infinity JSON tokens."""
    return json.dumps(
        self.to_dict(), indent=indent, sort_keys=True, allow_nan=False
    )


def paired_outcome_delta(
    baseline: xr.DataArray, candidate: xr.DataArray
) -> xr.DataArray:
  """Sums channels and subtracts within exactly matched posterior draws.

  Inputs must have exactly chain/draw/channel dimensions, finite values,
  unique explicit coordinates and matching pairing metadata. Chain and draw
  order must match. Channel order may differ and is aligned by label.
  Required attrs are ``posterior_id``, ``outcome_unit``, ``selected_geos``,
  ``selected_times`` and ``execution_semantics``. The caller must ensure that
  posterior_id identifies the same fitted posterior, not merely equal-sized
  arrays from different fits. ``audit_budget_decisions`` constructs this
  metadata from the Analyzer and the selected scope.
  """
  dims = (c.CHAIN, c.DRAW, c.CHANNEL)
  for name, data in (('baseline', baseline), ('candidate', candidate)):
    if set(data.dims) != set(dims):
      raise ValueError(f'`{name}` must have chain/draw/channel dimensions.')
    for dim in dims:
      if dim not in data.coords or not data.get_index(dim).is_unique:
        raise ValueError(f'`{name}` needs unique explicit {dim} coordinates.')
      if data.sizes[dim] < 1:
        raise ValueError(f'`{name}` has no {dim} values.')
    if not np.all(np.isfinite(data.values)):
      raise ValueError(f'`{name}` contains non-finite outcomes.')
  for key in _PAIR_METADATA:
    if key not in baseline.attrs or key not in candidate.attrs:
      raise ValueError(f'Pairing metadata `{key}` is required.')
    if baseline.attrs[key] != candidate.attrs[key]:
      raise ValueError(f'Pairing metadata `{key}` does not match.')
  for dim in (c.CHAIN, c.DRAW):
    if not baseline.get_index(dim).equals(candidate.get_index(dim)):
      raise ValueError(f'Posterior {dim} coordinates do not match exactly.')
  if set(baseline[c.CHANNEL].values) != set(candidate[c.CHANNEL].values):
    raise ValueError('Channel labels do not match.')
  candidate = candidate.sel({c.CHANNEL: baseline[c.CHANNEL]})
  delta = (candidate - baseline).sum(c.CHANNEL, skipna=False)
  if not np.all(np.isfinite(delta.values)):
    raise ValueError('Paired total gain contains non-finite values.')
  delta.attrs = dict(baseline.attrs)
  return delta.transpose(c.CHAIN, c.DRAW)


def summarize_paired_outcomes(
    baseline: xr.DataArray,
    candidate: xr.DataArray,
    *,
    policy: DownsidePolicy,
    confidence_level: float = 0.9,
    lower_quantile_probability: float = 0.05,
) -> dict[str, Any]:
  """Reports the posterior distribution of paired total expected gain."""
  _validate_probabilities(confidence_level, lower_quantile_probability)
  delta = paired_outcome_delta(baseline, candidate).values
  if not np.all(np.isfinite(delta)):
    raise ValueError('Paired total gain contains non-finite values.')
  alpha = (1 - confidence_level) / 2
  n_loss = int(np.count_nonzero(delta < -policy.loss_tolerance))
  probability = float(n_loss / delta.size)
  downside = float(np.maximum(-delta, 0).mean())
  result = {
      'mean_gain': float(delta.mean()),
      'median_gain': float(np.median(delta)),
      'central_interval': {
          'probability': float(confidence_level),
          'lower': float(np.quantile(delta, alpha)),
          'upper': float(np.quantile(delta, 1 - alpha)),
      },
      'lower_quantile': {
          'probability': float(lower_quantile_probability),
          'value': float(np.quantile(delta, lower_quantile_probability)),
      },
      'probability_loss_beyond_tolerance': probability,
      'n_loss_draws': n_loss,
      'n_posterior_draws': int(delta.size),
      'expected_downside': downside,
      'downside_policy_met_in_draws': (
          probability <= policy.max_loss_probability
          and downside <= policy.max_expected_downside
      ),
      'monte_carlo_uncertainty': 'not_assessed_for_decision_functionals',
  }
  finite_statistics = (
      result['mean_gain'], result['median_gain'], downside,
      result['central_interval']['lower'], result['central_interval']['upper'],
      result['lower_quantile']['value'],
  )
  if not np.all(np.isfinite(finite_statistics)):
    raise ValueError('Paired summary contains non-finite statistics.')
  return result


def audit_budget_decisions(
    specifications: Mapping[str, analyzer_module.Analyzer],
    *,
    baseline: Mapping[str, float],
    candidates: Mapping[str, Mapping[str, float]],
    policy: DownsidePolicy,
    outcome_unit: str,
    use_kpi: bool,
    selected_geos: Sequence[str] | None = None,
    selected_times: Sequence[str] | None = None,
    confidence_level: float = 0.9,
    lower_quantile_probability: float = 0.05,
    batch_size: int = c.DEFAULT_BATCH_SIZE,
) -> BudgetDecisionReport:
  """Audits named fitted specifications under the same historical flight.

  Supply one to eight Analyzer instances fitted to identical InputData
  (channel order may differ), a named baseline allocation and up to 32 named
  candidates. All allocations must use the same channel labels and budget.
  ``baseline`` is always included in results. Positive spend cannot be assigned
  to a channel with zero historical reference spend because no cost model can
  be inferred there. R&F models and new/future execution tensors are outside
  this first contract. ``use_kpi=False`` requires revenue conversion data and
  outcome_unit='revenue'; otherwise name the KPI unit explicitly.

  Strict sampling quality is assessed for EVERY specification. Failed or
  unavailable sampling remains visible, quantities are labeled exploratory,
  and sampling_ready is false. No allocation is recommended, even when all
  checks pass. Different specifications need not have equal draw counts;
  only baseline/candidate draws WITHIN each fit are paired.

  Raises:
    ValueError: For unsupported scope, inconsistent input data, malformed
      allocations, non-finite outcomes or invalid probabilities/tolerances.
  """
  _validate_probabilities(confidence_level, lower_quantile_probability)
  if not 1 <= len(specifications) <= 8:
    raise ValueError('Supply between one and eight fitted specifications.')
  if len(candidates) > 32:
    raise ValueError('Supply at most 32 candidate allocations.')
  for name in (*specifications, *candidates):
    if not isinstance(name, str) or not name.strip():
      raise ValueError('Specification and candidate names must be nonempty.')
  if _BASELINE in candidates:
    raise ValueError('Candidate name `baseline` is reserved.')
  if not isinstance(outcome_unit, str) or not outcome_unit.strip():
    raise ValueError('Name the expected-outcome unit explicitly.')
  if not isinstance(use_kpi, bool):
    raise ValueError('`use_kpi` must be an explicit boolean.')
  if not use_kpi and outcome_unit != 'revenue':
    raise ValueError('Revenue comparisons require outcome_unit="revenue".')
  if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
    raise ValueError('`batch_size` must be a positive integer.')

  first = next(iter(specifications.values()))
  input_data = first.model_context.input_data
  channels = list(input_data.get_all_paid_channels())
  if not channels or len(channels) != len(set(channels)):
    raise ValueError('At least one uniquely named paid media channel is required.')
  geos = _selected_coordinates(selected_geos, input_data.geo.values, 'geos')
  times = _selected_coordinates(selected_times, input_data.time.values, 'times')
  reference_data = _canonical_input_data(first)
  allocations = {_BASELINE: _allocation(baseline, channels)}
  for name, allocation in candidates.items():
    allocations[name] = _allocation(allocation, channels)
    if not np.isclose(
        allocations[name].sum(), allocations[_BASELINE].sum(),
        rtol=1e-10, atol=1e-8,
    ):
      raise ValueError(f'Candidate `{name}` must have the same total budget.')

  # Resolve the shared data contract before evaluating any candidate.
  for name, fitted in specifications.items():
    if not _canonical_input_data(fitted).equals(reference_data):
      raise ValueError(f'Specification `{name}` has different input data/history.')
    if fitted.model_context.input_data.kpi_type != input_data.kpi_type:
      raise ValueError(f'Specification `{name}` has a different KPI type.')
    if fitted.model_context.input_data.currency_code != input_data.currency_code:
      raise ValueError(f'Specification `{name}` has a different currency.')
    if use_kpi and input_data.kpi_type == c.REVENUE:
      raise ValueError('Revenue-type KPI requires use_kpi=False and revenue units.')
    if not use_kpi and fitted.model_context.input_data.revenue_per_kpi is None:
      raise ValueError('Revenue comparisons require revenue_per_kpi data.')

  fits = {}
  for name, fitted in specifications.items():
    fits[name] = _evaluate_fit(
        name=name, fitted=fitted, allocations=allocations, channels=channels,
        policy=policy, outcome_unit=outcome_unit, use_kpi=use_kpi,
        geos=geos, times=times, confidence_level=confidence_level,
        lower_quantile_probability=lower_quantile_probability,
        batch_size=batch_size,
    )
  ready = all(
      fit['sampling_quality']['sampling_passed']
      and fit['outcome_status'] == 'available'
      for fit in fits.values()
  )
  sensitivity = _sensitivity(fits, allocations)
  scope = {
      'outcome_unit': outcome_unit,
      'currency_code': input_data.currency_code,
      'outcome_kind': 'kpi' if use_kpi else 'revenue',
      'execution_semantics': _EXECUTION,
      'selected_geos': geos,
      'selected_times': times,
      'scaled_media_times_including_history': [
          str(value) for value in input_data.media_time.values
      ],
      'channels': [str(channel) for channel in channels],
      'total_budget': float(allocations[_BASELINE].sum()),
      'allocations': {
          name: {str(ch): float(value) for ch, value in zip(channels, allocation)}
          for name, allocation in allocations.items()
      },
      'input_data_sha256': _data_digest(reference_data),
      'confidence_level': float(confidence_level),
      'lower_quantile_probability': float(lower_quantile_probability),
  }
  return BudgetDecisionReport(scope, policy, fits, sensitivity, ready)


def _canonical_input_data(fitted: analyzer_module.Analyzer) -> xr.Dataset:
  model_context = fitted.model_context
  if model_context.n_rf_channels:
    raise ValueError('Reach/frequency models are not supported by this audit.')
  data = model_context.input_data.as_dataset()
  if c.MEDIA_CHANNEL in data.dims:
    data = data.sortby(c.MEDIA_CHANNEL)
  return data


def _allocation(
    values: Mapping[str, float], channels: Sequence[str]
) -> np.ndarray:
  if set(values) != set(channels):
    raise ValueError('Allocations must match every paid channel label exactly.')
  result = np.asarray([values[channel] for channel in channels], dtype=float)
  if result.shape != (len(channels),) or not np.all(np.isfinite(result)):
    raise ValueError('Allocations must contain finite scalar spend values.')
  if np.any(result < 0):
    raise ValueError('Allocations must be non-negative.')
  if not np.isfinite(result.sum()):
    raise ValueError('The total allocation budget must be finite.')
  return result


def _selected_coordinates(
    selected: Sequence[str] | None, available: np.ndarray, name: str
) -> list[str]:
  available = [str(value) for value in available]
  chosen = available if selected is None else list(selected)
  if (
      not chosen or len(chosen) != len(set(chosen))
      or not set(chosen).issubset(available)
  ):
    raise ValueError(f'Selected {name} must be a nonempty unique known subset.')
  return [value for value in available if value in chosen]


def _evaluate_fit(
    *, name, fitted, allocations, channels, policy, outcome_unit, use_kpi,
    geos, times, confidence_level, lower_quantile_probability, batch_size,
) -> dict[str, Any]:
  quality = sampling_quality.assess_sampling_quality(
      fitted.inference_data, model_context=fitted.model_context
  )
  report = {
      'sampling_quality': quality.to_dict(),
      'quantities_label': (
          'conditional_posterior' if quality.sampling_passed else 'exploratory'
      ),
      'outcome_status': 'unavailable',
      'candidates': {},
  }
  fit_channels = list(fitted.model_context.input_data.get_all_paid_channels())
  history = fitted.get_aggregated_spend(
      selected_geos=geos, selected_times=times, include_media=True,
      include_rf=False,
  ).sel({c.CHANNEL: fit_channels}).values.astype(float)
  if not np.all(np.isfinite(history)) or np.any(history < 0):
    raise ValueError(f'Specification `{name}` has invalid reference spend.')
  for spend in allocations.values():
    spend = np.asarray([spend[channels.index(ch)] for ch in fit_channels])
    if np.any((history == 0) & (spend > 0)):
      raise ValueError('Positive spend needs nonzero historical reference spend.')
  if c.POSTERIOR not in fitted.inference_data.groups():
    report['outcome_issue'] = 'Posterior draws are unavailable.'
    return report
  outcomes = {}
  posterior = fitted.inference_data.posterior
  for allocation_name, spend in allocations.items():
    spend = np.asarray([spend[channels.index(ch)] for ch in fit_channels])
    ratio = np.divide(spend, history, out=np.zeros_like(spend), where=history != 0)
    new_media = backend.to_tensor(ratio, dtype=backend.float_dtype) * (
        fitted.model_context.media_tensors.media
    )
    values = np.asarray(fitted.incremental_outcome(
        use_posterior=True, new_data=tensors.DataTensors(media=new_media),
        selected_geos=geos, selected_times=times, aggregate_geos=True,
        aggregate_times=True, use_kpi=use_kpi,
        include_non_paid_channels=False, batch_size=batch_size,
    ))
    outcomes[allocation_name] = xr.DataArray(
        values, dims=(c.CHAIN, c.DRAW, c.CHANNEL),
        coords={c.CHAIN: posterior[c.CHAIN], c.DRAW: posterior[c.DRAW],
                c.CHANNEL: fit_channels},
        attrs={
            'posterior_id': f'{name}:{id(posterior)}',
            'outcome_unit': outcome_unit, 'selected_geos': tuple(geos),
            'selected_times': tuple(times), 'execution_semantics': _EXECUTION,
        },
    )
  for allocation_name, outcome in outcomes.items():
    report['candidates'][allocation_name] = summarize_paired_outcomes(
        outcomes[_BASELINE], outcome, policy=policy,
        confidence_level=confidence_level,
        lower_quantile_probability=lower_quantile_probability,
    )
  report['outcome_status'] = 'available'
  report['historical_reference_spend'] = {
      str(channel): float(value) for channel, value in zip(fit_channels, history)
  }
  return report


def _sensitivity(fits, allocations) -> dict[str, Any]:
  result = {}
  for name in allocations:
    available = [fit['candidates'][name] for fit in fits.values()
                 if name in fit['candidates']]
    complete = len(available) == len(fits)
    means = [summary['mean_gain'] for summary in available]
    result[name] = {
        'n_available_specifications': len(available),
        'n_supplied_specifications': len(fits),
        'mean_gain_envelope': (
            {'minimum': min(means), 'maximum': max(means)} if means else None
        ),
        'unanimous_positive_mean_gain': (
            all(mean > 0 for mean in means) if complete else None
        ),
        'downside_policy_met_in_every_specification_draws': (
            all(item['downside_policy_met_in_draws'] for item in available)
            if complete else None
        ),
        'quantities_label': (
            'conditional_sensitivity'
            if complete and all(fit['sampling_quality']['sampling_passed']
                                for fit in fits.values())
            else 'exploratory_sensitivity'
        ),
    }
  return result


def _validate_probabilities(confidence_level, lower_quantile_probability):
  if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
    raise ValueError('`confidence_level` must be strictly between 0 and 1.')
  if (
      not np.isfinite(lower_quantile_probability)
      or not 0 <= lower_quantile_probability <= 0.5
  ):
    raise ValueError('`lower_quantile_probability` must be between 0 and 0.5.')


def _data_digest(data: xr.Dataset) -> str:
  digest = hashlib.sha256()
  for name in sorted(data.variables):
    variable = data[name]
    digest.update(name.encode())
    digest.update(str(variable.dims).encode())
    digest.update(str(variable.dtype).encode())
    digest.update(str(variable.shape).encode())
    if variable.dtype.kind in 'OUS':
      digest.update(json.dumps(variable.values.tolist(), default=str).encode())
    else:
      digest.update(np.ascontiguousarray(variable.values).tobytes())
  return digest.hexdigest()
