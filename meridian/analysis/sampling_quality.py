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

"""Strict, machine-readable posterior sampling-quality checks.

``assess_sampling_quality`` uses ArviZ's rank-normalized, folded split R-hat
(``rhat(method='rank')``), bulk ESS, tail ESS, and post-warmup NUTS divergence
flags. The default gate requires R-hat < 1.01, bulk/tail ESS >=
``max(400, 100 * n_chains)``, and zero divergences for every stochastic
posterior cell.

Missing or non-finite values never pass. Exactly constant draws are excluded
only when Meridian model metadata proves that cell is structurally
deterministic; otherwise they are unavailable because they could be stuck
chains. Passing these sampling checks does not establish causal
identification, model adequacy, prior appropriateness, or decision validity.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from typing import Any, Callable
import warnings

import arviz as az
from meridian import constants
import numpy as np

__all__ = [
    'SamplingQualityReport',
    'assess_sampling_quality',
    'STRICT_RANK_NORMALIZED_RHAT_THRESHOLD',
    'MIN_ESS',
    'MIN_ESS_PER_CHAIN',
    'MIN_DRAWS_PER_CHAIN',
]


# The strict 1.01 screen and 400-ESS recommendation follow Vehtari et al.
# (2021). The R-hat comparison is deliberately strict: exactly 1.01 fails.
STRICT_RANK_NORMALIZED_RHAT_THRESHOLD = 1.01
MIN_ESS = 400
MIN_ESS_PER_CHAIN = 100
MIN_DRAWS_PER_CHAIN = 4

_SCHEMA_VERSION = 1
_RANK_RHAT = 'rank_normalized_rhat'
_BULK_ESS = 'bulk_ess'
_TAIL_ESS = 'tail_ess'
_SAMPLE_STATS_GROUP = 'sample_stats'


@dataclasses.dataclass(frozen=True)
class SamplingQualityReport:
  """JSON-safe sampling evidence and its strict pass/fail status.

  ``status`` is ``"pass"``, ``"fail"``, or ``"unavailable"``. The latter
  means a required diagnostic could not be assessed and is blocking just like a
  failure. ``parameters`` contains one record per scalar posterior cell, with
  coordinate names and all available diagnostic values.
  """

  status: str
  thresholds: dict[str, Any]
  sample: dict[str, Any]
  divergences: dict[str, Any]
  parameters: tuple[dict[str, Any], ...]
  blocking_issues: tuple[dict[str, Any], ...]
  warnings: tuple[str, ...]

  @property
  def sampling_passed(self) -> bool:
    """Whether all required sampling checks passed."""
    return self.status == 'pass'

  def to_dict(self) -> dict[str, Any]:
    """Returns a deep-copied standards-compliant JSON representation."""
    return {
        'schema_version': _SCHEMA_VERSION,
        'status': self.status,
        'sampling_passed': self.sampling_passed,
        'thresholds': copy.deepcopy(self.thresholds),
        'sample': copy.deepcopy(self.sample),
        'divergences': copy.deepcopy(self.divergences),
        'parameters': copy.deepcopy(list(self.parameters)),
        'blocking_issues': copy.deepcopy(list(self.blocking_issues)),
        'warnings': list(self.warnings),
        'limitations': [
            'Passing sampling diagnostics does not establish causal '
            'identification, model adequacy, prior appropriateness, or '
            'decision validity.'
        ],
    }

  def to_json(self, *, indent: int | None = None) -> str:
    """Serializes with unavailable values represented as JSON ``null``."""
    return json.dumps(
        self.to_dict(), indent=indent, sort_keys=True, allow_nan=False
    )

  def format_summary(self, *, max_issues: int = 12) -> str:
    """Returns a compact human summary while retaining full JSON detail."""
    if max_issues < 1:
      raise ValueError('`max_issues` must be at least 1.')
    lines = [
        f'Strict sampling quality: {self.status.upper()}.',
        'Required: rank-normalized R-hat'
        f" < {self.thresholds['rank_normalized_rhat_lt']:g}; bulk/tail"
        f" ESS >= {self.thresholds['bulk_ess_gte']}; and zero post-warmup"
        ' divergences.',
        f"Sample: {self.sample.get('n_chains', 'unknown')} chains x"
        f" {self.sample.get('draws_per_chain', 'unknown')} draws per chain"
        f" ({self.sample.get('n_stochastic_parameter_cells', 0)} stochastic"
        ' parameter cells assessed).',
    ]
    if self.sampling_passed:
      lines.append(
          'All required sampling checks passed. This remains separate from '
          'causal and model-adequacy review.'
      )
      return '\n'.join(lines)
    lines.append(
        'Sampling checks did not pass; strict mode must block downstream '
        'decision outputs until these issues are resolved.'
    )
    for issue in self.blocking_issues[:max_issues]:
      lines.append(f"- {issue.get('id', 'global')}: {issue['message']}")
    if len(self.blocking_issues) > max_issues:
      lines.append(
          f'- {len(self.blocking_issues) - max_issues} additional issue(s) '
          'are retained in the JSON report.'
      )
    return '\n'.join(lines)


@dataclasses.dataclass(frozen=True)
class _Metadata:
  checker: Callable[[str], bool] | None
  prior_error: str | None
  baseline_geo_idx: int | None
  baseline_error: str | None


def assess_sampling_quality(
    inference_data: Any,
    *,
    model_context: Any | None = None,
    rhat_threshold: float = STRICT_RANK_NORMALIZED_RHAT_THRESHOLD,
    min_ess: int | None = None,
) -> SamplingQualityReport:
  """Assesses strict rank R-hat, ESS, and divergence requirements.

  Args:
    inference_data: ArviZ-compatible inference data with a posterior group.
    model_context: Optional Meridian model context used only to identify exact
      structural constants. It is never used to hide an unverified constant.
    rhat_threshold: Strict upper bound for ``az.rhat(method='rank')``.
    min_ess: Common lower bound for bulk/tail ESS. Defaults to
      ``max(400, 100 * n_chains)``.

  Returns:
    A report whose JSON includes per-cell values, coordinate names, failures,
    and unavailability evidence. Malformed/incomplete input returns an honest
    ``"unavailable"`` report instead of a pass.

  Raises:
    ValueError: If a supplied threshold is invalid.
  """
  rhat_threshold = _valid_rhat_threshold(rhat_threshold)
  min_ess = _valid_min_ess(min_ess)
  ess_rule = (
      f'explicit min_ess={min_ess}'
      if min_ess is not None
      else f'max({MIN_ESS}, {MIN_ESS_PER_CHAIN} * n_chains)'
  )
  posterior, error = _posterior(inference_data)
  if posterior is None:
    return _report(
        status='unavailable',
        rhat_threshold=rhat_threshold,
        min_ess=min_ess if min_ess is not None else MIN_ESS,
        n_chains=None,
        n_draws=None,
        divergences=_unavailable_divergences('Posterior is not available.'),
        parameters=(),
        issues=(
            _issue(
                'posterior_not_available',
                'No readable posterior group was supplied; sample the '
                'posterior before assessing sampling quality.',
                check='posterior',
                severity='unavailable',
                evidence=error,
            ),
        ),
        diagnostic_warnings=(),
        ess_rule=ess_rule,
    )

  n_chains, n_draws, shape_issues = _sample_shape(posterior)
  required_ess = (
      min_ess
      if min_ess is not None
      else max(MIN_ESS, MIN_ESS_PER_CHAIN * (n_chains or 0))
  )
  metrics, metric_errors, metric_warnings = _metrics(
      posterior, n_chains=n_chains, n_draws=n_draws
  )
  metadata = _metadata(model_context)
  parameter_cells = []
  issues = list(shape_issues)
  for name in sorted(posterior.data_vars):
    cells, cell_issues = _parameter_cells(
        name=name,
        data_array=posterior[name],
        metrics=metrics,
        metric_errors=metric_errors,
        metadata=metadata,
        rhat_threshold=rhat_threshold,
        min_ess=required_ess,
    )
    parameter_cells.extend(cells)
    issues.extend(cell_issues)

  divergences, divergence_issue = _divergences(
      inference_data, posterior, n_chains=n_chains, n_draws=n_draws
  )
  if divergence_issue is not None:
    issues.append(divergence_issue)
  n_stochastic = sum(cell['status'] != 'excluded' for cell in parameter_cells)
  if not n_stochastic:
    issues.append(
        _issue(
            'no_stochastic_posterior_parameters',
            'No stochastic posterior parameter cells were available for rank '
            'R-hat and ESS assessment.',
            check='posterior',
            severity='unavailable',
        )
    )
  return _report(
      status=_status(issues),
      rhat_threshold=rhat_threshold,
      min_ess=required_ess,
      n_chains=n_chains,
      n_draws=n_draws,
      divergences=divergences,
      parameters=tuple(parameter_cells),
      issues=tuple(issues),
      diagnostic_warnings=tuple(metric_warnings),
      ess_rule=ess_rule,
  )


def _valid_rhat_threshold(value: float) -> float:
  try:
    result = float(value)
  except (TypeError, ValueError) as error:
    raise ValueError(
        '`rhat_threshold` must be finite and greater than 1.'
    ) from error
  if not np.isfinite(result) or result <= 1:
    raise ValueError('`rhat_threshold` must be finite and greater than 1.')
  return result


def _valid_min_ess(value: int | None) -> int | None:
  if value is None:
    return None
  if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
    raise ValueError('`min_ess` must be a positive integer or None.')
  if value < 1:
    raise ValueError('`min_ess` must be a positive integer or None.')
  return int(value)


def _posterior(inference_data: Any) -> tuple[Any | None, str | None]:
  try:
    if constants.POSTERIOR not in inference_data.groups():
      return None, 'The inference data has no posterior group.'
    posterior = inference_data.posterior
    if len(posterior.data_vars) == 0:
      return None, 'The posterior group has no data variables.'
  except (AttributeError, TypeError, ValueError) as error:
    return None, f'{type(error).__name__}: {error}'
  return posterior, None


def _sample_shape(
    posterior: Any,
) -> tuple[int | None, int | None, list[dict[str, Any]]]:
  try:
    n_chains = int(posterior.sizes[constants.CHAIN])
    n_draws = int(posterior.sizes[constants.DRAW])
  except (KeyError, TypeError, ValueError) as error:
    return (
        None,
        None,
        [
            _issue(
                'posterior_chain_or_draw_dimension_missing',
                'The posterior must expose named chain and draw dimensions for '
                'rank-normalized diagnostics.',
                check='posterior',
                severity='unavailable',
                evidence=f'{type(error).__name__}: {error}',
            )
        ],
    )
  issues = []
  if n_chains < 2:
    issues.append(
        _issue(
            'insufficient_chains',
            'Rank-normalized R-hat requires at least 2 chains; found '
            f'{n_chains}.',
            check=_RANK_RHAT,
            severity='unavailable',
        )
    )
  if n_draws < MIN_DRAWS_PER_CHAIN:
    issues.append(
        _issue(
            'insufficient_draws',
            'Rank-split diagnostics require at least '
            f'{MIN_DRAWS_PER_CHAIN} post-warmup draws per chain; found '
            f'{n_draws}.',
            check=_RANK_RHAT,
            severity='unavailable',
        )
    )
  return n_chains, n_draws, issues


def _metrics(
    posterior: Any, *, n_chains: int | None, n_draws: int | None
) -> tuple[dict[str, Any | None], dict[str, str], list[str]]:
  """Computes each ArviZ metric once and retains warnings as evidence."""
  if n_chains is None or n_draws is None or n_chains < 2 or n_draws < 4:
    error = (
        'Need at least 2 chains and 4 draws per chain for rank-split '
        'diagnostics.'
    )
    return (
        {},
        {metric: error for metric in (_RANK_RHAT, _BULK_ESS, _TAIL_ESS)},
        [],
    )
  results, errors, warnings_seen = {}, {}, []
  for name, function, method in (
      (_RANK_RHAT, az.rhat, 'rank'),
      (_BULK_ESS, az.ess, 'bulk'),
      (_TAIL_ESS, az.ess, 'tail'),
  ):
    try:
      with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        results[name] = function(posterior, method=method)
    except (ArithmeticError, TypeError, ValueError) as error:
      errors[name] = f'{type(error).__name__}: {error}'
    else:
      warnings_seen.extend(
          f'{name}: {warning.category.__name__}: {warning.message}'
          for warning in caught
      )
  return results, errors, list(dict.fromkeys(warnings_seen))


def _metadata(model_context: Any | None) -> _Metadata:
  if model_context is None:
    return _Metadata(
        None,
        'Model context was not provided.',
        None,
        'Model context was not provided.',
    )
  try:
    checker = model_context.prior_broadcast.has_deterministic_param
  except (AttributeError, TypeError) as error:
    checker, prior_error = None, f'{type(error).__name__}: {error}'
  else:
    prior_error = None
  try:
    baseline_geo_idx = int(model_context.baseline_geo_idx)
  except (AttributeError, TypeError, ValueError) as error:
    baseline_geo_idx, baseline_error = None, f'{type(error).__name__}: {error}'
  else:
    baseline_error = None
  return _Metadata(checker, prior_error, baseline_geo_idx, baseline_error)


def _parameter_cells(
    *,
    name: str,
    data_array: Any,
    metrics: dict[str, Any | None],
    metric_errors: dict[str, str],
    metadata: _Metadata,
    rhat_threshold: float,
    min_ess: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  if (
      constants.CHAIN not in data_array.dims
      or constants.DRAW not in data_array.dims
  ):
    cell = _cell(name, {})
    issue = _cell_issue(
        cell,
        'parameter_chain_or_draw_dimension_missing',
        'This posterior variable lacks a named chain or draw dimension.',
        check='posterior',
        severity='unavailable',
    )
    cell['reasons'] = [issue]
    return [cell], [issue]
  dims = tuple(
      dim
      for dim in data_array.dims
      if dim not in (constants.CHAIN, constants.DRAW)
  )
  try:
    draws = np.asarray(
        data_array.transpose(constants.CHAIN, constants.DRAW, *dims).values,
        dtype=float,
    )
  except (TypeError, ValueError) as error:
    cell = _cell(name, {})
    issue = _cell_issue(
        cell,
        'non_numeric_posterior_draws',
        'Posterior draws could not be interpreted as numeric values.',
        check='posterior',
        severity='unavailable',
        evidence=f'{type(error).__name__}: {error}',
    )
    cell['reasons'] = [issue]
    return [cell], [issue]

  metric_arrays = {
      metric: _metric_array(
          metrics.get(metric),
          metric_error=metric_errors.get(metric),
          name=name,
          data_array=data_array,
          dims=dims,
          shape=draws.shape[2:],
      )
      for metric in (_RANK_RHAT, _BULK_ESS, _TAIL_ESS)
  }
  cells, issues = [], []
  for index in np.ndindex(draws.shape[2:]):
    cell = _cell(name, _coordinates(data_array, dims, index))
    values = draws[(slice(None), slice(None), *index)]
    deterministic, source, metadata_error = _deterministic(
        metadata, name, dims, index
    )
    if not values.size:
      cell['reasons'] = [
          _cell_issue(
              cell,
              'empty_posterior_draws',
              'This posterior cell contains no sampled transitions.',
              check='posterior',
              severity='unavailable',
          )
      ]
    elif not np.all(np.isfinite(values)):
      issue = _cell_issue(
          cell,
          'nonfinite_posterior_draws',
          'Posterior draws contain NaN or infinity.',
          check='posterior',
          severity='unavailable',
          evidence={
              'n_nonfinite_draws': int(values.size - np.isfinite(values).sum()),
              'n_draws_total': int(values.size),
          },
      )
      cell['reasons'] = [issue]
    elif deterministic is True and np.all(values == values.flat[0]):
      cell.update(status='excluded', deterministic_metadata={'source': source})
    elif deterministic is True:
      issue = _cell_issue(
          cell,
          'deterministic_metadata_conflicts_with_posterior',
          'Model metadata marks this cell deterministic, but its draws vary.',
          check='deterministic_metadata',
          severity='unavailable',
          evidence={'source': source},
      )
      cell['reasons'] = [issue]
    elif np.all(values == values.flat[0]):
      code = (
          'constant_draws_not_confirmed_deterministic'
          if deterministic is False
          else 'constant_draws_metadata_unavailable'
      )
      issue = _cell_issue(
          cell,
          code,
          'Posterior draws are exactly constant without sufficient structural '
          'determinism evidence; this may be a stuck chain.',
          check='deterministic_metadata',
          severity='unavailable',
          evidence={'source': source, 'metadata_error': metadata_error},
      )
      cell['reasons'] = [issue]
    else:
      _set_metric_values(
          cell,
          metric_arrays,
          index,
          rhat_threshold=rhat_threshold,
          min_ess=min_ess,
      )
    cells.append(cell)
    issues.extend(cell['reasons'])
  return cells, issues


def _metric_array(
    dataset: Any | None,
    *,
    metric_error: str | None,
    name: str,
    data_array: Any,
    dims: tuple[str, ...],
    shape: tuple[int, ...],
) -> tuple[np.ndarray | None, str | None]:
  if dataset is None:
    return None, metric_error or 'Metric was not computed.'
  try:
    if name not in dataset.data_vars:
      return None, 'Metric output does not contain this posterior variable.'
    metric_data_array = dataset[name]
    values = np.asarray(metric_data_array.transpose(*dims).values, dtype=float)
  except (AttributeError, KeyError, TypeError, ValueError) as exc:
    return None, f'{type(exc).__name__}: {exc}'
  if values.shape != shape:
    return None, f'Metric shape {values.shape!r} does not match {shape!r}.'
  for dim in dims:
    source_has_coord = dim in data_array.coords
    metric_has_coord = dim in metric_data_array.coords
    if source_has_coord != metric_has_coord:
      return None, f'Metric coordinate for {dim!r} does not match posterior.'
    if source_has_coord and not np.array_equal(
        data_array.coords[dim].values, metric_data_array.coords[dim].values
    ):
      return (
          None,
          f'Metric coordinate values for {dim!r} do not match posterior.',
      )
  return values, None


def _deterministic(
    metadata: _Metadata,
    name: str,
    dims: tuple[str, ...],
    index: tuple[int, ...],
) -> tuple[bool | None, str, str | None]:
  """Identifies only documented, explicit structural constants.

  ``posterior_sampler._compute_tau_g`` inserts zero at ``baseline_geo_idx``;
  that cell is excluded. Other ``tau_g`` cells remain diagnostic targets unless
  the corresponding ``tau_g_excl_baseline`` prior is explicitly deterministic.
  """
  if name == constants.TAU_G:
    baseline_source = (
        'model_context.baseline_geo_idx + posterior_sampler._compute_tau_g'
    )
    if constants.GEO not in dims:
      return None, baseline_source, 'tau_g has no geo dimension.'
    if metadata.baseline_geo_idx is None:
      return None, baseline_source, metadata.baseline_error
    if index[dims.index(constants.GEO)] == metadata.baseline_geo_idx:
      return True, baseline_source, None
    source = (
        'model_context.prior_broadcast.has_deterministic_param('
        'tau_g_excl_baseline)'
    )
    if metadata.checker is None:
      return None, source, metadata.prior_error
    try:
      return (
          bool(metadata.checker(constants.TAU_G_EXCL_BASELINE)),
          source,
          None,
      )
    except (AttributeError, TypeError, ValueError) as error:
      return None, source, f'{type(error).__name__}: {error}'
  source = 'model_context.prior_broadcast.has_deterministic_param'
  if metadata.checker is None:
    return None, source, metadata.prior_error
  try:
    return bool(metadata.checker(name)), source, None
  except (AttributeError, TypeError, ValueError) as error:
    return None, source, f'{type(error).__name__}: {error}'


def _set_metric_values(
    cell: dict[str, Any],
    metric_arrays: dict[str, tuple[np.ndarray | None, str | None]],
    index: tuple[int, ...],
    *,
    rhat_threshold: float,
    min_ess: int,
) -> None:
  checks = (
      (
          _RANK_RHAT,
          '<',
          rhat_threshold,
          'rank_normalized_rhat_not_below_threshold',
      ),
      (_BULK_ESS, '>=', min_ess, 'bulk_ess_below_threshold'),
      (_TAIL_ESS, '>=', min_ess, 'tail_ess_below_threshold'),
  )
  unavailable = failed = False
  for metric, operator, threshold, failure_code in checks:
    values, error = metric_arrays[metric]
    if values is None:
      cell['reasons'].append(
          _cell_issue(
              cell,
              f'{metric}_unavailable',
              f'{metric.replace("_", " ")} could not be computed: {error}',
              check=metric,
              severity='unavailable',
          )
      )
      unavailable = True
      continue
    value = float(values[index])
    cell[metric] = _json_value(value)
    if not np.isfinite(value):
      cell['reasons'].append(
          _cell_issue(
              cell,
              f'{metric}_nonfinite',
              f'{metric.replace("_", " ")} is non-finite.',
              check=metric,
              severity='unavailable',
          )
      )
      unavailable = True
    elif value < threshold if operator == '<' else value >= threshold:
      continue
    else:
      comparison = 'below' if operator == '<' else 'at least'
      cell['reasons'].append(
          _cell_issue(
              cell,
              failure_code,
              f'{metric.replace("_", " ")}={value:.6g} must be {comparison} '
              f'{threshold:.6g}.',
              check=metric,
              severity='fail',
              evidence={'value': value, 'threshold': threshold},
          )
      )
      failed = True
  cell['status'] = (
      'unavailable' if unavailable else 'fail' if failed else 'pass'
  )


def _divergences(
    inference_data: Any,
    posterior: Any,
    *,
    n_chains: int | None,
    n_draws: int | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
  if not n_chains or not n_draws:
    return _missing_divergences(
        'No posterior transitions are available to assess divergence flags.'
    )
  try:
    if _SAMPLE_STATS_GROUP not in inference_data.groups():
      return _missing_divergences('No sample_stats group is present.')
    data_array = inference_data.sample_stats[constants.DIVERGING]
  except (AttributeError, KeyError, TypeError, ValueError) as error:
    return _missing_divergences(
        "No readable sample_stats['diverging'] values are present.",
        evidence=f'{type(error).__name__}: {error}',
    )
  if (
      n_chains is None
      or n_draws is None
      or set(data_array.dims) != {constants.CHAIN, constants.DRAW}
      or data_array.sizes[constants.CHAIN] != n_chains
      or data_array.sizes[constants.DRAW] != n_draws
  ):
    return _missing_divergences(
        'Divergence flags are not aligned to posterior chain/draw dimensions.'
    )
  try:
    aligned = np.array_equal(
        data_array.coords[constants.CHAIN].values,
        posterior.coords[constants.CHAIN].values,
    ) and np.array_equal(
        data_array.coords[constants.DRAW].values,
        posterior.coords[constants.DRAW].values,
    )
    values = np.asarray(
        data_array.transpose(constants.CHAIN, constants.DRAW).values,
        dtype=float,
    )
  except (AttributeError, KeyError, TypeError, ValueError) as error:
    return _missing_divergences(
        'Divergence flags could not be read as aligned numeric values.',
        evidence=f'{type(error).__name__}: {error}',
    )
  if (
      not aligned
      or not np.all(np.isfinite(values))
      or not np.all((values == 0) | (values == 1))
  ):
    return _missing_divergences(
        'Divergence flags must be aligned finite binary values (0 or 1).'
    )
  per_chain = values.sum(axis=1).astype(int)
  count = int(per_chain.sum())
  report = {
      'status': 'pass' if count == 0 else 'fail',
      'reported': True,
      'n_divergences': count,
      'rate': float(count / (n_chains * n_draws)),
      'per_chain': [
          {'chain': _json_value(chain), 'n_divergences': int(total)}
          for chain, total in zip(
              posterior.coords[constants.CHAIN].values, per_chain
          )
      ],
  }
  if count == 0:
    return report, None
  return report, _issue(
      'divergent_transitions_detected',
      f'{count} post-warmup divergent transition(s) were reported. Resolve '
      'sampler geometry or reparameterize before using these draws for '
      'decisions.',
      check='divergences',
      severity='fail',
      evidence={'n_divergences': count, 'rate': report['rate']},
  )


def _missing_divergences(
    message: str, *, evidence: Any | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
  return _unavailable_divergences(message), _issue(
      'divergence_not_reported',
      message,
      check='divergences',
      severity='unavailable',
      evidence=evidence,
  )


def _unavailable_divergences(message: str) -> dict[str, Any]:
  return {
      'status': 'unavailable',
      'reported': False,
      'n_divergences': None,
      'rate': None,
      'per_chain': [],
      'message': message,
  }


def _cell(name: str, coordinates: dict[str, Any]) -> dict[str, Any]:
  return {
      'id': _cell_id(name, coordinates),
      'parameter': name,
      'coordinates': coordinates,
      'status': 'unavailable',
      _RANK_RHAT: None,
      _BULK_ESS: None,
      _TAIL_ESS: None,
      'reasons': [],
  }


def _coordinates(
    data_array: Any, dims: tuple[str, ...], index: tuple[int, ...]
) -> dict[str, Any]:
  return {
      dim: (
          _json_value(data_array.coords[dim].values[position])
          if dim in data_array.coords
          else int(position)
      )
      for dim, position in zip(dims, index)
  }


def _cell_id(name: str, coordinates: dict[str, Any]) -> str:
  if not coordinates:
    return name
  rendered = ', '.join(
      f'{key}={json.dumps(value)}' for key, value in coordinates.items()
  )
  return f'{name}[{rendered}]'


def _cell_issue(
    cell: dict[str, Any], code: str, message: str, **kwargs: Any
) -> dict[str, Any]:
  return _issue(
      code,
      message,
      parameter=cell['parameter'],
      coordinates=cell['coordinates'],
      cell_id=cell['id'],
      **kwargs,
  )


def _issue(
    code: str,
    message: str,
    *,
    check: str,
    severity: str,
    parameter: str | None = None,
    coordinates: dict[str, Any] | None = None,
    cell_id: str | None = None,
    evidence: Any | None = None,
) -> dict[str, Any]:
  issue = {
      'code': code,
      'message': message,
      'check': check,
      'severity': severity,
  }
  if parameter is not None:
    issue['parameter'] = parameter
  if coordinates is not None:
    issue['coordinates'] = coordinates
  if cell_id is not None:
    issue['id'] = cell_id
  if evidence is not None:
    issue['evidence'] = _json_value(evidence)
  return issue


def _status(issues: list[dict[str, Any]]) -> str:
  if any(issue['severity'] == 'unavailable' for issue in issues):
    return 'unavailable'
  if any(issue['severity'] == 'fail' for issue in issues):
    return 'fail'
  return 'pass'


def _report(
    *,
    status: str,
    rhat_threshold: float,
    min_ess: int,
    n_chains: int | None,
    n_draws: int | None,
    divergences: dict[str, Any],
    parameters: tuple[dict[str, Any], ...],
    issues: tuple[dict[str, Any], ...],
    diagnostic_warnings: tuple[str, ...],
    ess_rule: str,
) -> SamplingQualityReport:
  n_stochastic = sum(cell['status'] != 'excluded' for cell in parameters)
  return SamplingQualityReport(
      status=status,
      thresholds={
          'rank_normalized_rhat_method': "arviz.rhat(method='rank')",
          'rank_normalized_rhat_lt': rhat_threshold,
          'bulk_ess_gte': min_ess,
          'tail_ess_gte': min_ess,
          'post_warmup_divergences_eq': 0,
          'minimum_ess_rule': ess_rule,
      },
      sample={
          'n_chains': n_chains,
          'draws_per_chain': n_draws,
          'total_post_warmup_draws': (
              n_chains * n_draws
              if n_chains is not None and n_draws is not None
              else None
          ),
          'n_parameter_cells': len(parameters),
          'n_stochastic_parameter_cells': n_stochastic,
          'n_excluded_deterministic_cells': len(parameters) - n_stochastic,
      },
      divergences=divergences,
      parameters=parameters,
      blocking_issues=issues,
      warnings=diagnostic_warnings,
  )


def _json_value(value: Any) -> Any:
  if isinstance(value, dict):
    return {str(key): _json_value(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_value(item) for item in value]
  if isinstance(value, np.generic):
    return _json_value(value.item())
  if isinstance(value, float):
    return value if np.isfinite(value) else None
  if isinstance(value, (str, int, bool)) or value is None:
    return value
  return str(value)
