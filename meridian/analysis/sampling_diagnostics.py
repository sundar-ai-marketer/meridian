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

"""Whether a fitted model's NUTS posterior sample can be trusted.

Meridian's `posterior_sampler.py` already asks NUTS to report divergent
transitions -- they land in `inference_data.sample_stats['diverging']`, the
single most direct signal that the sampler struggled -- but nothing in the
library surfaces them. Effective sample size (ESS) is not computed anywhere
either, even though `arviz` is already a dependency. Low ESS with an
acceptable r-hat is common for weakly identified adstock/Hill parameters, and
it directly governs whether a quantile-based credible interval (Meridian
reports 90% intervals throughout) is trustworthy: an interval built from too
few effective draws is noisier than its reported width implies.

This module answers three questions about a fitted model:

  1. Did NUTS hit any divergent transitions? Even one means the sampler could
     not explore part of the posterior; see `DIVERGENCE_RATE_SERIOUS_THRESHOLD`
     for when a *rate* of divergences, not just their presence, is serious.
  2. Is Meridian's upstream TFP potential-scale-reduction R-hat below a
     defensible threshold? `ConvergenceCheck`
     (meridian/analysis/review/checks.py) already answers this with a 1.2
     cutoff inherited from upstream; this module reports that value unchanged
     (`UPSTREAM_RHAT_THRESHOLD`) alongside a stricter 1.01 heuristic
     (`MODERN_RHAT_THRESHOLD`, whose historical API name is retained).
     This is not ArviZ's rank-normalized split R-hat; use
     `arviz.rhat(..., method='rank')` if that separate diagnostic is wanted.
  3. Is ESS high enough to trust the posterior's quantiles? See
     `MIN_BULK_ESS` / `MIN_TAIL_ESS`.

All three are heuristics from the MCMC-diagnostics literature, not universal
correctness guarantees -- see each constant's docstring for its source and
reasoning.

```python
diag = sampling_diagnostics.SamplingDiagnostics(mmm)
print(diag.verdict)
diag.summary()
diag.divergence_summary()
```
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import arviz as az
from meridian import constants as c
from meridian.analysis import analyzer as analyzer_module
from meridian.analysis.review import configs as review_configs
from meridian.common import errors
from meridian.model import model
import numpy as np
import pandas as pd

__all__ = [
    'SamplingDiagnostics',
    'MODERN_RHAT_THRESHOLD',
    'UPSTREAM_RHAT_THRESHOLD',
    'MIN_BULK_ESS',
    'MIN_TAIL_ESS',
    'DIVERGENCE_RATE_SERIOUS_THRESHOLD',
]

# Strict heuristic, not a hard correctness bound. Its historical public name
# predates the clarification below. `Analyzer.get_rhat()` calls TensorFlow
# Probability's conventional `potential_scale_reduction`, not the
# rank-normalized split R-hat introduced by Vehtari et al. (2021), so 1.01 here
# is a deliberately strict screen on Meridian's upstream statistic, not a claim
# that this module implements that paper's estimator.
MODERN_RHAT_THRESHOLD = 1.01

# Heuristic. Matches Meridian's own `ConvergenceCheck`
# (meridian/analysis/review/checks.py) exactly -- pulled from its default
# config rather than duplicated as a literal, so the two thresholds cannot
# silently drift apart. Below this, `ConvergenceCheck` calls the model
# "converged"; this module reports it unchanged, alongside the stricter 1.01
# heuristic, rather than overriding upstream's judgment call.
UPSTREAM_RHAT_THRESHOLD = (
    review_configs.ConvergenceConfig().convergence_threshold
)

# Heuristic, not a hard correctness bound. Vehtari et al. (2021) recommend a
# bulk-ESS (and tail-ESS; see `MIN_TAIL_ESS`) of at least 400 -- roughly 100
# per chain for the 4-chain default -- before treating quantile-based
# posterior summaries (Meridian reports 90% credible intervals) as stable.
# Below this, the reported interval can move noticeably if the sampler were
# re-run with a different seed.
MIN_BULK_ESS = 400

# Heuristic; same source and reasoning as `MIN_BULK_ESS`, applied to tail-ESS
# (the ESS estimate for the 5th/95th-percentile region that bounds a 90%
# interval specifically, rather than the bulk of the distribution).
MIN_TAIL_ESS = 400

# Heuristic. There is no single universally-cited number for "how many
# divergences are too many"; this follows the widely-used Stan/PyMC
# community guidance (e.g. the Stan User's Guide's divergent-transitions
# chapter, and Betancourt (2017), "A Conceptual Introduction to Hamiltonian
# Monte Carlo") that *any* divergence means NUTS failed to explore part of
# the posterior and is worth investigating, while a divergence rate above
# roughly 0.5% of post-warmup draws indicates a systematic geometry problem
# (e.g. an unidentified adstock/Hill parameter) rather than a handful of
# unlucky proposals.
DIVERGENCE_RATE_SERIOUS_THRESHOLD = 0.005

# `az.InferenceData` group name for sampler statistics. Written as the
# literal string `"sample_stats"` by `posterior_sampler.py` (there is no
# `meridian.constants` entry for the group name itself, only for the metric
# keys within it, e.g. `constants.DIVERGING`).
_SAMPLE_STATS_GROUP = 'sample_stats'

_PARAMETER = 'parameter'
_RHAT = 'rhat'
_BULK_ESS = 'bulk_ess'
_TAIL_ESS = 'tail_ess'
_BULK_ESS_PER_DRAW = 'bulk_ess_per_draw'
_TAIL_ESS_PER_DRAW = 'tail_ess_per_draw'
_RHAT_OK_MODERN = 'rhat_ok_modern'
_RHAT_OK_UPSTREAM = 'rhat_ok_upstream'
_ESS_OK = 'ess_ok'


def max_rank_normalized_rhat(posterior) -> float:
  """Worst ArviZ rank-normalized split R-hat across a posterior's variables.

  This is `az.rhat(posterior, method='rank')` reduced to a single number, and
  it is a DIFFERENT statistic from `SamplingDiagnostics.rhat_max`, which
  reports the upstream TFP potential-scale-reduction R-hat via
  `Analyzer.get_rhat()`. The two do not agree in value, so they are not
  interchangeable and neither can be swapped in for the other to remove a
  duplicate.

  Deterministic parameters -- hierarchical terms pinned to zero in a national
  model, Hill parameters for a linear channel -- have no between-chain
  variance, so ArviZ returns an all-NaN R-hat for them. Those variables are
  dropped rather than reducing over a NaN slice. An infinite R-hat is kept: it
  has to stay visible and fail convergence.

  Args:
    posterior: An ArviZ posterior group, e.g. `inference_data.posterior`.

  Returns:
    The maximum finite-or-infinite R-hat, or `NaN` when no variable has one.
  """
  rhat = az.rhat(posterior, method='rank')  # pytype: disable=attribute-error
  per_variable = []
  for name in rhat.data_vars:  # pytype: disable=attribute-error
    values = np.asarray(rhat[name].values, dtype=float)
    non_nan = values[~np.isnan(values)]
    if non_nan.size:
      per_variable.append(float(non_nan.max()))
  return max(per_variable) if per_variable else float('nan')


def _nan_reduce(values: np.ndarray, reducer) -> float:
  """Reduces `values` over non-NaN cells, `NaN` if every cell is NaN.

  Deterministic parameters -- Hill parameters phantom-masked for linear
  channels in `get_rhat`, or hierarchical terms pinned to zero in a national
  model -- can produce `NaN`. So can diagnostics that are unavailable because
  there are too few usable draws or chains. Drop only `NaN`: an infinite R-hat
  must remain visible and fail convergence, while ESS callers separately reject
  a non-finite result as invalid.

  Args:
    values: A 1-D array of per-cell diagnostic values for one parameter.
    reducer: `np.max` (for r-hat, where larger is worse) or `np.min` (for
      ESS, where smaller is worse).

  Returns:
    The reduced value, or `NaN` if `values` has no non-NaN entries.
  """
  non_nan = values[~np.isnan(values)]
  if non_nan.size == 0:
    return float('nan')
  return float(reducer(non_nan))


@dataclasses.dataclass(frozen=True)
class _Divergences:
  """Per-chain divergence counts, or a "not reported" marker."""

  reported: bool
  per_chain_counts: np.ndarray  # float, so "not reported" can be all-NaN.


class SamplingDiagnostics:
  """MCMC trust signals absent from Meridian: divergences and ESS.

  Reuses `Analyzer.get_rhat()` for R-hat rather than recomputing it. That path
  currently calls TFP's conventional `potential_scale_reduction`, so this class
  and `meridian.analysis.review.checks.ConvergenceCheck` never disagree on the
  upstream statistic itself -- only, deliberately, on which threshold to flag
  it against. It does not calculate ArviZ's rank-normalized split R-hat.
  """

  def __init__(self, meridian: model.Meridian):
    """Initializes the diagnostic.

    Args:
      meridian: A fitted `Meridian` model.

    Raises:
      NotFittedModelError: If the model has no posterior draws.
    """
    if c.POSTERIOR not in meridian.inference_data.groups():
      raise errors.NotFittedModelError(
          'The model has no posterior draws. Call `sample_posterior()`'
          ' first.'
      )
    self._meridian = meridian
    self._analyzer = analyzer_module.Analyzer(
        model_context=meridian.model_context,
        inference_data=meridian.inference_data,
    )
    self._posterior = meridian.inference_data.posterior
    self._n_chains = int(self._posterior.sizes[c.CHAIN])
    self._n_draws = int(self._posterior.sizes[c.DRAW])
    self._divergences = self._compute_divergences()

  def _compute_divergences(self) -> _Divergences:
    """Reads per-chain divergence counts, degrading gracefully if absent.

    Divergences live in `inference_data.sample_stats['diverging']`, written
    by `posterior_sampler.py` for TFP's `windowed_adaptive_nuts()`. A model
    fit with a different sampler, or an older inference-data payload, may
    have no `sample_stats` group at all, or one without a `diverging`
    variable. Both are legitimate "not reported" states, not errors.
    """
    groups = self._meridian.inference_data.groups()
    if _SAMPLE_STATS_GROUP not in groups:
      return _Divergences(
          reported=False,
          per_chain_counts=np.full(self._n_chains, np.nan),
      )
    sample_stats = self._meridian.inference_data.sample_stats
    if c.DIVERGING not in sample_stats.data_vars:
      return _Divergences(
          reported=False,
          per_chain_counts=np.full(self._n_chains, np.nan),
      )
    diverging = np.asarray(sample_stats[c.DIVERGING].values, dtype=bool)
    return _Divergences(
        reported=True,
        per_chain_counts=diverging.sum(axis=1).astype(float),
    )

  @property
  def n_chains(self) -> int:
    """Number of MCMC chains in the posterior sample."""
    return self._n_chains

  @property
  def n_draws(self) -> int:
    """Post-warmup draws kept per chain."""
    return self._n_draws

  @property
  def total_draws(self) -> int:
    """Total post-warmup draws across all chains (`n_chains * n_draws`)."""
    return self._n_chains * self._n_draws

  # ---------------------------------------------------------------------
  # Divergent transitions.
  # ---------------------------------------------------------------------

  @property
  def divergences_reported(self) -> bool:
    """Whether this posterior sample reports divergent transitions at all."""
    return self._divergences.reported

  @property
  def n_divergences(self) -> Optional[int]:
    """Total divergences across all chains, or `None` if unreported."""
    if not self._divergences.reported:
      return None
    return int(np.sum(self._divergences.per_chain_counts))

  @property
  def divergence_rate(self) -> Optional[float]:
    """Divergences as a fraction of total post-warmup draws, or `None`."""
    if not self._divergences.reported:
      return None
    return self.n_divergences / self.total_draws

  @property
  def divergence_rate_is_serious(self) -> Optional[bool]:
    """Whether the divergence rate exceeds the serious-rate threshold."""
    rate = self.divergence_rate
    if rate is None:
      return None
    return rate > DIVERGENCE_RATE_SERIOUS_THRESHOLD

  def divergence_summary(self) -> pd.DataFrame:
    """Returns a per-chain divergence table.

    Returns:
      A DataFrame with one row per chain: divergence count, draws kept, and
      divergence rate. When `divergences_reported` is `False`, the count and
      rate columns are `NaN` for every chain -- check `reported` before
      relying on them.
    """
    if self._divergences.reported:
      counts = self._divergences.per_chain_counts
    else:
      counts = np.full(self._n_chains, np.nan)
    rates = counts / self._n_draws
    return pd.DataFrame(
        {
            c.CHAIN: np.arange(self._n_chains),
            'n_divergences': counts,
            'n_draws': self._n_draws,
            'divergence_rate': rates,
            'reported': self._divergences.reported,
            'serious': rates > DIVERGENCE_RATE_SERIOUS_THRESHOLD,
        }
    )

  # ---------------------------------------------------------------------
  # r-hat and effective sample size.
  # ---------------------------------------------------------------------

  def summary(self) -> pd.DataFrame:
    """Returns a per-parameter r-hat/ESS table, worst ESS first.

    Each row reduces a (possibly multi-dimensional, e.g. per-channel)
    parameter down to its single worst cell: the largest r-hat and the
    smallest bulk/tail ESS. That mirrors `ConvergenceCheck`, which reports
    the single worst r-hat per parameter rather than every cell.

    Returns:
      A DataFrame with columns: `parameter`, `rhat`, `bulk_ess`, `tail_ess`,
      `bulk_ess_per_draw`, `tail_ess_per_draw`, `rhat_ok_modern` (finite and
      below the strict 1.01 heuristic), `rhat_ok_upstream` (finite and below
      `UPSTREAM_RHAT_THRESHOLD`), and `ess_ok` (bulk and tail ESS both finite
      and at or above their thresholds). `rhat` is Meridian's conventional TFP
      potential-scale-reduction statistic, rather than rank-normalized R-hat.
      A parameter with an unavailable diagnostic gets `NaN` in that column and
      reads `False` for the corresponding `_ok` flag; infinite diagnostic
      values are also invalid.
    """
    rhat_by_param = self._rhat_per_parameter()
    bulk_by_param = self._ess_per_parameter(method='bulk')
    tail_by_param = self._ess_per_parameter(method='tail')

    parameters = sorted(
        set(rhat_by_param) | set(bulk_by_param) | set(tail_by_param)
    )
    rhat = np.array([rhat_by_param.get(p, float('nan')) for p in parameters])
    bulk_ess = np.array(
        [bulk_by_param.get(p, float('nan')) for p in parameters]
    )
    tail_ess = np.array(
        [tail_by_param.get(p, float('nan')) for p in parameters]
    )

    frame = pd.DataFrame(
        {
            _PARAMETER: parameters,
            _RHAT: rhat,
            _BULK_ESS: bulk_ess,
            _TAIL_ESS: tail_ess,
            _BULK_ESS_PER_DRAW: bulk_ess / self.total_draws,
            _TAIL_ESS_PER_DRAW: tail_ess / self.total_draws,
        }
    )
    frame[_RHAT_OK_MODERN] = np.isfinite(frame[_RHAT]) & (
        frame[_RHAT] < MODERN_RHAT_THRESHOLD
    )
    frame[_RHAT_OK_UPSTREAM] = np.isfinite(frame[_RHAT]) & (
        frame[_RHAT] < UPSTREAM_RHAT_THRESHOLD
    )
    frame[_ESS_OK] = (
        np.isfinite(frame[_BULK_ESS])
        & np.isfinite(frame[_TAIL_ESS])
        & (frame[_BULK_ESS] >= MIN_BULK_ESS)
        & (frame[_TAIL_ESS] >= MIN_TAIL_ESS)
    )
    # Worst (lowest) bulk ESS first; unavailable values sort last rather than
    # to either extreme.
    return frame.sort_values(
        _BULK_ESS, ascending=True, na_position='last'
    ).reset_index(drop=True)

  def _rhat_per_parameter(self) -> dict[str, float]:
    """Worst (max, non-NaN) r-hat per top-level parameter name."""
    rhat = self._analyzer.get_rhat()
    return {
        name: _nan_reduce(np.asarray(value, dtype=float).ravel(), np.max)
        for name, value in rhat.items()
    }

  def _ess_per_parameter(self, method: str) -> dict[str, float]:
    """Worst (min, non-NaN) ESS per top-level parameter name, via ArviZ."""
    ess_dataset = az.ess(self._posterior, method=method)
    return {
        name: _nan_reduce(
            np.asarray(data_array.values, dtype=float).ravel(), np.min
        )
        for name, data_array in ess_dataset.data_vars.items()
    }

  @property
  def rhat_max(self) -> float:
    """Worst upstream TFP potential-scale-reduction R-hat across parameters.

    `NaN` if R-hat is unavailable for every parameter (see `_nan_reduce`).
    Identical in value to what `ConvergenceCheck.run().max_r_hat` computes,
    since both call `Analyzer.get_rhat()`. This is not ArviZ's rank-normalized
    split R-hat.
    """
    values = np.array(list(self._rhat_per_parameter().values()))
    return _nan_reduce(values, np.max) if values.size else float('nan')

  @property
  def rhat_max_ok_modern(self) -> bool:
    """Whether upstream R-hat clears the strict 1.01 heuristic."""
    rhat_max = self.rhat_max
    return bool(np.isfinite(rhat_max) and rhat_max < MODERN_RHAT_THRESHOLD)

  @property
  def rhat_max_ok_upstream(self) -> bool:
    """Whether upstream R-hat is below `UPSTREAM_RHAT_THRESHOLD`."""
    rhat_max = self.rhat_max
    return bool(np.isfinite(rhat_max) and rhat_max < UPSTREAM_RHAT_THRESHOLD)

  @property
  def min_bulk_ess(self) -> float:
    """Smallest bulk-ESS across all parameters, or `NaN` if unavailable."""
    values = np.array(list(self._ess_per_parameter('bulk').values()))
    return _nan_reduce(values, np.min) if values.size else float('nan')

  @property
  def min_tail_ess(self) -> float:
    """Smallest tail-ESS across all parameters, or `NaN` if unavailable."""
    values = np.array(list(self._ess_per_parameter('tail').values()))
    return _nan_reduce(values, np.min) if values.size else float('nan')

  @property
  def min_bulk_ess_per_draw(self) -> float:
    """`min_bulk_ess` divided by `total_draws`."""
    return self.min_bulk_ess / self.total_draws

  @property
  def ess_ok(self) -> bool:
    """Whether all reported ESS values are finite and clear their thresholds."""
    bulk_values = np.asarray(
        list(self._ess_per_parameter('bulk').values()), dtype=float
    )
    tail_values = np.asarray(
        list(self._ess_per_parameter('tail').values()), dtype=float
    )
    reported_bulk = bulk_values[~np.isnan(bulk_values)]
    reported_tail = tail_values[~np.isnan(tail_values)]
    return bool(
        reported_bulk.size
        and reported_tail.size
        and np.all(np.isfinite(reported_bulk))
        and np.all(np.isfinite(reported_tail))
        and _nan_reduce(bulk_values, np.min) >= MIN_BULK_ESS
        and _nan_reduce(tail_values, np.min) >= MIN_TAIL_ESS
    )

  def _has_nonfinite_ess(self) -> bool:
    """Whether any reported per-parameter ESS is infinite."""
    for method in ('bulk', 'tail'):
      values = np.asarray(
          list(self._ess_per_parameter(method).values()), dtype=float
      )
      reported = values[~np.isnan(values)]
      if np.any(~np.isfinite(reported)):
        return True
    return False

  # ---------------------------------------------------------------------
  # Verdict.
  # ---------------------------------------------------------------------

  @property
  def verdict(self) -> str:
    """A short reading of whether this posterior sample can be trusted."""
    parts = []

    if not self.divergences_reported:
      parts.append(
          'Divergent transitions are not reported for this posterior'
          ' sample (no `sample_stats["diverging"]`), so that check is'
          ' skipped.'
      )
    elif self.n_divergences == 0:
      parts.append(
          f'No divergent transitions across {self.total_draws} post-warmup'
          ' draws.'
      )
    elif self.divergence_rate_is_serious:
      parts.append(
          f'{self.n_divergences} divergent transitions'
          f' ({self.divergence_rate:.2%} of {self.total_draws} draws) --'
          f' above the {DIVERGENCE_RATE_SERIOUS_THRESHOLD:.1%} heuristic for'
          ' a systematic geometry problem (e.g. an unidentified'
          ' adstock/Hill parameter). Treat downstream estimates as'
          ' unreliable until this is fixed.'
      )
    else:
      parts.append(
          f'{self.n_divergences} divergent transitions'
          f' ({self.divergence_rate:.2%} of {self.total_draws} draws).'
          ' Below the serious-rate heuristic, but any divergence means NUTS'
          ' could not fully explore the posterior; worth a closer look at'
          ' `divergence_summary()`.'
      )

    rhat_max = self.rhat_max
    if np.isnan(rhat_max):
      parts.append(
          'Upstream TFP R-hat is undefined for all parameters; convergence'
          ' cannot be assessed.'
      )
    elif not np.isfinite(rhat_max):
      parts.append(
          'Upstream TFP R-hat is non-finite; convergence cannot be assessed.'
      )
    elif self.rhat_max_ok_modern:
      parts.append(
          f'Max upstream TFP R-hat {rhat_max:.4f} clears both the strict'
          f' ({MODERN_RHAT_THRESHOLD}) heuristic and upstream'
          f' ({UPSTREAM_RHAT_THRESHOLD}) thresholds.'
      )
    elif self.rhat_max_ok_upstream:
      parts.append(
          f"Max upstream TFP R-hat {rhat_max:.4f} clears upstream's"
          f' ({UPSTREAM_RHAT_THRESHOLD}) threshold but not the strict'
          f' {MODERN_RHAT_THRESHOLD} heuristic.'
      )
    else:
      parts.append(
          f"Max upstream TFP R-hat {rhat_max:.4f} exceeds even upstream's"
          f' {UPSTREAM_RHAT_THRESHOLD} threshold. This model has not'
          ' converged.'
      )

    min_bulk = self.min_bulk_ess
    min_tail = self.min_tail_ess
    if np.isnan(min_bulk) and np.isnan(min_tail):
      parts.append(
          'ESS is undefined for all parameters; sampling precision cannot be'
          ' assessed.'
      )
    elif np.isnan(min_bulk) or np.isnan(min_tail):
      parts.append(
          'ESS is unavailable for one or more required diagnostics; sampling'
          ' precision cannot be assessed.'
      )
    elif self._has_nonfinite_ess():
      parts.append('ESS is non-finite; sampling precision cannot be assessed.')
    elif self.ess_ok:
      parts.append(
          f'Min bulk/tail ESS {min_bulk:.0f}/{min_tail:.0f} both clear'
          f' {MIN_BULK_ESS}; 90% credible intervals should be stable.'
      )
    else:
      parts.append(
          f'Min bulk/tail ESS {min_bulk:.0f}/{min_tail:.0f} -- below the'
          f' {MIN_BULK_ESS}-draw heuristic for a stable 90% credible'
          ' interval on at least one parameter, even if r-hat looks fine.'
          ' Consider more draws or reparameterizing that parameter.'
      )

    return ' '.join(parts)
