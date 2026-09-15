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

"""Whether geo-level budget allocation is supportable on a given model.

`BudgetOptimizer` allocates across channels in aggregate, not across geos.
google/meridian#1668 asked why, and whether per-geo response curves are simply
too noisy to allocate on. That is an empirical question about a particular
model, not a universal one, so this module measures it rather than asserting an
answer.

The metric is the posterior coefficient of variation (CV) of incremental
outcome -- the posterior standard deviation divided by the absolute posterior
mean. Comparing the per-geo CV against the aggregated CV shows directly how
much precision is lost by splitting the estimate by geo. Allocating a budget on
estimates whose CV exceeds `RELIABLE_CV_THRESHOLD` (0.5) means ranking geos
largely on posterior noise. That threshold is a heuristic, not a value derived
from theory or validated against held-out data -- treat it as a starting point
for judgment, not a settled cutoff.

CV alone cannot tell apart three situations that all present as "high CV,
unreliable":

1.  No media execution in that geo/channel. Incremental outcome is
    deterministically zero, so CV is meaningless there (see
    `_coefficient_of_variation`).
2.  A posterior that straddles zero: `|mean|` is small and `sd` is large
    because the effect's *sign* is genuinely uncertain, not merely its size.
3.  A posterior that is centred clearly away from zero but is, by ordinary
    standards, imprecise.

`reliability_data` and `summary()` add `prob_positive` (the posterior
probability the effect is positive) and `ci_excludes_zero` (whether a 90%
equal-tailed credible interval -- Meridian's own convention for "credible
interval" elsewhere in this library, not a literal highest-density interval --
excludes zero) so case 2 can be told apart from case 3 directly from the
posterior's sign, instead of inferred from the CV alone.

```python
diag = geo_diagnostics.GeoAllocationReliability(mmm)
print(diag.verdict)
diag.summary().head()
diag.plot_reliability()
```
"""

from __future__ import annotations

import dataclasses

import altair as alt
from meridian import constants as c
from meridian.analysis import analyzer as analyzer_module
from meridian.common import errors
from meridian.model import model
from meridian.templates import formatter
import numpy as np
import pandas as pd
import xarray as xr


__all__ = [
    'GeoAllocationReliability',
    'RELIABLE_CV_THRESHOLD',
    'SIGN_CI_PROBABILITY',
    'resolve_use_kpi',
]

# Above this posterior coefficient of variation, a per-geo estimate carries
# less signal than noise for ranking purposes. This is a heuristic rule of
# thumb (see module docstring), not a validated statistical cutoff.
RELIABLE_CV_THRESHOLD = 0.5

# Width of the equal-tailed credible interval used to decide whether an
# effect's sign is settled. 0.9 matches the confidence level Meridian's own
# reports typically use elsewhere; it is a convention, not a derived value.
SIGN_CI_PROBABILITY = 0.9

_MEAN = 'mean'
_SD = 'sd'
_CV = 'cv'
_PROB_POSITIVE = 'prob_positive'
_CI_EXCLUDES_ZERO = 'ci_excludes_zero'


def _coefficient_of_variation(
    draws: np.ndarray, axis: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns (mean, sd, cv) reduced over `axis`.

  Cells carrying no signal get `NaN`, not a number. A channel with no media
  execution in a geo -- a channel launched in only some geos, or discontinued
  in one -- produces deterministically zero incremental outcome, so both mean
  and standard deviation collapse to floating-point residue rather than exact
  zero. Dividing one by the other yields noise: it can land anywhere from 1e-16
  to 1e+2 depending on rounding, and either extreme corrupts the ranking, the
  reliable/unreliable split, and the precision ratio. `_calculate_vif` treats
  constant columns the same way.
  """
  mean = np.mean(draws, axis=axis)
  sd = np.std(draws, axis=axis)

  # "Negligible" is relative to the largest cell, so the test is scale-free.
  scale = float(np.nanmax(np.abs(mean))) if mean.size else 0.0
  floor = max(scale * 1e-10, float(np.finfo(np.float64).tiny))
  no_signal = np.abs(mean) <= floor

  with np.errstate(divide='ignore', invalid='ignore'):
    cv = np.where(no_signal, np.nan, sd / np.abs(mean))
  return mean, sd, cv


def _sign_precision(
    draws: np.ndarray,
    axis: tuple[int, ...],
    probability: float = SIGN_CI_PROBABILITY,
) -> tuple[np.ndarray, np.ndarray]:
  """Returns (prob_positive, ci_excludes_zero) reduced over `axis`.

  CV cannot distinguish a posterior whose sign is genuinely uncertain (small
  `|mean|`, large `sd`) from one that is merely imprecise (mean clearly one
  sign, sd just large relative to it) -- both produce a large CV. Reading the
  sign directly off the draws resolves the two: `prob_positive` is the
  fraction of draws above zero, and `ci_excludes_zero` is whether the
  `probability`-width equal-tailed interval around the draws excludes zero.

  Args:
    draws: Draws to reduce.
    axis: Axis or axes to reduce over.
    probability: Width of the equal-tailed interval, between zero and one.

  Returns:
    A tuple `(prob_positive, ci_excludes_zero)`, each an array with `axis`
    reduced away.
  """
  prob_positive = np.mean(draws > 0, axis=axis)
  lo_q = (1.0 - probability) / 2.0
  hi_q = 1.0 - lo_q
  lo = np.quantile(draws, lo_q, axis=axis)
  hi = np.quantile(draws, hi_q, axis=axis)
  ci_excludes_zero = (lo > 0) | (hi < 0)
  return prob_positive, ci_excludes_zero


def resolve_use_kpi(
    analyzer: analyzer_module.Analyzer, use_kpi: bool
) -> bool:
  """Resolves `use_kpi` through `Analyzer._use_kpi`, guarding a private call.

  `Analyzer._use_kpi` is not part of the public API; it encodes the
  warn-and-fall-back rules around `kpi_type` and `revenue_per_kpi` that this
  module and `prior_predictive` both need rather than reimplementing. A
  rebase that renames or removes it would otherwise surface as a bare
  `AttributeError` from deep inside `__init__`. This wrapper fails loudly
  instead, naming the missing symbol and where to record the break, so both
  callers get one clear message instead of duplicating this guard.

  Args:
    analyzer: The `Analyzer` instance to resolve against.
    use_kpi: The caller-supplied `use_kpi` flag.

  Returns:
    The resolved `use_kpi` flag.

  Raises:
    AttributeError: If `Analyzer._use_kpi` no longer exists.
  """
  resolve = getattr(analyzer, '_use_kpi', None)
  if resolve is None:
    raise AttributeError(
        '`Analyzer._use_kpi` no longer exists -- an upstream rebase likely'
        ' renamed or removed it. `geo_diagnostics.py` and'
        ' `prior_predictive.py` both depend on it through'
        ' `geo_diagnostics.resolve_use_kpi` for this fork'
        " KPI/revenue resolution logic. Find its replacement in"
        ' `meridian/analysis/analyzer.py`, update `resolve_use_kpi`'
        ' accordingly, and record the break in TRIAGE.md.'
    )
  return resolve(use_kpi)


@dataclasses.dataclass(frozen=True)
class _Reduced:
  mean: np.ndarray
  sd: np.ndarray
  cv: np.ndarray
  prob_positive: np.ndarray
  ci_excludes_zero: np.ndarray


class GeoAllocationReliability:
  """Measures the precision lost by estimating incremental outcome per geo."""

  def __init__(
      self,
      meridian: model.Meridian,
      use_kpi: bool = False,
      use_posterior: bool = True,
  ):
    """Initializes the diagnostic.

    Args:
      meridian: A fitted `Meridian` model.
      use_kpi: If `True`, measures KPI units rather than revenue.
      use_posterior: If `True`, uses posterior draws; otherwise prior draws.

    Raises:
      NotFittedModelError: If the required inference group is missing.
      ValueError: If the model is national, where the question does not arise.
    """
    group = c.POSTERIOR if use_posterior else c.PRIOR
    if group not in meridian.inference_data.groups():
      raise errors.NotFittedModelError(
          f'The model has no {group} draws. Call'
          f' `sample_{group}()` first.'
      )
    if meridian.model_context.is_national:
      raise ValueError(
          'Geo allocation reliability is meaningless for a national model,'
          ' which has a single geo.'
      )

    self._meridian = meridian
    self._use_posterior = use_posterior
    self._analyzer = analyzer_module.Analyzer(
        model_context=meridian.model_context,
        inference_data=meridian.inference_data,
    )
    self._use_kpi = resolve_use_kpi(self._analyzer, use_kpi)
    self._by_geo, self._aggregated, self._geos, self._channels = (
        self._compute()
    )

  def _compute(
      self,
  ) -> tuple[_Reduced, _Reduced, list[str], list[str]]:
    """Reduces per-geo and aggregated incremental outcome draws."""
    per_geo = self._analyzer.incremental_outcome_xr(
        use_posterior=self._use_posterior,
        aggregate_geos=False,
        aggregate_times=True,
        use_kpi=self._use_kpi,
    )
    aggregated = self._analyzer.incremental_outcome_xr(
        use_posterior=self._use_posterior,
        aggregate_geos=True,
        aggregate_times=True,
        use_kpi=self._use_kpi,
    )

    geos = [str(g) for g in per_geo.coords[c.GEO].values]
    channel_dim = c.CHANNEL
    channels = [str(m) for m in per_geo.coords[channel_dim].values]

    # Draw axes are (chain, draw); reduce over both.
    per_geo_values = per_geo.transpose(
        c.CHAIN, c.DRAW, c.GEO, channel_dim
    ).values
    agg_values = aggregated.transpose(c.CHAIN, c.DRAW, channel_dim).values

    return (
        _Reduced(
            *_coefficient_of_variation(per_geo_values, axis=(0, 1)),
            *_sign_precision(per_geo_values, axis=(0, 1)),
        ),
        _Reduced(
            *_coefficient_of_variation(agg_values, axis=(0, 1)),
            *_sign_precision(agg_values, axis=(0, 1)),
        ),
        geos,
        channels,
    )

  @property
  def reliability_data(self) -> xr.Dataset:
    """Per geo and channel posterior diagnostics of incremental outcome.

    - **`mean`, `sd`, `cv`:** Posterior mean, standard deviation, and
      coefficient of variation.
    - **`prob_positive`:** Posterior probability the effect is positive.
    - **`ci_excludes_zero`:** Whether the `SIGN_CI_PROBABILITY`-width
      equal-tailed credible interval excludes zero.
    """
    return xr.Dataset(
        data_vars={
            _MEAN: ((c.GEO, c.CHANNEL), self._by_geo.mean),
            _SD: ((c.GEO, c.CHANNEL), self._by_geo.sd),
            _CV: ((c.GEO, c.CHANNEL), self._by_geo.cv),
            _PROB_POSITIVE: ((c.GEO, c.CHANNEL), self._by_geo.prob_positive),
            _CI_EXCLUDES_ZERO: (
                (c.GEO, c.CHANNEL),
                self._by_geo.ci_excludes_zero,
            ),
        },
        coords={c.GEO: self._geos, c.CHANNEL: self._channels},
    )

  def summary(self) -> pd.DataFrame:
    """Returns a per geo and channel table, worst precision first.

    Adds `has_signal` (media execution occurred), `reliable` (CV clears
    `RELIABLE_CV_THRESHOLD`), and `sign_uncertain`: `True` for an informative,
    unreliable cell whose credible interval still straddles zero -- the
    effect's direction is unsettled, not just its magnitude. This is case 2
    from the module docstring, distinct from a cell that is merely imprecise
    but clearly signed (case 3).
    """
    frame = self.reliability_data.to_dataframe().reset_index()
    frame['has_signal'] = ~frame[_CV].isna()
    frame['reliable'] = frame[_CV] <= RELIABLE_CV_THRESHOLD
    frame['sign_uncertain'] = (
        frame['has_signal']
        & ~frame['reliable']
        & ~frame[_CI_EXCLUDES_ZERO]
    )
    # Worst precision first; cells with no media execution sort to the end
    # rather than to either extreme of the ranking.
    return (
        frame.sort_values([_CV], ascending=False, na_position='last')
        .reset_index(drop=True)
    )

  def precision_loss(self) -> pd.DataFrame:
    """Compares aggregated CV against the median per-geo CV, per channel.

    Returns:
      A DataFrame with one row per channel: the aggregated CV, the median
      per-geo CV, and their ratio. A ratio of 4 means splitting by geo makes
      the estimate four times noisier in relative terms.
    """
    with np.errstate(invalid='ignore'):
      # nanmedian: a channel absent from some geos still has a meaningful
      # median across the geos where it ran.
      median_geo_cv = np.nanmedian(self._by_geo.cv, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
      ratio = np.where(
          np.isfinite(self._aggregated.cv) & (self._aggregated.cv != 0),
          median_geo_cv / self._aggregated.cv,
          np.nan,
      )
    return pd.DataFrame({
        c.CHANNEL: self._channels,
        'aggregated_cv': self._aggregated.cv,
        'median_geo_cv': median_geo_cv,
        'cv_ratio': ratio,
    })

  @property
  def fraction_reliable(self) -> float:
    """Fraction of *informative* estimates at or below the CV threshold.

    Cells with no media execution carry no information and are excluded rather
    than counted as either reliable or unreliable.
    """
    cv = self._by_geo.cv
    informative = cv[~np.isnan(cv)]
    if informative.size == 0:
      return 0.0
    return float(np.mean(informative <= RELIABLE_CV_THRESHOLD))

  @property
  def verdict(self) -> str:
    """A short reading of whether per-geo allocation is supportable."""
    fraction = self.fraction_reliable
    frame = self.summary()
    informative = frame[frame['has_signal']]
    sign_uncertain_fraction = (
        float(informative['sign_uncertain'].mean())
        if len(informative)
        else 0.0
    )
    sign_note = (
        f' {sign_uncertain_fraction:.0%} of informative estimates have a'
        f' {SIGN_CI_PROBABILITY:.0%} credible interval that straddles zero'
        ' -- the effect direction itself is unsettled there, not just its'
        ' size.'
        if sign_uncertain_fraction > 0
        else ''
    )
    ratios = self.precision_loss()['cv_ratio'].to_numpy()
    finite_ratios = ratios[np.isfinite(ratios)]
    ratio = float(np.median(finite_ratios)) if finite_ratios.size else float('nan')
    if fraction >= 0.8:
      return (
          f'{fraction:.0%} of geo-channel estimates have CV <='
          f' {RELIABLE_CV_THRESHOLD}, and splitting by geo raises the CV by'
          f' {ratio:.1f}x. Per-geo allocation is defensible on this model,'
          ' though Meridian does not optimize across geos natively.'
          f'{sign_note}'
      )
    if fraction >= 0.5:
      return (
          f'Only {fraction:.0%} of geo-channel estimates have CV <='
          f' {RELIABLE_CV_THRESHOLD} (geo split raises CV by {ratio:.1f}x).'
          ' Allocate across geos only for the channels that clear the'
          ' threshold; treat the rest as aggregate-only.'
          f'{sign_note}'
      )
    return (
        f'Only {fraction:.0%} of geo-channel estimates have CV <='
        f' {RELIABLE_CV_THRESHOLD}, and splitting by geo raises the CV by'
        f' {ratio:.1f}x. Ranking geos on these estimates would largely be'
        ' ranking posterior noise. This is why aggregate optimization is the'
        ' supported path.'
        f'{sign_note}'
    )

  def plot_reliability(self) -> alt.Chart:
    """Plots the share of reliable channel estimates, one bar per geo.

    Each bar is the fraction of that geo's *informative* geo-channel
    estimates (channels with media execution) whose CV clears
    `RELIABLE_CV_THRESHOLD`, sorted worst to best. A geo with no informative
    channel at all still gets a bar, at zero, rather than being silently
    dropped.

    Returns:
      An Altair bar chart, one bar per geo.
    """
    frame = self.summary()
    informative = frame[frame['has_signal']]
    per_geo = (
        informative.groupby(c.GEO, as_index=False).agg(
            fraction_reliable=('reliable', 'mean'),
            n_channels=('reliable', 'size'),
            n_sign_uncertain=('sign_uncertain', 'sum'),
        )
    )
    all_geos = pd.DataFrame({c.GEO: self._geos})
    per_geo = all_geos.merge(per_geo, on=c.GEO, how='left').fillna({
        'fraction_reliable': 0.0,
        'n_channels': 0,
        'n_sign_uncertain': 0,
    })
    per_geo = per_geo.sort_values(
        'fraction_reliable', ascending=True
    ).reset_index(drop=True)

    geo_axis = alt.Axis(
        title=None, labelAngle=-45, **formatter.AXIS_CONFIG
    )
    fraction_axis = alt.Axis(
        title='Share of channels reliable',
        domain=False,
        format='%',
        **formatter.AXIS_CONFIG,
    )

    bar = (
        alt.Chart(per_geo)
        .mark_bar(
            size=c.BAR_SIZE, cornerRadiusEnd=c.CORNER_RADIUS, tooltip=True
        )
        .encode(
            x=alt.X(
                f'{c.GEO}:N',
                sort=None,
                axis=geo_axis,
                scale=alt.Scale(padding=c.BAR_SIZE),
            ),
            y=alt.Y(
                'fraction_reliable:Q',
                axis=fraction_axis,
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.condition(
                alt.datum.fraction_reliable >= 0.5,
                alt.value(c.BLUE_600),
                alt.value(c.GREY_600),
            ),
            tooltip=[
                alt.Tooltip(f'{c.GEO}:N', title='Geo'),
                alt.Tooltip(
                    'fraction_reliable:Q', title='Share reliable', format='.0%'
                ),
                alt.Tooltip('n_channels:Q', title='Informative channels'),
                alt.Tooltip(
                    'n_sign_uncertain:Q', title='Sign-uncertain channels'
                ),
            ],
        )
    )

    return (
        bar.properties(
            title=formatter.custom_title_params(
                'Per-geo allocation reliability'
            ),
            width=formatter.bar_chart_width(len(per_geo) + 2),
            height=300,
        )
        .configure_axis(**formatter.TEXT_CONFIG)
    )
