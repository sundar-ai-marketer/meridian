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
estimates whose CV exceeds roughly 0.5 means ranking geos largely on posterior
noise.

```python
diag = geo_diagnostics.GeoAllocationReliability(mmm)
print(diag.verdict)
diag.summary().head()
```
"""

from __future__ import annotations

import dataclasses

from meridian import constants as c
from meridian.analysis import analyzer as analyzer_module
from meridian.common import errors
from meridian.model import model
import numpy as np
import pandas as pd
import xarray as xr


__all__ = [
    'GeoAllocationReliability',
    'RELIABLE_CV_THRESHOLD',
]

# Above this posterior coefficient of variation, a per-geo estimate carries
# less signal than noise for ranking purposes.
RELIABLE_CV_THRESHOLD = 0.5

_MEAN = 'mean'
_SD = 'sd'
_CV = 'cv'


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


@dataclasses.dataclass(frozen=True)
class _Reduced:
  mean: np.ndarray
  sd: np.ndarray
  cv: np.ndarray


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
    self._use_kpi = self._analyzer._use_kpi(use_kpi)  # pylint: disable=protected-access
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
        _Reduced(*_coefficient_of_variation(per_geo_values, axis=(0, 1))),
        _Reduced(*_coefficient_of_variation(agg_values, axis=(0, 1))),
        geos,
        channels,
    )

  @property
  def reliability_data(self) -> xr.Dataset:
    """Per geo and channel posterior mean, sd and CV of incremental outcome."""
    return xr.Dataset(
        data_vars={
            _MEAN: ((c.GEO, c.CHANNEL), self._by_geo.mean),
            _SD: ((c.GEO, c.CHANNEL), self._by_geo.sd),
            _CV: ((c.GEO, c.CHANNEL), self._by_geo.cv),
        },
        coords={c.GEO: self._geos, c.CHANNEL: self._channels},
    )

  def summary(self) -> pd.DataFrame:
    """Returns a per geo and channel table, worst precision first."""
    frame = self.reliability_data.to_dataframe().reset_index()
    frame['has_signal'] = ~frame[_CV].isna()
    frame['reliable'] = frame[_CV] <= RELIABLE_CV_THRESHOLD
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
    ratios = self.precision_loss()['cv_ratio'].to_numpy()
    finite_ratios = ratios[np.isfinite(ratios)]
    ratio = float(np.median(finite_ratios)) if finite_ratios.size else float('nan')
    if fraction >= 0.8:
      return (
          f'{fraction:.0%} of geo-channel estimates have CV <='
          f' {RELIABLE_CV_THRESHOLD}, and splitting by geo raises the CV by'
          f' {ratio:.1f}x. Per-geo allocation is defensible on this model,'
          ' though Meridian does not optimize across geos natively.'
      )
    if fraction >= 0.5:
      return (
          f'Only {fraction:.0%} of geo-channel estimates have CV <='
          f' {RELIABLE_CV_THRESHOLD} (geo split raises CV by {ratio:.1f}x).'
          ' Allocate across geos only for the channels that clear the'
          ' threshold; treat the rest as aggregate-only.'
      )
    return (
        f'Only {fraction:.0%} of geo-channel estimates have CV <='
        f' {RELIABLE_CV_THRESHOLD}, and splitting by geo raises the CV by'
        f' {ratio:.1f}x. Ranking geos on these estimates would largely be'
        ' ranking posterior noise. This is why aggregate optimization is the'
        ' supported path.'
    )
