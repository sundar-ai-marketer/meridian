# Copyright 2026 Meridian fork contributors.
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

"""Prior predictive checks for Meridian.

`ModelFit` compares the *posterior* expected outcome against the observed
outcome, which is only available once `sample_posterior` has run -- often hours
after a bad prior was chosen. This module does the same comparison against the
*prior*, so an implausible prior can be caught from `sample_prior` alone, in
seconds.

Addresses google/meridian#647.

Typical use:

```python
mmm.sample_prior(500)
check = prior_predictive.PriorPredictiveCheck(mmm)
print(check.summary())
check.plot_prior_predictive()
```

A prior that implies an outcome far from the observed scale will not be fixed
by sampling; it needs different priors. See also
`google/meridian#1469`, where a total-treatment-contribution prior probability
of 1.0 was diagnosed only after a full fit.
"""

from collections.abc import Sequence
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
    'PriorPredictiveCheck',
    'PriorPredictiveSummary',
]

_MEAN = 'mean'
_CI_LO = 'ci_lo'
_CI_HI = 'ci_hi'
_EXPECTED = 'expected'
_ACTUAL = 'actual'


@dataclasses.dataclass(frozen=True)
class PriorPredictiveSummary:
  """Scalar diagnostics comparing the prior predictive to the observed data.

  Attributes:
    coverage: Fraction of time periods whose observed outcome falls inside the
      prior credible interval. A well-calibrated prior sits near
      `confidence_level`; near 0 means the prior disagrees with the data.
    total_actual: Observed outcome summed over geos and time.
    total_prior_median: Median prior predictive total outcome.
    total_ratio: `total_prior_median / total_actual`. A value of 10 means the
      prior expects an order of magnitude more outcome than was observed.
    total_actual_within_ci: Whether the observed total falls inside the prior
      credible interval for the total.
    confidence_level: The credible interval width used.
    n_draws: Number of prior draws the check is based on.
  """

  coverage: float
  total_actual: float
  total_prior_median: float
  total_ratio: float
  total_actual_within_ci: bool
  confidence_level: float
  n_draws: int

  def to_frame(self) -> pd.DataFrame:
    """Returns the summary as a one-column DataFrame for display."""
    return pd.DataFrame(
        {
            'value': [
                self.coverage,
                self.total_actual,
                self.total_prior_median,
                self.total_ratio,
                self.total_actual_within_ci,
                self.confidence_level,
                self.n_draws,
            ]
        },
        index=pd.Index([
            'coverage',
            'total_actual',
            'total_prior_median',
            'total_ratio',
            'total_actual_within_ci',
            'confidence_level',
            'n_draws',
        ]),
    )

  @property
  def verdict(self) -> str:
    """A short, human-readable reading of the diagnostics."""
    if self.total_ratio > 10 or self.total_ratio < 0.1:
      return (
          'The prior predictive total is off by more than an order of magnitude'
          f' ({self.total_ratio:.2g}x the observed total). Sampling will not'
          ' fix this -- revise the priors.'
      )
    if not self.total_actual_within_ci:
      return (
          'The observed total falls outside the prior credible interval. The'
          ' priors disagree with the data before any fitting has happened.'
      )
    if self.coverage < 0.5 * self.confidence_level:
      return (
          f'Only {self.coverage:.0%} of time periods fall inside the prior'
          ' credible interval. The prior is too narrow or centred wrongly.'
      )
    return (
        'The prior predictive is consistent with the observed outcome at this'
        ' confidence level.'
    )


class PriorPredictiveCheck:
  """Compares the prior predictive outcome against the observed outcome."""

  def __init__(
      self,
      meridian: model.Meridian,
      use_kpi: bool = False,
      confidence_level: float = c.DEFAULT_CONFIDENCE_LEVEL,
  ):
    """Initializes the check from a model with prior draws.

    Args:
      meridian: A `Meridian` model on which `sample_prior` has been called. No
        posterior is required.
      use_kpi: If `True`, compares KPI units. Otherwise compares revenue, using
        `revenue_per_kpi` when available.
      confidence_level: Width of the prior credible interval, between zero and
        one.

    Raises:
      NotFittedModelError: If `sample_prior` has not been called.
      ValueError: If `confidence_level` is not in (0, 1).
    """
    if not 0.0 < confidence_level < 1.0:
      raise ValueError(
          f'`confidence_level` must be in (0, 1), got {confidence_level}.'
      )
    if c.PRIOR not in meridian.inference_data.groups():
      raise errors.NotFittedModelError(
          'The model has no prior draws. Call `sample_prior()` before running'
          ' a prior predictive check.'
      )

    self._meridian = meridian
    self._confidence_level = confidence_level
    self._analyzer = analyzer_module.Analyzer(
        model_context=meridian.model_context,
        inference_data=meridian.inference_data,
    )
    self._use_kpi = self._analyzer._use_kpi(use_kpi)  # pylint: disable=protected-access
    self._data = self._build_data()

  @property
  def confidence_level(self) -> float:
    return self._confidence_level

  def _actual_outcome_by_time(self) -> xr.DataArray:
    """Observed outcome aggregated over geos, indexed by time."""
    input_data = self._meridian.input_data
    kpi = input_data.kpi
    revenue_per_kpi = input_data.revenue_per_kpi
    if self._use_kpi or revenue_per_kpi is None:
      outcome = kpi
    else:
      outcome = kpi * revenue_per_kpi
    return outcome.sum(dim=c.GEO)

  def _build_data(self) -> xr.Dataset:
    """Builds the prior predictive dataset."""
    # Shape: (chain, draw, time). The prior group always has a single chain.
    prior_outcome = np.asarray(
        self._analyzer.expected_outcome(
            use_posterior=False,
            aggregate_geos=True,
            aggregate_times=False,
            use_kpi=self._use_kpi,
        )
    )
    draws = prior_outcome.reshape(-1, prior_outcome.shape[-1])

    lo_q = (1.0 - self._confidence_level) / 2.0
    hi_q = 1.0 - lo_q
    expected = np.stack(
        [
            np.mean(draws, axis=0),
            np.quantile(draws, lo_q, axis=0),
            np.quantile(draws, hi_q, axis=0),
        ],
        axis=-1,
    )

    actual = self._actual_outcome_by_time()
    times = [str(t) for t in actual.coords[c.TIME].values]

    # Totals are computed per draw, then summarized -- summing the per-period
    # quantiles would understate the interval.
    totals = draws.sum(axis=-1)

    return xr.Dataset(
        data_vars={
            _EXPECTED: ((c.TIME, c.METRIC), expected),
            _ACTUAL: ((c.TIME,), actual.values),
        },
        coords={
            c.TIME: times,
            c.METRIC: [_MEAN, _CI_LO, _CI_HI],
        },
        attrs={
            'confidence_level': self._confidence_level,
            'n_draws': int(draws.shape[0]),
            'total_draws': totals,
            'use_kpi': self._use_kpi,
        },
    )

  @property
  def prior_predictive_data(self) -> xr.Dataset:
    """Prior predictive and observed outcome over time.

    - **Coordinates:** `time`, `metric` (`mean`, `ci_lo`, `ci_hi`)
    - **Data variables:** `expected` (prior predictive), `actual` (observed)
    """
    return self._data

  def summary(self) -> PriorPredictiveSummary:
    """Returns scalar diagnostics comparing the prior to the observed data."""
    data = self._data
    actual = data[_ACTUAL].values
    lo = data[_EXPECTED].sel({c.METRIC: _CI_LO}).values
    hi = data[_EXPECTED].sel({c.METRIC: _CI_HI}).values
    coverage = float(np.mean((actual >= lo) & (actual <= hi)))

    totals = data.attrs['total_draws']
    total_actual = float(actual.sum())
    total_median = float(np.median(totals))
    lo_q = (1.0 - self._confidence_level) / 2.0
    total_lo = float(np.quantile(totals, lo_q))
    total_hi = float(np.quantile(totals, 1.0 - lo_q))

    # Guard against a degenerate observed total.
    ratio = total_median / total_actual if total_actual else float('inf')

    return PriorPredictiveSummary(
        coverage=coverage,
        total_actual=total_actual,
        total_prior_median=total_median,
        total_ratio=ratio,
        total_actual_within_ci=bool(total_lo <= total_actual <= total_hi),
        confidence_level=self._confidence_level,
        n_draws=int(data.attrs['n_draws']),
    )

  def _plot_frame(self, selected_times: Sequence[str] | None) -> pd.DataFrame:
    data = self._data
    if selected_times is not None:
      data = data.sel({c.TIME: list(selected_times)})
    return pd.DataFrame({
        c.TIME: data.coords[c.TIME].values,
        _MEAN: data[_EXPECTED].sel({c.METRIC: _MEAN}).values,
        _CI_LO: data[_EXPECTED].sel({c.METRIC: _CI_LO}).values,
        _CI_HI: data[_EXPECTED].sel({c.METRIC: _CI_HI}).values,
        _ACTUAL: data[_ACTUAL].values,
    })

  def plot_prior_predictive(
      self, selected_times: Sequence[str] | None = None
  ) -> alt.LayerChart:
    """Plots the prior predictive interval against the observed outcome.

    Args:
      selected_times: Optional subset of time coordinates to plot.

    Returns:
      An Altair chart layering the prior credible band, the prior mean, and the
      observed outcome.
    """
    df = self._plot_frame(selected_times)
    outcome_label = 'KPI' if self._use_kpi else 'Revenue'

    base = alt.Chart(df).encode(
        x=alt.X(f'{c.TIME}:T', title='Time', axis=alt.Axis(**formatter.AXIS_CONFIG))
    )
    band = base.mark_area(opacity=0.25).encode(
        y=alt.Y(f'{_CI_LO}:Q', title=outcome_label),
        y2=alt.Y2(f'{_CI_HI}:Q'),
    )
    prior_mean = base.mark_line(strokeDash=[4, 3]).encode(y=f'{_MEAN}:Q')
    observed = base.mark_line().encode(y=f'{_ACTUAL}:Q')

    return (
        alt.layer(band, prior_mean, observed)
        .properties(
            title=formatter.custom_title_params(
                f'Prior predictive vs observed {outcome_label.lower()}'
            ),
            width=formatter.bar_chart_width(len(df)),
        )
        .configure_axis(**formatter.TEXT_CONFIG)
    )
