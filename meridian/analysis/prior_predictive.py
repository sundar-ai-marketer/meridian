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

**This is a predictive check, not just a check of the conditional mean.**
Meridian's likelihood is `y ~ Normal(y_pred, sigma)`, with `sigma` a free
parameter carrying its own prior (`prior_distribution.py`,
`posterior_sampler.py`'s `y ~ Normal(y_pred, sigma_gt)`). A correct prior
predictive check draws `y_rep ~ p(y|theta)` with `theta ~ prior` -- mean *and*
observation noise -- so this module adds sigma noise to
`analyzer.expected_outcome`'s conditional mean before computing any interval
or coverage figure below; see `_prior_predictive_noise_sd` for the derivation
and `analysis/review/checks.py`'s `BayesianPPPCheck._calculate_total_sigma`,
which this follows, for the upstream precedent. Skipping this step computes an interval
for the *mean function*, which is systematically narrower than the true
predictive interval and biases the check toward false "prior disagrees with
data" verdicts.

`coverage` and the verdict thresholds below are heuristics, not derived
statistics: with roughly 100 time periods, empirical coverage has real
binomial sampling spread (for a nominal 90% interval, a fifty-period series
easily reads 84-96% by chance alone), so a near-miss against
`_COVERAGE_FLOOR_FACTOR` is not on its own evidence of a miscalibrated prior.
"""

from collections.abc import Sequence
import dataclasses

import altair as alt
from meridian import constants as c
from meridian.analysis import analyzer as analyzer_module
from meridian.analysis import geo_diagnostics
from meridian.common import currency as currency_module
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
_SERIES = 'series'
_VALUE = 'value'

# Heuristic verdict thresholds (see module docstring): none of these are
# derived from a formal test, and near-misses should be read with the
# binomial sampling spread of `coverage` in mind.

# An order-of-magnitude mismatch between the prior predictive total and the
# observed total signals priors on the wrong scale -- sampling will not fix
# it -- rather than a calibration nuance.
_TOTAL_RATIO_HIGH = 10.0
_TOTAL_RATIO_LOW = 0.1

# Coverage below this fraction of the nominal confidence level flags a prior
# that is too narrow or centred wrongly.
_COVERAGE_FLOOR_FACTOR = 0.5

# Grey band, amber prior mean, blue observed: distinguishable in
# greyscale and for the common colour-vision deficiencies, and the two
# lines also differ by dash pattern so colour is never the only cue.
_SERIES_COLORS = ['#CFD8DC', '#EA8600', '#1A73E8']


@dataclasses.dataclass(frozen=True)
class PriorPredictiveSummary:
  """Scalar diagnostics comparing the prior predictive to the observed data.

  Attributes:
    coverage: Fraction of time periods whose observed outcome falls inside the
      prior *predictive* interval -- the mean prediction plus observation
      noise (see module docstring), not just the interval of the mean
      function. A well-calibrated prior sits near `confidence_level`; near 0
      means the prior disagrees with the data. With ~100 time periods this
      has real binomial sampling spread; see the module docstring before
      reading a near-miss as evidence of miscalibration.
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
    """A short, human-readable reading of the diagnostics.

    The thresholds behind this reading (`_TOTAL_RATIO_HIGH`,
    `_TOTAL_RATIO_LOW`, `_COVERAGE_FLOOR_FACTOR`) are heuristics, not derived
    statistics -- see the module docstring.
    """
    if (
        self.total_ratio > _TOTAL_RATIO_HIGH
        or self.total_ratio < _TOTAL_RATIO_LOW
    ):
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
    if self.coverage < _COVERAGE_FLOOR_FACTOR * self.confidence_level:
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
    self._use_kpi = geo_diagnostics.resolve_use_kpi(self._analyzer, use_kpi)
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

  def _prior_predictive_noise_sd(self, mean_draws: np.ndarray) -> np.ndarray:
    """Standard deviation of the prior predictive noise, per draw and time.

    Meridian's likelihood is `y_scaled ~ Normal(y_pred_scaled, sigma)` on the
    population-scaled KPI, independent across geo and time (see the module
    docstring and `posterior_sampler.py`'s `y ~ Normal(y_pred, sigma_gt)`).
    `KpiTransformer.inverse` (`model/transformers.py`) converts a scaled cell
    back to unscaled KPI as `tensor * population_scaled_stdev *
    population_g + population_scaled_mean * population_g`, so a zero-mean
    perturbation `eps_g ~ Normal(0, sigma_g)` in scaled space becomes, in
    unscaled KPI, `eps_g * population_scaled_stdev * population_g`.
    Multiplying by `revenue_per_kpi_{g,t}` when comparing revenue and summing
    independent geos gives, per prior draw and time period:

    `Var(sum_g Outcome_{g,t}) = sum_g sigma_g^2 * weight_{g,t}^2`

    where `weight_{g,t} = population_scaled_stdev * population_g *
    revenue_per_kpi_{g,t}` (`revenue_per_kpi` dropped when `use_kpi=True`).
    This is the same weight `BayesianPPPCheck._calculate_total_sigma` derives
    in `analysis/review/checks.py` for the fully-aggregated total; this
    method keeps the geo sum separate per time period, instead of also
    summing over time, and keeps sigma varying per prior draw rather than
    per posterior draw.

    Args:
      mean_draws: The `(n_draws, n_times)` conditional-mean draws this noise
        will be added to, used only to validate the time dimension matches.

    Returns:
      A `(n_draws, n_times)` array of standard deviations, aligned with
      `mean_draws`.
    """
    model_context = self._meridian.model_context
    input_data = self._meridian.input_data
    stdev = float(model_context.kpi_transformer.population_scaled_stdev)
    pop = np.asarray(model_context.population)  # (n_geos,)
    kpi = np.asarray(input_data.kpi)  # (n_geos, n_times)
    revenue_per_kpi = input_data.revenue_per_kpi

    if self._use_kpi or revenue_per_kpi is None:
      weight = stdev * pop[:, np.newaxis] * np.ones_like(kpi)
    else:
      weight = stdev * pop[:, np.newaxis] * np.asarray(revenue_per_kpi)

    sigma_da = self._meridian.inference_data.prior[c.SIGMA]
    if c.GEO in sigma_da.dims:
      sigma = sigma_da.transpose(c.CHAIN, c.DRAW, c.GEO).values
      sigma_flat = sigma.reshape(-1, sigma.shape[-1])  # (n_draws, n_geos)
    else:
      # `unique_sigma_for_each_geo=False`, or a single-geo model: one sigma
      # shared by every geo.
      sigma = sigma_da.transpose(c.CHAIN, c.DRAW).values
      sigma_flat = np.broadcast_to(
          sigma.reshape(-1, 1), (sigma.size, weight.shape[0])
      )

    if sigma_flat.shape[0] != mean_draws.shape[0]:
      raise ValueError(
          'Prior sigma draws'
          f' ({sigma_flat.shape[0]}) do not align with the conditional-mean'
          f' draws ({mean_draws.shape[0]}). This should not happen for a'
          ' prior group, which always has a single chain.'
      )

    # sum_g sigma_{d,g}^2 * weight_{g,t}^2, per prior draw d and time t.
    variance = np.einsum('dg,gt->dt', sigma_flat**2, weight**2)
    return np.sqrt(variance)

  def _build_data(self) -> xr.Dataset:
    """Builds the prior predictive dataset."""
    # Shape: (chain, draw, time). The prior group always has a single chain.
    prior_mean = np.asarray(
        self._analyzer.expected_outcome(
            use_posterior=False,
            aggregate_geos=True,
            aggregate_times=False,
            use_kpi=self._use_kpi,
        )
    )
    mean_draws = prior_mean.reshape(-1, prior_mean.shape[-1])

    # A true prior *predictive* draw adds observation noise to the
    # conditional mean -- see module docstring. A fixed seed keeps
    # `summary()` and `plot_prior_predictive()` deterministic across repeated
    # calls on the same fitted model, rather than resampling (and therefore
    # changing) the coverage figure on every call.
    noise_sd = self._prior_predictive_noise_sd(mean_draws)
    rng = np.random.default_rng(0)
    predictive_draws = (
        mean_draws + rng.normal(size=mean_draws.shape) * noise_sd
    )

    lo_q = (1.0 - self._confidence_level) / 2.0
    hi_q = 1.0 - lo_q
    expected = np.stack(
        [
            # The `mean` line is the low-noise conditional mean; only the
            # interval widens to reflect predictive, not just mean, coverage.
            np.mean(mean_draws, axis=0),
            np.quantile(predictive_draws, lo_q, axis=0),
            np.quantile(predictive_draws, hi_q, axis=0),
        ],
        axis=-1,
    )

    actual = self._actual_outcome_by_time()
    times = [str(t) for t in actual.coords[c.TIME].values]

    # Totals are computed per draw, then summarized -- summing the per-period
    # quantiles would understate the interval. The median total uses the
    # low-noise mean draws; the total interval uses the predictive draws.
    mean_totals = mean_draws.sum(axis=-1)
    predictive_totals = predictive_draws.sum(axis=-1)

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
            'n_draws': int(mean_draws.shape[0]),
            'total_draws': mean_totals,
            'total_predictive_draws': predictive_totals,
            'use_kpi': self._use_kpi,
        },
    )

  @property
  def prior_predictive_data(self) -> xr.Dataset:
    """Prior predictive and observed outcome over time.

    - **Coordinates:** `time`, `metric` (`mean`, `ci_lo`, `ci_hi`)
    - **Data variables:** `expected` (prior predictive), `actual` (observed)

    `expected`'s `mean` is the deterministic conditional mean,
    `E(Outcome|theta)`; `ci_lo`/`ci_hi` are the prior *predictive* interval,
    which also accounts for the observation-noise parameter `sigma` and is
    therefore wider than an interval built from `mean` alone -- see the
    module docstring.
    """
    return self._data

  def summary(self) -> PriorPredictiveSummary:
    """Returns scalar diagnostics comparing the prior to the observed data."""
    data = self._data
    actual = data[_ACTUAL].values
    lo = data[_EXPECTED].sel({c.METRIC: _CI_LO}).values
    hi = data[_EXPECTED].sel({c.METRIC: _CI_HI}).values
    coverage = float(np.mean((actual >= lo) & (actual <= hi)))

    mean_totals = data.attrs['total_draws']
    predictive_totals = data.attrs['total_predictive_draws']
    total_actual = float(actual.sum())
    total_median = float(np.median(mean_totals))
    lo_q = (1.0 - self._confidence_level) / 2.0
    total_lo = float(np.quantile(predictive_totals, lo_q))
    total_hi = float(np.quantile(predictive_totals, 1.0 - lo_q))

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
      available = set(str(t) for t in data.coords[c.TIME].values)
      unknown = [t for t in selected_times if str(t) not in available]
      if unknown:
        # xarray would raise a bare KeyError naming neither the bad value nor
        # what was expected.
        raise ValueError(
            f'`selected_times` contains values not in the model time'
            f' coordinates: {unknown}. Times run from'
            f' {min(available)} to {max(available)}.'
        )
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
      observed outcome, with a legend identifying each.

    Raises:
      ValueError: If `selected_times` contains unknown time coordinates.
    """
    df = self._plot_frame(selected_times)
    outcome_label = 'KPI' if self._use_kpi else 'Revenue'
    interval_label = f'Prior {self._confidence_level:.0%} interval'
    currency = getattr(
        self._meridian.input_data, 'currency_code', None
    )
    symbol = '' if self._use_kpi else currency_module.get_currency_symbol(
        currency
    )

    # Long form so the two series can share a colour scale and one legend.
    lines = df.melt(
        id_vars=[c.TIME],
        value_vars=[_MEAN, _ACTUAL],
        var_name=_SERIES,
        value_name=_VALUE,
    ).replace({_SERIES: {_MEAN: 'Prior mean', _ACTUAL: 'Observed'}})

    x = alt.X(
        f'{c.TIME}:T',
        title='Time',
        axis=alt.Axis(
            tickCount=8, format=c.QUARTER_FORMAT, **formatter.AXIS_CONFIG
        ),
    )
    y_axis = alt.Axis(
        labelExpr=formatter.compact_number_expr(currency=symbol),
        **formatter.AXIS_CONFIG,
    )

    # A constant column so the band joins the same colour scale as the lines
    # and therefore appears in the one shared legend.
    band_df = df.assign(**{_SERIES: interval_label})
    band = (
        alt.Chart(band_df)
        .mark_area(opacity=0.3)
        .encode(
            x=x,
            y=alt.Y(f'{_CI_LO}:Q', title=outcome_label, axis=y_axis),
            y2=alt.Y2(f'{_CI_HI}:Q'),
            color=alt.Color(
                f'{_SERIES}:N',
                scale=alt.Scale(
                    domain=[interval_label, 'Prior mean', 'Observed'],
                    range=_SERIES_COLORS,
                ),
                legend=alt.Legend(title=None),
            ),
        )
    )
    series = (
        alt.Chart(lines)
        .mark_line()
        .encode(
            x=x,
            y=alt.Y(f'{_VALUE}:Q', title=outcome_label, axis=y_axis),
            color=alt.Color(
                f'{_SERIES}:N',
                scale=alt.Scale(
                    domain=[interval_label, 'Prior mean', 'Observed'],
                    range=_SERIES_COLORS,
                ),
                legend=alt.Legend(title=None),
            ),
            strokeDash=alt.StrokeDash(
                f'{_SERIES}:N',
                scale=alt.Scale(
                    domain=['Prior mean', 'Observed'], range=[[4, 3], [1, 0]]
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip(f'{c.TIME}:T', title='Time'),
                alt.Tooltip(f'{_SERIES}:N', title='Series'),
                alt.Tooltip(f'{_VALUE}:Q', title=outcome_label, format=',.0f'),
            ],
        )
    )

    return (
        alt.layer(band, series)
        .properties(
            title=formatter.custom_title_params(
                f'Prior predictive vs observed {outcome_label.lower()}'
            ),
            # A fixed width, as the sibling charts use. Sizing by point count
            # is for bar charts: 156 weekly periods would be ~9700px wide.
            width=c.VEGALITE_FACET_EXTRA_LARGE_WIDTH,
            height=300,
        )
        .configure_axis(**formatter.TEXT_CONFIG)
    )
