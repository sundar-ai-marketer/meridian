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

"""Does Meridian recover a known ROI at *your* data shape?

A model that fits, converges and round-trips still tells you nothing about
whether its numbers are right. This generates data whose true ROI is known by
construction, fits Meridian, and reports whether the posterior recovers it.

Run it at the shape of a client's data before trusting a model built on that
data.

## The result that motivated this module

Recovery degrades when the simulated response is not the concave shape Meridian
assumes. At 5 geos, 104 weeks, 3 channels, seed 7, the highest-ROI channel
(true ROI 4.0, lowest spend) recovers like this:

| response shape | carryover | ROI error | true value inside 90% CI |
|---|---|---|---|
| concave        | none      | +9%       | yes |
| concave        | geometric | -43%      | yes |
| linear         | none      | **+71%**  | **no** |
| linear         | geometric | +39%      | yes |

Read the no-carryover rows against each other: response shape is the only thing
that differs, and a linear truth overstates ROI by 71% with an interval that
excludes the true value. That is not a Meridian defect. It is what fitting a
concave saturation curve to a response that is not concave does, in any MMM
that assumes saturation. The consequence worth internalising:

**Meridian's credible intervals quantify parameter uncertainty conditional on
the assumed saturation shape. They do not cover being wrong about that shape.**

So a channel far from saturation -- typically one at low spend, whose real
response is still close to linear -- can have its ROI overstated by tens of
percent with an interval that excludes the truth. Presenting such an interval
as the total uncertainty overstates what the method can support.

The carryover rows carry a second, separate lesson: estimating adstock and
saturation jointly from 520 geo-weeks costs real precision even when the shape
assumption holds. Channel ordering survived in all four runs; the level did not.

NOTE ON CONFIG: these figures are `max_lag`-specific, and the CLI defaults to
`max_lag=4`. Pass `--max-lag 0` to reproduce the no-carryover rows.

Measured on JAX 0.10.2 / TFP 0.26.0-dev20260130 / numpy 2.3.5, float64, macOS
arm64. NUTS is not bit-reproducible across library or hardware versions even at
a fixed seed, so re-measure rather than quoting these figures.

**Each row above is one simulated dataset and one fit** (`--replications 1`,
the default). That cannot separate systematic misspecification bias from a
single unlucky draw -- it is a sizing exercise at your own data's shape, not a
calibration statement.

## Turning the sizing exercise into an empirical recovery study

Pass `--replications N` to simulate and fit N independent datasets, each from
a seed deterministically derived from `--seed` via
`numpy.random.SeedSequence(seed).spawn(N)`. Across the N fits this reports,
per channel:

  * the median and interquartile range of relative ROI error;
  * **empirical coverage** -- the fraction of replications where the true ROI
    fell inside the nominal credible interval at this configured truth, data
    shape, prior, and response. It describes recovery at that setting; it is
    not a general guarantee that Bayesian interval coverage equals the nominal
    `--confidence-level` (0.9 by default), even when the response shape is
    correctly specified. Compare relevant truths and configurations rather
    than treating one point estimate as a universal calibration result.
  * a 95% Wilson score interval on that coverage estimate. With N
    replications the coverage estimate is itself noisy (binomial with only N
    trials); report the interval, not a point estimate, and do not treat N=20
    as enough to resolve 0.90 from 0.80.

It also records a rank fraction per channel and replication: the fraction of
posterior ROI draws below the realised true ROI. These ranks help show whether
the fitted estimates tend to land above or below the configured truth.

**This fixed-truth recovery experiment is not simulation-based calibration
(SBC).** Classical SBC draws each data-generating parameter from the same
prior used to fit the model; this module deliberately fixes `true_roi` and
can deliberately simulate a response shape outside the fitted model. Its rank
fractions therefore need not be Uniform(0, 1), even when the recovery setup is
working as intended. They are descriptive recovery output, not a calibration
test or a basis for a uniform-null threshold.

This module does not implement SBC, rank histograms, or a calibrated rank
test. What is above is the whole of it.

Multi-replication runs take N times as long as a single fit (each fit is
several minutes); `--replications` prints a per-replication progress line and
a running estimate of total runtime so you know what you signed up for before
it is done. **No results for `--replications > 1` are published in this
docstring** -- generating them takes the time it takes, and this module does
not report numbers nobody measured.

```
python -m meridian.validation --n-geos 5 --n-times 104 --response linear
python -m meridian.validation --replications 20 --response linear
```
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import dataclasses
import sys
import time
from typing import Literal

import numpy as np
import pandas as pd

from meridian import backend
from meridian import constants as c
from meridian.analysis import analyzer as analyzer_module
from meridian.data import data_frame_input_data_builder as dfb
from meridian.model import model
from meridian.model import prior_distribution
from meridian.model import spec

__all__ = [
    'RecoveryConfig',
    'RecoveryResult',
    'ChannelRecovery',
    'ChannelReplicationSummary',
    'MultiRecoveryResult',
    'simulate',
    'run_recovery',
    'run_recovery_replications',
    'derive_replication_seeds',
]

ResponseShape = Literal['linear', 'concave']


def _require_integer_at_least(name: str, value: object, minimum: int) -> None:
  """Raises a clear error unless `value` is an integer at least `minimum`."""
  if (
      isinstance(value, bool)
      or not isinstance(value, (int, np.integer))
      or value < minimum
  ):
    raise ValueError(
        f'`{name}` must be an integer >= {minimum}, got {value!r}.'
    )


def _require_finite_at_least(
    name: str, value: object, minimum: float, *, strict: bool = False
) -> None:
  """Raises a clear error unless a scalar is finite and above a bound."""
  try:
    number = float(value)
  except (TypeError, ValueError) as error:
    raise ValueError(f'`{name}` must be a finite number.') from error
  violates_bound = number <= minimum if strict else number < minimum
  if not np.isfinite(number) or violates_bound:
    comparison = '>' if strict else '>='
    raise ValueError(
        f'`{name}` must be finite and {comparison} {minimum}, got {value!r}.'
    )


def _require_positive_vector(name: str, values: object) -> None:
  """Raises a clear error unless a vector is nonempty, finite, and positive."""
  try:
    array = np.asarray(values, dtype=float)
  except (TypeError, ValueError) as error:
    raise ValueError(
        f'`{name}` must be a nonempty sequence of finite values > 0.'
    ) from error
  if (
      array.ndim != 1
      or array.size == 0
      or not np.all(np.isfinite(array))
      or np.any(array <= 0.0)
  ):
    raise ValueError(
        f'`{name}` must be a nonempty sequence of finite values > 0.'
    )


@dataclasses.dataclass(frozen=True)
class RecoveryConfig:
  """Shape of the synthetic experiment.

  Attributes:
    true_roi: True ROI per channel, known by construction.
    n_geos: Number of geos.
    n_times: Time periods in the modelling window.
    max_lag: Carryover length. Media history is extended to cover it, and the
      decay weights are normalized so total incremental outcome -- and so true
      ROI -- is unchanged.
    alpha: Geometric decay rate, used only when `max_lag > 0`.
    response: `'concave'` matches the saturation Meridian assumes.
      `'linear'` deliberately violates it, to size the misspecification bias.
    noise_fraction: Observation noise as a fraction of mean baseline level.
    spend_scale: Mean spend per channel, before the gamma draw.
    confidence_level: Credible interval width.
    n_chains: MCMC chains.
    n_adapt: MCMC adaptation steps.
    n_burnin: MCMC burn-in steps.
    n_keep: MCMC retained draws per chain.
    prior_roi_median: Median of the shared ROI prior applied to every channel.
      The same prior prevents channel-specific prior information from deciding
      their relative recovery, but it remains informative about ROI levels.
    prior_roi_sigma: Log-scale standard deviation of that prior.
    seed: Base seed for both simulation and sampling. With `replications > 1`
      each replication derives its own seed from this one via
      `numpy.random.SeedSequence(seed).spawn(replications)`, so the whole run
      -- including which seed each replication gets -- is reproducible.
    replications: Number of independent simulate-and-fit replications.
      `1` (the default) is the single-run path: `run_recovery` behaves
      exactly as it did before this field existed. Values `> 1` are handled
      by `run_recovery_replications`, which turns a sizing exercise into an
      empirical-coverage estimate.
  """

  true_roi: tuple[float, ...] = (1.0, 2.0, 4.0)
  n_geos: int = 5
  n_times: int = 104
  max_lag: int = 4
  alpha: float = 0.6
  response: ResponseShape = 'concave'
  noise_fraction: float = 0.05
  spend_scale: tuple[float, ...] = (3_000.0, 6_000.0, 2_000.0)
  confidence_level: float = 0.9
  n_chains: int = 4
  n_adapt: int = 1_000
  n_burnin: int = 500
  n_keep: int = 500
  prior_roi_median: float = 2.0
  prior_roi_sigma: float = 0.7
  seed: int = 7
  replications: int = 1

  def __post_init__(self):
    try:
      n_true_roi = len(self.true_roi)
      n_spend_scale = len(self.spend_scale)
    except TypeError as error:
      raise ValueError(
          '`true_roi` and `spend_scale` must be sequences.'
      ) from error
    if n_true_roi != n_spend_scale:
      raise ValueError(
          f'`true_roi` has {n_true_roi} entries but `spend_scale` has'
          f' {n_spend_scale}.'
      )
    _require_positive_vector('true_roi', self.true_roi)
    _require_positive_vector('spend_scale', self.spend_scale)
    _require_integer_at_least('n_geos', self.n_geos, 1)
    _require_integer_at_least('n_times', self.n_times, 1)
    _require_integer_at_least('max_lag', self.max_lag, 0)
    _require_integer_at_least('n_chains', self.n_chains, 1)
    _require_integer_at_least('n_adapt', self.n_adapt, 0)
    _require_integer_at_least('n_burnin', self.n_burnin, 0)
    _require_integer_at_least('n_keep', self.n_keep, 1)
    _require_integer_at_least('replications', self.replications, 1)
    _require_finite_at_least('alpha', self.alpha, 0.0)
    _require_finite_at_least('noise_fraction', self.noise_fraction, 0.0)
    _require_finite_at_least(
        'prior_roi_median', self.prior_roi_median, 0.0, strict=True
    )
    _require_finite_at_least(
        'prior_roi_sigma', self.prior_roi_sigma, 0.0, strict=True
    )
    if self.response not in ('linear', 'concave'):
      raise ValueError(
          f"`response` must be 'linear' or 'concave', got {self.response!r}."
      )
    if not 0.0 < self.confidence_level < 1.0:
      raise ValueError('`confidence_level` must be in (0, 1).')

  @property
  def channels(self) -> list[str]:
    return [f'channel_{i}' for i in range(len(self.true_roi))]

  @property
  def n_media_times(self) -> int:
    return self.n_times + self.max_lag


@dataclasses.dataclass(frozen=True)
class ChannelRecovery:
  """Recovery of one channel's ROI."""

  channel: str
  true_roi: float
  median: float
  ci_low: float
  ci_high: float

  @property
  def covered(self) -> bool:
    return self.ci_low <= self.true_roi <= self.ci_high

  @property
  def relative_error(self) -> float:
    return (self.median - self.true_roi) / self.true_roi


@dataclasses.dataclass(frozen=True)
class RecoveryResult:
  """Outcome of one recovery experiment."""

  config: RecoveryConfig
  channels: tuple[ChannelRecovery, ...]
  # ArviZ rank-normalized split R-hat, unlike SamplingDiagnostics' upstream
  # TFP potential-scale-reduction statistic.
  max_r_hat: float

  @property
  def converged(self) -> bool:
    return self.max_r_hat < 1.2

  @property
  def all_covered(self) -> bool:
    return all(ch.covered for ch in self.channels)

  @property
  def ordering_recovered(self) -> bool:
    """Whether the estimated ranking matches the true ranking."""
    true_rank = np.argsort([ch.true_roi for ch in self.channels])
    est_rank = np.argsort([ch.median for ch in self.channels])
    return bool(np.array_equal(true_rank, est_rank))

  @property
  def passed(self) -> bool:
    return self.converged and self.all_covered and self.ordering_recovered

  def to_frame(self) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                'channel': ch.channel,
                'true_roi': ch.true_roi,
                'median': ch.median,
                'ci_low': ch.ci_low,
                'ci_high': ch.ci_high,
                'covered': ch.covered,
                'relative_error': ch.relative_error,
            }
            for ch in self.channels
        ]
    )

  def format_report(self) -> str:
    lines = [
        f'Recovery: response={self.config.response}'
        f' max_lag={self.config.max_lag}'
        f' n_geos={self.config.n_geos} n_times={self.config.n_times}',
        '-' * 72,
        f'max rank-normalized r_hat = {self.max_r_hat:.4f}'
        f"  {'converged' if self.converged else 'NOT CONVERGED'}",
        '',
        f"{'channel':<12}{'true':>8}{'median':>9}{'90% CI':>22}"
        f"{'cov':>6}{'rel.err':>9}",
    ]
    for ch in self.channels:
      lines.append(
          f'{ch.channel:<12}{ch.true_roi:8.3f}{ch.median:9.3f}'
          f'  [{ch.ci_low:7.3f},{ch.ci_high:7.3f}]'
          f'{str(ch.covered):>6}{ch.relative_error:+8.1%}'
      )
    lines += [
        '',
        f'ordering recovered: {self.ordering_recovered}',
        f'all true ROIs inside interval: {self.all_covered}',
        f'VERDICT: {"PASS" if self.passed else "FAIL"}',
    ]
    return '\n'.join(lines)


def _decay_weights(max_lag: int, alpha: float) -> np.ndarray:
  """Normalized decay weights: carryover moves outcome in time, not in total."""
  if max_lag <= 0:
    return np.array([1.0])
  weights = alpha ** np.arange(max_lag + 1)
  return weights / weights.sum()


def simulate(config: RecoveryConfig) -> tuple[pd.DataFrame, np.ndarray]:
  """Generates data whose true ROI is known.

  Args:
    config: Experiment shape.

  Returns:
    A tidy DataFrame, and the REALISED true ROI of each channel over the
    modelling window, which is the quantity Meridian reports. With carryover
    this differs slightly from `config.true_roi`, because burn-in spend
    contributes incremental outcome inside the window while spend near the end
    contributes outside it. Recovery is judged against the realised figure.
  """
  rng = np.random.default_rng(config.seed)
  n_channels = len(config.true_roi)
  true_roi = np.asarray(config.true_roi, dtype=float)
  n_media_times = config.n_media_times

  dates = pd.date_range('2023-01-02', periods=n_media_times, freq='W-MON')
  dates = dates.strftime('%Y-%m-%d').tolist()
  window = set(dates[config.max_lag :])

  geos = [f'geo_{i}' for i in range(config.n_geos)]
  population = np.linspace(5e5, 2e6, config.n_geos)

  spend = rng.gamma(
      shape=4.0, scale=1.0, size=(config.n_geos, n_media_times, n_channels)
  ) * np.asarray(config.spend_scale)

  if config.response == 'concave':
    # Rescaled so total incremental outcome, and so true ROI, is unchanged.
    shaped = np.sqrt(spend)
    shaped = shaped * (spend.sum(axis=(0, 1)) / shaped.sum(axis=(0, 1)))
  else:
    shaped = spend

  contribution = shaped * true_roi
  weights = _decay_weights(config.max_lag, config.alpha)
  incremental = np.zeros_like(contribution)
  for lag, weight in enumerate(weights):
    incremental[:, lag:, :] += (
        weight * contribution[:, : n_media_times - lag, :]
    )

  baseline_level = 6.0e5 * (population / population.mean())
  trend = 1.0 + 0.15 * np.sin(np.linspace(0, 4 * np.pi, n_media_times))
  control = rng.normal(size=(config.n_geos, n_media_times))
  baseline = baseline_level[:, None] * trend[None, :] * (1.0 + 0.03 * control)

  revenue = baseline + incremental.sum(axis=-1)
  revenue = revenue + rng.normal(
      scale=config.noise_fraction * baseline_level.mean(),
      size=(config.n_geos, n_media_times),
  )
  revenue = np.maximum(revenue, 1.0)

  win = slice(config.max_lag, n_media_times)
  true_roi_window = incremental[:, win, :].sum(axis=(0, 1)) / spend[
      :, win, :
  ].sum(axis=(0, 1))

  rows = []
  for gi, geo in enumerate(geos):
    for ti, date in enumerate(dates):
      inside = date in window
      row = {
          'geo': geo,
          'date': date,
          'population': population[gi],
          # KPI, controls and spend exist only in the modelling window;
          # impressions extend earlier to supply carryover history.
          'revenue': revenue[gi, ti] if inside else np.nan,
          'control_1': control[gi, ti] if inside else np.nan,
      }
      for ci, channel in enumerate(config.channels):
        row[f'{channel}_impressions'] = spend[gi, ti, ci] * 50.0
        row[f'{channel}_spend'] = spend[gi, ti, ci] if inside else np.nan
      rows.append(row)

  return pd.DataFrame(rows), true_roi_window


def _build_model(df: pd.DataFrame, config: RecoveryConfig) -> model.Meridian:
  data = (
      dfb.DataFrameInputDataBuilder(kpi_type=c.REVENUE)
      .with_kpi(
          df.dropna(subset=['revenue']),
          kpi_col='revenue',
          time_col='date',
          geo_col='geo',
      )
      .with_population(df, population_col='population', geo_col='geo')
      .with_controls(
          df.dropna(subset=['control_1']),
          control_cols=['control_1'],
          time_col='date',
          geo_col='geo',
      )
      .with_media(
          df,
          media_cols=[f'{ch}_impressions' for ch in config.channels],
          media_spend_cols=[f'{ch}_spend' for ch in config.channels],
          media_channels=config.channels,
          time_col='date',
          geo_col='geo',
      )
      .build()
  )
  prior = prior_distribution.PriorDistribution(
      roi_m=backend.tfd.LogNormal(
          backend.np_float_dtype(np.log(config.prior_roi_median)),
          backend.np_float_dtype(config.prior_roi_sigma),
          name=c.ROI_M,
      )
  )
  return model.Meridian(
      input_data=data,
      model_spec=spec.ModelSpec(
          prior=prior, media_prior_type=c.ROI, max_lag=config.max_lag
      ),
  )


def _fit_and_recover(
    config: RecoveryConfig,
) -> tuple[RecoveryResult, np.ndarray, np.ndarray]:
  """Simulates and fits one replication.

  Factored out of `run_recovery` so that `run_recovery_replications` can
  reuse the same fit for its rank fractions (see `ChannelReplicationSummary`)
  without fitting twice.

  Returns:
    A tuple `(result, draws, true_roi)`: the single-run `RecoveryResult`
    (identical to what `run_recovery` returns), the posterior ROI draws with
    shape `(n_draws, n_channels)`, and the realised true ROI per channel used
    for the descriptive rank fraction.
  """
  df, true_roi = simulate(config)
  mmm = _build_model(df, config)

  mmm.sample_prior(500, seed=config.seed)
  mmm.sample_posterior(
      n_chains=config.n_chains,
      n_adapt=config.n_adapt,
      n_burnin=config.n_burnin,
      n_keep=config.n_keep,
      seed=config.seed,
  )

  import arviz as az  # pylint: disable=g-import-not-at-top

  r_hat = az.rhat(  # pytype: disable=attribute-error
      mmm.inference_data.posterior, method='rank'
  )
  # Deterministic parameters -- hierarchical terms pinned to zero in a national
  # model, for instance -- have no between-chain variance, so arviz returns
  # all-NaN r_hat for them. Drop those rather than reducing over a NaN slice.
  per_variable = []
  for name in r_hat.data_vars:
    values = np.asarray(r_hat[name].values, dtype=float)
    non_nan = values[~np.isnan(values)]
    if non_nan.size:
      per_variable.append(float(non_nan.max()))
  # NaN when every parameter is deterministic; `converged` then reads False,
  # which is the safe direction.
  max_r_hat = max(per_variable) if per_variable else float('nan')

  roi = np.asarray(
      analyzer_module.Analyzer(
          model_context=mmm.model_context,
          inference_data=mmm.inference_data,
      ).roi()
  )
  draws = roi.reshape(-1, roi.shape[-1])
  tail = (1.0 - config.confidence_level) / 2.0
  low = np.quantile(draws, tail, axis=0)
  high = np.quantile(draws, 1.0 - tail, axis=0)
  median = np.median(draws, axis=0)

  channels = tuple(
      ChannelRecovery(
          channel=name,
          true_roi=float(true_roi[i]),
          median=float(median[i]),
          ci_low=float(low[i]),
          ci_high=float(high[i]),
      )
      for i, name in enumerate(config.channels)
  )
  result = RecoveryResult(config=config, channels=channels, max_r_hat=max_r_hat)
  return result, draws, true_roi


def run_recovery(config: RecoveryConfig | None = None) -> RecoveryResult:
  """Simulates, fits, and reports whether the true ROI was recovered.

  This is the single-replication path: it ignores `config.replications` and
  always runs exactly one simulate-and-fit, exactly as before this module
  gained repeated-replication support. For `replications > 1`, use
  `run_recovery_replications`.
  """
  config = config or RecoveryConfig()
  result, _, _ = _fit_and_recover(config)
  return result


def derive_replication_seeds(seed: int, replications: int) -> tuple[int, ...]:
  """Deterministically derives one child seed per replication.

  Uses `numpy.random.SeedSequence(seed).spawn(replications)`, NumPy's
  documented mechanism for generating independent, reproducible streams from
  a base seed. Each spawned `SeedSequence` is reduced to a single Python int
  (via `generate_state`) because both `simulate` and the backend's MCMC
  sampler take a plain int seed, not a `SeedSequence`.

  The same `seed` and `replications` always produce the same child seeds, so
  a multi-replication run is as reproducible as the single-run path.

  Args:
    seed: Base seed, e.g. `RecoveryConfig.seed`.
    replications: Number of child seeds to derive.

  Returns:
    A tuple of `replications` distinct int seeds.
  """
  parent = np.random.SeedSequence(seed)
  children = parent.spawn(replications)
  # `generate_state` returns unsigned 64-bit words; fold each one into
  # [0, 2**32 - 1). Both backends' seeding ultimately bottoms out in
  # `numpy.random.seed`, which the TensorFlow backend calls directly and
  # which requires a uint32 -- so the derived seeds have to fit that range
  # even though `simulate`'s own `default_rng` would accept a wider one.
  return tuple(
      int(child.generate_state(1, dtype=np.uint64)[0] % (2**32 - 1))
      for child in children
  )


def _wilson_interval(
    successes: int, n: int, z: float = 1.959963984540054
) -> tuple[float, float]:
  """Wilson score interval for a binomial proportion.

  Preferred over the normal (Wald) approximation because it stays inside
  [0, 1] and remains reasonable at the small N and near-0/1 proportions a
  handful of replications can produce. `z` defaults to the 97.5th percentile
  of the standard normal, i.e. a 95% interval.

  Args:
    successes: Number of replications where the true value was covered.
    n: Number of replications.
    z: Normal quantile for the desired interval width.

  Returns:
    `(low, high)`, clipped to [0, 1].
  """
  if n == 0:
    return float('nan'), float('nan')
  phat = successes / n
  denom = 1.0 + z**2 / n
  center = phat + z**2 / (2 * n)
  margin = z * np.sqrt(phat * (1.0 - phat) / n + z**2 / (4 * n**2))
  low = (center - margin) / denom
  high = (center + margin) / denom
  return max(0.0, float(low)), min(1.0, float(high))


def _ks_statistic_vs_uniform(values: np.ndarray) -> float:
  """Returns a legacy, uncalibrated distance from Uniform(0, 1).

  `rank_ks_statistic` is retained in `ChannelReplicationSummary` for API
  compatibility. This recovery experiment fixes the data-generating truth
  instead of drawing it from the fitting prior, so this value is not an SBC
  statistic and has no calibrated uniform-null threshold. Use the rank-fraction
  summaries instead.
  """
  x = np.sort(np.asarray(values, dtype=float))
  n = x.size
  if n == 0:
    return float('nan')
  ecdf_upper = np.arange(1, n + 1) / n
  ecdf_lower = np.arange(0, n) / n
  return float(max(np.max(ecdf_upper - x), np.max(x - ecdf_lower)))


@dataclasses.dataclass(frozen=True)
class ChannelReplicationSummary:
  """One channel's recovery statistics aggregated across replications.

  `rank_ks_statistic` and `rank_ks_threshold` are retained for API
  compatibility. The former is an uncalibrated descriptive distance from a
  uniform distribution; the latter is `NaN` because a fixed-truth recovery
  experiment has no valid uniform-null threshold. Read the rank-fraction
  summaries rather than using either field as an SBC test.
  """

  channel: str
  n: int
  median_relative_error: float
  iqr_low: float
  iqr_high: float
  coverage: float
  coverage_ci_low: float
  coverage_ci_high: float
  median_ci_width: float
  rank_ks_statistic: float
  rank_ks_threshold: float
  rank_fraction_median: float = float('nan')
  rank_fraction_iqr_low: float = float('nan')
  rank_fraction_iqr_high: float = float('nan')


@dataclasses.dataclass(frozen=True)
class MultiRecoveryResult:
  """Outcome of N repeated simulate-and-fit replications.

  Attributes:
    config: The base config passed to `run_recovery_replications`. Its
      `.seed` is the base seed the per-replication seeds were derived from,
      not any individual replication's seed -- see `seeds`.
    seeds: The per-replication seeds, in replication order, as derived by
      `derive_replication_seeds`.
    replications: Each replication's full single-run `RecoveryResult`, in the
      same order as `seeds`.
    ranks: Per replication, per channel, the descriptive rank fraction -- the
      fraction of that replication's posterior ROI draws falling below the
      fixed, realised true value. This is not an SBC rank. Shape
      `(len(replications), n_channels)`.
  """

  config: RecoveryConfig
  seeds: tuple[int, ...]
  replications: tuple[RecoveryResult, ...]
  ranks: tuple[tuple[float, ...], ...]

  @property
  def n(self) -> int:
    return len(self.replications)

  @property
  def all_converged(self) -> bool:
    return all(rep.converged for rep in self.replications)

  @property
  def passed(self) -> bool:
    """Whether every replication converged.

    Deliberately not a claim about coverage: whether empirical coverage is
    "good enough" is a judgment call for whoever reads the report, not a
    threshold this module bakes in.
    """
    return self.all_converged

  def channel_summaries(self) -> tuple[ChannelReplicationSummary, ...]:
    """Aggregates per-channel statistics across all replications."""
    summaries = []
    for i, channel in enumerate(self.config.channels):
      errors = np.array(
          [rep.channels[i].relative_error for rep in self.replications]
      )
      covered = np.array([rep.channels[i].covered for rep in self.replications])
      widths = np.array(
          [
              rep.channels[i].ci_high - rep.channels[i].ci_low
              for rep in self.replications
          ]
      )
      rank_fracs = np.array([rep_ranks[i] for rep_ranks in self.ranks])
      coverage_low, coverage_high = _wilson_interval(
          int(covered.sum()), covered.size
      )
      summaries.append(
          ChannelReplicationSummary(
              channel=channel,
              n=self.n,
              median_relative_error=float(np.median(errors)),
              iqr_low=float(np.quantile(errors, 0.25)),
              iqr_high=float(np.quantile(errors, 0.75)),
              coverage=float(covered.mean()),
              coverage_ci_low=coverage_low,
              coverage_ci_high=coverage_high,
              median_ci_width=float(np.median(widths)),
              rank_ks_statistic=_ks_statistic_vs_uniform(rank_fracs),
              rank_ks_threshold=float('nan'),
              rank_fraction_median=float(np.median(rank_fracs)),
              rank_fraction_iqr_low=float(np.quantile(rank_fracs, 0.25)),
              rank_fraction_iqr_high=float(np.quantile(rank_fracs, 0.75)),
          )
      )
    return tuple(summaries)

  def to_frame(self) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                'channel': s.channel,
                'n': s.n,
                'median_relative_error': s.median_relative_error,
                'iqr_low': s.iqr_low,
                'iqr_high': s.iqr_high,
                'coverage': s.coverage,
                'coverage_ci_low': s.coverage_ci_low,
                'coverage_ci_high': s.coverage_ci_high,
                'median_ci_width': s.median_ci_width,
                'rank_ks_statistic': s.rank_ks_statistic,
                'rank_ks_threshold': s.rank_ks_threshold,
                'rank_fraction_median': s.rank_fraction_median,
                'rank_fraction_iqr_low': s.rank_fraction_iqr_low,
                'rank_fraction_iqr_high': s.rank_fraction_iqr_high,
            }
            for s in self.channel_summaries()
        ]
    )

  def format_report(self) -> str:
    nominal = self.config.confidence_level
    n_converged = sum(1 for rep in self.replications if rep.converged)
    lines = [
        f'Recovery (repeated): response={self.config.response}'
        f' max_lag={self.config.max_lag}'
        f' n_geos={self.config.n_geos} n_times={self.config.n_times}'
        f' replications={self.n}',
        f'base seed={self.config.seed}, per-replication seeds derived via'
        f' SeedSequence({self.config.seed}).spawn({self.n})',
        '-' * 78,
        f'{n_converged}/{self.n} replications converged'
        ' (max rank-normalized r_hat < 1.2)',
        '',
        f"{'channel':<12}{'med.err':>9}{'IQR':>18}"
        f"{f'cov({nominal:.0%})':>11}{'95% Wilson CI':>17}{'med.width':>11}"
        f"{'rank med.':>11}",
    ]
    for s in self.channel_summaries():
      lines.append(
          f'{s.channel:<12}{s.median_relative_error:+8.1%} '
          f' [{s.iqr_low:+6.1%},{s.iqr_high:+6.1%}]'
          f'{s.coverage:10.0%}'
          f'  [{s.coverage_ci_low:5.0%},{s.coverage_ci_high:5.0%}]'
          f'{s.median_ci_width:11.3f}'
          f'{s.rank_fraction_median:11.3f}'
      )
    lines += [
        '',
        'coverage = fraction of replications where the true ROI fell inside'
        f' the nominal {nominal:.0%} interval; the Wilson CI is a 95%'
        ' interval on that coverage estimate itself, not on ROI -- with only'
        f' {self.n} replications, coverage is a noisy estimate, not a'
        ' precise one.',
        'rank fraction = fraction of posterior ROI draws below the fixed,'
        ' realised true ROI. This is a recovery experiment, not SBC: truth is'
        ' configured rather than drawn from the fitting prior, so ranks are'
        ' descriptive and do not have a Uniform(0, 1) calibration target.',
        f'VERDICT: {"ALL CONVERGED" if self.passed else "NOT ALL CONVERGED"}'
        ' -- whether the coverage above is acceptable is a judgment call for'
        ' the reader, not something this module scores pass/fail.',
    ]
    return '\n'.join(lines)


def run_recovery_replications(
    config: RecoveryConfig,
    progress: Callable[[str], None] | None = None,
) -> MultiRecoveryResult:
  """Runs `config.replications` independent simulate-and-fit replications.

  Each replication uses a distinct seed derived from `config.seed` (see
  `derive_replication_seeds`) and is otherwise a full, independent
  `run_recovery`-equivalent fit: fresh simulated dataset, fresh model, fresh
  MCMC run. Runtime therefore scales linearly with `config.replications` --
  this prints a per-replication progress line and a running estimate of
  total runtime so that is visible before the run finishes.

  Args:
    config: Experiment shape, with `config.replications` replications.
    progress: Callback for progress lines; defaults to printing to stderr so
      stdout stays free for the final report. Pass a no-op to silence it.

  Returns:
    The aggregated `MultiRecoveryResult`.
  """
  if progress is None:
    progress = lambda line: print(line, file=sys.stderr)

  n = config.replications
  seeds = derive_replication_seeds(config.seed, n)
  progress(
      f'Running {n} replications from base seed={config.seed}'
      f' (seeds derived via SeedSequence.spawn). Each simulate+fit takes'
      ' minutes; an estimated total appears after replication 1.'
  )

  results = []
  ranks = []
  start = time.monotonic()
  for i, seed_i in enumerate(seeds):
    rep_config = dataclasses.replace(config, seed=seed_i, replications=1)
    rep_start = time.monotonic()
    result, draws, true_roi = _fit_and_recover(rep_config)
    rep_elapsed = time.monotonic() - rep_start
    results.append(result)
    ranks.append(
        tuple(
            float((draws[:, c_i] < true_roi[c_i]).mean())
            for c_i in range(len(config.channels))
        )
    )

    elapsed_total = time.monotonic() - start
    avg = elapsed_total / (i + 1)
    remaining = avg * (n - i - 1)
    status = 'converged' if result.converged else 'NOT CONVERGED'
    progress(
        f'replication {i + 1}/{n}: {rep_elapsed:.0f}s, seed={seed_i},'
        f' max r_hat={result.max_r_hat:.3f} ({status}) | elapsed'
        f' {elapsed_total / 60:.1f} min, est. remaining'
        f' {remaining / 60:.1f} min, est. total'
        f' {(elapsed_total + remaining) / 60:.1f} min'
    )

  return MultiRecoveryResult(
      config=config,
      seeds=seeds,
      replications=tuple(results),
      ranks=tuple(ranks),
  )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      prog='python -m meridian.validation',
      description='Meridian ROI recovery check.',
  )
  defaults = RecoveryConfig()
  parser.add_argument('--n-geos', type=int, default=defaults.n_geos)
  parser.add_argument('--n-times', type=int, default=defaults.n_times)
  parser.add_argument('--max-lag', type=int, default=defaults.max_lag)
  parser.add_argument(
      '--response',
      choices=['concave', 'linear'],
      default=defaults.response,
      help="'linear' deliberately violates Meridian's saturation assumption.",
  )
  parser.add_argument(
      '--noise-fraction', type=float, default=defaults.noise_fraction
  )
  parser.add_argument('--n-chains', type=int, default=defaults.n_chains)
  parser.add_argument('--n-adapt', type=int, default=defaults.n_adapt)
  parser.add_argument('--n-burnin', type=int, default=defaults.n_burnin)
  parser.add_argument('--n-keep', type=int, default=defaults.n_keep)
  parser.add_argument('--seed', type=int, default=defaults.seed)
  parser.add_argument(
      '--true-roi',
      type=float,
      nargs='+',
      default=list(defaults.true_roi),
      help='True ROI per channel.',
  )
  parser.add_argument(
      '--replications',
      type=int,
      default=defaults.replications,
      help=(
          'Number of independent simulate-and-fit replications. 1 (default)'
          ' runs the original single-fit check. N > 1 derives N seeds from'
          ' --seed, fits each independently, and reports empirical coverage'
          ' with a Wilson interval on it -- see the module docstring.'
      ),
  )
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = _parse_args(argv)
  true_roi = tuple(args.true_roi)
  defaults = RecoveryConfig()
  # Reuse the default spend scales, extending if more channels were asked for.
  scales = tuple(
      defaults.spend_scale[i % len(defaults.spend_scale)]
      for i in range(len(true_roi))
  )
  config = RecoveryConfig(
      true_roi=true_roi,
      spend_scale=scales,
      n_geos=args.n_geos,
      n_times=args.n_times,
      max_lag=args.max_lag,
      response=args.response,
      noise_fraction=args.noise_fraction,
      n_chains=args.n_chains,
      n_adapt=args.n_adapt,
      n_burnin=args.n_burnin,
      n_keep=args.n_keep,
      seed=args.seed,
      replications=args.replications,
  )
  if config.replications <= 1:
    # Unchanged path: same call, same report, same exit-code rule as before
    # this module supported repeated replications.
    result = run_recovery(config)
    print(result.format_report())
    return 0 if result.passed else 1

  multi_result = run_recovery_replications(config)
  print(multi_result.format_report())
  return 0 if multi_result.passed else 1


if __name__ == '__main__':
  raise SystemExit(main())
