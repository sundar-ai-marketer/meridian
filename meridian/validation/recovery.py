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

Recovery is good when the simulated response has the concave shape Meridian
assumes, and bad when it does not:

| response shape | premium-channel ROI error | true value inside 90% CI |
|---|---|---|
| concave        | +2.8%                     | yes |
| linear         | +71.4%                    | no  |

Cutting the noise tenfold on linear data did not fix it -- the interval
narrowed to about +/-3% and still missed every true value. Removing a
population confound from the generator did not fix it either. Only matching
the response shape did.

That is not a Meridian defect. It is what fitting a concave saturation curve
to a response that is not concave does, in any MMM that assumes saturation.
The consequence for reporting is the part worth internalising:

**Meridian's credible intervals quantify parameter uncertainty conditional on
the assumed saturation shape. They do not cover being wrong about that shape.**

So a channel far from saturation -- typically one at low spend, whose real
response is still close to linear -- can have its ROI overstated by tens of
percent, with an interval that excludes the truth and narrows as you add data.
Presenting such an interval as the total uncertainty overstates what the method
can support.

Use `response='linear'` here to see how large that effect is at your own data
shape, and treat the gap as a floor on the uncertainty you should carry into a
recommendation.

```
python -m meridian.validation.recovery --n-geos 5 --n-times 104 --response linear
```
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import dataclasses
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
    'simulate',
    'run_recovery',
]

ResponseShape = Literal['linear', 'concave']


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
    prior_roi_median: Median of the ROI prior applied to EVERY channel. Left
      deliberately uninformative so that separating the channels is the data's
      work, not the prior's.
    prior_roi_sigma: Log-scale standard deviation of that prior.
    seed: Seed for both simulation and sampling.
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

  def __post_init__(self):
    if len(self.true_roi) != len(self.spend_scale):
      raise ValueError(
          f'`true_roi` has {len(self.true_roi)} entries but `spend_scale` has'
          f' {len(self.spend_scale)}.'
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
    return pd.DataFrame([
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
    ])

  def format_report(self) -> str:
    lines = [
        f'Recovery: response={self.config.response}'
        f' max_lag={self.config.max_lag}'
        f' n_geos={self.config.n_geos} n_times={self.config.n_times}',
        '-' * 72,
        f'max r_hat = {self.max_r_hat:.4f}'
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
    shaped = shaped * (
        spend.sum(axis=(0, 1)) / shaped.sum(axis=(0, 1))
    )
  else:
    shaped = spend

  contribution = shaped * true_roi
  weights = _decay_weights(config.max_lag, config.alpha)
  incremental = np.zeros_like(contribution)
  for lag, weight in enumerate(weights):
    incremental[:, lag:, :] += weight * contribution[
        :, : n_media_times - lag, :
    ]

  baseline_level = 6.0e5 * (population / population.mean())
  trend = 1.0 + 0.15 * np.sin(np.linspace(0, 4 * np.pi, n_media_times))
  control = rng.normal(size=(config.n_geos, n_media_times))
  baseline = (
      baseline_level[:, None] * trend[None, :] * (1.0 + 0.03 * control)
  )

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


def run_recovery(config: RecoveryConfig | None = None) -> RecoveryResult:
  """Simulates, fits, and reports whether the true ROI was recovered."""
  config = config or RecoveryConfig()
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

  r_hat = az.rhat(mmm.inference_data.posterior)  # pytype: disable=attribute-error
  # Deterministic parameters -- hierarchical terms pinned to zero in a national
  # model, for instance -- have no between-chain variance, so arviz returns
  # all-NaN r_hat for them. Drop those rather than reducing over a NaN slice.
  per_variable = []
  for name in r_hat.data_vars:
    values = np.asarray(r_hat[name].values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size:
      per_variable.append(float(finite.max()))
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
  return RecoveryResult(
      config=config, channels=channels, max_r_hat=max_r_hat
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
  )
  result = run_recovery(config)
  print(result.format_report())
  return 0 if result.passed else 1


if __name__ == '__main__':
  raise SystemExit(main())
