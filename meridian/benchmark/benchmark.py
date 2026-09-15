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

"""A reproducible performance benchmark for Meridian.

Addresses google/meridian#1396, which asked for a macOS benchmark suite. That
request listed "first-token latency" and "tokens/sec", which are language-model
metrics and do not exist here. This measures what actually governs how long a
Meridian job takes:

  * cold import time, measured in a fresh interpreter
  * `InputData` construction
  * model construction
  * `sample_prior` throughput
  * `sample_posterior` throughput, the dominant cost
  * peak resident memory
  * the backend and devices actually in use

It runs on any platform, not only macOS. Reporting the resolved backend and
device list matters because several upstream reports -- google/meridian#1746
(10 min on one T4, over an hour on another) and google/meridian#1585 -- turned
out to be environment differences that a benchmark record would have surfaced
immediately.

Run it as:

```
python -m meridian.benchmark.benchmark --n-geos 20 --n-times 156 --json out.json
```
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import dataclasses
import json
import platform
import resource
import subprocess
import sys
import time
from typing import Any


__all__ = [
    'BenchmarkConfig',
    'BenchmarkResult',
    'run_benchmark',
    'environment_info',
    'cold_import_seconds',
]


def _peak_rss_bytes() -> int:
  """Returns peak resident set size in bytes.

  `ru_maxrss` is bytes on macOS but kilobytes on Linux, so normalize.
  """
  peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  return peak if sys.platform == 'darwin' else peak * 1024


def cold_import_seconds(timeout: float = 300.0) -> float:
  """Measures `import meridian.model.model` in a fresh interpreter.

  Timing an import in-process reports ~0 once the module is already loaded,
  which is what happens when this benchmark runs via `-m`. A subprocess gives
  the real cold-start cost.

  Args:
    timeout: Seconds to wait for the subprocess.

  Returns:
    Import time in seconds, or NaN if the subprocess failed.
  """
  snippet = (
      'import time;'
      't=time.perf_counter();'
      'import meridian.model.model;'
      'print(time.perf_counter()-t)'
  )
  try:
    completed = subprocess.run(
        [sys.executable, '-c', snippet],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    return float(completed.stdout.strip().splitlines()[-1])
  except (
      subprocess.SubprocessError,
      ValueError,
      IndexError,
  ):
    return float('nan')


def environment_info() -> dict[str, Any]:
  """Collects versions, platform and resolved compute devices."""
  info: dict[str, Any] = {
      'python': sys.version.split()[0],
      'platform': platform.platform(),
      'machine': platform.machine(),
      'processor': platform.processor() or platform.machine(),
  }
  try:
    import meridian  # pylint: disable=g-import-not-at-top

    info['meridian'] = meridian.__version__
  except Exception as e:  # pylint: disable=broad-except
    info['meridian'] = f'unavailable: {e!r}'

  try:
    import jax  # pylint: disable=g-import-not-at-top

    info['jax'] = jax.__version__
    info['jax_devices'] = [str(d) for d in jax.devices()]
    info['jax_x64'] = bool(jax.config.read('jax_enable_x64'))
  except Exception as e:  # pylint: disable=broad-except
    info['jax'] = f'unavailable: {e!r}'

  try:
    import tensorflow as tf  # pylint: disable=g-import-not-at-top

    info['tensorflow'] = tf.__version__
    info['tf_gpus'] = [d.name for d in tf.config.list_physical_devices('GPU')]
  except Exception as e:  # pylint: disable=broad-except
    info['tensorflow'] = f'unavailable: {e!r}'

  try:
    import numpy as np  # pylint: disable=g-import-not-at-top

    info['numpy'] = np.__version__
  except Exception as e:  # pylint: disable=broad-except
    info['numpy'] = f'unavailable: {e!r}'

  return info


@dataclasses.dataclass(frozen=True)
class BenchmarkConfig:
  """Shape of the synthetic workload to benchmark.

  Attributes:
    n_geos: Number of geos. Use 1 for a national model.
    n_times: Number of time periods in the modeling window.
    n_media_channels: Number of paid media channels.
    n_controls: Number of control variables.
    max_lag: Adstock `max_lag`. Media history is sized to cover it.
    n_prior_draws: Draws for `sample_prior`.
    n_chains: MCMC chains.
    n_adapt: MCMC adaptation steps.
    n_burnin: MCMC burn-in steps.
    n_keep: MCMC retained draws per chain.
    seed: Seed for both sampling calls.
  """

  n_geos: int = 10
  n_times: int = 104
  n_media_channels: int = 4
  n_controls: int = 2
  max_lag: int = 8
  n_prior_draws: int = 100
  n_chains: int = 2
  n_adapt: int = 100
  n_burnin: int = 100
  n_keep: int = 100
  seed: int = 0

  @property
  def n_media_times(self) -> int:
    """Media history long enough to fill the adstock window."""
    return self.n_times + self.max_lag


@dataclasses.dataclass(frozen=True)
class BenchmarkResult:
  """Timings and environment for one benchmark run."""

  config: dict[str, Any]
  environment: dict[str, Any]
  timings_seconds: dict[str, float]
  throughput: dict[str, float]
  peak_rss_bytes: int

  def to_json(self, indent: int = 2) -> str:
    return json.dumps(dataclasses.asdict(self), indent=indent, default=str)

  def format_table(self) -> str:
    """Returns a readable report."""
    env = self.environment
    lines = [
        'Meridian benchmark',
        '=' * 58,
        f'  meridian     {env.get("meridian")}',
        f'  python       {env.get("python")}',
        f'  platform     {env.get("platform")}',
        f'  processor    {env.get("processor")}',
        f'  jax          {env.get("jax")}  devices={env.get("jax_devices")}',
        f'  jax x64      {env.get("jax_x64")}',
        f'  tensorflow   {env.get("tensorflow")}  gpus={env.get("tf_gpus")}',
        '',
        'Workload',
        '-' * 58,
    ]
    for key, value in self.config.items():
      lines.append(f'  {key:<20} {value}')
    lines += ['', 'Timings (seconds)', '-' * 58]
    for key, value in self.timings_seconds.items():
      lines.append(f'  {key:<20} {value:10.2f}')
    lines += ['', 'Throughput', '-' * 58]
    for key, value in self.throughput.items():
      lines.append(f'  {key:<20} {value:10.2f}')
    lines += [
        '',
        f'Peak RSS: {self.peak_rss_bytes / 1e9:.2f} GB',
    ]
    return '\n'.join(lines)


def _build_input_data(config: BenchmarkConfig):
  """Builds synthetic input data of the requested shape."""
  from meridian.data import test_utils as data_test_utils  # pylint: disable=g-import-not-at-top

  return data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
      n_geos=config.n_geos,
      n_times=config.n_times,
      n_media_times=config.n_media_times,
      n_media_channels=config.n_media_channels,
      n_controls=config.n_controls,
  )


def run_benchmark(config: BenchmarkConfig | None = None) -> BenchmarkResult:
  """Runs the benchmark and returns timings.

  Args:
    config: Workload shape. Defaults to a modest geo model.

  Returns:
    A `BenchmarkResult`.
  """
  config = config or BenchmarkConfig()
  timings: dict[str, float] = {}

  from meridian.model import model  # pylint: disable=g-import-not-at-top
  from meridian.model import spec  # pylint: disable=g-import-not-at-top

  timings['cold_import'] = cold_import_seconds()

  t0 = time.perf_counter()
  data = _build_input_data(config)
  timings['build_input_data'] = time.perf_counter() - t0

  t0 = time.perf_counter()
  mmm = model.Meridian(
      input_data=data, model_spec=spec.ModelSpec(max_lag=config.max_lag)
  )
  timings['build_model'] = time.perf_counter() - t0

  t0 = time.perf_counter()
  mmm.sample_prior(config.n_prior_draws, seed=config.seed)
  timings['sample_prior'] = time.perf_counter() - t0

  t0 = time.perf_counter()
  mmm.sample_posterior(
      n_chains=config.n_chains,
      n_adapt=config.n_adapt,
      n_burnin=config.n_burnin,
      n_keep=config.n_keep,
      seed=config.seed,
  )
  timings['sample_posterior'] = time.perf_counter() - t0
  timings['total'] = sum(
      v for k, v in timings.items() if k != 'cold_import'
  )

  total_mcmc_draws = config.n_chains * (
      config.n_adapt + config.n_burnin + config.n_keep
  )
  throughput = {
      'prior_draws_per_sec': (
          config.n_prior_draws / timings['sample_prior']
          if timings['sample_prior']
          else float('inf')
      ),
      'mcmc_draws_per_sec': (
          total_mcmc_draws / timings['sample_posterior']
          if timings['sample_posterior']
          else float('inf')
      ),
      'seconds_per_1k_mcmc_draws': (
          1000.0 * timings['sample_posterior'] / total_mcmc_draws
          if total_mcmc_draws
          else float('nan')
      ),
  }

  return BenchmarkResult(
      config=dataclasses.asdict(config),
      environment=environment_info(),
      timings_seconds=timings,
      throughput=throughput,
      peak_rss_bytes=_peak_rss_bytes(),
  )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  defaults = BenchmarkConfig()
  for field in dataclasses.fields(BenchmarkConfig):
    parser.add_argument(
        f'--{field.name.replace("_", "-")}',
        type=int,
        default=getattr(defaults, field.name),
        help=f'default: {getattr(defaults, field.name)}',
    )
  parser.add_argument(
      '--json', dest='json_path', default=None, help='Write results as JSON.'
  )
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = _parse_args(argv)
  config = BenchmarkConfig(**{
      f.name: getattr(args, f.name) for f in dataclasses.fields(BenchmarkConfig)
  })
  result = run_benchmark(config)
  print(result.format_table())
  if args.json_path:
    with open(args.json_path, 'w') as f:
      f.write(result.to_json())
    print(f'\nWrote {args.json_path}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
