#!/usr/bin/env python3
# Copyright 2026 Sundar Ramesh Kumar.
# Licensed under the Apache License, Version 2.0.
# NOTICE: This file is new in this fork and does not exist in the original
# google/meridian source.

"""Run reproducible fixed-truth recovery replications in fresh processes.

The recovery module's in-process repeated runner is useful for exploration,
but numerical backends retain compiled state.  This script gives an audit or
user study a small, reproducible command: each replication is a new Python
process, receives a seed from ``derive_replication_seeds``, and writes its
single-run dataclass result before the next fit starts.

The reported R-hat comparison is a screening check only.  Fixed-truth ranks
are descriptive and are not simulation-based calibration (SBC).
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from scripts.evidence import (  # pylint: disable=g-import-not-at-top,g-bad-import-order
    git_head as _git_head,
    json_safe as _json_safe,
    package_versions as _package_versions,
    sha256 as _sha256,
)
_RECOVERY_SOURCE = Path('meridian/validation/recovery.py')
_R_HAT_SCREENING_THRESHOLD = 1.2
_DEFAULT_OUTPUT_DIR = Path('recovery-study')
_STUDY_ARTIFACT_NAMES = frozenset(
    ('metadata.json', 'aggregate.json', 'manifest.json', 'report.txt')
)


def _load_recovery():
  """Loads the recovery module lazily so orchestration tests stay lightweight."""
  if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
  from meridian.validation import recovery  # pylint: disable=g-import-not-at-top

  return recovery


def _positive_integer(value: str) -> int:
  try:
    parsed = int(value)
  except ValueError as error:
    raise argparse.ArgumentTypeError('must be a positive integer') from error
  if parsed < 1:
    raise argparse.ArgumentTypeError('must be a positive integer')
  return parsed


def _write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False)
      + '\n'
  )


def _read_json(path: Path) -> Any:
  return json.loads(path.read_text())


def _backend_info(recovery_module) -> dict[str, str | None]:
  try:
    backend = recovery_module.backend.computation_backend().name
  except Exception:  # pylint: disable=broad-exception-caught
    backend = None
  try:
    precision = recovery_module.backend.computation_precision().name
  except Exception:  # pylint: disable=broad-exception-caught
    precision = None
  return {'backend': backend, 'precision': precision}


def _config_as_dict(config) -> dict[str, Any]:
  return dataclasses.asdict(config)


def _config_from_dict(recovery_module, data: Mapping[str, Any]):
  values = dict(data)
  for field in ('true_roi', 'spend_scale'):
    if field in values:
      values[field] = tuple(values[field])
  return recovery_module.RecoveryConfig(**values)


def _float_or_nan(value: Any) -> float:
  return float('nan') if value is None else float(value)


def _result_from_payload(recovery_module, payload: Mapping[str, Any]):
  result_data = payload['result']
  channels = tuple(
      recovery_module.ChannelRecovery(
          channel=channel['channel'],
          true_roi=_float_or_nan(channel['true_roi']),
          median=_float_or_nan(channel['median']),
          ci_low=_float_or_nan(channel['ci_low']),
          ci_high=_float_or_nan(channel['ci_high']),
      )
      for channel in result_data['channels']
  )
  return recovery_module.RecoveryResult(
      config=_config_from_dict(recovery_module, result_data['config']),
      channels=channels,
      max_r_hat=_float_or_nan(result_data.get('max_r_hat')),
  )


def _redact_log(text: str) -> str:
  """Removes local checkout/interpreter paths from retained child logs."""
  replacements = {
      str(REPO_ROOT): '$REPO_ROOT',
      str(Path(sys.prefix)): '$PYTHON_ENV',
  }
  for source, replacement in replacements.items():
    text = text.replace(source, replacement)
  return text


def _run_child(arguments: Sequence[str]) -> int:
  """Fits one replication for the parent process and writes its payload."""
  if len(arguments) != 4:
    raise ValueError('child mode expects index, seed, result path, and config')
  recovery_module = _load_recovery()
  index = int(arguments[0])
  seed = int(arguments[1])
  output_path = Path(arguments[2])
  config_data = json.loads(arguments[3])
  config = _config_from_dict(recovery_module, config_data)
  started = time.monotonic()
  result, draws, true_roi = recovery_module._fit_and_recover(config)
  elapsed = time.monotonic() - started
  payload = {
      'dataclass': 'meridian.validation.recovery.RecoveryResult',
      'replication_index': index,
      'seed': seed,
      'elapsed_seconds': elapsed,
      'backend': recovery_module.backend.computation_backend().name,
      'precision': recovery_module.backend.computation_precision().name,
      'config': dataclasses.asdict(config),
      'result': dataclasses.asdict(result),
      'ranks': [
          float((draws[:, i] < true_roi[i]).mean())
          for i in range(len(config.channels))
      ],
      'draws_shape': list(draws.shape),
  }
  _write_json(output_path, payload)
  print(
      f'replication {index}: seed={seed} elapsed={elapsed:.1f}s '
      f'max_r_hat={result.max_r_hat:.6f}',
      flush=True,
  )
  return 0


def _child_command(
    python_executable: str,
    index: int,
    seed: int,
    result_path: Path,
    config: Mapping[str, Any],
) -> list[str]:
  return [
      python_executable,
      '-u',
      str(Path(__file__).resolve()),
      '--child',
      str(index),
      str(seed),
      str(result_path),
      json.dumps(_json_safe(config), sort_keys=True, allow_nan=False),
  ]


def _screen_description() -> str:
  return (
      'finite ArviZ rank-normalized r_hat < 1.2 is a screening check only; '
      'it is not an unqualified convergence claim or a calibration test.'
  )


def _rank_description() -> str:
  return (
      'Rank fractions are descriptive fixed-truth recovery output. Truth is '
      'configured rather than drawn from the fitting prior, so this study is '
      'not simulation-based calibration (SBC).'
  )


def _coverage_description(n: int) -> str:
  return (
      f'Coverage is empirical at this configured truth, data shape, prior, '
      f'and response. With {n} replications, the Wilson interval is broad '
      'and does not establish a general interval-coverage guarantee.'
  )


def _metadata(recovery_module, config, seeds: Sequence[int]) -> dict[str, Any]:
  source_hash = None
  module_file = getattr(recovery_module, '__file__', None)
  if module_file:
    try:
      source_hash = _sha256(Path(module_file))
    except OSError:
      source_hash = None
  info = _backend_info(recovery_module)
  return {
      'schema': 'meridian.recovery-study.v1',
      'status': 'running',
      'started_at_utc': datetime.now(timezone.utc).isoformat(),
      'git_head': _git_head(),
      'source': {
          'recovery_module': _RECOVERY_SOURCE.as_posix(),
          'recovery_module_sha256': source_hash,
      },
      'package_versions': _package_versions(),
      'python_version': platform.python_version(),
      'platform': platform.platform(),
      'backend': info['backend'],
      'precision': info['precision'],
      'config': _config_as_dict(config),
      'derived_seeds': list(seeds),
      'fresh_subprocess_per_replication': True,
      'r_hat_screening': _screen_description(),
      'rank_note': _rank_description(),
      'coverage_note': _coverage_description(config.replications),
  }


def _artifact_name(index: int, suffix: str = '') -> str:
  return f'replication-{index:03d}{suffix}'


def _existing_study_artifacts(output_dir: Path) -> tuple[str, ...]:
  if not output_dir.is_dir():
    return ()
  names = []
  for path in output_dir.iterdir():
    if path.name in _STUDY_ARTIFACT_NAMES or (
        path.name.startswith('replication-')
        and path.name.endswith(('.json', '.stdout.log', '.stderr.log'))
    ):
      names.append(path.name)
  return tuple(sorted(names))


def _prepare_output_dir(output_dir: Path) -> None:
  """Creates an empty evidence directory and protects prior evidence."""
  if output_dir.exists() and not output_dir.is_dir():
    raise RuntimeError(f'output path is not a directory: {output_dir}')
  existing = _existing_study_artifacts(output_dir)
  if existing:
    shown = ', '.join(existing[:5])
    if len(existing) > 5:
      shown += f', ... ({len(existing)} total)'
    raise RuntimeError(
        'output directory already contains recovery-study artifacts: '
        f'{shown}; choose a new --output-dir to preserve the existing evidence'
    )
  output_dir.mkdir(parents=True, exist_ok=True)


def run_study(
    config,
    output_dir: Path,
    *,
    recovery_module=None,
    python_executable: str | None = None,
    run_process: Callable[..., subprocess.CompletedProcess] | None = None,
) -> dict[str, Any]:
  """Runs the configured study and writes its evidence bundle.

  Args:
    config: A `recovery.RecoveryConfig` with `replications` > 0.
    output_dir: Directory for JSON results, redacted logs, and the manifest.
    recovery_module: Optional injected recovery module for lightweight tests.
    python_executable: Interpreter used for each fresh child process.
    run_process: Optional injected `subprocess.run` for orchestration tests.

  Returns:
    The JSON-safe aggregate mapping. A child process failure raises
    `RuntimeError` after writing the partial-run metadata.
  """
  recovery_module = recovery_module or _load_recovery()
  python_executable = python_executable or sys.executable
  run_process = run_process or subprocess.run
  output_dir = Path(output_dir).expanduser().resolve()
  _prepare_output_dir(output_dir)
  seeds = tuple(
      recovery_module.derive_replication_seeds(config.seed, config.replications)
  )
  if len(seeds) != config.replications:
    raise RuntimeError(
        'derive_replication_seeds returned the wrong number of seeds'
    )

  metadata = _metadata(recovery_module, config, seeds)
  metadata_path = output_dir / 'metadata.json'
  _write_json(metadata_path, metadata)
  payloads = []
  results = []
  ranks = []
  started = time.monotonic()
  try:
    for index, seed in enumerate(seeds, 1):
      stem = _artifact_name(index)
      result_path = output_dir / f'{stem}.json'
      stdout_path = output_dir / f'{stem}.stdout.log'
      stderr_path = output_dir / f'{stem}.stderr.log'
      # A rerun may reuse the directory. Do not let a stale result mask a
      # child that exits successfully without writing its required artifact.
      result_path.unlink(missing_ok=True)
      child_config = dataclasses.replace(config, seed=seed, replications=1)
      command = _child_command(
          python_executable,
          index,
          seed,
          result_path,
          _config_as_dict(child_config),
      )
      print(
          f'BEGIN replication {index}/{config.replications} seed={seed}',
          flush=True,
      )
      completed = run_process(
          command,
          cwd=REPO_ROOT,
          env={
              **os.environ,
              'MPLCONFIGDIR': str(output_dir / 'mplconfig'),
              'MLFLOW_DISABLE_AGENT_HINT': '1',
          },
          capture_output=True,
          text=True,
          check=False,
      )
      stdout_path.write_text(_redact_log(completed.stdout or ''))
      stderr_path.write_text(_redact_log(completed.stderr or ''))
      if completed.returncode != 0:
        raise RuntimeError(
            f'fresh recovery subprocess failed for replication {index} '
            f'(exit {completed.returncode}); see {stderr_path.name}'
        )
      if not result_path.is_file():
        raise RuntimeError(
            f'replication {index} completed without writing {result_path.name}'
        )
      payload = _read_json(result_path)
      if (
          payload.get('replication_index') != index
          or payload.get('seed') != seed
      ):
        raise RuntimeError(
            f'replication {index} result identity does not match its derived seed'
        )
      payloads.append(payload)
      result = _result_from_payload(recovery_module, payload)
      results.append(result)
      replication_ranks = tuple(
          _float_or_nan(value) for value in payload.get('ranks', [])
      )
      if len(replication_ranks) != len(config.channels):
        raise RuntimeError(
            f'replication {index} result has {len(replication_ranks)} rank '
            f'values for {len(config.channels)} channels'
        )
      ranks.append(replication_ranks)
      print(
          f'END replication {index}/{config.replications} seed={seed} '
          f'max_r_hat={result.max_r_hat:.6f}',
          flush=True,
      )
  except Exception as error:
    metadata['status'] = 'failed'
    metadata['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    metadata['elapsed_seconds'] = time.monotonic() - started
    metadata['failure'] = _redact_log(str(error))
    _write_json(metadata_path, metadata)
    raise

  multi = recovery_module.MultiRecoveryResult(
      config=config,
      seeds=seeds,
      replications=tuple(results),
      ranks=tuple(ranks),
  )
  summaries = multi.channel_summaries()
  screen_pass = tuple(bool(result.converged) for result in results)
  report = (
      'R-hat comparison: '
      + _screen_description()
      + '\n'
      + multi.format_report()
  )
  aggregate = _json_safe(
      {
          'schema': 'meridian.recovery-study.v1',
          'dataclass': 'meridian.validation.recovery.MultiRecoveryResult',
          'config': _config_as_dict(config),
          'seeds': list(seeds),
          'n': multi.n,
          'r_hat_screening': {
              'threshold': _R_HAT_SCREENING_THRESHOLD,
              'description': _screen_description(),
              'replications_passing': int(sum(screen_pass)),
              'all_replications_pass': all(screen_pass),
          },
          'rank_note': _rank_description(),
          'coverage_note': _coverage_description(multi.n),
          'replications': [
              {
                  'replication_index': payload['replication_index'],
                  'seed': payload['seed'],
                  'backend': payload.get('backend'),
                  'precision': payload.get('precision'),
                  'elapsed_seconds': payload.get('elapsed_seconds'),
                  'max_r_hat': result.max_r_hat,
                  'r_hat_screen_pass': result.converged,
                  'all_covered': result.all_covered,
                  'ordering_recovered': result.ordering_recovered,
              }
              for payload, result in zip(payloads, results)
          ],
          'channel_summaries': [
              dataclasses.asdict(summary) for summary in summaries
          ],
          'format_report': report,
      }
  )
  aggregate_path = output_dir / 'aggregate.json'
  report_path = output_dir / 'report.txt'
  _write_json(aggregate_path, aggregate)
  report_path.write_text(report + '\n')

  metadata['status'] = 'complete'
  metadata['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
  metadata['elapsed_seconds'] = time.monotonic() - started
  metadata['artifact_files'] = sorted(
      path.name
      for path in output_dir.iterdir()
      if path.is_file() and path.name != 'manifest.json'
  )
  _write_json(metadata_path, metadata)
  manifest = {
      'schema': 'meridian.recovery-study.v1',
      'metadata': metadata,
      'sha256': {
          path.name: _sha256(path)
          for path in sorted(output_dir.iterdir())
          if path.is_file() and path.name != 'manifest.json'
      },
  }
  _write_json(output_dir / 'manifest.json', manifest)
  return aggregate


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--replications', type=_positive_integer, default=10)
  parser.add_argument('--seed', type=int, default=7)
  parser.add_argument('--output-dir', type=Path, default=_DEFAULT_OUTPUT_DIR)
  parser.add_argument('--n-geos', type=_positive_integer, default=5)
  parser.add_argument('--n-times', type=_positive_integer, default=104)
  parser.add_argument('--max-lag', type=int, default=0)
  parser.add_argument(
      '--response', choices=('concave', 'linear'), default='concave'
  )
  parser.add_argument('--n-chains', type=_positive_integer, default=4)
  parser.add_argument('--n-adapt', type=int, default=1000)
  parser.add_argument('--n-burnin', type=int, default=500)
  parser.add_argument('--n-keep', type=_positive_integer, default=1000)
  parser.add_argument('--noise-fraction', type=float, default=0.05)
  parser.add_argument('--confidence-level', type=float, default=0.9)
  return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace, recovery_module=None):
  recovery_module = recovery_module or _load_recovery()
  return recovery_module.RecoveryConfig(
      n_geos=args.n_geos,
      n_times=args.n_times,
      max_lag=args.max_lag,
      response=args.response,
      n_chains=args.n_chains,
      n_adapt=args.n_adapt,
      n_burnin=args.n_burnin,
      n_keep=args.n_keep,
      noise_fraction=args.noise_fraction,
      confidence_level=args.confidence_level,
      seed=args.seed,
      replications=args.replications,
  )


def main(argv: Sequence[str] | None = None) -> int:
  args = parse_args(argv)
  recovery_module = _load_recovery()
  config = config_from_args(args, recovery_module)
  output_dir = args.output_dir.expanduser().resolve()
  aggregate = run_study(config, output_dir, recovery_module=recovery_module)
  print(aggregate['format_report'])
  print(f'EVIDENCE_DIR={output_dir}')
  return 0 if aggregate['r_hat_screening']['all_replications_pass'] else 1


if __name__ == '__main__':
  try:
    if len(sys.argv) > 1 and sys.argv[1] == '--child':
      raise SystemExit(_run_child(sys.argv[2:]))
    raise SystemExit(main())
  except KeyboardInterrupt:
    raise SystemExit(130) from None
