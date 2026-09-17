#!/usr/bin/env python3
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

"""Checks that a Meridian environment is actually usable, and says how to fix
it when it is not.

Run this immediately after installing, before trusting anything:

    python scripts/verify_environment.py

Every check here corresponds to a real failure that cost time. The most
expensive was a `tensorflow-metal` plugin left over in the environment, which
makes `import meridian` die with a symbol-not-found error inside
`libmetal_plugin.dylib` and says nothing about the actual cause. The
accelerator check reports the same class of answer for `jax-metal`, which
registers a Metal device that cannot compile this model.

Exit code is 0 when everything required passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import pathlib
import platform
import re
import sys
import tomllib
from collections.abc import Sequence


# (module to import, distribution name, extra that provides it, required?)
_OPTIONAL_DEPENDENCIES = [
    ('mmm.v1', 'mmm-proto-schema', 'schema', True),
    ('meridian_geox', 'meridian-geox', 'geox', False),
    ('mlflow', 'mlflow', 'mlflow', False),
    ('googleapiclient', 'google-api-python-client', 'scenarioplanner', False),
]

_GREEN = 'PASS'
_RED = 'FAIL'
_AMBER = 'WARN'


def _normalized(name: str | None) -> str | None:
  """PEP 503 name normalization, so `-`/`_`/case differences do not read as
  two distributions."""
  if name is None:
    return None
  return re.sub(r'[-_.]+', '-', name).lower()


def _expected_distribution_name(repo_root: pathlib.Path) -> str | None:
  """Reads the current `[project].name` from the repo's pyproject.toml."""
  try:
    with (repo_root / 'pyproject.toml').open('rb') as f:
      return tomllib.load(f)['project']['name']
  except Exception:  # pylint: disable=broad-except
    return None


class Report:
  """Collects check outcomes."""

  def __init__(self) -> None:
    self.rows: list[tuple[str, str, str]] = []
    self.failed = False

  def add(self, status: str, name: str, detail: str = '') -> None:
    self.rows.append((status, name, detail))
    if status == _RED:
      self.failed = True

  def render(self) -> str:
    width = max(len(name) for _, name, _ in self.rows) + 2
    lines = []
    for status, name, detail in self.rows:
      line = f'  [{status}] {name:<{width}}{detail}'
      lines.append(line.rstrip())
    return '\n'.join(lines)


def check_python(report: Report) -> None:
  major, minor = sys.version_info[:2]
  version = f'{major}.{minor}.{sys.version_info[2]}'
  if (major, minor) < (3, 11):
    report.add(
        _RED,
        'Python version',
        f'{version} -- Meridian requires 3.11 or newer.',
    )
  elif (major, minor) > (3, 13):
    report.add(
        _AMBER,
        'Python version',
        f'{version} -- only 3.11 to 3.13 are declared supported.',
    )
  else:
    report.add(_GREEN, 'Python version', version)


def check_meridian_import(report: Report, repo_root: pathlib.Path) -> bool:
  try:
    meridian = importlib.import_module('meridian')
  except Exception as e:  # pylint: disable=broad-except
    detail = f'{type(e).__name__}: {e}'
    hint = ''
    if 'libmetal_plugin' in str(e) or 'tensorflow-plugins' in str(e):
      hint = (
          '  -> A leftover `tensorflow-metal` plugin is incompatible with the'
          ' installed TensorFlow. Fix: pip uninstall -y tensorflow-metal'
      )
    report.add(_RED, 'import meridian', detail)
    if hint:
      report.add(_RED, '', hint)
    return False

  report.add(_GREEN, 'import meridian', f'version {meridian.__version__}')

  # Importing a PyPI copy instead of this checkout means edits and
  # fixes in the working tree simply do not take effect.
  module_file = getattr(meridian, '__file__', None)
  if module_file is None:
    report.add(_AMBER, 'meridian source', 'namespace package, cannot locate')
    return True
  module_path = pathlib.Path(module_file).resolve()
  try:
    module_path.relative_to(repo_root.resolve())
  except ValueError:
    report.add(
        _RED,
        'meridian source',
        f'importing {module_path}, NOT this checkout at {repo_root}.'
        ' Fix: pip install -e ".[dev,schema]" from the repository root.',
    )
  else:
    report.add(_GREEN, 'meridian source', 'this checkout (editable install)')
  return True


def check_backend(report: Report) -> None:
  try:
    from meridian import backend  # pylint: disable=g-import-not-at-top

    dtype = backend.standardize_dtype(backend.float_dtype)
    report.add(_GREEN, 'backend float dtype', dtype)
    if dtype == 'float32':
      # float32 means the TensorFlow backend resolved; JAX defaults to 64-bit.
      report.add(
          _AMBER,
          'backend',
          'TensorFlow backend is active and is deprecated upstream. Unset'
          ' MERIDIAN_BACKEND to use JAX.',
      )
  except Exception as e:  # pylint: disable=broad-except
    report.add(_RED, 'backend', f'{type(e).__name__}: {e}')


def check_conflicting_packages(report: Report) -> None:
  """tensorflow-metal is the one that silently breaks the import."""
  try:
    metal = metadata.version('tensorflow-metal')
  except metadata.PackageNotFoundError:
    report.add(_GREEN, 'tensorflow-metal', 'not installed (correct)')
    return
  report.add(
      _RED,
      'tensorflow-metal',
      f'{metal} installed. It is not compatible with the pinned TensorFlow'
      ' and breaks `import meridian`. Fix: pip uninstall -y tensorflow-metal',
  )


def check_accelerator(report: Report) -> None:
  """Reports what this machine's accelerator can do for Meridian.

  On Apple Silicon the honest answer is "nothing yet", and saying so here
  saves the afternoon it otherwise takes to find out. Both Metal plugins were
  measured on an M4 Max on 17 September 2026; see
  `docs/validation/gpu-metal-*.json` for the operation-by-operation record and
  `scripts/gpu_validation.py --capability-only` to reproduce it on your own
  machine.
  """
  try:
    import jax  # pylint: disable=g-import-not-at-top

    backend = jax.default_backend()
    devices = [str(d) for d in jax.devices()]
  except Exception as e:  # pylint: disable=broad-except
    report.add(_RED, 'accelerator', f'{type(e).__name__}: {e}')
    return

  try:
    metal = metadata.version('jax-metal')
  except metadata.PackageNotFoundError:
    metal = None

  if metal is not None:
    # jax-metal registers a device and then fails to compile for it, so a
    # device list alone would read as success.
    report.add(
        _RED,
        'jax-metal',
        f'{metal} installed. It registers a Metal device that cannot run this'
        ' model: on the supported JAX range it rejects the compiler output'
        ' outright, and on the JAX version it was built for it cannot lower'
        ' matrix multiplication or Cholesky. Fix: pip uninstall -y jax-metal',
    )
    return

  if backend.lower() == 'cpu':
    detail = f'CPU ({", ".join(devices)})'
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
      detail += ' -- the supported path on Apple Silicon; no Metal GPU backend'
    report.add(_GREEN, 'compute device', detail)
    return

  report.add(_GREEN, 'compute device', f'{backend} ({", ".join(devices)})')


def check_duplicate_distribution(
    report: Report, repo_root: pathlib.Path
) -> None:
  """Fails if more than one installed distribution owns `import meridian`.

  This repo was renamed `google-meridian` -> `meridian-mmm-fork`. A venv
  created before the rename and reused afterwards can end up with both
  installed editable from the same source tree; which one Python resolves is
  then filesystem-order-dependent, not something to rely on.

  Uses `packages_distributions()`, which maps the top-level `meridian` import
  to every distribution that claims it, rather than hardcoding the two known
  names -- a future rename away from `meridian-mmm-fork` is caught the same
  way, and an editable install's own `*.egg-info` (which is not reliably on
  `sys.path` and so is not something `Distribution.files` can be trusted to
  see) never has to be located directly.
  """
  expected = _normalized(_expected_distribution_name(repo_root))
  # Deduplicate by normalized name. `packages_distributions()` reports one
  # entry per metadata directory that claims the import, so a single editable
  # install can appear twice -- once from its `*.dist-info` in site-packages
  # and once from the `*.egg-info` at the repo root, whenever the repo root is
  # on `sys.path`. Counting registrations rather than distributions made this
  # check's verdict depend on how it was invoked (`python scripts/x.py` put
  # `scripts/` on the path and passed; anything with the repo root on the path
  # failed) and told the user to uninstall the only copy they had.
  # A real conflict is two *different* names owning the same import.
  owners = sorted({
      _normalized(name)
      for name in metadata.packages_distributions().get('meridian', [])
  })
  if len(owners) <= 1:
    report.add(
        _GREEN,
        'meridian distribution',
        owners[0] if owners else 'not installed as a distribution',
    )
    return
  stale = [name for name in owners if name != expected]
  uninstall_targets = stale or owners[1:]
  report.add(
      _RED,
      'meridian distribution',
      f'multiple installed distributions own the `meridian` import path:'
      f' {", ".join(owners)}. Only one may be installed at a time.'
      f' Fix: pip uninstall -y {" ".join(uninstall_targets)}',
  )


def check_stale_egg_info(report: Report, repo_root: pathlib.Path) -> None:
  """Warns about a root `*.egg-info` left over from a previous project name.

  `make clean` removes these; a stale one from before a rename does not break
  anything by itself, but it is a sign the build tree was never cleaned.
  """
  expected = _expected_distribution_name(repo_root)
  if expected is None:
    return
  expected_dir = expected.replace('-', '_') + '.egg-info'
  stale = sorted(
      p.name
      for p in repo_root.glob('*.egg-info')
      if p.is_dir() and p.name != expected_dir
  )
  if stale:
    report.add(
        _AMBER,
        'stale egg-info',
        f'{", ".join(stale)} at the repo root does not match the current'
        f' project ({expected_dir}). Fix: make clean',
    )


def check_core_versions(report: Report) -> None:
  for dist in ('tensorflow', 'jax', 'numpy', 'protobuf', 'arviz'):
    try:
      report.add(_GREEN, dist, metadata.version(dist))
    except metadata.PackageNotFoundError:
      report.add(_RED, dist, 'not installed')


def check_optional_dependencies(report: Report) -> None:
  for module, dist, extra, required in _OPTIONAL_DEPENDENCIES:
    try:
      importlib.import_module(module)
    except ImportError:
      status = _RED if required else _AMBER
      report.add(
          status,
          f'extra [{extra}]',
          f'{dist} missing. Needed for the full test suite to pass.'
          f' Fix: pip install -e ".[{extra}]"',
      )
    else:
      report.add(_GREEN, f'extra [{extra}]', dist)


def check_report_stylesheet(report: Report, repo_root: pathlib.Path) -> None:
  """Checks that the HTML report stylesheet was generated.

  `summary.html.jinja` inlines `style.css` with `ignore missing`, so when the
  file is absent every generated report silently comes out with an empty
  `<style>` tag and no formatting at all. An editable install never runs the
  `compile_scss` build command that generates it.
  """
  stylesheet = repo_root / 'meridian' / 'templates' / 'style.css'
  if not stylesheet.is_file():
    report.add(
        _RED,
        'report stylesheet',
        'meridian/templates/style.css is missing, so generated reports will'
        ' have no styling at all (the template ignores it silently).'
        ' Fix: python scripts/compile_report_css.py',
    )
    return
  size = stylesheet.stat().st_size
  if size == 0:
    report.add(
        _RED,
        'report stylesheet',
        'meridian/templates/style.css is empty.'
        ' Fix: python scripts/compile_report_css.py --force',
    )
    return
  report.add(_GREEN, 'report stylesheet', f'compiled ({size:,} bytes)')


def check_end_to_end(report: Report) -> None:
  """A tiny real fit. Slow-ish, but it is the only check that proves the
  install can actually produce a number."""
  try:
    import numpy as np  # pylint: disable=g-import-not-at-top

    from meridian.analysis import analyzer as analyzer_module  # pylint: disable=g-import-not-at-top
    from meridian.data import test_utils as data_test_utils  # pylint: disable=g-import-not-at-top
    from meridian.model import model  # pylint: disable=g-import-not-at-top
    from meridian.model import spec  # pylint: disable=g-import-not-at-top

    data = data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
        n_geos=2, n_times=20, n_media_times=22, n_media_channels=2,
        n_controls=1,
    )
    mmm = model.Meridian(
        input_data=data, model_spec=spec.ModelSpec(max_lag=2)
    )
    mmm.sample_prior(20, seed=0)
    mmm.sample_posterior(
        n_chains=2, n_adapt=20, n_burnin=20, n_keep=20, seed=0
    )
    roi = np.asarray(
        analyzer_module.Analyzer(
            model_context=mmm.model_context,
            inference_data=mmm.inference_data,
        ).roi()
    )
    if not np.all(np.isfinite(roi)):
      report.add(_RED, 'end-to-end fit', 'ROI contained non-finite values.')
    else:
      report.add(
          _GREEN,
          'end-to-end fit',
          f'fitted and produced finite ROI for {roi.shape[-1]} channels',
      )
  except Exception as e:  # pylint: disable=broad-except
    report.add(_RED, 'end-to-end fit', f'{type(e).__name__}: {e}')


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
      prog='python scripts/verify_environment.py',
      description='Check that a Meridian environment is usable.',
  )
  parser.add_argument(
      '--skip-fit',
      action='store_true',
      help='Skip the end-to-end fit (faster, but proves less).',
  )
  args = parser.parse_args(argv)

  repo_root = pathlib.Path(__file__).resolve().parent.parent
  report = Report()

  print('Meridian environment check')
  print('=' * 70)

  check_python(report)
  imported = check_meridian_import(report, repo_root)
  check_conflicting_packages(report)
  check_duplicate_distribution(report, repo_root)
  check_stale_egg_info(report, repo_root)
  if imported:
    check_backend(report)
    check_accelerator(report)
    check_core_versions(report)
    check_optional_dependencies(report)
    check_report_stylesheet(report, repo_root)
    if not args.skip_fit:
      print('  ... running a small model fit, this takes a moment')
      check_end_to_end(report)

  print(report.render())
  print('=' * 70)
  if report.failed:
    print('RESULT: FAIL -- fix the items marked FAIL above, then re-run.')
    return 1
  print('RESULT: PASS -- environment is ready.')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
