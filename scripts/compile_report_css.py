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

"""Compiles the HTML report stylesheet, which an editable install skips.

`meridian/templates/summary.html.jinja` inlines `style.css` with
`{% include "style.css" ignore missing %}`. That file is not in the source
tree; it is generated from `style.scss` by the `compile_scss` command in
`setup.py`, which `setuptools` runs as part of `build`.

A PEP 660 editable install -- what `README.md` and `scripts/setup.sh` both
tell you to do -- never runs `build`, so `style.css` is never generated. The
`ignore missing` then swallows the absence, and every generated report comes
out with an empty `<style></style>` tag. Nothing errors. The report renders as
unstyled text, because it is built from custom elements (`<card>`,
`<chart-embed>`, `<charts>`) that have no default browser styling at all.

Running this after an editable install produces the same stylesheet a wheel
would have contained.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Sequence

_TEMPLATES = pathlib.Path(__file__).resolve().parent.parent / 'meridian' / 'templates'
_SOURCE = _TEMPLATES / 'style.scss'
_OUTPUT = _TEMPLATES / 'style.css'


def compile_stylesheet(force: bool = False) -> pathlib.Path:
  """Compiles `style.scss` to `style.css`, returning the output path."""
  if not _SOURCE.is_file():
    raise FileNotFoundError(f'No SCSS source at {_SOURCE}')

  if _OUTPUT.is_file() and not force:
    if _OUTPUT.stat().st_mtime >= _SOURCE.stat().st_mtime:
      return _OUTPUT

  try:
    import sass  # pylint: disable=g-import-not-at-top
  except ImportError as e:
    raise ImportError(
        'libsass is required to compile the report stylesheet. Install it'
        ' with `pip install libsass`, or install this checkout with the'
        ' [dev] extra, which declares it.'
    ) from e

  _OUTPUT.write_text(
      sass.compile(string=_SOURCE.read_text(encoding='utf-8')),
      encoding='utf-8',
  )
  return _OUTPUT


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
      prog='python scripts/compile_report_css.py',
      description='Compile the HTML report stylesheet for an editable install.',
  )
  parser.add_argument(
      '--force',
      action='store_true',
      help='Recompile even if the output is already newer than the source.',
  )
  args = parser.parse_args(argv)

  try:
    output = compile_stylesheet(force=args.force)
  except (FileNotFoundError, ImportError) as e:
    print(f'ERROR: {e}', file=sys.stderr)
    return 1

  print(f'    {output} ({output.stat().st_size} bytes)')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
