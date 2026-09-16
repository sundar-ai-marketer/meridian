# Copyright 2026 Sundar Ramesh Kumar.
# Licensed under the Apache License, Version 2.0.
# NOTICE: This file is new in this fork; see NOTICE at the repository root.

"""Integrity-checked, packaged resources for self-contained HTML reports."""

import base64
import functools
import hashlib
import json
from pathlib import Path

_ASSET_ROOT = Path(__file__).with_name('assets')


@functools.lru_cache(maxsize=1)
def _manifest() -> dict:
  return json.loads((_ASSET_ROOT / 'manifest.json').read_text())


def _read_asset(name: str) -> bytes:
  if Path(name).name != name or name not in _manifest():
    raise ValueError(f'Unknown report asset: {name!r}')
  content = (_ASSET_ROOT / name).read_bytes()
  if hashlib.sha256(content).hexdigest() != _manifest()[name]['sha256']:
    raise RuntimeError(f'Report asset failed its integrity check: {name}')
  return content


@functools.lru_cache(maxsize=16)
def data_uri(name: str) -> str:
  """Embed a fixed, verified asset; callers cannot request arbitrary paths."""
  content = _read_asset(name)
  mime = _manifest()[name]['mime']
  return f'data:{mime};base64,{base64.b64encode(content).decode("ascii")}'


@functools.lru_cache(maxsize=1)
def licenses() -> str:
  """Carry third-party notices with every standalone report."""
  return '\n\n'.join(
      f'{name}\n{_read_asset(name).decode("utf-8")}'
      for name in _manifest()
      if name.endswith('-LICENSE.txt')
  )
