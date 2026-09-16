#!/usr/bin/env bash
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
#
# One-command setup: build an isolated environment, install this checkout with
# the supported CPU extras from uv.lock, and verify the result actually works.
#
#   ./scripts/setup.sh                 # venv at ./.venv
#   ./scripts/setup.sh ~/.venvs/mmm    # venv somewhere else
#   MERIDIAN_INSTALL_MODE=pip ./scripts/setup.sh  # explicit unlocked fallback
#
# Safe to re-run. Exits non-zero if the environment does not come out usable.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${1:-$REPO_ROOT/.venv}"
EXTRAS='[dev,colab,schema,mlflow,geox,scenarioplanner]'
UV_VERSION="${MERIDIAN_UV_VERSION:-0.11.14}"
INSTALL_MODE="${MERIDIAN_INSTALL_MODE:-uv}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
die() { printf '\nERROR: %s\n' "$1" >&2; exit 1; }

case "$INSTALL_MODE" in
  uv|pip) ;;
  *) die "MERIDIAN_INSTALL_MODE must be uv (default) or pip (explicit unlocked fallback)." ;;
esac

say "Finding a supported Python (3.11, 3.12 or 3.13)"
PYTHON=""
for candidate in python3.11 python3.12 python3.13 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    version="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    case "$version" in
      3.11|3.12|3.13) PYTHON="$candidate"; break ;;
    esac
  fi
done
[ -n "$PYTHON" ] || die "No Python 3.11-3.13 found. Install one and re-run.
On macOS:  brew install python@3.11
On Debian: sudo apt install python3.11 python3.11-venv"
printf '    using %s (%s)\n' "$PYTHON" "$("$PYTHON" --version)"

if ! command -v git >/dev/null 2>&1; then
  die "Git is required to build the sibling proto package. Install Git and re-run.
On macOS:  brew install git
On Debian: sudo apt install git ca-certificates"
fi

# Resolve the venv path before changing into the repository. Without this,
# invoking the script from another directory with a relative path creates the
# venv in the caller's directory, then tries to use that same relative path
# after `cd "$REPO_ROOT"` and fails to find its interpreter.
VENV="$($PYTHON -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$VENV")"

say "Creating virtual environment at $VENV"
if [ -d "$VENV" ]; then
  printf '    already exists, reusing it\n'
else
  "$PYTHON" -m venv "$VENV"
fi
VENV_PY="$VENV/bin/python"
[ -x "$VENV_PY" ] || die "venv looks broken: no interpreter at $VENV_PY"

UV=""
if [ "$INSTALL_MODE" = uv ]; then
  # Prefer an already-installed uv only when it is the version used to create
  # this lock. Otherwise keep the bootstrap inside the task-owned target tree;
  # installing uv into the target venv would let `uv sync` remove its own
  # package as an extraneous dependency.
  if command -v uv >/dev/null 2>&1 \
      && [ "$(uv --version 2>/dev/null || true)" = "uv $UV_VERSION" ]; then
    UV="$(command -v uv)"
  else
    UV_BOOTSTRAP_VENV="$VENV/.meridian-uv"
    UV_BOOTSTRAP_PY="$UV_BOOTSTRAP_VENV/bin/python"
    if [ ! -x "$UV_BOOTSTRAP_PY" ]; then
      say "Creating the pinned uv bootstrap environment at $UV_BOOTSTRAP_VENV"
      "$PYTHON" -m venv "$UV_BOOTSTRAP_VENV"
    fi
    if ! "$UV_BOOTSTRAP_PY" -c \
        'import importlib.metadata as m, sys; sys.exit(m.version("uv") != sys.argv[1] or m.version("pip") != "26.2.1")' \
        "$UV_VERSION" >/dev/null 2>&1; then
      say "Installing uv $UV_VERSION in the task-owned bootstrap environment"
      "$UV_BOOTSTRAP_PY" -m pip install -q --disable-pip-version-check \
        --no-cache-dir 'pip==26.2.1' "uv==$UV_VERSION"
    fi
    UV="$UV_BOOTSTRAP_VENV/bin/uv"
  fi
  [ -x "$UV" ] || die "uv $UV_VERSION bootstrap failed; re-run with network access."
fi

# tensorflow-metal is the single most confusing failure: it makes
# `import meridian` die inside libmetal_plugin.dylib with a symbol error that
# names nothing useful. Remove it before it can bite.
if "$VENV_PY" -m pip show tensorflow-metal >/dev/null 2>&1; then
  say "Removing tensorflow-metal (incompatible with the pinned TensorFlow)"
  "$VENV_PY" -m pip uninstall -y tensorflow-metal
fi

cd "$REPO_ROOT"
if [ -n "$UV" ]; then
  printf '    using %s (%s)\n' "$UV" "$("$UV" --version)"
  say "Installing this checkout from uv.lock (selected extras; several minutes)"
  # uv.lock is universal and includes the optional dependency metadata needed
  # for every declared extra. Install only the supported CPU extras here:
  # `and-cuda` is intentionally excluded because it has no Apple Silicon
  # wheels and is not part of the supported all-extras setup.
  UV_PROJECT_ENVIRONMENT="$VENV" "$UV" sync \
    --python "$VENV_PY" \
    --frozen \
    --no-default-groups \
    --extra dev \
    --extra colab \
    --extra schema \
    --extra mlflow \
    --extra geox \
    --extra scenarioplanner
else
  say "Upgrading pip tooling (explicit pip fallback; dependency resolution is UNLOCKED)"
  "$VENV_PY" -m pip install -q -U pip setuptools wheel

  say "Installing this checkout with selected extras (pip fallback; UNLOCKED)"
  # Install the sibling schema package first. Without this explicit local
  # install, pip satisfies the root package's `mmm-proto-schema` extra from
  # PyPI and the checkout silently runs against a different schema build.
  "$VENV_PY" -m pip install -e proto --config-settings editable_mode=strict
  "$VENV_PY" -m pip install -e ".$EXTRAS"
fi

# `python -m venv` normally seeds pip, but uv is allowed to remove packages it
# does not find in the lock. Restore it if a future uv version treats pip as
# extraneous so the diagnostic below remains available on every run.
if ! "$VENV_PY" -m pip --version >/dev/null 2>&1; then
  say "Restoring pip for dependency checks"
  "$VENV_PY" -m ensurepip --upgrade >/dev/null
fi

# An editable install never runs setup.py's `build`, so the `compile_scss`
# command that generates the report stylesheet never fires. The report template
# includes it with `ignore missing`, so the absence is silent and every
# generated report comes out unstyled. Compile it explicitly.
say "Compiling the report stylesheet"
"$VENV_PY" scripts/compile_report_css.py

say "Verifying the environment"
if ! "$VENV_PY" scripts/verify_environment.py; then
  die "Environment verification failed. Fix the items marked FAIL above."
fi
"$VENV_PY" -m pip check

# These commands are meant to be copied into a shell. Quote paths for that
# shell, including custom environment paths and checkouts containing spaces.
printf -v REPO_COMMAND_PATH '%q' "$REPO_ROOT"
printf -v ACTIVATE_COMMAND_PATH '%q' "$VENV/bin/activate"
printf -v PYTHON_COMMAND_PATH '%q' "$VENV_PY"

cat <<EOF

Setup complete.

  Repository: cd $REPO_COMMAND_PATH
  Activate:   source $ACTIVATE_COMMAND_PATH
  Try it:     $PYTHON_COMMAND_PATH examples/quickstart.py
  Run tests:  $PYTHON_COMMAND_PATH scripts/run_tests.py

EOF
