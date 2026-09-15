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
# every extra, and verify the result actually works.
#
#   ./scripts/setup.sh                 # venv at ./.venv
#   ./scripts/setup.sh ~/.venvs/mmm    # venv somewhere else
#
# Safe to re-run. Exits non-zero if the environment does not come out usable.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${1:-$REPO_ROOT/.venv}"
EXTRAS='[dev,colab,schema,mlflow,geox,scenarioplanner]'

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
die() { printf '\nERROR: %s\n' "$1" >&2; exit 1; }

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

say "Creating virtual environment at $VENV"
if [ -d "$VENV" ]; then
  printf '    already exists, reusing it\n'
else
  "$PYTHON" -m venv "$VENV"
fi
VENV_PY="$VENV/bin/python"
[ -x "$VENV_PY" ] || die "venv looks broken: no interpreter at $VENV_PY"

say "Upgrading pip tooling"
"$VENV_PY" -m pip install -q -U pip setuptools wheel

# tensorflow-metal is the single most confusing failure: it makes
# `import meridian` die inside libmetal_plugin.dylib with a symbol error that
# names nothing useful. Remove it before it can bite.
if "$VENV_PY" -m pip show tensorflow-metal >/dev/null 2>&1; then
  say "Removing tensorflow-metal (incompatible with the pinned TensorFlow)"
  "$VENV_PY" -m pip uninstall -y tensorflow-metal
fi

say "Installing this checkout with all extras (several minutes)"
cd "$REPO_ROOT"
"$VENV_PY" -m pip install -e ".$EXTRAS"

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

cat <<EOF

Setup complete.

  Activate:   source $VENV/bin/activate
  Try it:     $VENV_PY examples/quickstart.py
  Run tests:  $VENV_PY -m pytest meridian -q -n 8

EOF
