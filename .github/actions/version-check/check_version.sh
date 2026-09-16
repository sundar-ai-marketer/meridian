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
# NOTICE: This file is new in this fork and does not exist in the original
# google/meridian source.

# Shared implementation for the version-check composite action.

set -euo pipefail

: "${CURRENT_VERSION:?CURRENT_VERSION must be set}"
: "${SEMVER_PREFIX:?SEMVER_PREFIX must be set}"
: "${GITHUB_OUTPUT:?GITHUB_OUTPUT must be set}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! "$PYTHON_BIN" -m semver check "$CURRENT_VERSION"; then
  printf 'Error: Invalid semantic version: %s\n' "$CURRENT_VERSION" >&2
  exit 1
fi

# `git describe` exits non-zero when a newly published repository has no
# matching tag. That is the normal first-release case, so handle it before
# comparing versions.
if ! PREVIOUS_TAG="$(
  git describe --tags --abbrev=0 --match "${SEMVER_PREFIX}*" 2>/dev/null
)"; then
  printf 'Warning: no matching previous tag for %s*. Assuming first version.\n' \
    "$SEMVER_PREFIX"
  printf 'new_version=%s\n' "$CURRENT_VERSION" >> "$GITHUB_OUTPUT"
  exit 0
fi

PREVIOUS_VERSION="${PREVIOUS_TAG:${#SEMVER_PREFIX}}"

printf 'CURRENT_VERSION: %s\n' "$CURRENT_VERSION"
printf 'PREVIOUS_TAG: %s\n' "$PREVIOUS_TAG"
printf 'PREVIOUS_VERSION: %s\n' "$PREVIOUS_VERSION"

if [[ $("$PYTHON_BIN" -m semver compare "$CURRENT_VERSION" \
  "$PREVIOUS_VERSION") -eq 1 ]]; then
  printf 'New version detected: %s\n' "$CURRENT_VERSION"
  printf 'new_version=%s\n' "$CURRENT_VERSION" >> "$GITHUB_OUTPUT"
else
  printf 'No version increment detected.\n'
  printf 'new_version=\n' >> "$GITHUB_OUTPUT"
fi
