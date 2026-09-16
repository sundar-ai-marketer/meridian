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

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CHECKER="$SCRIPT_DIR/check_version.sh"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ "$PYTHON_BIN" == */* ]]; then
  PYTHON_BIN="$(cd -- "$(dirname -- "$PYTHON_BIN")" && pwd -P)/$(basename -- "$PYTHON_BIN")"
else
  PYTHON_BIN="$(command -v "$PYTHON_BIN")"
fi
TEST_ROOT="$(mktemp -d)"
trap 'rm -rf "$TEST_ROOT"' EXIT

new_repo() {
  local repo="$1"
  mkdir "$repo"
  git -C "$repo" init -q
  git -C "$repo" config user.email test@example.invalid
  git -C "$repo" config user.name version-check-test
  printf 'seed\n' > "$repo/seed"
  git -C "$repo" add seed
  git -C "$repo" commit -q -m seed
}

run_check() {
  local repo="$1" current="$2" prefix="$3" expected="$4"
  local output="$repo/output"
  : > "$output"
  (
    cd "$repo"
    CURRENT_VERSION="$current" \
      SEMVER_PREFIX="$prefix" \
      GITHUB_OUTPUT="$output" \
      PYTHON_BIN="$PYTHON_BIN" \
      bash "$CHECKER"
  )
  local actual
  actual="$(sed -n 's/^new_version=//p' "$output")"
  if [[ "$actual" != "$expected" ]]; then
    printf 'Expected new_version=%q, got %q for %s\n' \
      "$expected" "$actual" "$current" >&2
    exit 1
  fi
}

run_check_failure() {
  local repo="$1" current="$2" prefix="$3"
  local output="$repo/output"
  : > "$output"
  if (
    cd "$repo"
    CURRENT_VERSION="$current" \
      SEMVER_PREFIX="$prefix" \
      GITHUB_OUTPUT="$output" \
      PYTHON_BIN="$PYTHON_BIN" \
      bash "$CHECKER"
  ); then
    printf 'Expected version-check to reject %s\n' "$current" >&2
    exit 1
  fi
}

# A repository with no tags is the first-release path and must succeed.
first_repo="$TEST_ROOT/first"
new_repo "$first_repo"
run_check "$first_repo" 1.0.0 v 1.0.0

# An existing release with the same version must produce no new version.
same_repo="$TEST_ROOT/same"
new_repo "$same_repo"
git -C "$same_repo" tag v1.0.0
run_check "$same_repo" 1.0.0 v ''

# A larger valid version must be surfaced for publication.
increment_repo="$TEST_ROOT/increment"
new_repo "$increment_repo"
git -C "$increment_repo" tag v1.0.0
run_check "$increment_repo" 1.1.0 v 1.1.0

# Invalid versions must fail even when no previous release tag exists.
invalid_first_repo="$TEST_ROOT/invalid-first"
new_repo "$invalid_first_repo"
run_check_failure "$invalid_first_repo" invalid v

# The same validation must apply on the tagged path.
invalid_tagged_repo="$TEST_ROOT/invalid-tagged"
new_repo "$invalid_tagged_repo"
git -C "$invalid_tagged_repo" tag v1.0.0
run_check_failure "$invalid_tagged_repo" invalid v

# A tag with a different prefix must not be treated as the previous release.
mismatch_repo="$TEST_ROOT/mismatch"
new_repo "$mismatch_repo"
git -C "$mismatch_repo" tag proto-v1.0.0
run_check "$mismatch_repo" 1.0.0 v 1.0.0

printf 'version-check contract tests passed\n'
