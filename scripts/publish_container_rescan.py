#!/usr/bin/env python3
# Copyright 2026 Sundar Ramesh Kumar.
# Licensed under the Apache License, Version 2.0.
# NOTICE: This file is new in this fork; see NOTICE at the repository root.

"""Publish regenerated evidence to a review branch without rewriting history.

The workflow retains its artifacts before invoking this helper. Copy the exact
generated staging set to private temporary storage before restoring those
paths, then integrate main into the automation branch and apply the snapshot.
Only regenerated aliases can resolve a merge conflict automatically. Existing
dated evidence with different bytes is never overwritten; a same-day collision
stops publication with the fresh measurement retained in the run artifact.

GITHUB_TOKEN may be forbidden to create PRs by repository settings. That exact
permission failure leaves the pushed evidence branch and a compare link for a
maintainer. Other failures remain failures. This helper never pushes main,
force-pushes, changes repository settings, approves checks, or merges a PR.
"""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

BRANCH = 'automation/container-rescan'
_ALIASES = (
    'container-reachability-latest.json', 'os-triage-summary-latest.json',
    'os-triage-latest.md',
)
_FORBIDDEN_PR = 'GitHub Actions is not permitted to create or approve pull requests'
_TOKEN_DOC = 'https://docs.github.com/en/actions/concepts/security/github_token#when-github_token-triggers-workflow-runs'
_PERMISSION_DOC = (
    'https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/'
    'enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository'
)


def _run(args, *, check=True):
  result = subprocess.run(args, text=True, capture_output=True, check=False)
  if check and result.returncode:
    raise RuntimeError(f'{args[0]} {args[1]} failed: {result.stderr.strip()}')
  return result


def _git(*args, check=True):
  return _run(['git', *args], check=check)


def _summary(text):
  path = os.environ.get('GITHUB_STEP_SUMMARY')
  if path:
    with Path(path).open('a', encoding='utf-8') as handle:
      handle.write(text + '\n')
  print(text)


def _snapshot(date, runner_temp):
  """Copy first, then restore only the exact regenerated staging paths."""
  names = (*_ALIASES, f'container-reachability-{date}.json',
           f'os-triage-summary-{date}.json')
  snapshot = Path(tempfile.mkdtemp(prefix='rescan-evidence-', dir=runner_temp))
  paths = [Path('docs/validation') / name for name in names]
  for path in paths:
    if path.is_symlink() or not path.is_file():
      raise ValueError(f'Regenerated evidence is not a regular file: {path}')
    shutil.copyfile(path, snapshot / path.name)
  for family in ('container-reachability', 'os-triage-summary'):
    if (snapshot / f'{family}-latest.json').read_bytes() != (
        snapshot / f'{family}-{date}.json').read_bytes():
      raise ValueError(f'{family} alias and regenerated dated copy differ.')
  # Reset only this staging set, including newly added files, before checking
  # which paths already belong to HEAD. Unrelated staged changes stay visible.
  _git('reset', '--', *(str(path) for path in paths))
  for path in paths:
    tracked = _git('ls-files', '--error-unmatch', '--', str(path), check=False)
    if tracked.returncode == 0:
      _git('restore', '--source=HEAD', '--staged', '--worktree', '--', str(path))
    else:
      path.unlink()
  if _git('status', '--porcelain').stdout.strip():
    raise RuntimeError('Unrelated working-tree changes prevent evidence publication.')
  return snapshot, paths


def _apply(snapshot, paths):
  """Preserve every existing dated file; aliases reflect the new snapshot."""
  for path in paths:
    if path.name not in _ALIASES and path.exists():
      if path.read_bytes() != (snapshot / path.name).read_bytes():
        raise RuntimeError(f'Dated evidence collision; original preserved: {path}')
    if path.name not in _ALIASES:
      family = path.name[:-16]  # Remove '-YYYY-MM-DD.json'.
      date = path.stem[-10:]
      newer = [item for item in path.parent.glob(f'{family}-????-??-??.json')
               if item.stem[-10:] > date]
      if newer:
        raise RuntimeError('Newer dated evidence exists; stale scan was not published.')
  for path in paths:
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(snapshot / path.name, path)
  _git('add', '--', *(str(path) for path in paths))


def _prepare_branch(snapshot, paths):
  # Identity must exist before a merge can create its commit.
  _git('config', 'user.name', 'github-actions[bot]')
  _git('config', 'user.email', '41898282+github-actions[bot]@users.noreply.github.com')
  _git('fetch', 'origin', 'refs/heads/main:refs/remotes/origin/main')
  remote = _git('ls-remote', '--exit-code', '--heads', 'origin', BRANCH, check=False)
  if remote.returncode not in (0, 2):
    raise RuntimeError(f'Cannot inspect evidence branch: {remote.stderr.strip()}')
  if remote.returncode == 2:
    _git('switch', '--create', BRANCH, 'origin/main')
    return
  _git('fetch', 'origin', f'refs/heads/{BRANCH}:refs/remotes/origin/{BRANCH}')
  local = _git('show-ref', '--verify', '--quiet', f'refs/heads/{BRANCH}', check=False)
  if local.returncode == 0:
    _git('switch', BRANCH)
    _git('merge', '--ff-only', f'origin/{BRANCH}')
  else:
    _git('switch', '--create', BRANCH, '--track', f'origin/{BRANCH}')
  merged = _git('merge', '--no-edit', 'origin/main', check=False)
  if merged.returncode:
    conflicts = set(_git('diff', '--name-only', '--diff-filter=U').stdout.splitlines())
    resolvable = {str(Path('docs/validation') / name) for name in _ALIASES}
    if not conflicts or not conflicts.issubset(resolvable):
      _git('merge', '--abort', check=False)
      raise RuntimeError('Main merge failed outside regenerated aliases; no push made.')
    # Resolve only aliases, using the retained measurement. Dated snapshots and
    # all unrelated code still use Git's ordinary merge semantics.
    try:
      _apply(snapshot, paths)
    except (OSError, RuntimeError):
      _git('merge', '--abort', check=False)
      raise
    _git('commit', '--no-edit')


def _publish_pr(*, runner_temp, repository_url, run_url):
  title = 'Refresh container OS advisory evidence'
  body = runner_temp / 'container-rescan-pr.md'
  body.write_text(
      f'This PR contains the refreshed advisory inventory, reachability '
      f'measurement and triage from [this rescan]({run_url}).\n\n'
      f'The scanner found {os.environ.get("UNFIXED_COUNT", "0")} unfixed '
      f'advisory IDs and {os.environ.get("FIXABLE_COUNT", "0")} high or '
      'critical findings with a vendor fix.\n\n'
      'The weekly job updates this evidence branch. Main retains all required '
      'CI checks. A PR created or synchronized using GITHUB_TOKEN requires a '
      f'maintainer to approve its Actions runs; see [GitHub event rules]({_TOKEN_DOC}). '
      'A manually dispatched run does not substitute for the PR required checks.\n\n'
      'If repository settings prohibit Actions from creating PRs, a maintainer '
      f'can [open this comparison]({repository_url}/compare/main...{BRANCH}?expand=1) '
      f'without changing [workflow permission settings]({_PERMISSION_DOC}).\n',
      encoding='utf-8',
  )
  existing = _run([
      'gh', 'pr', 'list', '--state', 'open', '--base', 'main', '--head', BRANCH,
      '--json', 'number', '--jq', '.[0].number // empty',
  ]).stdout.strip()
  if existing:
    command = ['gh', 'pr', 'edit', existing, '--title', title, '--body-file', str(body)]
  else:
    command = ['gh', 'pr', 'create', '--base', 'main', '--head', BRANCH,
               '--title', title, '--body-file', str(body)]
  result = _run(command, check=False)
  if result.returncode:
    if _FORBIDDEN_PR in result.stderr:
      _summary(
          'Evidence branch published. Repository settings prohibit Actions '
          'from creating or approving pull requests. A maintainer must open '
          'the comparison above and run the required PR checks. Settings were '
          'preserved.'
      )
      return
    raise RuntimeError(f'PR publication failed: {result.stderr.strip()}')
  _summary(f'Evidence PR {"updated" if existing else "created"}: {result.stdout.strip()}')


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--date', required=True)
  args = parser.parse_args(argv)
  if datetime.date.fromisoformat(args.date).isoformat() != args.date:
    raise ValueError('The evidence date must use YYYY-MM-DD format.')
  runner_temp = Path(os.environ['RUNNER_TEMP'])
  repository_url = (
      os.environ.get('GITHUB_SERVER_URL', 'https://github.com').rstrip('/')
      + '/' + os.environ['GITHUB_REPOSITORY']
  )
  run_url = repository_url + '/actions/runs/' + os.environ['GITHUB_RUN_ID']
  _summary(
      f'Container rescan evidence: [retained run artifact]({run_url}), '
      f'[evidence branch]({repository_url}/tree/{BRANCH}), '
      f'[open/review PR comparison]({repository_url}/compare/main...{BRANCH}?expand=1).'
  )
  snapshot, paths = _snapshot(args.date, runner_temp)
  _prepare_branch(snapshot, paths)
  _apply(snapshot, paths)
  if _git('diff', '--cached', '--quiet', check=False).returncode:
    _git('commit', '-m', 'Refresh the container OS advisory evidence', '-m',
         f'Clean rebuild and rescan for {args.date}. '
         f'Unfixed advisories: {os.environ.get("UNFIXED_COUNT", "0")}. '
         f'Fixable high/critical: {os.environ.get("FIXABLE_COUNT", "0")}.')
  else:
    _summary('The evidence branch already contains this rescan output.')
  _git('push', 'origin', f'HEAD:refs/heads/{BRANCH}')
  _summary('Evidence branch published; main still requires its configured PR checks.')
  if _git('rev-list', '--count', 'origin/main..HEAD').stdout.strip() == '0':
    _summary('This evidence is already on main; no empty PR was requested.')
  else:
    _publish_pr(runner_temp=runner_temp, repository_url=repository_url, run_url=run_url)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
