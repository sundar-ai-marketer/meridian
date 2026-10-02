# Copyright 2026 Sundar Ramesh Kumar.
# Licensed under the Apache License, Version 2.0.
# NOTICE: This file is new in this fork; see NOTICE at the repository root.

"""Publication contract tests using real temporary Git repos and bare remotes.

Only the GitHub CLI is replaced: no GitHub credentials, network, container
build or scanner are needed. Tests exercise generated dirty files, staging,
branch integration, immutable snapshots, idempotence and explicit PR controls.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts import publish_container_rescan

_HELPER = Path(publish_container_rescan.__file__).resolve()
_WORKFLOW = _HELPER.parents[1] / '.github/workflows/container-rescan.yml'
_DATE = '2026-10-02'


class RescanPublicationTest(unittest.TestCase):

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.root = Path(self.temporary.name)
    self.remote = self.root / 'remote.git'
    self.seed = self.root / 'seed'
    self.seed.mkdir()
    self.env = {
        **os.environ,
        'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': os.devnull,
        'RUNNER_TEMP': str(self.root / 'runner-temp'),
        'GITHUB_STEP_SUMMARY': str(self.root / 'summary.md'),
        'GITHUB_REPOSITORY': 'example/meridian', 'GITHUB_RUN_ID': '123',
        'GITHUB_SERVER_URL': 'https://github.com',
        'GH_LOG': str(self.root / 'gh.jsonl'), 'GH_EXISTING_PR': '',
        'GH_PR_FAILURE': '',
    }
    for name in ('GIT_AUTHOR_NAME', 'GIT_AUTHOR_EMAIL', 'GIT_COMMITTER_NAME',
                 'GIT_COMMITTER_EMAIL'):
      self.env.pop(name, None)
    Path(self.env['RUNNER_TEMP']).mkdir()
    self._git(self.root, 'init', '--bare', '--initial-branch=main', str(self.remote))
    self._git(self.remote, 'config', 'receive.denyNonFastForwards', 'true')
    self._git(self.remote, 'config', 'receive.denyDeletes', 'true')
    self._git(self.seed, 'init', '--initial-branch=main')
    self._git(self.seed, 'config', 'user.name', 'Fixture maintainer')
    self._git(self.seed, 'config', 'user.email', 'fixture@example.invalid')
    self._evidence(self.seed, '2026-10-01', 'initial')
    (self.seed / 'app.txt').write_text('initial application\n')
    self._git(self.seed, 'add', '.')
    self._git(self.seed, 'commit', '-m', 'Initial main')
    self._git(self.seed, 'remote', 'add', 'origin', str(self.remote))
    self._git(self.seed, 'push', 'origin', 'main')
    self.main_head = self._git(self.remote, 'rev-parse', 'main')
    self._fake_gh()
    self.clone_index = 0

  def _git(self, cwd, *args):
    return subprocess.run(
        ['git', *args], cwd=cwd, env=self.env, check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()

  def _clone(self):
    self.clone_index += 1
    clone = self.root / f'checkout-{self.clone_index}'
    self._git(self.root, 'clone', str(self.remote), str(clone))
    return clone

  def _fake_gh(self):
    directory = self.root / 'bin'
    directory.mkdir()
    cli = directory / 'gh'
    cli.write_text(
        f'#!{sys.executable}\n'
        'import json, os, pathlib, sys\n'
        'args = sys.argv[1:]\n'
        'record = {"args": args}\n'
        'if "--body-file" in args:\n'
        '    path = pathlib.Path(args[args.index("--body-file") + 1])\n'
        '    record["body"] = path.read_text()\n'
        'with open(os.environ["GH_LOG"], "a") as log:\n'
        '    log.write(json.dumps(record) + "\\n")\n'
        'if args[:2] == ["pr", "list"]:\n'
        '    print(os.environ.get("GH_EXISTING_PR", ""))\n'
        'elif os.environ.get("GH_PR_FAILURE"):\n'
        '    print(os.environ["GH_PR_FAILURE"], file=sys.stderr)\n'
        '    sys.exit(1)\n'
        'else:\n'
        '    print("https://github.com/example/meridian/pull/17")\n'
    )
    cli.chmod(0o755)
    self.env['PATH'] = str(directory) + os.pathsep + self.env['PATH']

  def _evidence(self, repository, date, value):
    directory = repository / 'docs/validation'
    directory.mkdir(parents=True, exist_ok=True)
    for family in ('container-reachability', 'os-triage-summary'):
      payload = json.dumps({'date': date, 'value': value}) + '\n'
      (directory / f'{family}-latest.json').write_text(payload)
      (directory / f'{family}-{date}.json').write_text(payload)
    (directory / 'os-triage-latest.md').write_text(value + '\n')

  def _publish(self, repository, date=_DATE, value='fresh', *, expected=0):
    self._evidence(repository, date, value)
    result = subprocess.run(
        [sys.executable, str(_HELPER), '--date', date], cwd=repository,
        env=self.env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
    return result

  def _branch_head(self):
    return self._git(self.remote, 'rev-parse', publish_container_rescan.BRANCH)

  def _assert_published(self, date, value):
    branch = publish_container_rescan.BRANCH
    self.assertEqual(self._git(self.remote, 'rev-parse', 'main'), self.main_head)
    for family in ('container-reachability', 'os-triage-summary'):
      dated = self._git(self.remote, 'show', f'{branch}:docs/validation/{family}-{date}.json')
      latest = self._git(self.remote, 'show', f'{branch}:docs/validation/{family}-latest.json')
      self.assertEqual(dated, latest)
      self.assertEqual(json.loads(dated), {'date': date, 'value': value})
      historical = self._git(
          self.remote, 'show', f'{branch}:docs/validation/{family}-2026-10-01.json'
      )
      self.assertEqual(json.loads(historical)['value'], 'initial')

  def _gh_calls(self):
    return [json.loads(line) for line in Path(self.env['GH_LOG']).read_text().splitlines()]

  def _advance_main(self, *, evidence_date=None):
    (self.seed / 'app.txt').write_text('main advanced\n')
    if evidence_date:
      self._evidence(self.seed, evidence_date, 'main measurement')
    self._git(self.seed, 'add', '.')
    self._git(self.seed, 'commit', '-m', 'Main advances independently')
    self._git(self.seed, 'push', 'origin', 'main')
    self.main_head = self._git(self.remote, 'rev-parse', 'main')

  def test_first_run_cleans_generated_staging_and_opens_pr(self):
    clone = self._clone()
    self._evidence(clone, _DATE, 'fresh')
    self._git(clone, 'add', 'docs/validation')
    self._publish(clone)
    self._assert_published(_DATE, 'fresh')
    self.assertEqual(self._git(clone, 'status', '--porcelain'), '')
    self.assertEqual(self._git(clone, 'log', '-1', '--format=%an'), 'github-actions[bot]')
    calls = self._gh_calls()
    self.assertEqual(calls[-1]['args'][:2], ['pr', 'create'])
    self.assertIn('--body-file', calls[-1]['args'])
    self.assertIn('approve its Actions runs', calls[-1]['body'])
    self.assertIn('manually dispatched run does not substitute', calls[-1]['body'])

  def test_repeat_updates_existing_branch_and_pr_then_is_idempotent(self):
    self._publish(self._clone())
    first = self._branch_head()
    self.env['GH_EXISTING_PR'] = '17'
    self._publish(self._clone(), date='2026-10-03', value='newer')
    second = self._branch_head()
    self.assertNotEqual(first, second)
    self._git(self.remote, 'merge-base', '--is-ancestor', first, second)
    self._assert_published('2026-10-03', 'newer')
    self._publish(self._clone(), date='2026-10-03', value='newer')
    self.assertEqual(self._branch_head(), second)
    self.assertEqual(self._gh_calls()[-1]['args'][:3], ['pr', 'edit', '17'])
    old = self._git(self.remote, 'show',
                    f'{publish_container_rescan.BRANCH}:docs/validation/'
                    'container-reachability-2026-10-02.json')
    self.assertEqual(json.loads(old)['value'], 'fresh')

  def test_main_advancement_merges_with_bot_identity_before_commit(self):
    self._publish(self._clone())
    previous = self._branch_head()
    self._advance_main()
    self._publish(self._clone(), date='2026-10-03', value='newer')
    current = self._branch_head()
    self._git(self.remote, 'merge-base', '--is-ancestor', previous, current)
    self._git(self.remote, 'merge-base', '--is-ancestor', self.main_head, current)
    self.assertEqual(self._git(self.remote, 'show', f'{current}:app.txt'), 'main advanced')
    self._assert_published('2026-10-03', 'newer')

  def test_alias_only_merge_conflicts_use_snapshot_and_keep_both_histories(self):
    self._publish(self._clone())
    previous = self._branch_head()
    self._advance_main(evidence_date='2026-10-03')
    self._publish(self._clone(), date='2026-10-04', value='freshest')
    current = self._branch_head()
    self._git(self.remote, 'merge-base', '--is-ancestor', previous, current)
    self._git(self.remote, 'merge-base', '--is-ancestor', self.main_head, current)
    self._assert_published('2026-10-04', 'freshest')
    for date, expected in ((_DATE, 'fresh'), ('2026-10-03', 'main measurement')):
      payload = self._git(self.remote, 'show',
                          f'{current}:docs/validation/container-reachability-{date}.json')
      self.assertEqual(json.loads(payload)['value'], expected)

  def test_same_date_different_bytes_preserves_original_and_retains_snapshot(self):
    self._publish(self._clone())
    previous = self._branch_head()
    result = self._publish(self._clone(), value='different same-day scan', expected=1)
    self.assertIn('Dated evidence collision', result.stderr)
    self.assertEqual(self._branch_head(), previous)
    self._assert_published(_DATE, 'fresh')
    snapshots = Path(self.env['RUNNER_TEMP']).glob('rescan-evidence-*')
    self.assertTrue(any(
        json.loads((path / f'container-reachability-{_DATE}.json').read_text())['value']
        == 'different same-day scan' for path in snapshots
    ))

  def test_stale_scan_cannot_make_latest_alias_older_than_dated_history(self):
    self._publish(self._clone(), date='2026-10-04', value='newest')
    previous = self._branch_head()
    result = self._publish(self._clone(), date='2026-10-03', value='stale', expected=1)
    self.assertIn('Newer dated evidence exists', result.stderr)
    self.assertEqual(self._branch_head(), previous)
    self._assert_published('2026-10-04', 'newest')

  def test_forbidden_actions_pr_creation_leaves_branch_and_actionable_summary(self):
    self.env['GH_PR_FAILURE'] = (
        'GraphQL: GitHub Actions is not permitted to create or approve pull requests'
    )
    self._publish(self._clone())
    self._assert_published(_DATE, 'fresh')
    summary = Path(self.env['GITHUB_STEP_SUMMARY']).read_text()
    self.assertIn('Settings were preserved', summary)
    self.assertIn('compare/main...automation/container-rescan?expand=1', summary)
    self.assertIn('actions/runs/123', summary)

  def test_already_merged_evidence_does_not_request_an_empty_pr(self):
    self._publish(self._clone())
    self._git(self.seed, 'fetch', 'origin', publish_container_rescan.BRANCH)
    self._git(self.seed, 'merge', '--ff-only', 'FETCH_HEAD')
    self._git(self.seed, 'push', 'origin', 'main')
    self.main_head = self._git(self.remote, 'rev-parse', 'main')
    previous_calls = len(self._gh_calls())
    result = self._publish(self._clone())
    self.assertEqual(len(self._gh_calls()), previous_calls)
    self.assertIn('already on main; no empty PR', result.stdout)
    self._assert_published(_DATE, 'fresh')

  def test_unrelated_github_failure_is_not_swallowed(self):
    self.env['GH_PR_FAILURE'] = 'GraphQL: Resource not accessible by integration'
    result = self._publish(self._clone(), expected=1)
    self.assertIn('PR publication failed', result.stderr)
    self._assert_published(_DATE, 'fresh')
    self.assertIn('retained run artifact', Path(self.env['GITHUB_STEP_SUMMARY']).read_text())

  def test_unrelated_staged_work_is_preserved_and_blocks_branch_operations(self):
    clone = self._clone()
    (clone / 'app.txt').write_text('unrelated pending work\n')
    self._git(clone, 'add', 'app.txt')
    result = self._publish(clone, expected=1)
    self.assertIn('Unrelated working-tree changes', result.stderr)
    self.assertEqual((clone / 'app.txt').read_text(), 'unrelated pending work\n')
    self.assertEqual(self._git(clone, 'diff', '--cached', '--name-only'), 'app.txt')
    self.assertEqual(self._git(clone, 'branch', '--show-current'), 'main')

  def test_unrelated_merge_conflict_is_not_resolved_or_pushed(self):
    self._publish(self._clone())
    branch_owner = self._clone()
    self._git(branch_owner, 'switch', '--track', f'origin/{publish_container_rescan.BRANCH}')
    self._git(branch_owner, 'config', 'user.name', 'Fixture branch owner')
    self._git(branch_owner, 'config', 'user.email', 'owner@example.invalid')
    (branch_owner / 'app.txt').write_text('unrelated automation-branch edit\n')
    self._git(branch_owner, 'add', 'app.txt')
    self._git(branch_owner, 'commit', '-m', 'Independent branch edit')
    self._git(branch_owner, 'push', 'origin', publish_container_rescan.BRANCH)
    previous = self._branch_head()
    self._advance_main()
    clone = self._clone()
    result = self._publish(clone, date='2026-10-03', expected=1)
    self.assertIn('outside regenerated aliases', result.stderr)
    self.assertEqual(self._branch_head(), previous)
    self.assertEqual(self._git(clone, 'diff', '--name-only', '--diff-filter=U'), '')
    self.assertEqual((clone / 'app.txt').read_text(), 'unrelated automation-branch edit\n')

  def test_workflow_retains_artifacts_before_the_publication_helper(self):
    content = _WORKFLOW.read_text()
    self.assertLess(content.index('name: Retain the complete report'),
                    content.index('python3 scripts/publish_container_rescan.py'))
    self.assertIn('pull-requests: write', content)
    self.assertNotIn('git push origin HEAD:main', content)


if __name__ == '__main__':
  unittest.main()
