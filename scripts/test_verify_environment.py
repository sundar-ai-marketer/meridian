# Copyright 2026 Sundar Ramesh Kumar.
# Licensed under the Apache License, Version 2.0.
# NOTICE: This file is new in this fork; see NOTICE at the repository root.

"""Contracts for the environment checks that decide whether setup succeeded.

These checks exist to replace a confusing failure with an instruction, so the
dangerous direction is a check that fires when nothing is wrong: a developer
told to uninstall their only working install will do it. Both cases below were
real -- the duplicate-distribution check originally counted metadata
registrations rather than distributions, so a single editable install seen
twice (once from its `dist-info`, once from the repo-root `egg-info`) read as a
conflict, and the verdict changed with how the script was invoked.
"""

import pathlib
import unittest
from unittest import mock

from scripts import verify_environment as verify

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _owners(*names):
  """Patches the import-to-distribution map `packages_distributions` returns."""
  return mock.patch.object(
      verify.metadata,
      "packages_distributions",
      return_value={"meridian": list(names)},
  )


def _run(check, *args):
  report = verify.Report()
  check(report, *args)
  return report


class NameNormalizationTest(unittest.TestCase):

  def test_pep503_spellings_collapse(self):
    self.assertEqual(
        verify._normalized("Meridian_MMM.Fork"), "meridian-mmm-fork"
    )

  def test_none_survives(self):
    self.assertIsNone(verify._normalized(None))


class DuplicateDistributionTest(unittest.TestCase):

  def test_one_distribution_passes(self):
    with _owners("meridian-mmm-fork"):
      self.assertFalse(_run(verify.check_duplicate_distribution, _REPO_ROOT).failed)

  def test_the_same_distribution_registered_twice_is_not_a_conflict(self):
    # The regression. One editable install can be reported once from its
    # `dist-info` and again from the repo-root `egg-info`.
    with _owners("meridian-mmm-fork", "meridian-mmm-fork"):
      self.assertFalse(_run(verify.check_duplicate_distribution, _REPO_ROOT).failed)

  def test_the_same_name_spelled_two_ways_is_not_a_conflict(self):
    with _owners("meridian-mmm-fork", "meridian_mmm_fork"):
      self.assertFalse(_run(verify.check_duplicate_distribution, _REPO_ROOT).failed)

  def test_two_different_distributions_fail(self):
    with _owners("google-meridian", "meridian-mmm-fork"):
      report = _run(verify.check_duplicate_distribution, _REPO_ROOT)
    self.assertTrue(report.failed)
    self.assertIn("google-meridian", report.render())

  def test_the_fix_names_the_stale_distribution_not_the_current_one(self):
    # Naming the wrong one here is how a developer uninstalls the copy they
    # need, so the message has to single out the distribution that is stale.
    with _owners("google-meridian", "meridian-mmm-fork"):
      rendered = _run(verify.check_duplicate_distribution, _REPO_ROOT).render()
    self.assertIn("pip uninstall -y google-meridian", rendered)
    self.assertNotIn("pip uninstall -y meridian-mmm-fork", rendered)

  def test_no_distribution_at_all_is_reported_not_crashed(self):
    with _owners():
      self.assertFalse(_run(verify.check_duplicate_distribution, _REPO_ROOT).failed)


class PathIndependentImportTest(unittest.TestCase):
  """An install that only works from the repo root is not an install.

  The real case: the editable `.pth` in site-packages carried the macOS
  `hidden` flag, which CPython's `site` module silently skips. Every other
  check still passed -- `import meridian` succeeded because the repository
  root was on `sys.path` -- while a notebook in any other directory could not
  import the package at all.
  """

  def test_a_successful_import_from_elsewhere_passes(self):
    report = _run(verify.check_import_is_path_independent, _REPO_ROOT)
    self.assertFalse(report.failed)
    self.assertIn("this checkout", report.render())

  def test_a_failing_import_from_elsewhere_fails_with_the_cause(self):
    completed = mock.Mock(returncode=1, stdout="", stderr="ModuleNotFoundError: No module named 'meridian'")
    with mock.patch.object(verify.subprocess, "run", return_value=completed):
      report = _run(verify.check_import_is_path_independent, _REPO_ROOT)
    self.assertTrue(report.failed)
    rendered = report.render()
    self.assertIn("outside the repository", rendered)
    # The message has to name the macOS cause, or the reader has no next step.
    self.assertIn("hidden", rendered)
    self.assertIn("chflags nohidden", rendered)

  def test_an_import_resolving_outside_the_checkout_warns(self):
    # A PyPI copy winning over the checkout is a different failure, and a
    # warning rather than an error: the environment works, it is just not
    # testing this source tree.
    completed = mock.Mock(returncode=0, stdout="/usr/lib/python3/meridian/__init__.py", stderr="")
    with mock.patch.object(verify.subprocess, "run", return_value=completed):
      report = _run(verify.check_import_is_path_independent, _REPO_ROOT)
    self.assertFalse(report.failed)
    self.assertIn("outside this checkout", report.render())


class AcceleratorTest(unittest.TestCase):
  """What this machine will actually compute on, and what to do about it."""

  def _with_jax(self, backend, devices=("cpu:0",)):
    module = mock.MagicMock()
    module.default_backend.return_value = backend
    module.devices.return_value = list(devices)
    return mock.patch.dict("sys.modules", {"jax": module})

  def test_a_cpu_backend_passes_and_says_so(self):
    with self._with_jax("cpu"), mock.patch.object(
        verify.metadata,
        "version",
        side_effect=verify.metadata.PackageNotFoundError,
    ):
      report = _run(verify.check_accelerator)
    self.assertFalse(report.failed)
    self.assertIn("CPU", report.render())

  def test_jax_metal_fails_with_the_uninstall_command(self):
    # It registers a Metal device and then cannot compile for it, so a device
    # list alone would read as success.
    with self._with_jax("METAL", ["METAL:0"]), mock.patch.object(
        verify.metadata, "version", return_value="0.1.1"
    ):
      report = _run(verify.check_accelerator)
    self.assertTrue(report.failed)
    rendered = report.render()
    self.assertIn("jax-metal", rendered)
    self.assertIn("pip uninstall -y jax-metal", rendered)

  def test_a_real_accelerator_is_reported_as_the_device_in_use(self):
    with self._with_jax("cuda", ["cuda:0"]), mock.patch.object(
        verify.metadata,
        "version",
        side_effect=verify.metadata.PackageNotFoundError,
    ):
      report = _run(verify.check_accelerator)
    self.assertFalse(report.failed)
    self.assertIn("cuda", report.render())

  def test_an_unimportable_jax_fails_rather_than_passing_silently(self):
    with mock.patch.dict("sys.modules", {"jax": None}):
      report = _run(verify.check_accelerator)
    self.assertTrue(report.failed)


if __name__ == "__main__":
  unittest.main()
