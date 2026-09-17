# Copyright 2026 Sundar Ramesh Kumar.
# Licensed under the Apache License, Version 2.0.
# NOTICE: This file is new in this fork; see NOTICE at the repository root.

"""End-to-end contracts for the prior predictive audit, on synthetic data.

The audit's claim -- that the shipped prior generates data the library rejects
-- only carries weight if the measurement is real. So these tests build actual
Meridian models on a small synthetic dataset, sample their priors, and check
the reported percentages against what the draws contain. They deliberately do
not mock the model.

Kept small (two geos, sixteen periods, a handful of prior draws) because
`sample_prior` is cheap; no MCMC runs here.
"""

import contextlib
import io
import json
import pathlib
import tempfile
import unittest

import numpy as np

from scripts import prior_predictive_audit


_TIGHT = {
    "sigma_scale": 0.1,
    "baseline_sd": 0.1,
    "hier_sd": 0.1,
    "roi_median": 2.0,
    "roi_sigma": 0.7,
}
_DEFAULT_WIDTH = {
    "sigma_scale": 5.0,
    "baseline_sd": 5.0,
    "roi_median": 2.0,
    "roi_sigma": 0.7,
}


def _fixture(n_geos=2, n_times=16):
  from meridian.validation import recovery

  config = recovery.RecoveryConfig(
      true_roi=(1.0, 2.0),
      spend_scale=(3_000.0, 6_000.0),
      n_geos=n_geos,
      n_times=n_times,
      max_lag=2,
      seed=11,
  )
  template, _ = recovery.simulate(config)
  return template, config


class ModelConstructionTest(unittest.TestCase):

  def test_no_override_leaves_the_library_defaults_in_place(self):
    from meridian import constants as c

    template, config = _fixture()
    model = prior_predictive_audit._build_model(template, config, None)
    sigma = model.model_spec.prior.sigma
    # HalfNormal(5) is the shipped default; the audit's headline rests on it.
    self.assertAlmostEqual(float(np.asarray(sigma.scale)), 5.0, places=5)
    self.assertEqual(model.model_spec.media_prior_type, c.ROI)

  def test_overrides_reach_the_model_spec(self):
    template, config = _fixture()
    model = prior_predictive_audit._build_model(template, config, _TIGHT)
    prior = model.model_spec.prior
    self.assertAlmostEqual(float(np.asarray(prior.sigma.scale)), 0.1, places=6)
    self.assertAlmostEqual(
        float(np.asarray(prior.knot_values.scale)), 0.1, places=6
    )
    self.assertAlmostEqual(float(np.asarray(prior.eta_m.scale)), 0.1, places=6)
    self.assertAlmostEqual(float(np.asarray(prior.xi_c.scale)), 0.1, places=6)

  def test_beta_m_is_not_overridden_because_roi_parameterization_ignores_it(self):
    # Meridian warns that a custom `beta_m` is ignored when media_prior_type
    # is "roi". Setting it would make the sweep look like it tightened
    # something it did not.
    template, config = _fixture()
    model = prior_predictive_audit._build_model(template, config, _TIGHT)
    self.assertAlmostEqual(
        float(np.asarray(model.model_spec.prior.beta_m.scale)), 5.0, places=6
    )

  def test_omitting_hier_sd_leaves_the_hierarchical_scales_alone(self):
    # This is the distinction the audit's sweep turns on: tightening the
    # population-level terms alone is not the same as tightening everything.
    template, config = _fixture()
    model = prior_predictive_audit._build_model(
        template, config, {**_DEFAULT_WIDTH, "baseline_sd": 0.1}
    )
    self.assertAlmostEqual(
        float(np.asarray(model.model_spec.prior.eta_m.scale)), 1.0, places=6
    )


class MeasurementTest(unittest.TestCase):

  def test_measurement_reports_the_keys_the_evidence_file_needs(self):
    template, config = _fixture()
    model = prior_predictive_audit._build_model(template, config, _TIGHT)
    measured = prior_predictive_audit._measure(model, draws=8, seed=11)

    for key in (
        "draws",
        "observed_kpi",
        "conditional_mean",
        "prior_predictive",
        "sigma_prior_draws",
        "scaled_value_where_revenue_reaches_zero",
    ):
      self.assertIn(key, measured)
    self.assertEqual(measured["draws"], 8)
    json.dumps(measured)

  def test_percentages_are_within_range(self):
    template, config = _fixture()
    model = prior_predictive_audit._build_model(template, config, _TIGHT)
    measured = prior_predictive_audit._measure(model, draws=8, seed=11)
    for section in ("conditional_mean", "prior_predictive"):
      for key, value in measured[section].items():
        if key.endswith("_pct"):
          self.assertGreaterEqual(value, 0.0)
          self.assertLessEqual(value, 100.0)

  def test_the_observed_kpi_is_positive_so_negativity_is_a_prior_property(self):
    # If the fixture itself contained negative revenue the finding would be
    # about the fixture, not about the prior.
    template, config = _fixture()
    model = prior_predictive_audit._build_model(template, config, _TIGHT)
    measured = prior_predictive_audit._measure(model, draws=4, seed=11)
    self.assertGreater(measured["observed_kpi"]["min"], 0.0)

  def test_a_wide_prior_produces_more_negative_mass_than_a_tight_one(self):
    # The direction is the audit's whole claim. Exact percentages move with
    # the fixture; the ordering must not.
    template, config = _fixture()
    tight = prior_predictive_audit._measure(
        prior_predictive_audit._build_model(template, config, _TIGHT),
        draws=16,
        seed=11,
    )
    wide = prior_predictive_audit._measure(
        prior_predictive_audit._build_model(template, config, _DEFAULT_WIDTH),
        draws=16,
        seed=11,
    )
    self.assertGreater(
        wide["conditional_mean"]["negative_cells_pct"],
        tight["conditional_mean"]["negative_cells_pct"],
    )

  def test_the_shipped_default_prior_generates_rejected_data(self):
    # The finding recorded in AUDIT.md and TRIAGE.md, asserted directly.
    template, config = _fixture()
    model = prior_predictive_audit._build_model(template, config, None)
    measured = prior_predictive_audit._measure(model, draws=16, seed=11)
    self.assertGreater(
        measured["prior_predictive"]["draws_with_a_negative_cell_pct"], 50.0
    )

  def test_measurement_is_reproducible_for_a_fixed_seed(self):
    template, config = _fixture()
    first = prior_predictive_audit._measure(
        prior_predictive_audit._build_model(template, config, _TIGHT),
        draws=8,
        seed=11,
    )
    second = prior_predictive_audit._measure(
        prior_predictive_audit._build_model(template, config, _TIGHT),
        draws=8,
        seed=11,
    )
    self.assertEqual(
        first["prior_predictive"]["negative_cells_pct"],
        second["prior_predictive"]["negative_cells_pct"],
    )


class CommandLineTest(unittest.TestCase):

  def test_the_sweep_writes_evidence_covering_every_variant(self):
    with tempfile.TemporaryDirectory() as directory:
      out = pathlib.Path(directory) / "audit.json"
      with contextlib.redirect_stdout(io.StringIO()):
        status = prior_predictive_audit.main(
            ["--draws", "4", "--geos", "2", "--times", "16",
             "--output", str(out)]
        )
      self.assertEqual(status, 0)
      evidence = json.loads(out.read_text(encoding="utf-8"))

    self.assertEqual(len(evidence["variants"]), 5)
    labels = [variant["variant"] for variant in evidence["variants"]]
    self.assertEqual(labels[0], "meridian defaults")
    self.assertIn("hierarchical scales tightened too", labels)
    self.assertIn("HalfNormal(5)", evidence["library_default_priors_on_the_scaled_kpi"]["sigma"])
    self.assertTrue(evidence["limitations"])


if __name__ == "__main__":
  unittest.main()
