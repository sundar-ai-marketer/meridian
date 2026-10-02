# Copyright 2026 Sundar Ramesh Kumar.
# Licensed under the Apache License, Version 2.0.
# NOTICE: This file is new in this fork; see NOTICE at the repository root.

"""Fail-closed contracts for refreshing reports from existing smoke models."""

import dataclasses
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from meridian import constants as c
from meridian.model import spec
from scripts import test_end_to_end as e2e


def _saved_fixture():
  context = SimpleNamespace(
      n_geos=2,
      n_times=20,
      n_media_times=22,
      n_controls=1,
      n_media_channels=2,
      n_rf_channels=0,
      n_organic_media_channels=0,
      n_organic_rf_channels=0,
      n_non_media_channels=0,
  )
  return SimpleNamespace(
      model_context=context,
      input_data=e2e._fixture_data(e2e._DEFAULT_CASE),
      model_spec=spec.ModelSpec(max_lag=2),
      inference_data=SimpleNamespace(
          posterior=SimpleNamespace(sizes={c.CHAIN: 2, c.DRAW: 20})
      ),
  )


class ReuseFixtureTest(unittest.TestCase):

  def test_valid_saved_fixture_contract(self):
    e2e._validate_saved_case(_saved_fixture(), e2e._DEFAULT_CASE)

  def test_unknown_case_is_rejected(self):
    with self.assertRaisesRegex(ValueError, 'Unknown smoke case'):
      e2e._validate_saved_case(_saved_fixture(), 'arbitrary')

  def test_mismatched_dimensions_are_rejected(self):
    model = _saved_fixture()
    model.model_context.n_geos = 1
    with self.assertRaisesRegex(ValueError, 'n_geos'):
      e2e._validate_saved_case(model, e2e._DEFAULT_CASE)

  def test_mismatched_specification_is_rejected(self):
    model = _saved_fixture()
    model.model_spec = dataclasses.replace(model.model_spec, max_lag=4)
    with self.assertRaisesRegex(ValueError, 'specification'):
      e2e._validate_saved_case(model, e2e._DEFAULT_CASE)

  def test_same_shape_different_values_are_rejected(self):
    model = _saved_fixture()
    controls = model.input_data.controls.copy(deep=True)
    controls.values.flat[0] += 1.0
    model.input_data = dataclasses.replace(model.input_data, controls=controls)
    with self.assertRaisesRegex(ValueError, 'input field controls'):
      e2e._validate_saved_case(model, e2e._DEFAULT_CASE)

  def test_mismatched_draws_are_rejected(self):
    model = _saved_fixture()
    model.inference_data.posterior.sizes[c.DRAW] = 40
    with self.assertRaisesRegex(ValueError, '20 draws'):
      e2e._validate_saved_case(model, e2e._DEFAULT_CASE)

  def test_missing_model_does_not_fall_back_to_sampling(self):
    with tempfile.TemporaryDirectory() as directory:
      with mock.patch.object(e2e, '_fit_case') as fit:
        with self.assertRaisesRegex(FileNotFoundError, 'No saved smoke model'):
          e2e.run(Path(directory), reuse_models=True)
      fit.assert_not_called()


if __name__ == '__main__':
  unittest.main()
