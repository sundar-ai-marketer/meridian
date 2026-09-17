# Copyright 2026 Sundar Ramesh Kumar.
# Licensed under the Apache License, Version 2.0.
# NOTICE: This file is new in this fork; see NOTICE at the repository root.

"""Aggregation contracts for the prior-sensitivity sweep.

The sweep's headline number is how far the posterior median moves across
priors, expressed as a fraction of the channel's true ROI. These tests run that
aggregation on synthetic fits, where the right answer is known by hand, because
the number is only worth quoting if it is computed the way the document says.
"""

import json
import math
import unittest

from scripts import prior_sensitivity


def _fit(label, medians, lows=None, highs=None):
  n = len(medians)
  lows = lows if lows is not None else [m - 1.0 for m in medians]
  highs = highs if highs is not None else [m + 1.0 for m in medians]
  return {
      "label": label,
      "channels": [
          {
              "channel": f"channel_{i}",
              "median": medians[i],
              "ci_low": lows[i],
              "ci_high": highs[i],
          }
          for i in range(n)
      ],
  }


class PriorGridTest(unittest.TestCase):

  def test_the_grid_spans_sceptical_through_optimistic(self):
    medians = [median for _, median, _ in prior_sensitivity._PRIOR_GRID]
    self.assertLess(min(medians), 1.0)
    self.assertGreater(max(medians), 4.0)

  def test_every_grid_entry_is_a_usable_lognormal(self):
    for label, median, sigma in prior_sensitivity._PRIOR_GRID:
      self.assertIsInstance(label, str)
      self.assertGreater(median, 0.0, label)
      self.assertGreater(sigma, 0.0, label)

  def test_grid_labels_are_unique(self):
    # They key the per-channel median map; duplicates would silently drop fits.
    labels = [label for label, _, _ in prior_sensitivity._PRIOR_GRID]
    self.assertEqual(len(labels), len(set(labels)))


class SummarizeTest(unittest.TestCase):

  def test_spread_is_the_range_of_medians_across_priors(self):
    fits = [_fit("a", [1.0]), _fit("b", [4.0]), _fit("c", [2.0])]
    summary = prior_sensitivity.summarize(fits, ["channel_0"], [2.0])
    self.assertAlmostEqual(summary[0]["spread"], 3.0)

  def test_prior_share_is_the_spread_relative_to_the_true_value(self):
    # Two channels with the same absolute spread are not equally sensitive if
    # their true ROIs differ; dividing by truth is what makes them comparable.
    fits = [_fit("a", [1.0, 1.0]), _fit("b", [2.0, 2.0])]
    summary = prior_sensitivity.summarize(fits, ["c0", "c1"], [1.0, 4.0])
    self.assertAlmostEqual(summary[0]["prior_share_of_true_roi"], 1.0)
    self.assertAlmostEqual(summary[1]["prior_share_of_true_roi"], 0.25)

  def test_a_prior_insensitive_channel_reports_zero_spread(self):
    fits = [_fit("a", [2.0]), _fit("b", [2.0])]
    summary = prior_sensitivity.summarize(fits, ["channel_0"], [2.0])
    self.assertEqual(summary[0]["spread"], 0.0)
    self.assertEqual(summary[0]["prior_share_of_true_roi"], 0.0)

  def test_zero_true_roi_yields_nan_rather_than_dividing_by_zero(self):
    fits = [_fit("a", [1.0]), _fit("b", [2.0])]
    summary = prior_sensitivity.summarize(fits, ["channel_0"], [0.0])
    self.assertTrue(math.isnan(summary[0]["prior_share_of_true_roi"]))

  def test_coverage_is_true_only_when_every_interval_contains_the_truth(self):
    covering = [
        _fit("a", [2.0], lows=[1.0], highs=[3.0]),
        _fit("b", [2.2], lows=[1.2], highs=[3.2]),
    ]
    self.assertTrue(
        prior_sensitivity.summarize(covering, ["c"], [2.0])[0][
            "every_interval_covers_truth"
        ]
    )

    missing = covering + [_fit("c", [9.0], lows=[8.0], highs=[10.0])]
    self.assertFalse(
        prior_sensitivity.summarize(missing, ["c"], [2.0])[0][
            "every_interval_covers_truth"
        ]
    )

  def test_interval_endpoints_count_as_covering(self):
    fits = [_fit("a", [2.0], lows=[2.0], highs=[3.0])]
    self.assertTrue(
        prior_sensitivity.summarize(fits, ["c"], [2.0])[0][
            "every_interval_covers_truth"
        ]
    )

  def test_median_by_prior_is_keyed_by_label(self):
    fits = [_fit("sceptical", [1.0]), _fit("optimistic", [3.0])]
    summary = prior_sensitivity.summarize(fits, ["channel_0"], [2.0])
    self.assertEqual(
        summary[0]["median_by_prior"], {"sceptical": 1.0, "optimistic": 3.0}
    )

  def test_summary_covers_every_channel(self):
    fits = [_fit("a", [1.0, 2.0, 3.0]), _fit("b", [1.5, 2.5, 3.5])]
    summary = prior_sensitivity.summarize(fits, ["x", "y", "z"], [1.0, 2.0, 3.0])
    self.assertEqual([entry["channel"] for entry in summary], ["x", "y", "z"])

  def test_summary_is_json_serializable(self):
    fits = [_fit("a", [1.0]), _fit("b", [2.0])]
    summary = prior_sensitivity.summarize(fits, ["channel_0"], [1.0])
    json.dumps(prior_sensitivity._json_safe(summary))


class JsonSafeTest(unittest.TestCase):

  def test_booleans_survive_as_booleans(self):
    result = prior_sensitivity._json_safe({"covered": True})
    self.assertIs(result["covered"], True)

  def test_nan_becomes_null(self):
    result = prior_sensitivity._json_safe({"share": float("nan")})
    self.assertIsNone(result["share"])
    json.dumps(result)


if __name__ == "__main__":
  unittest.main()
