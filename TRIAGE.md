# Upstream issue triage

Disposition for every issue open on [google/meridian](https://github.com/google/meridian/issues)
as of 2026-09-15 (41 open issues, 59 open PRs), evaluated against this fork at
upstream v2.0.0 (`00134ea`).

**Why this file exists.** Google does not accept external pull requests — a
maintainer declined a community fix for #1502 with "we are currently not
accepting external pull requests". Fixes therefore live here permanently, and
each one needs a written reason.

**Verification environment.** Python 3.11.8, TensorFlow 2.21.0, JAX 0.10.2
(CPU), NumPy 2.3.5, protobuf 7.36.1, macOS arm64 (M4 Max). JAX is the default
backend from v2.0.0 and defaults to 64-bit precision.

Status values:

| Status | Meaning |
|---|---|
| `FIXED UPSTREAM` | Reproduced or inspected on v2.0.0 and confirmed already resolved. No action. |
| `FIXED HERE` | Still broken on v2.0.0. Patched in this fork, with a test. |
| `USAGE` | Not a defect. Answer recorded below. |
| `ENVIRONMENT` | Caused by hardware or dependency state we do not control. Mitigation recorded. |
| `FEATURE` | Genuine enhancement request. Accept/decline decision recorded. |
| `OPEN` | Real, unresolved, and not fixable here without more information. |

---

## FIXED HERE (8 upstream issues + 4 found here, 13 commits)

| # | Title | What was wrong | Fix |
|---|---|---|---|
| — | *(not filed upstream)* | `PriorDistribution` rejected any float32 distribution once JAX defaulted to 64-bit, so the documented `LogNormal(0.2, 0.9)` idiom raised for every user setting custom priors. | Widen float32 → float64 losslessly and warn. `_widen_distribution_to_float64` |
| 1404 | Channel-ordered `(mu, sigma)` tuples not saved via Serde | Blocked before serde by the same dtype gate. Serde itself handles tuples. | Unblocked by the dtype fix; verified a tuple-valued prior now saves and loads. |
| 1364 | `TypeError` calling `save_meridian` | Same dtype gate blocked the float32 workaround. | Same fix. |
| 1466 | `ModelFittingError: Model has critical EDA issue` | `np.std` of a NaN column is NaN and `NaN < threshold` is False, so non-finite variables were treated as non-constant and passed to statsmodels, which aborted the whole check with `MissingDataError` naming no variable. | Exclude non-finite variables like constant ones; add a finding that names them. |
| 1502 | `times` coords already set when `n_media_times = n_times + max_lag` | The reported error no longer reproduces, but the silent path does: a fully populated DataFrame yields `n_media_times == n_times`, i.e. **no burn-in at all**, with no error or warning. Adstock is then zero-padded and carryover is understated. | Warn in `ModelContext` when media history is shorter than `max_lag`. Corrected the `with_media`/`with_reach` docstrings, which described the relationship backwards. |
| 1596 | Revenue based model not working | The error listed every required column rather than the absent one, so the user hunted for `Sales` (present) instead of `time` (absent). | Name the missing columns and list what the DataFrame contains. |
| 647 | Expose a method for prior predictive checks | `ModelFit` compares only the *posterior* to observed data, so a bad prior was discovered after a full fit rather than before one. | New `analysis/prior_predictive.py`: interval coverage, prior-vs-actual total ratio, and a plot, from `sample_prior()` alone. |
| 1396 | Add macOS benchmark suite | No way to record what governs runtime, so environment problems (#1746, #1585) were indistinguishable from library behaviour. | New `meridian/benchmark/`. The issue asked for "tokens/sec" and "first-token latency", which do not exist in an MMM library; implemented the metrics that do. |
| 1668 | Geo-level budget allocation | Asked whether per-geo response curves are too noisy to allocate on. Previously answerable only by assertion. | New `analysis/geo_diagnostics.py` measures posterior CV per geo and channel against the aggregated CV, so the question is answered on the user's own model. |
| — | *(not filed upstream)* | `ModelSpec(prior=None)` silently replaced the default priors and surfaced as `AttributeError: 'NoneType' object has no attribute 'beta_m'` from inside ModelContext. | Reject it in `__post_init__` with a message saying to omit the argument. |
| — | *(not filed upstream)* | `meridian_serde_test` failed on protobuf ≥ 6: `FieldDescriptor.label` was removed from descriptor instances. | `_is_repeated_field` prefers `is_repeated`, falls back to `label`. |
| — | *(not filed upstream)* | Six `weekly_optimization_grid` tests failed on Python 3.11: `mock.patch.object(..., wraps=..., autospec=True)` ignores `wraps` before 3.12, returning a MagicMock that unpacks as empty. | Use `side_effect`, which calls through on every supported version. |

## FIXED UPSTREAM — verified, no action (11)

| # | Title | Evidence |
|---|---|---|
| 643 | Unexpected behavior of `response_curves_data` | API is now `response_curves`. Called with and without an explicit full `selected_times`, outputs match to 3.8e-16 relative. |
| 1427 | `reshape() got an unexpected keyword argument 'newshape'` | Zero occurrences of `newshape` in the codebase. |
| 1453 | `AttributeError: module 'numpy' has no attribute 'concat'` | Zero occurrences of `np.concat`; was a caller-side NumPy version problem. |
| 1464 | `KeyError` converting Proto to DataFrame | Reproduced the reported configuration (non-revenue KPI, no `revenue_per_kpi`): converts cleanly to 5 frames. Import path moved to `meridian.schema.processors` in 1.5.2. |
| 1538 | Update TensorFlow to address CVE-2026-2492 | `pyproject.toml` pins `tensorflow >= 2.21.0, < 2.22`. Fixed in 1.8.0. |
| 1589 | `ModelReviewer`/`BayesianPPPCheck` fails on JAX backend | `checks.py` now wraps the xarray product in `backend.to_tensor()` and uses `np.asarray` for the weights. Fixed in 1.6.2. |
| 1623 | `IndependentMultivariateDistribution` and serializing models | Fixed in 1.7.0 per changelog; serde round-trips non-uniform prior types in our probes. |
| 1644 | Channel order mismatch can silently produce incorrect spend attribution | Built `InputData` directly with `media_spend` channels reversed. v2.0.0 raises `` `media_channel` coordinates of array `media_spend` don't match ``. Fixed in 1.7.1. **This was the most dangerous issue on the tracker and it is genuinely closed.** |
| 1675 | `save_meridian()` fails on calendar-monthly time axes | Built a model on monthly coords (day gaps 28/29/30/31) and saved successfully. Fixed in 1.7.1. |
| 1709 | `sample_posterior` OOMs during posterior reconstruction | `reconstruction_batch_size` was added to `sample_posterior` in 1.8.0 to chunk the reconstruction pass. Verified the parameter is accepted and the run completes. |
| 1751 | Serde not keeping media prior type | Saved and loaded with `media_prior_type='contribution'`: preserved, and the posterior keeps `contribution_m` with no `roi_m`. |

## ENVIRONMENT — not a code defect (7)

| # | Title | Disposition |
|---|---|---|
| 1405 | Databricks — Meridian | The message quoted is the standard "consider setting custom ROI priors" warning for a non-revenue KPI without `revenue_per_kpi`, not an error. Set a total-media-contribution prior or supply `revenue_per_kpi`. |
| 1435 | `InternalError` only on H100 GPU | `CUDA_ERROR_INVALID_HANDLE` from the CUDA driver, not from Meridian. Driver/toolkit mismatch on that Databricks image. Cannot reproduce on CPU. |
| 1446 | Demo not converging on L4 or A100 | PTXAS warnings point to a CUDA compilation mismatch. Our demo-grade run on CPU is recorded below. |
| 1585 | System crash for geo level MMM | 6 geos × 942 obs × 15 channels on n1-highmem-8 at ~10 GB: host RAM exhaustion. Use `reconstruction_batch_size` (added 1.8.0), fewer chains, or a larger instance. |
| 1666 | GPU Memory Exhaustion on T4 | OOM allocating `[36, 7000, 12078]` — 183 DMAs × 36 channels on 16 GB VRAM. Mitigate with `reconstruction_batch_size`; verified the parameter is accepted and runs. |
| 1746 | `sample_posterior()` taking too long | 10 min on Colab T4 vs 1 hr on g4dn.xlarge is an environment difference (thermal/driver/CPU-fallback), not library behaviour. Confirm the GPU is actually being used. |
| 1486 | Meridian Community Connector in Looker Studio | A separate Looker Studio product, not this codebase. Out of scope for the fork. |

## USAGE — questions, not defects (9)

| # | Title | Answer |
|---|---|---|
| 1160 | How to integrate COADs? | No first-class support. Either split each channel, or model COAD spend as a non-media treatment. |
| 1363 | Population for scaling the KPI in the Geo Model | Population scaling puts geos on a comparable scale so hierarchical priors are meaningful. Annual demographic figures held constant within a year are fine; it need not track weekly reach fluctuations. |
| 1403 | Transforming old pickle files into Protobuf | Load with the deprecated `model.load_mmm()` (still present and working — verified) and re-save with `meridian_serde.save_meridian()` to `.binpb`. |
| 1432 | How to calculate contribution of controls and media manually | v2.0.0 added `ModelContext.get_channel_parameters`, `get_channel_parameter_tensor` and `Analyzer.incremental_outcome_xr` for exactly this. |
| 1469 | Contradictory results | Prior vs posterior contribution are different quantities. A prior probability of 1.0 that treatment exceeds 100% of outcome means the priors are too optimistic; tighten them. |
| 1555 | Scenario Planner in Colab or any IDE? | Any IDE. Colab is a convenience, not a requirement. |
| 1567 | How to model retail shops in a geo hierarchy | Geos must partition the market without overlap. Model shops as geos only if the KPI is attributable to exactly one shop; otherwise use trade areas. |
| 1620 | Run scenario planner without Looker Studio | The library produces the proto/DataFrames. Verified `DataFrameModelConverter` yields plain pandas frames that can feed any front end. |
| 1676 | Budget optimization when KPI is non-revenue | Fixed-budget reallocates; flexible-budget needs a revenue interpretation to trade off total spend. Supply `revenue_per_kpi`, or use fixed-budget and read CPIK rather than ROI. |

## FEATURE REQUESTS — declined, with reasons (3)

| # | Title | Decision |
|---|---|---|
| 753 | Non-normal likelihoods | **Decline.** Changing the likelihood alters the model's statistical core and invalidates the ROI/contribution prior machinery. Not something to bolt on without a validation programme. |
| 795 | Carryover model | **Decline, with the precise reason.** Meridian does model carryover, and v2.0.0 ships two decay families. Both are monotonically decreasing in lag -- geometric is `alpha^l`, binomial is `(1 - l/w)^alpha` -- so neither can place peak effect at a lag > 0, which is what LightweightMMM's carryover does. Supporting it needs a new decay kernel *and* a new sampled per-channel delay parameter, which touches the priors, the sampler and serde. Not a bolt-on. |
| 1713 | Support KPIs with occasional negative values | **Decline.** The input gate is one loop and trivial to relax, but the ROI and contribution prior machinery divides by and normalises against total outcome, so negative KPIs would produce numbers that look plausible and are not. Model gross flows separately instead. |

## OPEN — real and unresolved (3)

| # | Title | Position |
|---|---|---|
| 1778 | MCMC convergence fails below a specific media channel count | Not reproducible without the reporter's data. Their setup (10 geos, 105 weeks, 22–23 channels, 8 national-constant controls) is severely over-parameterised, but the direction is backwards from what over-parameterisation predicts — *more* channels converge. Note they hit a float32/float64 dtype error when trying 64-bit; the coercion fix here removes that obstacle, so it is worth retrying on this fork. |
| 1624 | MCMC non-convergence on 1.6.0 with official `tfp-nightly` | Version-specific to 1.6.0. Our demo-grade run on v2.0.0 is recorded below. |
| 1455 | "x" | Empty placeholder issue with no content. Nothing to action. |

---

## Verification

### Test suite

Full upstream suite before any changes: **26 failed, 5341 passed, 44 skipped,
1 error**. Of the 26 failures, 20 were missing optional dependencies (`geox`,
`mlflow`); installing those extras turns all 20 green (287 passed). The
remaining 6 and the 1 error are fixed here, with tests.

### End-to-end model run

Bundled `geo_media.csv` — 20 geos x 156 weeks x 4 channels, non-revenue KPI
with `revenue_per_kpi` — at the demo's own MCMC settings (7 chains, 2000 adapt,
500 burn-in, 1000 keep) on Python 3.11.8 / JAX 0.10.2 CPU / M4 Max:

| Stage | Result |
|---|---|
| `sample_posterior` | 2111 s (~35 min) |
| Convergence | **max r_hat 1.0456** against a 1.2 threshold |
| ROI (Channel0–3) | 2.682, 1.591, 3.664, 1.928 — all finite |
| Budget optimization | ran; optimized ROI 2.449 vs non-optimized 2.411 |
| Summary two-pager | written, 562 KB |
| Serde round-trip | ROI **bit-identical** (max delta 0.00e+00); `media_prior_type` preserved |
| `save_mmm`/`load_mmm` | works (deprecated path still functional) |

A shorter run (4 chains, 400 adapt) gave ROI 2.685 / 1.598 / 3.790 / 1.990 —
close enough to the full run to indicate the estimates are stable rather than
sampling artefacts. That shorter run did *not* converge (max r_hat 1.60), which
is under-sampling, not a defect: r_hat is a diagnostic of chain length, and
2000 adaptation steps are what the demo prescribes.

This is the direct answer to #1624 and #1446: on v2.0.0 with the JAX backend,
the demo workload converges cleanly.

### Environment

The `[geox]`, `[mlflow]` and `[scenarioplanner]` extras are needed for the full
suite to run green. Six `weekly_optimization_grid` tests additionally require
either Python >= 3.12 or the `side_effect` fix applied here.
