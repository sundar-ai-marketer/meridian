<!-- NOTICE: This file was modified from the original google/meridian source.
     See the NOTICE file at the repository root for details. -->

# How to Contribute

This is a personal fork of
[google/meridian](https://github.com/google/meridian), maintained by Sundar
Ramesh Kumar. It is not affiliated with, endorsed by, or supported by Google.

Reproducible bug reports, documentation improvements, and focused pull requests
are welcome. Keep changes within the fork's documented scope and include
evidence that a proposed fix works. Review and response times depend on
maintainer availability.

**There is no Contributor License Agreement for this fork.** The upstream
project requires one; this one does not. Anything merged here is under the
Apache License 2.0, the same licence as the code. By opening a pull request you
confirm you have the right to contribute the work under those terms.

## Where a change belongs

Before opening a pull request, decide whether the change belongs here or
upstream.

*   **Upstream**, if it is a defect in Google's code unrelated to this fork's
    changes. Reproduce it against a clean `pip install google-meridian` first,
    then report it at [google/meridian](https://github.com/google/meridian/issues).
    Check the upstream contribution policy before proposing a patch.
*   **Here**, if it is a defect in this fork's own modules
    (`meridian/validation`, `meridian/benchmark`,
    `meridian.analysis.prior_predictive`, `meridian.analysis.geo_diagnostics`),
    or a fix for something [`TRIAGE.md`](TRIAGE.md) records as unresolved.

## What a change needs

*   **A test.** Behaviour changes need one that fails before the change.
*   **A green suite on both backends.** JAX is the default; the TensorFlow
    backend is deprecated upstream but still exercised in CI:

    ```sh
    make test
    make test-tf
    ```

    These use the same process-isolated phases as CI, with two workers by
    default. Use `TEST_WORKERS=1` on smaller machines. All phases run even if
    a test fails; the final exit code preserves failure. Run `make test-e2e`
    for the real fit-to-report integration check.
    Report changes also need a browser check against the generated HTML:

    ```sh
    python -m pip install playwright
    python -m playwright install chromium
    python scripts/test_end_to_end.py --output-dir /tmp/meridian-e2e
    python scripts/test_report_browser.py /tmp/meridian-e2e/summary.html
    ```

    For an optimization report, add `--min-desktop-chart-width 500` to reflect
    its two-column desktop layout. The default 600px check applies to the
    model-summary and EDA reports; mobile overflow and keyboard checks remain
    the same for all three. The browser check disables networking and rejects
    external resource requests, so an accidentally reintroduced CDN dependency
    fails the check.

    Use `backend.tfd` and `backend.np_float_dtype` rather than importing a TFP
    substrate directly, or your code will only work on one backend.
*   **An updated [`TRIAGE.md`](TRIAGE.md)** if the change alters the
    disposition of an upstream issue, and an updated
    [`NOTICE`](NOTICE) if it adds or modifies a file — Apache 2.0 section 4(b)
    requires modified files to say so.

## Two tests to take seriously

*   `meridian/upstream_issues_test.py` makes the code-testable dispositions in
    `TRIAGE.md` executable — 15 of the 47 open upstream issues. A failure means
    a patch was dropped or upstream regressed something this fork relies on.
*   `meridian/math_invariants_test.py` asserts the arithmetic behind reported
    ROI — that `roi == incremental_outcome / spend`, that a geo breakdown sums
    to its total, that adstock weights sum to one. A failure here means the
    numbers changed. Do not adjust the test to make it pass without
    understanding why.

## Code style

Match the surrounding code: two-space indent, Google-style docstrings,
`pyink`/`pylint` are declared in the `[dev]` extra.

## Maintaining a release

1. Reproduce a reported defect before changing behavior, then retain a focused
   regression test. Keep statistical fixes separate from changes to model
   assumptions.
2. For dependency updates, check compatibility with both numerical backends.
   Run the full suites, integration check, package builds, and a clean Docker
   build; review GitHub's Python-version matrix before releasing.
   Keep the frozen dependency lock and its package metadata consistent. GitHub
   Actions and schema includes use full commit pins; update them deliberately
   and verify source provenance. Report assets have their own source and hash
   manifest in `meridian/templates/assets/` and must retain their licenses.
   Automated uv updates preserve the declared compatibility bounds; widening
   those bounds or changing Docker's Python minor version requires a deliberate
   compatibility review. Weekly proposals are grouped to bound CI workload.
3. For reports, run the browser check above. For sampling or diagnostic changes,
   inspect convergence, effective sample size, divergences, and recovery results
   as appropriate. Do not loosen thresholds just to obtain a passing result.
4. Review dependency advisories and scan the intended Git history for secrets
   before publication. Record the date, environment, test results, and remaining
   limits in `AUDIT.md`; update attribution in `NOTICE`.
5. Keep changes in reviewable commits so a faulty release can be reverted.
   The inherited PyPI publishing workflows are disabled for this fork; a GitHub
   push does not publish a Python package to PyPI.
