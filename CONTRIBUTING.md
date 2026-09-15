# How to Contribute

This is a personal fork of
[google/meridian](https://github.com/google/meridian), maintained by Sundar
Ramesh Kumar. It is not affiliated with, endorsed by, or supported by Google.

It is published so the fixes are usable and auditable, not to run a community
project. Issues are welcome; pull requests may or may not be merged, depending
on whether the change fits how this fork is used. If you need a guarantee,
fork it yourself — that is what Apache 2.0 is for.

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
    Note that Google states it is not currently accepting external pull
    requests, so expect to file an issue rather than a patch.
*   **Here**, if it is a defect in this fork's own modules
    (`meridian/validation`, `meridian/benchmark`,
    `meridian.analysis.prior_predictive`, `meridian.analysis.geo_diagnostics`),
    or a fix for something [`TRIAGE.md`](TRIAGE.md) records as unresolved.

## What a change needs

*   **A test.** Behaviour changes need one that fails before the change.
*   **A green suite on both backends.** JAX is the default; the TensorFlow
    backend is deprecated upstream but still exercised in CI:

    ```sh
    pytest meridian -q -n 8
    MERIDIAN_BACKEND=tensorflow pytest meridian -q -n 8
    ```

    Use `backend.tfd` and `backend.np_float_dtype` rather than importing a TFP
    substrate directly, or your code will only work on one backend.
*   **An updated [`TRIAGE.md`](TRIAGE.md)** if the change alters the
    disposition of an upstream issue, and an updated
    [`NOTICE`](NOTICE) if it adds or modifies a file — Apache 2.0 section 4(b)
    requires modified files to say so.

## Two tests to take seriously

*   `meridian/upstream_issues_test.py` makes every disposition in `TRIAGE.md`
    executable. A failure means a patch was dropped or upstream regressed
    something this fork relies on.
*   `meridian/math_invariants_test.py` asserts the arithmetic behind reported
    ROI — that `roi == incremental_outcome / spend`, that a geo breakdown sums
    to its total, that adstock weights sum to one. A failure here means the
    numbers changed. Do not adjust the test to make it pass without
    understanding why.

## Code style

Match the surrounding code: two-space indent, Google-style docstrings,
`pyink`/`pylint` are declared in the `[dev]` extra.
