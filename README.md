# About Meridian

Marketing mix modeling (MMM) is a statistical analysis technique that measures
the impact of marketing campaigns and activities to guide budget planning
decisions and improve overall media effectiveness. MMM uses aggregated data to
measure impact across marketing channels and account for non-marketing factors
that impact sales and other key performance indicators (KPIs). MMM is
privacy-safe and does not use any cookie or user-level information.

Meridian is an MMM framework that enables advertisers to set up and run their
own in-house models. Meridian helps you answer key questions such as:

*   How did the marketing channels drive my revenue or other KPI?
*   What was my marketing return on investment (ROI)?
*   How do I optimize my marketing budget allocation for the future?

Meridian is a highly customizable modeling framework that is based on
[Bayesian causal inference](https://developers.google.com/meridian/docs/causal-inference/bayesian-inference).
It is capable of handling large scale geo-level data, which is encouraged if
available, but it can also be used for national-level modeling. Meridian
provides clear insights and visualizations to inform business decisions around
marketing budget and planning. Additionally, Meridian provides methodologies to
support calibration of MMM with experiments and other prior information, and to
optimize target ad frequency by utilizing reach and frequency data.

If you are using LightweightMMM, see the
[migration guide](https://developers.google.com/meridian/docs/migrate) to help
you understand the differences between these MMM projects.

## Quickstart for this fork

The upstream instructions below install the published PyPI package. To work
with *this* repository, install it from the checkout instead.

Python 3.11, 3.12 or 3.13. Do not install into an environment that already has
TensorFlow — see the note on `tensorflow-metal` below.

```sh
# 1. Clone
git clone <your-repo-url> meridian && cd meridian

# 2. A clean, isolated environment. Do not skip this.
python3.11 -m venv ~/.venvs/meridian
source ~/.venvs/meridian/bin/activate
python -m pip install -U pip setuptools wheel

# 3. Install the checkout, with every extra the test suite needs
pip install -e ".[dev,colab,schema,mlflow,geox,scenarioplanner]"

# 4. Prove the environment works before trusting anything
python scripts/verify_environment.py
```

Step 4 should end with `RESULT: PASS`. If it does not, it names the fix. Then:

```sh
pytest meridian -q -n 8           # full suite, about 6 minutes
```

### Things that will bite you otherwise

*   **`tensorflow-metal`.** If it is present and does not match the installed
    TensorFlow, `import meridian` dies inside `libmetal_plugin.dylib` with a
    symbol-not-found error that says nothing about the real cause. Remove it:
    `pip uninstall -y tensorflow-metal`. `verify_environment.py` checks for it.
*   **Install the extras.** Without `[schema,mlflow,geox,scenarioplanner]`,
    about 20 tests fail on missing imports. They are not real failures, but
    they look like them.
*   **Editable install.** If a PyPI `google-meridian` is also present, Python
    may import that instead of your checkout and your edits will appear to do
    nothing. `verify_environment.py` checks which copy is live.
*   **JAX is the default backend** from v2.0.0, at 64-bit precision. The
    TensorFlow backend still works but is deprecated upstream.
*   **One harmless install warning.** pip prints `tfp-nightly
    0.26.0.dev20260130 does not provide the extra 'substrates-jax'`. That extra
    does not exist in this nightly build, so pip ignores it and installs the
    package anyway. The JAX substrate is present regardless and the full suite
    passes on it. The declaration comes from upstream's `pyproject.toml`; it is
    left as-is rather than diverging for a cosmetic warning.
*   **No GPU on Apple Silicon.** Fits run on CPU. For scale: 20 geos x 156
    weeks x 4 channels at the demo's MCMC settings takes about 35 minutes on an
    M4 Max. Use `python -m meridian.benchmark.benchmark` to measure your own
    hardware.

## Install Meridian

Python 3.11-3.13 is required to use Meridian. We also recommend using a
minimum of 1 GPU.

Note: This project has been tested on T4 GPU using 16 GB of RAM.

To install Meridian, run the following command to automatically install the
latest release from PyPI.

*   For Linux-GPU users:

    Note: CUDA toolchain and a compatible GPU device is necessary for
    `[and-cuda]` extra to activate.

    ```sh
    $ pip install --upgrade google-meridian[and-cuda]
    ```

*   For macOS and general CPU users:

    Note: There is no official GPU support for macOS.

    ```sh
    $ pip install --upgrade google-meridian
    ```

Alternatively, run the following command to install the most recent, unreleased
version from GitHub.

*   For GPU users:

    ```sh
    $ pip install --upgrade "google-meridian[and-cuda] @ git+https://github.com/google/meridian.git"
    ```

*   For CPU users:

    ```sh
    $ pip install --upgrade git+https://github.com/google/meridian.git
    ```

We recommend to install Meridian in a fresh
[virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments)
to make sure that correct versions of all the dependencies are installed, as
defined in [pyproject.toml](https://github.com/google/meridian/blob/main/pyproject.toml).

## How to use the Meridian library

To get started with Meridian, you can run the code programmatically using sample
data with the [Getting Started Colab][3].

The Meridian model uses a holistic MCMC sampling approach called
[No U Turn Sampler (NUTS)](https://www.tensorflow.org/probability/api_docs/python/tfp/experimental/mcmc/NoUTurnSampler)
which can be compute intensive. To help with this, GPU support has been
developed across the library (out-of-the-box) using tensors. We recommend
running your Meridian model on GPUs to get real time optimization results and
significantly reduce training time.

## Differences from upstream

This is a fork of [google/meridian](https://github.com/google/meridian).
Upstream does not accept external pull requests, so fixes live here.

[`TRIAGE.md`](TRIAGE.md) records a disposition for every issue open on the
upstream tracker: verified already fixed, fixed here, a usage question, an
environment problem, a declined feature with its reason, or genuinely
unresolved.

Behaviour changes to existing code:

*   Float32 prior distributions are widened to float64 rather than rejected, so
    the documented `LogNormal(0.2, 0.9)` idiom works on the 64-bit JAX default.
*   A warning when media history is too short to fill the adstock window, which
    otherwise silently zero-pads and understates carryover.
*   Non-finite variables are excluded from the VIF check instead of aborting it.
*   Input validation errors name the offending column or coordinate.

Added modules:

*   `meridian.analysis.prior_predictive` — check priors against observed data
    from `sample_prior()` alone, before committing to a full fit.
*   `meridian.analysis.geo_diagnostics` — measure whether per-geo estimates on
    your model are precise enough to allocate budget on.
*   `meridian.benchmark` — record what actually governs runtime on your
    hardware: `python -m meridian.benchmark.benchmark --help`.
*   `meridian.validation.recovery` — generate data whose true ROI is known by
    construction, fit it, and report whether the posterior recovers the truth.
    Run it at the shape of your own data before trusting a model built on it:
    `python -m meridian.validation.recovery --help`.

Added test modules:

*   `meridian/upstream_issues_test.py` — every disposition in
    [`TRIAGE.md`](TRIAGE.md) as an executable assertion, so a rebase that
    silently regresses one of them fails the suite.
*   `meridian/math_invariants_test.py` — the arithmetic identities behind
    reported figures, verified at machine precision: `roi ==
    incremental_outcome / spend` (7.1e-15), per-geo incremental summing to the
    aggregate (4.5e-13), normalized adstock weights summing to 1, and the Hill
    closed form.

## Maintaining this fork

Upstream does not accept external pull requests, so these fixes live here
permanently and the main long-term risk is drift. Upstream moves quickly —
v1.4 to v2.0 inside nine months.

Branch layout:

*   `main` — this fork's line of development.
*   `upstream-main` — a pristine mirror of `google/meridian`, tracking the
    `upstream` remote. Never commit to it.

To take upstream changes:

```sh
git fetch upstream
git checkout upstream-main && git merge --ff-only upstream/main
git checkout main && git rebase upstream-main
pytest meridian -q -n 8                      # then the same on MERIDIAN_BACKEND=tensorflow
```

If `meridian/upstream_issues_test.py` fails after a rebase, that is the point
of it. Each test there is named for the upstream issue it guards, so a failure
tells you either that one of this fork's patches was dropped in the rebase, or
that upstream regressed something this fork depends on. Read
[`TRIAGE.md`](TRIAGE.md) for the reasoning behind that specific issue before
changing the test.

`meridian/math_invariants_test.py` failing after a rebase is more serious: it
means the arithmetic behind reported ROI changed. Do not paper over it.

## Reading ROI intervals honestly

Recovery testing on synthetic data found a limit worth knowing before any of
these numbers reach a client.

When the simulated response has the concave shape Meridian assumes, recovery is
good: the premium channel's ROI came back within 2.8% of truth, inside the 90%
credible interval. When the true response is **linear** instead, the same
channel's ROI was overstated by 71% and the interval did not contain the truth.
Cutting the noise tenfold did not help — the interval narrowed to about ±3% and
still missed every true value.

This is not a Meridian defect. It is what fitting a concave saturation curve to
a response that is not concave does, in any MMM that assumes saturation. The
consequence for reporting is:

> Meridian's credible intervals quantify parameter uncertainty **conditional on
> the assumed saturation shape**. They do not cover being wrong about that
> shape.

A channel far from saturation — typically one at low spend, whose real response
is still close to linear — can therefore have its ROI overstated substantially
while its interval looks reassuringly tight. Use
`meridian.validation.recovery` with `--response linear` to size that effect at
your own data shape, and treat the gap as a floor on the uncertainty you carry
into a recommendation.

## Meridian Documentation & Tutorials

The following documentation, colab, and video resources will help you get
started quickly with using Meridian:

| Resource                    | Description                                    |
| --------------------------- | ---------------------------------------------- |
| [Meridian documentation][1] | Main landing page for Meridian documentation.  |
| [Meridian basics][2]        | Learn about Meridian features, methodologies, and the model math. |
| [Getting started colab][3]  | Install and quickly learn how to use Meridian with this colab tutorial using sample data. |
| [User guide][4]             | A detailed walk-through of how to use Meridian and generating visualizations using your own data. |
| [Pre-modeling][5]           | Prepare and analyze your data before modeling. |
| [Modeling][6]               | Modeling guidance for model refinement and edge cases. |
| [Post-modeling][7]          | Post-modeling guidance for model fit, visualizations, optimizations, refreshing the model, and debugging. |
| [Scenario planning][8]      | Plan budget allocation with Meridian on Colab & Data Studio interactively. |
| [Migrate from LMMM][9]      | Learn about the differences between Meridian and LightweightMMM as you consider migrating. |
| [Full-funnel colab][10]     | Learn how to build a full-funnel MMM with intermediate mediator variables. |
| [API Reference][11]         | API reference documentation for the Meridian package. |
| [Reference list][12]        | White papers and other referenced material.    |

[1]: https://developers.google.com/meridian
[2]: https://developers.google.com/meridian/docs/basics/meridian-introduction
[3]: https://developers.google.com/meridian/notebook/meridian-getting-started
[4]: https://developers.google.com/meridian/docs/user-guide/installing
[5]: https://developers.google.com/meridian/docs/pre-modeling/collect-data
[6]: https://developers.google.com/meridian/docs/advanced-modeling/control-variables
[7]: https://developers.google.com/meridian/docs/post-modeling/model-fit
[8]: https://developers.google.com/meridian/docs/scenario-planning/meridian-scenario-planner
[9]: https://developers.google.com/meridian/docs/migrate
[10]: https://colab.sandbox.google.com/github/google/meridian/blob/main/demo/Meridian_Full_Funnel.ipynb
[11]: https://developers.google.com/meridian/reference/api/meridian
[12]: https://developers.google.com/meridian/docs/reference-list

## Support

**Questions about methodology**: Please see the [Modeling](https://developers.google.com/meridian/docs/basics/meridian-introduction) tab in the technical documentation.

**Issues installing or using Meridian**: Feel free to post questions in the
[Discussions](https://github.com/google/meridian/discussions) or [Issues](https://github.com/google/meridian/issues) tabs of the Meridian GitHub repository. The Meridian team responds to
these questions weekly in batches, so please be patient and don't reach out
directly to your Google Account teams.

**Bug reports**: Please post bug reports to the [Issues](https://github.com/google/meridian/issues)
tab of the Meridian GitHub repository. We also encourage the community to share
tips and advice with each other on the [Issues](https://github.com/google/meridian/issues)
tab. When our team addresses or resolves a new bug, we will notify you through
the comments on the issue.

**Feature requests**: Please post these to the [Discussions](https://github.com/google/meridian/discussions)
tab of the Meridian GitHub repository. We have an internal roadmap for Meridian
development, but would love your inputs for new feature requests so that we can
prioritize them based on the roadmap.

**Pull requests**: These are appreciated but are very difficult for us to merge
because the code in this repository is linked to Google internal systems and has
to pass internal review. If you submit a pull request and we believe that we can
incorporate a change in the base code, we will reach out to you directly about
this.

## Citing Meridian

To cite this repository:

<!-- mdlint off(SNIPPET_INVALID_LANGUAGE) -->
```BibTeX
@software{meridian_github,
  author = {Google Meridian Marketing Mix Modeling Team},
  title = {Meridian: Marketing Mix Modeling},
  url = {https://github.com/google/meridian},
  version = {2.0.0},
  year = {2026},
}
```
