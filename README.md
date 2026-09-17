<!-- NOTICE: This file was modified from the original google/meridian source.
     See the NOTICE file at the repository root for details. -->

# Meridian: marketing mix modeling with practical diagnostics

[![CI](https://github.com/sundar-ai-marketer/meridian/actions/workflows/ci.yml/badge.svg)](https://github.com/sundar-ai-marketer/meridian/actions/workflows/ci.yml)

> **Unofficial fork.** This repository is a personal fork of
> [google/meridian](https://github.com/google/meridian), maintained by Sundar
> Ramesh Kumar. It is **not affiliated with, endorsed by, or supported by
> Google.** "Meridian" is Google's project name, used here only to identify the
> upstream work this is derived from.
>
> Fork-specific fixes are maintained here. See [`TRIAGE.md`](TRIAGE.md)
> for what changed and why, and
> [`NOTICE`](NOTICE) for attribution. Licensed under Apache 2.0, the same terms
> as the original.
>
> **Do not report problems with this fork to Google.** Use this repository's
> issue tracker.

Turn aggregated marketing data into channel ROI estimates, budget scenarios,
and an HTML report. This fork adds checks that help you judge when those
estimates deserve confidence, plus a one-command local setup and a Docker path.

* **Check assumptions before fitting:** prior predictive checks can reveal a
  mismatch between your priors and observed outcomes.
* **Inspect uncertainty after fitting:** convergence, effective sample size,
  divergences, and geographic precision diagnostics expose weak estimates.
* **Review the evidence:** [issue triage](TRIAGE.md) records the upstream
  problems addressed; [the release audit](AUDIT.md) records fresh checks,
  fixes, and remaining limitations.

This is a Python modeling library, not a hosted application. The included
sample is simulated data. Software tests cannot establish that a model is
causally valid for your business.


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

The setup helper targets Linux and Apple Silicon macOS. For Windows or Intel
Macs, use the Linux container path below. The required TensorFlow version has
no native Intel macOS wheel; see [TensorFlow's platform support](https://www.tensorflow.org/install/pip).

```sh
git clone https://github.com/sundar-ai-marketer/meridian.git meridian && cd meridian
make quickstart
```

One command: it finds a supported Python, builds an isolated environment,
removes `tensorflow-metal` if present, installs this checkout with its supported
CPU extras from `uv.lock`, compiles the report stylesheet, verifies the result,
then runs a real
end-to-end analysis on the bundled sample data. Installation and sampling can
take several minutes each. Setup can be rerun and exits with an error if its
environment checks fail. A pinned uv installer is bootstrapped inside the target
environment when needed. `MERIDIAN_INSTALL_MODE=pip make setup` is an explicit
unlocked fallback; it does not reproduce the audited lock.

`make help` lists everything:

| | |
|---|---|
| `make quickstart` | setup → stylesheet → verify → demo. The one-command path. |
| `make setup` | environment only |
| `make verify` | check an existing environment is usable |
| `make test` / `make test-tf` | core and Scenario Planner suites, on either backend |
| `make test-e2e` | real fit → budget optimization → save/load → styled report |
| `make demo` | the end-to-end example |
| `make build` | wheel + sdist |
| `make clean` | build artifacts and generated output. Never touches your venv. |

The demo builds input data, checks an ROI prior against the data before fitting,
saves the fitted model and `sampling-quality.json`, then produces exploratory
ROI, budget and HTML outputs. The first eight observed media weeks provide
carryover history; the following 148 weeks form the analysis window. The HTML
carries an exploratory-use note when shared. Runtime depends on hardware and
sampling settings; allow several minutes for a real fit.

For a stricter workflow, the following settings passed the sampling gate on the
bundled example during the audit (about 14 minutes for sampling on the audited
machine):

```sh
.venv/bin/python examples/quickstart.py --full --knots 13 \
  --n-chains 4 --n-adapt 2000 --n-burnin 500 --n-keep 4000 \
  --target-accept-prob 0.99 --sampling-mode strict \
  --output-dir quickstart_output/strict
```

Strict mode refuses to reuse a directory containing an earlier summary, so a
failed rerun cannot leave a stale report beside new diagnostic evidence.

Strict mode requires rank-normalized R-hat below 1.01, bulk and tail effective
sample sizes of at least `max(400, 100 × chains)`, and zero divergences. Missing
or non-finite diagnostics block the gate. Failure preserves the model and JSON
evidence and exits before ROI, optimization or HTML generation. Passing these
checks still requires a separate review of priors, causal assumptions and model
adequacy. The 13-knot specification is an explicit example choice, not a
universal setting. [The audit](AUDIT.md#strict-sampling-reference) records all
three trials, including the two that failed; longer sampling alone is no
guarantee.

### Without installing anything locally

*   **GitHub Codespaces / VS Code Dev Containers.** `.devcontainer/` builds the
    whole environment on container create. Open the repo in a Codespace, or
    "Reopen in Container" locally, and run `make demo`.
*   **Docker.** The image includes TensorFlow and JAX. A clean Linux arm64 build
    and real model-to-report smoke test passed during this audit; the measured
    image size was about 901 MB. Size and build time vary by platform.

Run the container demo and copy its outputs to your computer:

```sh
docker build -t meridian .
docker run --name meridian-demo meridian
docker cp meridian-demo:/app/quickstart_output ./quickstart_output
docker rm meridian-demo
```

For container security, review the [dated OS advisory inventory](AUDIT.md#container-os-security)
and [rebuild policy](SECURITY.md#container-operating-system-advisories). Available
OS fixes are applied; vendor-unfixed findings remain explicitly documented.

Open `quickstart_output/summary.html` to inspect the report. Review the printed
sampling diagnostics before interpreting the estimates. For a smaller
integration check, run `docker run --rm meridian python scripts/test_end_to_end.py`.

<details>
<summary>Manual setup, if you would rather not run a script</summary>

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m venv .venv/.meridian-uv
.venv/.meridian-uv/bin/python -m pip install pip==26.2.1 uv==0.11.14
.venv/.meridian-uv/bin/uv sync --frozen --no-default-groups --extra dev --extra colab \
  --extra schema --extra mlflow --extra geox --extra scenarioplanner
python scripts/compile_report_css.py     # see "Report stylesheet" below
python scripts/verify_environment.py     # must end with RESULT: PASS
```

</details>

To run the tests:

```sh
make test                                                # full JAX suite
make test-tf                                             # the other backend
make test-e2e                                            # small real integration run
```

### Things that will bite you otherwise

*   **`tensorflow-metal`.** If it is present and does not match the installed
    TensorFlow, `import meridian` dies inside `libmetal_plugin.dylib` with a
    symbol-not-found error that says nothing about the real cause. Remove it:
    `pip uninstall -y tensorflow-metal`. `verify_environment.py` checks for it.
*   **Report stylesheet.** `meridian/templates/style.css` is generated from
    `style.scss` by a `setup.py` build command. An editable install never runs
    that command, and the report template includes the file with `ignore
    missing` — so every generated report comes out with an empty `<style>` tag
    and no formatting, with no error. `scripts/setup.sh` compiles it for you;
    after a manual `pip install -e`, run `python scripts/compile_report_css.py`.
    `verify_environment.py` checks for it.
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
    M4 Max. Use `python -m meridian.benchmark` to measure your own
    hardware.

## Shaping your own data

`examples/quickstart.py` is the working reference — swap the one marked
DataFrame for yours. The builder call is:

```python
from meridian import constants as c
from meridian.data import data_frame_input_data_builder as dfb

data = (
    dfb.DataFrameInputDataBuilder(kpi_type=c.NON_REVENUE)
    .with_kpi(df, kpi_col='conversions', time_col='date', geo_col='geo')
    .with_revenue_per_kpi(df, revenue_per_kpi_col='revenue_per_conversion',
                          time_col='date', geo_col='geo')
    .with_population(df, population_col='population', geo_col='geo')
    .with_controls(df, control_cols=['price_index'], time_col='date',
                   geo_col='geo')
    .with_media(
        df,
        media_cols=[f'{ch}_impressions' for ch in channels],
        media_spend_cols=[f'{ch}_spend' for ch in channels],
        media_channels=channels,
        time_col='date',
        geo_col='geo',
    )
    .build()
)
```

### The one mistake that costs you accuracy silently

If media comes from the same fully-populated DataFrame as your KPI, you get
**zero adstock burn-in**. Meridian then adstocks the start of your modelling
window against zero-padded history, understating carryover and the ROI of
affected channels. Nothing errors.

Burn-in is expressed by *rows where media exists but spend does not*:

```python
# Wrong: every column populated for every date -> n_media_times == n_times
#        -> no burn-in, carryover understated

# Right: media history extends `max_lag` periods earlier, with KPI, controls
#        and spend left as NaN in those rows
df.loc[df['date'] < window_start, ['conversions', 'price_index',
                                   'Search_spend']] = pd.NA
```

Check it landed:

```python
print(len(data.media.coords[c.MEDIA_TIME]), len(data.kpi.coords[c.TIME]))
# want the first to exceed the second by at least max_lag
```

This fork warns when the gap is too small — upstream does not. See
[`TRIAGE.md`](TRIAGE.md) for the full story (google/meridian#1502).

## Install Meridian

> Working from this checkout? Use [Quickstart](#quickstart-for-this-fork)
> above instead. This section is upstream's plain PyPI path, and following it
> installs the published package rather than your clone.

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

This checkout ships nine runnable notebooks in [`demo/`](demo/) — no Colab
account needed:

```sh
jupyter notebook demo/Meridian_Getting_Started.ipynb
```

Note that the Getting Started notebook builds its `InputData` from a fully
populated DataFrame, which produces **zero adstock burn-in**. Read
[the one mistake that costs you accuracy silently](#the-one-mistake-that-costs-you-accuracy-silently)
before adapting it to your own data.

To get started with Meridian, you can run the code programmatically using sample
data with the [Getting Started Colab][3].

The Meridian model uses an MCMC sampling approach called
[No U Turn Sampler (NUTS)](https://www.tensorflow.org/probability/api_docs/python/tfp/experimental/mcmc/NoUTurnSampler)
which can be compute intensive. To help with this, GPU support has been
developed across the library (out-of-the-box) using tensors. We recommend
running your Meridian model on GPUs to get real time optimization results and
significantly reduce training time.

## Differences from upstream

This is a fork of [google/meridian](https://github.com/google/meridian).
Upstream does not accept external pull requests, so fixes live here.

[`TRIAGE.md`](TRIAGE.md) records a disposition for the 47 issues open on the
upstream tracker on 15 September 2026: verified already fixed, fixed here, a usage question, an
environment problem, a declined feature with its reason, or genuinely
unresolved.

Behaviour changes to existing code:

*   Float32 prior distributions are widened to float64 rather than rejected, so
    the documented `LogNormal(0.2, 0.9)` idiom works on the 64-bit JAX default.
*   A warning when media history is too short to fill the adstock window, which
    otherwise silently zero-pads and understates carryover.
*   Non-finite variables are excluded from the VIF check instead of aborting it.
*   Input validation errors name the offending column or coordinate.

Changes to the generated HTML report:

*   The response-curve chart shows its credible interval. Upstream passes
    `include_ci=False` for that one chart, so the single place where
    saturation-shape risk actually lives was the only point-estimate-only
    chart in a report where ROI bars do show intervals. It is now faceted per
    channel so seven bands stay readable.
*   The report carries the interval caveat. It said nothing about the
    limitation this README documents, which meant the artifact that reaches a
    client contradicted the repository it came from.
*   Multi-channel charts use an explicit colour-blind-safe categorical range.
    Every chart needing more than two colours previously fell through to
    Vega-Lite's default scheme, which fails both a chroma floor and an
    adjacent-pair separation check, and degrades further as channels are added.
*   Budget allocation is a labelled bar, not a pie. The old pie put its
    percentages in hover tooltips only, in a file whose whole purpose is being
    exported and sent to someone.
*   Generated reports are self-contained: pinned chart libraries, icons and the
    icon font are embedded with their license notices. Charts render without
    internet access. Desktop/mobile browser tests disable networking and reject
    external resource requests. Documentation links still need a connection
    when followed.
*   On small screens, wide charts stay readable in keyboard-accessible scroll
    regions instead of being clipped inside a narrow card.

Added modules:

*   `meridian.analysis.prior_predictive` — check priors against observed data
    from `sample_prior()` alone, before committing to a full fit:

    ```python
    mmm.sample_prior(500)
    check = prior_predictive.PriorPredictiveCheck(mmm)
    print(check.summary().verdict)
    check.plot_prior_predictive()
    ```

    The interval is a genuine *predictive* interval: Meridian's likelihood is
    `y ~ Normal(y_pred, sigma)`, and the observation-noise term is drawn and
    added per prior draw rather than comparing the observed data against an
    interval for the conditional mean alone. Leaving `sigma` out makes the
    interval too narrow, which biases the check toward reporting that a
    perfectly reasonable prior disagrees with the data.

*   `meridian.analysis.sampling_diagnostics` — the two MCMC trust signals the
    library computes but never surfaces: **divergent transitions** (already in
    `inference_data.sample_stats`, previously unread) and **effective sample
    size**:

    ```python
    diag = sampling_diagnostics.SamplingDiagnostics(mmm)
    print(diag.verdict)
    diag.summary().head()          # per parameter: r-hat, bulk/tail ESS
    ```

    This diagnostic uses upstream's TFP R-hat estimator and compares it with
    thresholds of 1.2 and 1.01. The latter is a stricter heuristic on that same
    estimator, not Vehtari's rank-normalized diagnostic. The quickstart and
    recovery tools separately report ArviZ's rank-normalized R-hat; their
    values can differ. Read ESS and divergences alongside either result:
    passing an R-hat threshold alone does not establish a reliable interval.

*   `meridian.analysis.geo_diagnostics` — measure whether per-geo estimates on
    your model are precise enough to allocate budget on:

    ```python
    diag = geo_diagnostics.GeoAllocationReliability(mmm)   # needs a fitted model
    print(diag.verdict)
    diag.summary().head()
    diag.plot_reliability()
    ```

    Reports `prob_positive` and `ci_excludes_zero` alongside the coefficient of
    variation, because CV alone cannot tell "this geo had no media execution"
    from "this geo's effect might genuinely be zero" — both blow CV up, and
    they mean entirely different things.

*   `meridian.benchmark` — record what actually governs runtime on your
    hardware: `python -m meridian.benchmark --help`.
*   `meridian.validation.recovery` — generate data whose true ROI is known by
    construction, fit it, and report whether the posterior recovers the truth.
    Run it at the shape of your own data before trusting a model built on it:
    `python -m meridian.validation --help`.

    A single fit sizes the error; it cannot tell a systematic bias from one
    unlucky dataset. `--replications N` fits N independent datasets from seeds
    derived from yours and reports **empirical coverage** — the fraction of
    replications where the true ROI fell inside the nominal interval — with a
    Wilson interval on that estimate, since coverage measured over a small
    number of replications is itself noisy. This quantifies the evidence and
    its uncertainty at the selected synthetic setting.

Added test modules:

*   `meridian/upstream_issues_test.py` — the code-testable dispositions in
    [`TRIAGE.md`](TRIAGE.md) as executable assertions, so a rebase that
    silently regresses one of them fails the suite. 15 of the 47 open upstream
    issues are guarded in that file. Three more (#647, #1396, #1668) are
    guarded by the test modules for the features that answered them —
    `prior_predictive_test.py`, `benchmark_test.py`, `geo_diagnostics_test.py`.
    The remaining 29 are usage questions, other people's hardware, or declined
    features, none of which can be asserted in code.
*   `meridian/math_invariants_test.py` — the arithmetic identities behind
    reported figures. Measured error on this fork: `roi ==
    incremental_outcome / spend` (7.1e-15), per-geo incremental summing to the
    aggregate (4.5e-13). The assertions themselves allow 1e-9 on float64, to
    leave headroom across backends. Also normalized adstock weights summing to 1, and the Hill
    closed form.

## Maintaining this fork

Upstream does not accept external pull requests, so these fixes live here
permanently and the main long-term risk is drift. Upstream moves quickly —
v1.4 to v2.0 inside nine months.

`main` is this fork's public line of development. A fresh clone only configures
the fork's `origin` remote. Add Google's repository once, then review upstream
changes on a separate branch:

```sh
git remote add upstream https://github.com/google/meridian.git  # once per clone
git fetch upstream
git switch -c sync-upstream
git merge upstream/main
make test
make test-tf
make test-e2e
```

Resolve any merge conflicts, review the changes and open a pull request into
`main`. Require the full CI matrix before merging; this preserves published
history and does not assume a local `upstream-main` branch exists.

If `meridian/upstream_issues_test.py` fails after an upstream merge, that is the point
of it. Each test there is named for the upstream issue it guards, so a failure
tells you either that one of this fork's patches was dropped in the rebase, or
that upstream regressed something this fork depends on. Read
[`TRIAGE.md`](TRIAGE.md) for the reasoning behind that specific issue before
changing the test.

`meridian/math_invariants_test.py` failing after a merge is more serious: it
means the arithmetic behind reported ROI changed. Do not paper over it.

### Standing checks

These checks exist because their claims decay if nobody re-runs them. Each one
writes dated evidence into `docs/validation/` rather than printing a verdict
and forgetting it.

```sh
# Container OS advisories: which of them the analysis workflow actually loads.
docker build --pull -t meridian:triage .
docker run --name probe --entrypoint python meridian:triage \
    scripts/container_reachability_probe.py --output /tmp/probe.json
docker cp probe:/tmp/probe.json docs/validation/container-reachability-latest.json
docker rm probe
python scripts/triage_container_os.py --scan scan.json \
    --probe docs/validation/container-reachability-latest.json \
    --output docs/validation/os-triage-latest.md

# Can the prior generate data the model would accept? For the shipped
# defaults the answer is no, which is why SBC is not run here.
python scripts/prior_predictive_audit.py --draws 400 \
    --output docs/validation/prior-predictive-audit-$(date +%F).json

# How much of a reported ROI is the prior rather than the data?
python scripts/prior_sensitivity.py \
    --output docs/validation/prior-sensitivity-$(date +%F).json

# Does the interval still cover at other dataset shapes? ~2 hours;
# --dry-run prints the plan and the estimate without fitting.
python scripts/coverage_grid.py --replications 10 \
    --output docs/validation/coverage-grid-$(date +%F).json

# Does a GPU fit agree with a CPU fit within Monte Carlo error?
# Needs a machine with a card; it refuses to run without one.
python scripts/gpu_validation.py --output docs/validation/gpu-$(date +%F).json
```

The [`container-rescan`](.github/workflows/container-rescan.yml) workflow runs
the first of these weekly and commits the refreshed evidence, so the published
advisory count describes a recent scanner database rather than the audit date.
It opens an issue only when a high or critical finding gains a vendor fix,
which is the only case with anything to do about it.

## Reading ROI intervals honestly

Recovery testing on synthetic data found a limit worth knowing before any of
these numbers reach a client.

`python -m meridian.validation --response {concave,linear}` generates data whose
true ROI is known by construction, fits Meridian, and reports recovery. Below is
the highest-ROI channel (true ROI 4.0, lowest spend) at 5 geos, 104 weeks, 3
channels, seed 7 — measured on the library versions recorded at the top of
[`TRIAGE.md`](TRIAGE.md):

| true response | carryover | ROI error | truth inside 90% CI |
|---|---|---|---|
| concave | none (`--max-lag 0`) | +9% | yes |
| concave | geometric (default, `max_lag=4`) | −43% | yes |
| **linear** | none (`--max-lag 0`) | **+71%** | **no** |
| linear | geometric (default) | +39% | yes |

All four met the recovery tool's loose R-hat threshold (reported maximum
R-hat ≤ 1.04) and recovered channel *ordering*. That does not establish that
every fit met the stricter 1.01 target or adequate effective sample size.
Two separate things are visible here, and they are worth keeping apart:

**1. Saturation misspecification biases the level and the interval misses it.**
On the no-carryover rows — the paired comparison, where the only thing that
changes is the response shape — a linear truth overstates ROI by 71% and the
90% interval excludes the true value. Concave truth lands within 9% and is
covered. These results illustrate the risk of fitting a concave saturation
curve to a response that is not concave; a single fit cannot establish the
size or direction of that effect generally. The reporting consequence:

> Meridian's credible intervals quantify parameter uncertainty **conditional on
> the assumed saturation shape**. They do not cover being wrong about that
> shape.

A channel far from saturation — typically one at low spend, whose real response
is still close to linear — can therefore have its ROI overstated substantially
while its interval looks reassuringly tight.

**2. Adding carryover costs a lot of precision on its own.** The same concave
data with `max_lag=4` lands 43% low rather than 9% high. Estimating adstock and
saturation jointly from 520 geo-weeks is simply harder than estimating
saturation alone. Rank channels and allocate on the interval; do not quote a
median ROI to two decimals off a model this size and call it a measurement.

**Re-measure before citing any of this.** Each row is one simulated dataset and
one fit — enough to size the effect, not enough to separate systematic bias from
an unlucky draw, and NUTS is not bit-reproducible across library or hardware
versions even at a fixed seed. Use many seeds to measure recovery coverage
and bias. The module also reports descriptive posterior rank fractions for
fixed synthetic truths. Those are not simulation-based calibration (SBC):
SBC requires drawing the truth from the same prior used to fit the model,
which this module does not implement. Run it at the shape and scale of *your*
data and investigate measured recovery gaps as sensitivity signals. Review
model specification and real-world validation before making recommendations.

What this does **not** do: it does not diagnose whether any particular channel's
real response is linear — which often is not identifiable from observational
spend data at all. To inspect a fitted channel's estimated response shape, see
[`demo/ROI_mROI_Response_Curves.ipynb`](demo/ROI_mROI_Response_Curves.ipynb).

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

**Problems with this fork** — installation, the added modules, or anything
listed in [`TRIAGE.md`](TRIAGE.md): use this repository's issue tracker. Run
`python scripts/verify_environment.py` first; it diagnoses the common install
failures and prints the fix.

**Questions about MMM methodology**: Google's
[technical documentation](https://developers.google.com/meridian/docs/basics/meridian-introduction)
remains the reference, and applies to this fork unchanged.

**Do not raise this fork's issues on google/meridian.** Google did not write
these changes and cannot support them. If you believe you have found a defect
in *upstream* code that is unrelated to this fork's changes, reproduce it
against a clean `google-meridian` install first, then report it there.

**A caution before you trust a number**: read
[Reading ROI intervals honestly](#reading-roi-intervals-honestly). On simulated
data where the answer is known, channel ordering recovered but absolute ROI was
off by about 40% in both directions.

## Citing Meridian

To cite Google's upstream framework:

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
