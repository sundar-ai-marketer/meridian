<!-- NOTICE: This file is new in this fork; see NOTICE at the repository root. -->

# Agent guide to this repository

An unaffiliated fork of [google/meridian](https://github.com/google/meridian),
a Bayesian marketing mix modeling library. Maintained by one person, developed
largely with AI coding agents. This file records the invariants, the commands
that prove a change, and where the authoritative facts live, so that work done
here lands in the state the maintainer would have left it.

## Commands that prove a change

```sh
make setup       # build .venv, install supported CPU extras, compile report CSS
make verify      # is this environment usable, and which device will it use
make test        # full suite, process-isolated phases, JAX backend
make test-tf     # full suite, TensorFlow backend
make test-e2e    # real fit -> optimize -> serialize -> report
make lint        # pylint, including the duplicate-code checker
make gpu-check   # what this machine's accelerator can do for Meridian
make bench       # sampling throughput on this machine
```

`make test`, `make test-tf` and `make test-e2e` all pass before a change is
done. Both backends stay green: JAX is the default, TensorFlow is deprecated
upstream and still shipped and tested here.

A change to sampling, priors, or diagnostics also means re-running the
relevant producer in `scripts/` and committing its JSON. See Evidence below.

## Invariants

These hold for the library's reported numbers to mean anything. When a change
makes one fail, the change is what moved.

- `roi == incremental_outcome / spend`; incremental outcome is additive over
  geos and over times; response curves are monotone in spend and zero at zero
  spend. Asserted in `meridian/math_invariants_test.py`.
- Normalized adstock weights sum to one, `max_lag=0` is the identity, and zero
  media gives zero adstock. Same file.
- `meridian/upstream_issues_test.py` makes every `TRIAGE.md` disposition
  executable, one test per upstream issue number. A failure there means a fix
  regressed or upstream behaviour this fork depends on changed — update the
  `TRIAGE.md` entry alongside the code.
- Reach for `backend.tfd` and `backend.np_float_dtype` rather than importing a
  TFP substrate directly. That indirection is what lets the same code run on
  both backends.
- Values reach the HTML report through `meridian/templates/formatter.py`, which
  applies Jinja2 autoescaping and `markupsafe.Markup`. Vega-Lite payloads go
  through `{{ chart_json|tojson }}` — JSON-inside-HTML is a different escaping
  problem from HTML text, and that filter is what handles it.

## Evidence: a number in prose cites a file

Every quantitative claim in `README.md` and `AUDIT.md` traces to a JSON file
under `docs/validation/`, written by a producer in `scripts/` that calls
`scripts/evidence.py` for `provenance()` — git HEAD, architecture, package
versions, and the producer's own SHA-256. To change a number:

1. Run the producer. It writes a dated file, e.g.
   `docs/validation/coverage-grid-2026-09-20.json`.
2. Quote the figures the file contains, at the precision the file records.
3. Keep any `-latest` alias identical to the newest dated file in its family.
4. Commit the JSON and the prose edit together.

Re-run the producer rather than adjusting a number in prose to round more
cleanly. Where a figure comes from a single hand-run fit rather than a
committed file, the prose says so and names the command — `README.md`'s
ROI-recovery table is the worked example.

## Apple Silicon

CPU is the supported path, and it is a good one: the full suite and a real
end-to-end fit both run natively on arm64. Keep the default float64
precision; the benchmark shows float32 trades throughput for a little memory
rather than winning speed.

On accelerators, report what `make gpu-check` measures rather than what a
plugin's README claims. The recorded state: Apple's `jax-metal` and
`tensorflow-metal` do not run this model at all, and `make verify` names
either one if installed; the third-party `jax-mps` does run it at float32 but
measured ~18x slower than this CPU, on fits that had not converged. See
`docs/validation/gpu-metal-*.json`. A GPU claim here needs a converged fit on
both devices, not a completed one.

## One-way doors — confirm before doing these

- Pushing a `v*` tag or bumping `meridian/version.py`. The publish jobs are
  gated to `google/meridian`'s own repository and do not fire here; treat a
  version bump as a release action regardless.
- Force-push or history rewrite on `main`. `AUDIT.md` and `TRIAGE.md` cite
  commit hashes as evidence anchors, and rewriting history orphans them.
- Deleting or overwriting anything under `docs/validation/` without
  regenerating it. That file is the only record the measurement was taken.
- Publishing to PyPI. The distribution name is `meridian-mmm-fork`; the
  `google-meridian` name belongs to Google.

## Where the authoritative facts live

| Question | File |
|---|---|
| What was measured, and what the limits are | `AUDIT.md` |
| Disposition of each upstream issue | `TRIAGE.md` |
| What differs from `google/meridian`, and why | `NOTICE`, README's "What differs from upstream" |
| Security posture and disclosure | `SECURITY.md` |
| What CI gates | `.github/workflows/ci.yml`, `container-rescan.yml` |
| Whether a fix belongs here or upstream | `CONTRIBUTING.md` |

## House style

Short declarative sentences, one claim each. Every number carries its source
or its scope in the same sentence. State a limitation next to the result it
limits rather than in a closing disclaimer. Avoid "seamless", "robust",
"leverage", "delve", "landscape". A measured limitation is a finding and stays
in the text with its figures intact.
