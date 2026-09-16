<!-- NOTICE: This file is new in this fork and does not exist in the
     original google/meridian source. -->

# Security Policy

This is a personal fork of
[google/meridian](https://github.com/google/meridian), not affiliated with or
supported by Google.

## Where to report

**Report a vulnerability in Google's Meridian code to Google**, not here. This
fork carries upstream's code essentially unchanged, so almost any security
issue you find is upstream's and everyone benefits from it being fixed there.
Use Google's private intake at [g.co/vulnz](https://g.co/vulnz), as directed
by [upstream's security policy](https://github.com/google/meridian/security).
Do not disclose vulnerability details in a public issue.

**Report it here only if it is specific to this fork's own changes** — the
modules listed in [`NOTICE`](NOTICE) under "Files added by this fork", or a
behavioural change listed in [`TRIAGE.md`](TRIAGE.md).

To report privately, use GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
on this repository rather than opening a public issue. If the private reporting
button is unavailable, open an issue asking for a private contact without
including vulnerability details.

## What to expect

This fork is maintained by one person alongside other work. Reports are
reviewed as maintainer availability permits; there is no response-time
commitment. Keep vulnerability details private while a fix is assessed.

## Dependency vulnerabilities

Meridian pins several dependencies with upper bounds, some of which exist to
work around upstream breakage rather than for security — see the comments in
[`pyproject.toml`](pyproject.toml). A CVE in a pinned dependency is worth
raising, but be aware that raising a ceiling can break the model: `arviz` and
`matplotlib` ceilings in particular are load-bearing.

The TensorFlow floor (`>= 2.21.0`) retains the upstream fix for
[CVE-2026-2492](https://www.zerodayinitiative.com/advisories/ZDI-26-116/)
and should not be lowered without reviewing that protection.

## Container operating-system advisories

The container pins its Python base image and applies available OS package
updates during the build. CI retains a complete Trivy OS advisory report and
blocks high or critical findings with a vendor-provided fix. Findings without
a vendor fix remain in that report; a passing gate does not mean that the
image has no advisories. See the dated findings in [AUDIT.md](AUDIT.md).

Refresh OS packages with a clean build (`docker build --pull --no-cache -t
meridian .`). Review the retained `container-os-advisories` CI artifact and
update the pinned base and scanner deliberately. Package-level findings do
not by themselves establish whether a vulnerable component is reachable in
Meridian's default workflow.

The findings that have no vendor fix are triaged rather than suppressed. The
default analysis workflow loads 11 of the image's 87 OS packages, measured by
`scripts/container_reachability_probe.py` reading `/proc/self/maps` inside the
built image. Each advisory is recorded with whether any affected package is in
that loaded set, in
[docs/validation/os-triage-latest.md](docs/validation/os-triage-latest.md).
Read that column as scope, not as a verdict: a loaded package is not an
exploit, an unloaded one is not proof of safety, and Debian attributes
advisories to source packages rather than to the specific binary at fault.

The [container-rescan](.github/workflows/container-rescan.yml) workflow repeats
the rebuild, scan, load measurement and triage every week, so the published
inventory reflects a recent scanner database rather than the audit date. It
raises an issue when a high or critical finding gains a vendor fix, which is
the point at which there is something to do.

Alternative base images were measured on 17 September 2026 and none was an
improvement; see the table in [AUDIT.md](AUDIT.md). If you are considering a
base change, measure it rather than assuming distroless is smaller: for this
image it carries more high-severity findings, not fewer.

## Scope

Meridian is a modelling library. It executes code you give it and reads files
you select; it provides neither authentication nor an isolated execution
environment. The default container runs local analysis as a non-root user and
opens no server port. Privileged execution and host mounts change that exposure.
Reports amounting to "this library runs the Python you pass it" are not
vulnerabilities.

Genuine issues would include: code execution triggered by loading a
`.binpb` model file from an untrusted source, a dependency vulnerability
reachable through normal use, or a flaw in this fork's own modules.
