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
Use [google/meridian](https://github.com/google/meridian/issues).

**Report it here only if it is specific to this fork's own changes** — the
modules listed in [`NOTICE`](NOTICE) under "Files added by this fork", or a
behavioural change listed in [`TRIAGE.md`](TRIAGE.md).

To report privately, use GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
on this repository rather than opening a public issue.

## What to expect

This fork is maintained by one person alongside other work. There is no
response-time commitment. If you need a guaranteed response, Apache 2.0 lets
you fork and patch it yourself.

## Dependency vulnerabilities

Meridian pins several dependencies with upper bounds, some of which exist to
work around upstream breakage rather than for security — see the comments in
[`pyproject.toml`](pyproject.toml). A CVE in a pinned dependency is worth
raising, but be aware that raising a ceiling can break the model: `arviz` and
`matplotlib` ceilings in particular are load-bearing.

The TensorFlow floor (`>= 2.21.0`) exists specifically to address
CVE-2026-2492 and should not be lowered.

## Scope

Meridian is a modelling library. It executes code you give it, reads files you
point it at, and does not run a network service, handle authentication, or
process untrusted input by design. Reports amounting to "this library runs the
Python you pass it" are not vulnerabilities.

Genuine issues would include: code execution triggered by loading a
`.binpb` model file from an untrusted source, a dependency vulnerability
reachable through normal use, or a flaw in this fork's own modules.
