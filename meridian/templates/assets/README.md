<!-- NOTICE: This file is new in this fork; see NOTICE at the repository root. -->

# Offline report resources

Generated reports embed these resources as data URLs, including the icon font.
They do not fetch chart scripts, fonts or icons from a CDN. `manifest.json`
records each exact source artifact, SHA-256, MIME type and license. npm archives
were checked against their registry SHA-512 integrity before extracting the
named files. Upstream resource files are unmodified; their licenses accompany
both the package and each standalone HTML report.

Versions verified on 16 September 2026: Vega 6.4.0, Vega-Lite 6.4.1,
Vega-Embed 7.0.2; Material Icons Outlined font v110 and Material SVG icons.

To update, obtain the exact version's archive metadata from the official npm
registry, verify its integrity, and extract only the named build and LICENSE
files. Review the license and update the manifest with the new source and
content digest. For icon/font updates, use the recorded upstream source and
retain the Apache license. Never overwrite a digest simply to silence a
failed integrity check. Update the explicit template filenames with chart
version changes, rebuild distributions, and run report tests plus the browser
smoke test with network access disabled. The latter verifies all charts and
keyboard navigation at phone and desktop widths.

External links in explanatory text still require a connection when followed.
Caller-supplied chart specifications that explicitly reference remote data are
outside the self-contained generated-report contract.
