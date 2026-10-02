<!-- NOTICE: This file is new in this fork; see NOTICE at the repository root. -->

# Offline report resources

Generated reports embed these resources as data URLs, including the icon font.
They do not fetch chart scripts, fonts or icons from a CDN. `manifest.json`
records each exact source artifact, SHA-256, MIME type and license. npm archives
were checked against their registry SHA-512 integrity before extracting the
named files. Third-party resource files are unmodified; their licenses accompany
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

The fork-owned `report-charts.js` uses the pinned runtime to render browser SVG,
retain chart-defined tooltips/selections, and provide local SVG and editable
Vega-Lite JSON downloads. No remote Vega Editor action is enabled. “Chart source
data” shows embedded rows before Vega transforms; it is not a table of the
transformed plotted values. Tables load on first open, cache fields per dataset,
and display at most 50 rows and 25 columns per page. The complete source remains
in the downloaded spec. Arbitrary remote-data specifications are outside the
offline contract. Native here means browser SVG and Vega-Lite, not Office chart
objects. Dense workloads should still be assessed before choosing SVG.

The interaction APIs were checked on 2 October 2026 against the pinned versions
and the primary [Vega View API](https://vega.github.io/vega/docs/api/view/),
[Vega Embed API](https://github.com/vega/vega-embed), and
[Vega-Lite data contract](https://vega.github.io/vega-lite/docs/data.html).
The optional `scripts/test_native_report_browser.py` checks all template paths,
source/spec/SVG reconciliation, an actual spec edit that changes a mark,
malicious labels, native selection/tooltips, pagination and async failures.
`scripts/test_report_browser.py` checks real fitted reports offline at phone and
desktop widths and records first-viewport/full-page screenshots. These checks
cover Chromium; they do not establish universal browser support or field speed.
