#!/usr/bin/env python3
# Copyright 2026 Sundar Ramesh Kumar.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NOTICE: This file is new in this fork and does not exist in the
# original google/meridian source.

"""Run a focused browser regression against a generated Meridian report.

The report is opened as a local file with the network disabled and checked at
phone and desktop widths. External resource requests fail the check.
The check waits for the rendered Vega visualizations, records browser errors,
checks for horizontal document overflow, and exercises keyboard panning on an
actually overflowing chart when one is present.

Examples:

  python scripts/test_report_browser.py /path/to/summary.html
  python scripts/test_report_browser.py /path/to/summary.html \
      --screenshot-dir /tmp/report-screenshots \
      --executable-path /path/to/chrome

Playwright is an optional test dependency. If ``--executable-path`` is
omitted, Playwright's default Chromium installation is used.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET

_VIEWPORTS = ((320, 844), (390, 844), (1440, 1000))
_CHART_READY_TIMEOUT_MS = 60_000


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      'report_html',
      type=Path,
      help='path to the generated report HTML file',
  )
  parser.add_argument(
      '--screenshot-dir',
      type=Path,
      help='optional directory for full-page viewport screenshots',
  )
  parser.add_argument(
      '--min-desktop-chart-width',
      type=float,
      default=600,
      help=(
          'expected minimum desktop chart viewport width in pixels '
          '(default: 600; use 500 for the two-column optimization report)',
      ),
  )
  parser.add_argument(
      '--executable-path',
      type=Path,
      help=(
          'optional browser executable override; defaults to Playwright '
          'Chromium'
      ),
  )
  return parser.parse_args(argv)


def _assert_no_horizontal_overflow(page: Any) -> dict[str, int | bool]:
  dimensions = page.evaluate("""() => ({
        innerWidth,
        bodyScrollWidth: document.body.scrollWidth,
        documentScrollWidth: document.documentElement.scrollWidth,
        bodyOverflow: document.body.scrollWidth > innerWidth,
        documentOverflow: document.documentElement.scrollWidth > innerWidth,
      })""")
  if dimensions['bodyOverflow'] or dimensions['documentOverflow']:
    raise AssertionError(
        'The report has horizontal document overflow: ' f'{dimensions}'
    )
  return dimensions


def _check_keyboard_pan(page: Any, expect: Any) -> dict[str, Any]:
  chart_index = page.evaluate(
      """() => Array.from(document.querySelectorAll('chart-embed'))
        .findIndex((chart) => chart.clientWidth > 0 &&
          chart.scrollWidth > chart.clientWidth + 1)"""
  )
  if chart_index < 0:
    return {
        'status': 'SKIP',
        'reason': 'No overflowing chart viewport at this width',
    }

  chart = page.locator('chart-embed').nth(chart_index)
  chart.wait_for(state='visible')
  tabindex = chart.get_attribute('tabindex')
  if tabindex is None or int(tabindex) < 0:
    raise AssertionError(
        'An overflowing chart viewport must be keyboard focusable.'
    )

  chart.evaluate('element => { element.scrollLeft = 0; element.focus(); }')
  expect(chart).to_be_focused()

  before = chart.evaluate('element => element.scrollLeft')
  chart.press('ArrowRight')
  after_arrow_right = chart.evaluate('element => element.scrollLeft')
  if after_arrow_right <= before:
    raise AssertionError(
        'ArrowRight did not advance the overflowing chart viewport: '
        f'{before} -> {after_arrow_right}'
    )

  max_scroll = chart.evaluate(
      'element => element.scrollWidth - element.clientWidth'
  )
  chart.press('End')
  after_end = chart.evaluate('element => element.scrollLeft')
  if after_end < max_scroll - 1:
    raise AssertionError(
        f'End did not reach the chart scroll end: {after_end} < {max_scroll}'
    )

  chart.press('Home')
  after_home = chart.evaluate('element => element.scrollLeft')
  if after_home > 1:
    raise AssertionError(
        f'Home did not return the chart to its start: {after_home}'
    )

  return {
      'status': 'PASS',
      'chart_index': chart_index,
      'client_width': chart.evaluate('element => element.clientWidth'),
      'scroll_width': chart.evaluate('element => element.scrollWidth'),
      'arrow_right': [before, after_arrow_right],
      'end': after_end,
      'home': after_home,
  }


def _check_stat_contrast(page: Any) -> list[dict[str, Any]]:
  """Check computed text contrast for the actually rendered scenario deltas."""
  colors = page.locator('delta').evaluate_all(
      """elements => elements.map(element => {
    const style = getComputedStyle(element);
    let parent = element;
    let background = 'rgb(255, 255, 255)';
    while (parent) {
      const candidate = getComputedStyle(parent).backgroundColor;
      if (candidate !== 'rgba(0, 0, 0, 0)' && candidate !== 'transparent') {
        background = candidate; break;
      }
      parent = parent.parentElement;
    }
    return {text: element.textContent.trim(), color: style.color, background,
            font_size: parseFloat(style.fontSize), font_weight: Number(style.fontWeight),
            semantic_class: element.className};
  })"""
  )

  def luminance(css_color):
    channels = [
        float(value) / 255 for value in re.findall(r'[\d.]+', css_color)[:3]
    ]
    linear = [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    ]
    return sum(
        value * weight
        for value, weight in zip(linear, (0.2126, 0.7152, 0.0722))
    )

  for color in colors:
    levels = sorted((luminance(color['color']), luminance(color['background'])))
    ratio = (levels[1] + 0.05) / (levels[0] + 0.05)
    large = color['font_size'] >= 24 or (
        color['font_size'] >= 18.667 and color['font_weight'] >= 700
    )
    minimum = 3.0 if large else 4.5
    if ratio < minimum:
      raise AssertionError(
          f'Scenario delta text contrast {ratio:.3f} < {minimum}: {color}'
      )
    if (
        color['text'] in ('$0', '0.00')
        and color['semantic_class'] != 'neutral-text'
    ):
      raise AssertionError(
          'An exact-zero scenario delta must have neutral color.'
      )
    color.update(contrast_ratio=round(ratio, 3), minimum_ratio=minimum)
  return colors


def _check_native_controls(page: Any, expect: Any) -> dict[str, Any]:
  """Exercise the real report's source view and both local native exports."""
  chart = page.locator('chart-embed').first
  host = chart.locator('..')
  controls = host.locator('.chart-controls').first
  show = controls.locator('button').nth(0)
  show.click()
  panel = host.locator('.chart-source-panel').first
  expect(panel).to_be_visible()
  rows = panel.locator('tbody tr').count()
  if rows > 50:
    raise AssertionError('The source table exceeded its 50-row page bound.')
  expect(panel.get_by_role('heading', name='Chart source data')).to_be_visible()
  with page.expect_download() as pending_spec:
    controls.locator('button').nth(2).click()
  specification = json.loads(Path(pending_spec.value.path()).read_text('utf-8'))
  source_specification = chart.evaluate(
      'element => element.__meridianChart.spec'
  )
  if specification != source_specification:
    raise AssertionError('Native spec export changed the chart source.')
  with page.expect_download() as pending_svg:
    controls.locator('button').nth(1).click()
  svg = ET.fromstring(Path(pending_svg.value.path()).read_text('utf-8'))
  if svg.tag != '{http://www.w3.org/2000/svg}svg':
    raise AssertionError('The SVG download must be a native SVG document.')
  # Keep the report screenshot's working comparison in its default state.
  show.click()
  return {
      'status': 'PASS',
      'source_table_rows_on_first_page': rows,
      'source_semantics': 'embedded source rows before plot transforms',
      'spec_export': 'exact native Vega-Lite specification',
      'svg_export': 'native SVG document',
  }


def run_browser_regression(
    report_html: Path,
    screenshot_dir: Path | None = None,
    executable_path: Path | None = None,
    min_desktop_chart_width: float = 600,
) -> dict[str, Any]:
  """Runs the report checks and returns JSON-serializable evidence."""
  if not math.isfinite(min_desktop_chart_width) or min_desktop_chart_width <= 0:
    raise ValueError('Minimum desktop chart width must be finite and positive.')
  try:
    from playwright.sync_api import expect, sync_playwright
  except ImportError as error:
    raise RuntimeError(
        'Playwright is required for this optional browser check. Install it '
        'with `pip install playwright` and run `playwright install chromium`.'
    ) from error

  report_html = report_html.expanduser().resolve()
  if not report_html.is_file():
    raise FileNotFoundError(f'Report HTML does not exist: {report_html}')
  if executable_path is not None:
    executable_path = executable_path.expanduser().resolve()
    if not executable_path.is_file():
      raise FileNotFoundError(
          f'Browser executable does not exist: {executable_path}'
      )
  if screenshot_dir is not None:
    screenshot_dir = screenshot_dir.expanduser().resolve()
    screenshot_dir.mkdir(parents=True, exist_ok=True)

  results: list[dict[str, Any]] = []
  with sync_playwright() as playwright:
    launch_args: dict[str, Any] = {'headless': True}
    if executable_path is not None:
      launch_args['executable_path'] = str(executable_path)
    browser = playwright.chromium.launch(**launch_args)
    try:
      for width, height in _VIEWPORTS:
        context = browser.new_context(
            viewport={'width': width, 'height': height},
            device_scale_factor=1,
            offline=True,
        )
        page = context.new_page()
        page_errors: list[str] = []
        console_errors: list[str] = []
        external_requests: list[str] = []
        page.on(
            'request',
            lambda request, requests=external_requests: (
                requests.append(request.url)
                if request.url.startswith(('https://', 'http://'))
                else None
            ),
        )
        page.on(
            'pageerror',
            lambda error, errors=page_errors: errors.append(str(error)),
        )
        page.on(
            'console',
            lambda message, errors=console_errors: (
                errors.append(message.text) if message.type == 'error' else None
            ),
        )
        try:
          page.goto(report_html.as_uri())

          expect(page.locator('.header .title')).to_be_visible()
          expect(page.get_by_role('heading', level=1)).to_be_visible()
          expect(page.get_by_role('main')).to_be_visible()
          page.wait_for_function(
              """() => {
                const charts = Array.from(
                  document.querySelectorAll('chart-embed')
                );
                return charts.length > 0 && charts.every(
                  (chart) => chart.__meridianChart?.view && chart.querySelector('svg')
                );
              }""",
              timeout=_CHART_READY_TIMEOUT_MS,
          )
          chart_count = page.locator('.vega-embed').count()
          rendered_count = page.locator('.vega-embed svg').count()
          if chart_count == 0 or rendered_count != chart_count:

            raise AssertionError(
                'No rendered Vega chart found '
                f'({chart_count=}, {rendered_count=}).'
            )
          if page_errors or console_errors or external_requests:
            raise AssertionError(
                f'Browser errors while rendering report: '
                f'{page_errors=}, {console_errors=}, {external_requests=}'
            )

          page.evaluate('() => document.fonts.ready.then(() => true)')

          overflow = _assert_no_horizontal_overflow(page)

          desktop: dict[str, Any] = {'status': 'SKIP'}
          if width >= 1000:
            first_chart = page.locator('chart-embed').first
            if first_chart.count():
              desktop_width = first_chart.evaluate(
                  'element => element.getBoundingClientRect().width'
              )
              if desktop_width < min_desktop_chart_width:
                raise AssertionError(
                    'Desktop chart viewport is unexpectedly narrow: '
                    f'{desktop_width}px'
                )
              desktop = {
                  'status': 'PASS',
                  'first_chart_viewport_width': desktop_width,
                  'minimum_chart_viewport_width': min_desktop_chart_width,
              }

          native = _check_native_controls(page, expect)
          stat_contrast = _check_stat_contrast(page)

          if screenshot_dir is not None:
            page.evaluate('() => window.scrollTo(0, 0)')
            page.screenshot(
                path=str(screenshot_dir / f'first-viewport-{width}.png'),
                full_page=False,
            )
            page.screenshot(
                path=str(screenshot_dir / f'report-{width}.png'),
                full_page=True,
            )

          keyboard = _check_keyboard_pan(page, expect)

          results.append(
              {
                  'viewport': {'width': width, 'height': height},
                  'chart_count': chart_count,
                  'rendered_count': rendered_count,
                  'page_errors': page_errors,
                  'console_errors': console_errors,
                  'external_requests': external_requests,
                  'overflow': overflow,
                  'keyboard_scroll': keyboard,
                  'desktop': desktop,
                  'native_controls': native,
                  'stat_contrast': stat_contrast,
              }
          )
        finally:
          page.close()
          context.close()
    finally:
      browser.close()

  return {
      'status': 'PASS',
      'network': 'disabled',
      'report': str(report_html),
      'viewports': results,
  }


def main(argv: list[str] | None = None) -> int:
  args = _parse_args(argv)
  try:
    evidence = run_browser_regression(
        args.report_html,
        screenshot_dir=args.screenshot_dir,
        executable_path=args.executable_path,
        min_desktop_chart_width=args.min_desktop_chart_width,
    )
  except (FileNotFoundError, RuntimeError, AssertionError, ValueError) as error:
    print(f'FAIL: {error}')
    return 1
  print(json.dumps(evidence, indent=2))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
