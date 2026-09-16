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

The report is opened as a local file and checked at phone and desktop widths.
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
from pathlib import Path
from typing import Any

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


def run_browser_regression(
    report_html: Path,
    screenshot_dir: Path | None = None,
    executable_path: Path | None = None,
) -> dict[str, Any]:
  """Runs the report checks and returns JSON-serializable evidence."""
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
        )
        page = context.new_page()
        page_errors: list[str] = []
        console_errors: list[str] = []
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
          page.wait_for_function(
              """() => {
                const charts = Array.from(
                  document.querySelectorAll('chart-embed')
                );
                return charts.length > 0 && charts.every(
                  (chart) => chart.querySelector('canvas, svg')
                );
              }""",
              timeout=_CHART_READY_TIMEOUT_MS,
          )
          chart_count = page.locator('.vega-embed').count()
          rendered_count = page.locator(
              '.vega-embed canvas, .vega-embed svg'
          ).count()
          if chart_count == 0 or rendered_count == 0:
            raise AssertionError(
                'No rendered Vega chart found '
                f'({chart_count=}, {rendered_count=}).'
            )
          if page_errors or console_errors:
            raise AssertionError(
                f'Browser errors while rendering report: '
                f'{page_errors=}, {console_errors=}'
            )

          overflow = _assert_no_horizontal_overflow(page)

          desktop: dict[str, Any] = {'status': 'SKIP'}
          if width >= 1000:
            first_chart = page.locator('chart-embed').first
            if first_chart.count():
              desktop_width = first_chart.evaluate(
                  'element => element.getBoundingClientRect().width'
              )
              if desktop_width < 600:
                raise AssertionError(
                    'Desktop chart viewport is unexpectedly narrow: '
                    f'{desktop_width}px'
                )
              desktop = {
                  'status': 'PASS',
                  'first_chart_viewport_width': desktop_width,
              }

          if screenshot_dir is not None:
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
                  'overflow': overflow,
                  'keyboard_scroll': keyboard,
                  'desktop': desktop,
              }
          )
        finally:
          page.close()
          context.close()
    finally:
      browser.close()

  return {
      'status': 'PASS',
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
    )
  except (FileNotFoundError, RuntimeError, AssertionError) as error:
    print(f'FAIL: {error}')
    return 1
  print(json.dumps(evidence, indent=2))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
