#!/usr/bin/env python3
# Copyright 2026 Sundar Ramesh Kumar.
# Licensed under the Apache License, Version 2.0.
# NOTICE: This file is new in this fork; see NOTICE at the repository root.

"""Optional Chromium contract tests for native, offline report interactions.

The normal numerical suite skips these when Playwright is unavailable. Run
with a Python environment containing Playwright and an installed Chromium;
NATIVE_REPORT_CHROME optionally selects the executable. The fixture is built
with the repository environment without fitting a model. Browser SVG and
editable Vega-Lite JSON are the tested native formats, not Office chart objects.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

_ROOT = Path(__file__).resolve().parents[1]
_LABEL = '<img src=x onerror="window.pwned=1"> & </script><script>window.pwned=2</script>'


def write_fixture(path: Path):
  """Exercise all report chart templates with a safe, bounded source fixture."""
  if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
  from meridian.templates import formatter  # pylint: disable=import-outside-toplevel

  env = formatter.create_template_env()
  spec = {
      '$schema': 'https://vega.github.io/schema/vega-lite/v6.json',
      'title': 'Source comparison',
      'width': 620,
      'height': 220,
      'data': {'name': 'source'},
      'datasets': {
          'source': [
              {
                  'index': i,
                  'value': 11 if i == 0 else 30 + i,
                  'label': _LABEL if i == 0 else f'Row {i}',
              }
              for i in range(120)
          ]
      },
      'mark': {'type': 'point', 'filled': True},
      'params': [
          {
              'name': 'highlight',
              'select': {
                  'type': 'point',
                  'on': 'pointerover',
                  'fields': ['index'],
              },
          }
      ],
      'encoding': {
          'x': {'field': 'index', 'type': 'quantitative'},
          'y': {
              'field': 'value',
              'type': 'quantitative',
              'scale': {'zero': True},
          },
          'tooltip': [
              {'field': 'index'},
              {'field': 'value'},
              {'field': 'label'},
          ],
          'color': {
              'condition': {'param': 'highlight', 'value': '#174ea6'},
              'value': '#5f6368',
          },
      },
  }
  chart_json = json.dumps(spec)
  cards = [
      formatter.create_card_html(
          env,
          formatter.CardSpec('native-source', 'Native chart comparison'),
          'Synthetic source rows verify rendering and exports.',
          [formatter.ChartSpec('native-chart', chart_json)],
          stats_specs=[
              formatter.StatsSpec('Neutral budget', '$91', '$0', 'neutral'),
              formatter.StatsSpec('Gain ROI', '0.69', '+0.04', 'positive'),
              formatter.StatsSpec('Small loss', '1.00', '-<0.01', 'negative'),
          ],
      )
  ]
  cards.append(
      env.get_template('channel_recommendation_card.html.jinja').render(
          implausible_roi_chart_json=chart_json,
          high_variance_chart_json=chart_json,
          potential_bias_chart_json=chart_json,
      )
  )
  for name in ('calibration_overview', 'calibration_details'):
    cards.append(
        env.get_template(f'{name}_card.html.jinja').render(
            plotted_channels=[{'chart_id': '0', 'chart_json': chart_json}],
            details_description='Synthetic calibration source fixture.',
            overview_description='Synthetic calibration source fixture.',
        )
    )
  path.write_text(
      formatter.create_summary_html(env, 'Native chart contracts', cards),
      encoding='utf-8',
  )


class NativeReportBrowserTest(unittest.TestCase):
  """Behavior tests are optional locally and executable in browser CI."""

  @classmethod
  def setUpClass(cls):
    try:
      from playwright.sync_api import sync_playwright  # pylint: disable=import-outside-toplevel
    except ImportError as error:
      raise unittest.SkipTest(
          'Playwright is an optional browser dependency.'
      ) from error
    cls.directory = tempfile.TemporaryDirectory(
        prefix='meridian-native-browser-'
    )
    cls.fixture = Path(cls.directory.name) / 'native.html'
    python = _ROOT / '.venv/bin/python'
    subprocess.run(
        [
            str(python) if python.is_file() else sys.executable,
            str(Path(__file__).resolve()),
            '--write-fixture',
            str(cls.fixture),
        ],
        cwd=_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    cls.playwright = sync_playwright().start()
    options = {'headless': True}
    if os.environ.get('NATIVE_REPORT_CHROME'):
      options['executable_path'] = os.environ['NATIVE_REPORT_CHROME']
    cls.browser = cls.playwright.chromium.launch(**options)

  @classmethod
  def tearDownClass(cls):
    cls.browser.close()
    cls.playwright.stop()
    cls.directory.cleanup()

  def setUp(self):
    self.context = self.browser.new_context(
        viewport={'width': 390, 'height': 844},
        offline=True,
        accept_downloads=True,
    )
    self.page = self.context.new_page()
    self.errors = []
    self.requests = []
    self.page.on('pageerror', lambda error: self.errors.append(str(error)))
    self.page.on(
        'request',
        lambda request: self.requests.append(request.url)
        if request.url.startswith(('http://', 'https://'))
        else None,
    )
    self.page.goto(self.fixture.as_uri())
    self.page.wait_for_function(
        """() => [...document.querySelectorAll('chart-embed')]
      .every(chart => chart.__meridianChart?.view)"""
    )
    self.chart = self.page.locator('#native-chart')
    self.host = self.chart.locator('..')

  def tearDown(self):
    self.assertEqual(self.errors, [])
    self.assertEqual(self.requests, [])
    self.context.close()

  def _download(self, name):
    with self.page.expect_download() as pending:
      self.host.get_by_role('button', name=name).click()
    return Path(pending.value.path()).read_text(encoding='utf-8')

  def test_every_specialized_chart_is_svg_with_local_controls(self):
    self.assertEqual(self.page.locator('chart-embed').count(), 6)
    self.assertEqual(self.page.locator('.vega-embed svg').count(), 6)
    self.assertEqual(self.page.locator('.vega-embed canvas').count(), 0)
    self.assertEqual(self.page.locator('.chart-controls').count(), 6)
    self.assertEqual(
        self.page.locator('.chart-source-panel tbody tr').count(), 0
    )
    self.assertEqual(
        self.chart.get_attribute('aria-label'),
        'Source comparison. Scrollable chart. Use left and right arrow keys to pan.',
    )
    self.assertIn(
        'Source comparison',
        self.host.get_by_role('group').get_attribute('aria-label'),
    )

  def test_lazy_bounded_source_table_and_malicious_labels(self):
    show = self.host.get_by_role(
        'button', name='Show chart source data for Source comparison'
    )
    show.focus()
    show.press('Enter')
    self.assertEqual(
        self.host.get_by_role(
            'button', name='Hide chart source data for Source comparison'
        ).get_attribute('aria-expanded'),
        'true',
    )
    panel = self.host.locator('.chart-source-panel')
    self.assertEqual(panel.locator('tbody tr').count(), 50)
    self.assertIn(_LABEL, panel.locator('tbody tr').first.inner_text())
    self.assertEqual(panel.locator('img, script').count(), 0)
    self.assertIsNone(self.page.evaluate('window.pwned'))
    self.assertIn('Rows 1–50 of 120', panel.inner_text())
    panel.get_by_role('button', name='Next rows').click()
    self.assertIn('Rows 51–100 of 120', panel.inner_text())
    self.assertEqual(panel.locator('tbody tr').count(), 50)
    panel.get_by_role('button', name='Next rows').click()
    self.assertEqual(panel.locator('tbody tr').count(), 20)
    self.assertTrue(panel.get_by_role('button', name='Next rows').is_disabled())
    panel.get_by_role('button', name='Previous rows').click()
    self.assertEqual(panel.locator('tbody tr').count(), 50)
    self.assertIn('Plotted values may use filters', panel.inner_text())

  def test_native_json_edit_rerenders_source_mark_and_svg_export(self):
    exported = json.loads(
        self._download('Download chart spec for Source comparison')
    )
    self.assertEqual(exported['datasets']['source'][0]['value'], 11)
    self.assertEqual(exported['datasets']['source'][0]['label'], _LABEL)
    mark = self.chart.locator('.mark-symbol path').first
    before = mark.get_attribute('transform')
    self.assertIn('value: 11', mark.get_attribute('aria-label'))
    svg = ET.fromstring(self._download('Download SVG for Source comparison'))
    self.assertTrue(svg.tag.endswith('svg'))
    self.assertTrue(
        self.host.get_by_role(
            'button', name='Download SVG for Source comparison'
        ).evaluate('element => document.activeElement === element')
    )
    self.assertTrue(
        any(
            'value: 11' in item.attrib.get('aria-label', '')
            for item in svg.iter()
        )
    )
    exported['datasets']['source'][0]['value'] = 22
    self.chart.evaluate(
        '(element, spec) => MeridianCharts.mount(element, spec)', exported
    )
    mark = self.chart.locator('.mark-symbol path').first
    self.assertIn('value: 22', mark.get_attribute('aria-label'))
    self.assertNotEqual(before, mark.get_attribute('transform'))
    self.host.get_by_role(
        'button', name='Show chart source data for Source comparison'
    ).click()
    self.assertEqual(
        self.host.locator('tbody tr').first.locator('td').nth(1).inner_text(),
        '22',
    )
    edited = json.loads(
        self._download('Download chart spec for Source comparison')
    )
    self.assertEqual(edited['datasets']['source'][0]['value'], 22)
    edited_svg = ET.fromstring(
        self._download('Download SVG for Source comparison')
    )
    self.assertTrue(
        any(
            'value: 22' in item.attrib.get('aria-label', '')
            for item in edited_svg.iter()
        )
    )

  def test_tooltip_and_original_native_selection_remain_interactive(self):
    mark = self.chart.locator('.mark-symbol path').first
    mark.hover()
    self.page.wait_for_function("""() => document.querySelector('#native-chart')
      .__meridianChart.view.signal('highlight').index?.includes(0)""")
    tooltip = self.page.locator('#vg-tooltip-element')
    self.assertIn(_LABEL, tooltip.inner_text())
    self.assertEqual(tooltip.locator('img, script').count(), 0)
    self.assertIsNone(self.page.evaluate('window.pwned'))

  def test_visible_loading_and_error_preserve_source_and_spec(self):
    self.chart.evaluate("""element => {
      const spec = element.__meridianChart.spec;
      window.originalEmbed = window.vegaEmbed;
      window.vegaEmbed = () => new Promise((resolve, reject) => { window.rejectEmbed = reject; });
      MeridianCharts.mount(element, spec);
    }""")
    self.assertEqual(
        self.host.get_by_role('status').inner_text(), 'Loading chart…'
    )
    self.assertTrue(
        self.host.get_by_role(
            'button', name='Download SVG for Source comparison'
        ).is_disabled()
    )
    self.page.evaluate(
        "window.rejectEmbed(new Error('<img src=x onerror=window.pwned=1>'))"
    )
    self.host.get_by_role('alert').wait_for()
    self.assertIn(
        'Chart unavailable:', self.host.get_by_role('alert').inner_text()
    )
    self.assertEqual(self.host.get_by_role('alert').locator('img').count(), 0)
    self.host.get_by_role(
        'button', name='Show chart source data for Source comparison'
    ).click()
    self.assertEqual(self.host.locator('tbody tr').count(), 50)
    self.assertEqual(
        json.loads(self._download('Download chart spec for Source comparison'))[
            'datasets'
        ]['source'][0]['value'],
        11,
    )

  def test_semantic_delta_colors_meet_actual_text_contrast(self):
    if str(_ROOT) not in sys.path:
      sys.path.insert(0, str(_ROOT))
    from scripts import test_report_browser  # pylint: disable=import-outside-toplevel

    colors = test_report_browser._check_stat_contrast(self.page)
    self.assertEqual(
        [color['semantic_class'] for color in colors],
        ['neutral-text', 'green-text', 'red-text'],
    )
    self.assertTrue(all(color['contrast_ratio'] >= 4.5 for color in colors))

  def test_invalid_json_has_visible_failure_and_no_export(self):
    self.chart.evaluate(
        "element => MeridianCharts.mount(element, 'invalid JSON')"
    )
    self.assertIn(
        'Chart unavailable:', self.host.get_by_role('alert').inner_text()
    )
    for button in self.host.locator('.chart-controls button').all():
      self.assertTrue(button.is_disabled())
    self.assertEqual(self.chart.locator('svg').count(), 0)


if __name__ == '__main__':
  if len(sys.argv) == 3 and sys.argv[1] == '--write-fixture':
    write_fixture(Path(sys.argv[2]))
  else:
    unittest.main()
