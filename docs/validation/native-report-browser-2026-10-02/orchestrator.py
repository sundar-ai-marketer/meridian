import json
from pathlib import Path
import sys
sys.path.insert(0, '/private/tmp/meridian-audit-20261002')
from scripts.test_report_browser import run_browser_regression
chrome=Path('/Users/sundar/Library/Caches/ms-playwright/chromium-1243/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing')
root=Path('/private/tmp/meridian-audit-final-e2e')
screens=Path('/private/tmp/meridian-audit-final-native-screenshots')
results=[]
for backend in ('jax','tensorflow'):
 for case in ('geo_nonrevenue_media','national_revenue_media','national_revenue_rf','geo_revenue_mixed'):
  for report in ('summary','optimization'):
   # The accepted representative optimization was already checked on final source.
   if backend=='jax' and case=='geo_nonrevenue_media' and report=='optimization':
    results.append(json.loads(Path('/private/tmp/meridian-audit-final-native-jax-optimization.json').read_text()))
    continue
   print(f'Checking {backend}/{case}/{report}',flush=True)
   result=run_browser_regression(root/backend/case/f'{report}.html',screenshot_dir=screens/backend/case/report,executable_path=chrome,min_desktop_chart_width=500 if report=='optimization' else 600)
   results.append(result)
   (screens/backend/case/f'{report}-browser.json').write_text(json.dumps(result,indent=2)+'\n')
   print('PASS',flush=True)
Path('/private/tmp/meridian-audit-final-native-browser-matrix.json').write_text(json.dumps({'status':'PASS','scope':'offline Chromium; synthetic integration-only fits; native browser SVG/Vega-Lite','reports':results},indent=2)+'\n')
