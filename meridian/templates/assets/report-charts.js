/* Copyright 2026 Sundar Ramesh Kumar.
 * Licensed under the Apache License, Version 2.0.
 * NOTICE: This file is new in this fork; see NOTICE at the repository root.
 */
(() => {
  'use strict';
  const PAGE_SIZE = 50;
  const MAX_COLUMNS = 25;
  let nextId = 0;

  function node(tag, text, className) {
    const element = document.createElement(tag);
    if (text !== undefined) element.textContent = text;
    if (className) element.className = className;
    return element;
  }

  function button(text, label) {
    const element = node('button', text);
    element.type = 'button';
    if (label) element.setAttribute('aria-label', label);
    return element;
  }

  function sourceSets(spec) {
    const sets = [];
    Object.entries(spec.datasets || {}).forEach(([name, rows]) => {
      if (Array.isArray(rows)) sets.push({name, rows});
    });
    function visit(part, path) {
      if (!part || typeof part !== 'object') return;
      if (Array.isArray(part.data?.values)) {
        sets.push({name: path, rows: part.data.values});
      }
      ['layer', 'concat', 'hconcat', 'vconcat'].forEach(key => {
        (Array.isArray(part[key]) ? part[key] : []).forEach((child, index) => {
          visit(child, `${path}.${key}[${index}]`);
        });
      });
      if (part.spec) visit(part.spec, `${path}.spec`);
    }
    visit(spec, 'data');
    return sets;
  }

  function download(content, mime, filename) {
    const url = URL.createObjectURL(new Blob([content], {type: mime}));
    const link = node('a');
    link.href = url;
    link.download = filename;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function sourcePanel(id, sets) {
    const panel = node('section', undefined, 'chart-source-panel');
    panel.id = `${id}-source`;
    panel.hidden = true;
    const heading = node('h4', 'Chart source data');
    heading.id = `${id}-source-title`;
    panel.setAttribute('aria-labelledby', heading.id);
    panel.append(heading, node('p',
        'Embedded source rows. Plotted values may use filters, aggregates or other transforms. The chart spec contains the complete source.',
        'chart-source-description'));
    if (!sets.length) {
      panel.append(node('p', 'No embedded source rows are available.'));
      return panel;
    }
    const select = node('select');
    select.setAttribute('aria-label', 'Chart source dataset');
    sets.forEach((set, index) => {
      const option = node('option', `Dataset ${index + 1} (${set.rows.length} rows)`);
      option.value = String(index);
      select.append(option);
    });
    if (sets.length > 1) panel.append(select);
    const wrapper = node('div', undefined, 'chart-source-scroll');
    wrapper.tabIndex = 0;
    wrapper.setAttribute('role', 'region');
    wrapper.setAttribute('aria-label', 'Chart source data table');
    const table = node('table');
    table.append(node('caption', 'Chart source data'));
    const head = node('thead');
    const body = node('tbody');
    table.append(head, body);
    wrapper.append(table);
    const pager = node('div', undefined, 'chart-source-pager');
    const previous = button('Previous rows');
    const next = button('Next rows');
    const range = node('span');
    range.setAttribute('aria-live', 'polite');
    pager.append(previous, range, next);
    const notice = node('p', undefined, 'chart-source-description');
    panel.append(wrapper, pager, notice);
    let page = 0;
    function render() {
      const dataset = sets[Number(select.value)];
      const rows = dataset.rows;
      if (!dataset.fields) {
        const fields = new Set();
        rows.forEach(row => {
          if (row && typeof row === 'object' && !Array.isArray(row)) {
            Object.keys(row).forEach(key => fields.add(key));
          } else fields.add('value');
        });
        dataset.fields = [...fields];
      }
      const columns = dataset.fields.slice(0, MAX_COLUMNS);
      head.replaceChildren();
      body.replaceChildren();
      const header = node('tr');
      columns.forEach(key => {
        const cell = node('th', key);
        cell.scope = 'col';
        header.append(cell);
      });
      head.append(header);
      const start = page * PAGE_SIZE;
      rows.slice(start, start + PAGE_SIZE).forEach(row => {
        const record = row && typeof row === 'object' && !Array.isArray(row)
            ? row : {value: row};
        const line = node('tr');
        columns.forEach(key => {
          const value = Object.hasOwn(record, key) ? record[key] : undefined;
          const text = value === undefined ? '—' :
              (typeof value === 'object' ? JSON.stringify(value) : String(value));
          const cell = node('td', text.length > 240 ? `${text.slice(0, 240)}…` : text);
          if (text.length > 240) cell.title = text;
          line.append(cell);
        });
        body.append(line);
      });
      range.textContent = rows.length
          ? `Rows ${start + 1}–${Math.min(start + PAGE_SIZE, rows.length)} of ${rows.length}`
          : '0 rows';
      previous.disabled = page === 0;
      next.disabled = start + PAGE_SIZE >= rows.length;
      notice.textContent = dataset.fields.length > MAX_COLUMNS
          ? `Showing the first ${MAX_COLUMNS} of ${dataset.fields.length} columns. Download the chart spec for all fields.`
          : '';
    }
    previous.addEventListener('click', () => { page--; render(); });
    next.addEventListener('click', () => { page++; render(); });
    select.addEventListener('change', () => { page = 0; render(); });
    let populated = false;
    panel.populate = () => {
      if (!populated) { render(); populated = true; }
    };
    return panel;
  }

  async function mount(element, serializedSpec) {
    if (!element) return;
    const old = element.__meridianChart;
    if (old) {
      old.view?.finalize();
      old.observer?.disconnect();
      old.controls.remove();
      old.status.remove();
      old.panel?.remove();
      element.removeEventListener('keydown', old.keydown);
    }
    element.replaceChildren();
    element.setAttribute('role', 'region');
    const id = `meridian-chart-${++nextId}`;
    const controls = node('div', undefined, 'chart-controls');
    controls.setAttribute('role', 'group');
    controls.setAttribute('aria-label', 'Chart data and downloads');
    const show = button('Show data', 'Show chart source data');
    const svg = button('Download SVG');
    const json = button('Download chart spec');
    [show, svg, json].forEach(control => { control.disabled = true; });
    controls.append(show, svg, json);
    const status = node('p', 'Loading chart…', 'chart-render-status');
    status.setAttribute('role', 'status');
    status.setAttribute('aria-live', 'polite');
    element.before(status, controls);
    const state = {controls, status, view: null, spec: null, panel: null};
    element.__meridianChart = state;
    function failed(message) {
      status.className = 'chart-render-status chart-render-error';
      status.setAttribute('role', 'alert');
      status.textContent = message;
    }
    try {
      const spec = typeof serializedSpec === 'string'
          ? JSON.parse(serializedSpec) : JSON.parse(JSON.stringify(serializedSpec));
      if (!spec || typeof spec !== 'object' || Array.isArray(spec)) {
        throw new Error('A chart specification must be a JSON object.');
      }
      state.spec = spec;
      const titleValue = typeof spec.title === 'object' ? spec.title?.text : spec.title;
      const title = (Array.isArray(titleValue) ? titleValue.join(' ') : titleValue)
          || element.parentElement.querySelector('.chart-title')?.textContent.trim()
          || `Chart ${nextId}`;
      const chartTitle = String(title);
      controls.setAttribute('aria-label', `${chartTitle}: chart data and downloads`);
      svg.setAttribute('aria-label', `Download SVG for ${chartTitle}`);
      json.setAttribute('aria-label', `Download chart spec for ${chartTitle}`);
      state.panel = sourcePanel(id, sourceSets(spec));
      element.after(state.panel);
      show.disabled = false;
      json.disabled = false;
      show.setAttribute('aria-expanded', 'false');
      show.setAttribute('aria-controls', state.panel.id);
      show.setAttribute('aria-label', `Show chart source data for ${chartTitle}`);
      show.addEventListener('click', () => {
        state.panel.populate?.();
        state.panel.hidden = !state.panel.hidden;
        show.setAttribute('aria-expanded', String(!state.panel.hidden));
        show.textContent = state.panel.hidden ? 'Show data' : 'Hide data';
        show.setAttribute('aria-label', state.panel.hidden
            ? `Show chart source data for ${chartTitle}` : `Hide chart source data for ${chartTitle}`);
      });
      const filename = (element.id || id).replace(/[^a-zA-Z0-9_-]/g, '_');
      json.addEventListener('click', () => {
        download(JSON.stringify(state.spec, null, 2) + '\n',
            'application/json', `${filename}.vl.json`);
      });
      const result = await vegaEmbed(element, JSON.parse(JSON.stringify(spec)), {
        renderer: 'svg', actions: false,
      });
      if (element.__meridianChart !== state) { result.view.finalize(); return; }
      state.view = result.view;
      svg.disabled = false;
      status.textContent = 'Chart ready.';
      status.className = 'chart-render-status chart-render-ready';
      svg.addEventListener('click', async () => {
        const hadFocus = document.activeElement === svg;
        svg.disabled = true;
        try {
          download(await state.view.toSVG(), 'image/svg+xml', `${filename}.svg`);
        } catch (error) {
          failed(`SVG download failed: ${error.message}. Try again.`);
        } finally {
          svg.disabled = false;
          if (hadFocus && document.activeElement === document.body) {
            svg.focus({preventScroll: true});
          }
        }
      });
      function overflow() {
        const scrollable = element.scrollWidth > element.clientWidth + 1;
        element.tabIndex = scrollable ? 0 : -1;
        element.setAttribute('aria-label', scrollable
            ? `${chartTitle}. Scrollable chart. Use left and right arrow keys to pan.`
            : chartTitle);
        if (scrollable) element.setAttribute('aria-keyshortcuts', 'ArrowLeft ArrowRight Home End');
        else element.removeAttribute('aria-keyshortcuts');
      }
      state.keydown = event => {
        if (event.target !== element || element.tabIndex !== 0) return;
        if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
          event.preventDefault();
          if (event.key === 'Home') element.scrollLeft = 0;
          else if (event.key === 'End') element.scrollLeft = element.scrollWidth;
          else element.scrollLeft += (event.key === 'ArrowRight' ? 1 : -1)
              * Math.max(80, element.clientWidth * 0.8);
        }
      };
      element.addEventListener('keydown', state.keydown);
      state.observer = new ResizeObserver(overflow);
      state.observer.observe(element);
      overflow();
    } catch (error) {
      if (element.__meridianChart === state) {
        failed(`Chart unavailable: ${error.message}. Source data and spec remain available when valid.`);
      }
    }
  }
  window.MeridianCharts = Object.freeze({mount});
})();
