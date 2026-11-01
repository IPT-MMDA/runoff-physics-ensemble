import test from 'node:test';
import assert from 'node:assert/strict';
import { errorColor, comparisonRequest, HydropowerComparisonView } from '../complexity_ensemble/studio/hydropower-comparison.js';

class Node {
  constructor() { this.children = []; this.value = ''; this.textContent = ''; this.disabled = false; this.listeners = {}; }
  get options() { return this.children; }
  addEventListener(event, fn) { this.listeners[event] = fn; }
  replaceChildren(...children) { this.children = children; this.value = children[0]?.value || ''; this.textContent = ''; }
  append(...children) { this.children.push(...children); }
  checkValidity() { return true; }
}
function fixture(api = async () => structuredClone(result), leaflet = async () => { throw new Error('Offline maps'); }) {
  const nodes = new Map();
  globalThis.document = {
    getElementById(id) { if (!nodes.has(id)) nodes.set(id, new Node()); return nodes.get(id); },
    createElement() { return new Node(); },
  };
  const node = id => document.getElementById(`hydro-compare-${id}`);
  const tables = {};
  const view = new HydropowerComparisonView(api, (parent, rows) => {
    for (const name of ['metrics', 'basins', 'plants']) if (parent === node(name)) tables[name] = rows;
  }, leaflet);
  for (const [name, value] of Object.entries({ head_m: '100', efficiency: '0.9', environmental_flow_m3_s: '0',
    max_turbine_flow_m3_s: '', shortfall_eur_mwh: '50', surplus_eur_mwh: '20', lead: '1' })) node(name).value = value;
  view.setRuns([{ run_id: 'one' }, { run_id: 'two' }]);
  return { view, node, tables };
}
const metrics = { mae_m3_s: 1, rmse_m3_s: 2, bias_m3_s: -1, mean_basin_nse: .8, nse_basins: 1 };
const result = { run_a: 'one', run_b: 'two', available_dates: ['2020-01-01', '2020-01-02'], target_date: '2020-01-02',
  period: { start: '2020-01-01', end: '2020-01-02' }, samples: 2, basins: 1, lead_day: 1,
  metrics_a: metrics, metrics_b: metrics, audit: { only_a: 0 }, basin_results: [],
  accuracy_change: { mae_reduction_pct: 10, rmse_reduction_pct: 8, mean_basin_nse_gain: .1 },
  economics: { cost_a_eur: 80, cost_b_eur: 100, saving_eur: 20, saving_pct: 20, cost_index_a: 80, cost_index_b: 100 },
  map: { features: [{ geometry: { coordinates: [7, 46] }, properties: { name: '<Test>', basin_id: 'basin', observed: 10,
    a: 11, b: 8, error_a_m3_s: 1, error_b_m3_s: -2, status: 'Paired forecast' } }], error_scale_m3_s: 2 } };

test('error gradients use a shared symmetric scale and distinguish missing data', () => {
  assert.equal(errorColor(null, 2), '#9ca3af');
  assert.equal(errorColor(0, 2), 'rgb(247,247,247)');
  assert.equal(errorColor(-2, 2), 'rgb(33,102,172)');
  assert.equal(errorColor(2, 2), 'rgb(178,24,43)');
  assert.notEqual(errorColor(1, 2), errorColor(2, 2));
});

test('request preserves assumptions, selected period and lead', () => {
  const { node } = fixture();
  node('start').value = '2020-01-01'; node('lead').value = '7';
  const request = comparisonRequest();
  assert.equal(request.run_b, 'two');
  assert.equal(request.scenario.max_turbine_flow_m3_s, null);
  assert.equal(request.scenario.efficiency, .9);
  assert.equal(request.start, '2020-01-01');
  assert.equal(request.lead_day, 7);
  node('head_m').value = '';
  assert.throws(comparisonRequest, /assumptions/);
});

test('comparison posts paired settings and renders period metrics and costs', async () => {
  let sent;
  const { view, node, tables } = fixture(async (path, options) => { sent = { path, ...options }; return result; });
  await view.update();
  assert.equal(sent.path, '/hydropower/compare');
  assert.equal(sent.method, 'POST');
  assert.equal(sent.body.run_a, 'one');
  assert.equal(node('date').value, '2020-01-02');
  assert.match(node('summary').textContent, /20%/);
  assert.match(node('accuracy').textContent, /MAE reduction 10%/);
  assert.equal(tables.metrics.length, 2);
  assert.equal(tables.plants[0]['error B (m³/s)'], -2);
  assert.equal(node('download').disabled, false);
  assert.equal(node('update').disabled, false);
  assert.match(node('map-status').textContent, /Offline/);
});

test('changing assumptions invalidates pending comparisons and clears stale costs', async () => {
  let resolve;
  const { view, node } = fixture(() => new Promise(done => { resolve = done; }));
  const pending = view.update();
  node('efficiency').value = '.8'; node('efficiency').listeners.input();
  resolve(result); await pending;
  assert.equal(view.result, null);
  assert.equal(node('summary').textContent, '');
  assert.equal(node('download').disabled, true);
  assert.equal(node('update').disabled, false);
});

test('refresh removes unavailable runs and comparison failures remain retryable', async () => {
  const { view, node } = fixture(async () => { throw new Error('No shared observations'); });
  await view.update();
  assert.match(node('status').textContent, /No shared observations/);
  assert.equal(node('download').disabled, true);
  assert.equal(node('update').disabled, false);
  view.setRuns([]);
  assert.equal(node('update').disabled, true);
});

test('an older server missing the comparison route explains how to load the update', async () => {
  for (const message of ['Not Found', 'Request failed (404)', 'run not found']) {
    const { view, node } = fixture(async () => { throw new Error(message); });
    await view.update();
    if (message === 'run not found') assert.equal(node('status').textContent, message);
    else assert.match(node('status').textContent, /Restart the Studio\/API server/);
    assert.equal(node('update').disabled, false);
  }
});

test('both maps render the same plants with their own errors and synchronize views', async () => {
  const markers = [], maps = [];
  const L = {
    map() {
      const map = { center: { lat: 46, lng: 7, equals(p) { return p.lat === this.lat && p.lng === this.lng; } }, zoom: 8,
        setView(center, zoom) { if (!Array.isArray(center)) this.center = center; this.zoom = zoom; return this; },
        getCenter() { return this.center; }, getZoom() { return this.zoom; },
        on(event, handler) { this[event] = handler; }, invalidateSize() {} };
      maps.push(map); return map;
    },
    tileLayer() { return { on() { return this; }, addTo() {} }; },
    featureGroup() { return { addTo() { return this; }, clearLayers() {} }; },
    circleMarker(coords, options) {
      markers.push({ coords, options });
      return { bindPopup(popup) { assert.equal(popup.children[0].textContent, '<Test>'); return this; }, addTo() {} };
    },
  };
  const { view } = fixture(undefined, async () => L);
  await view.drawMaps(result, view.version);
  assert.equal(maps.length, 2);
  assert.equal(markers.length, 2);
  assert.equal(markers[0].options.fillColor, errorColor(1, 2));
  assert.equal(markers[1].options.fillColor, errorColor(-2, 2));
  maps[0].zoom = 10; maps[0].moveend();
  assert.equal(maps[1].zoom, 10);
});
