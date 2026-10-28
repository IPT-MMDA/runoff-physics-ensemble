import test from 'node:test';
import assert from 'node:assert/strict';
import { dischargeColor, HydropowerView } from '../complexity_ensemble/studio/hydropower.js';

class Node {
  constructor() { this.children = []; this.value = ''; this.textContent = ''; this.disabled = false; this.listeners = {}; }
  get options() { return this.children; }
  addEventListener(event, fn) { this.listeners[event] = fn; }
  replaceChildren(...children) { this.children = children; this.value = children[0]?.value || ''; }
  append(...children) { this.children.push(...children); }
  removeAttribute(name) { delete this[name]; }
}
function fixture(api) {
  const nodes = new Map();
  globalThis.document = {
    getElementById(id) { if (!nodes.has(id)) nodes.set(id, new Node()); return nodes.get(id); },
    createElement() { return new Node(); },
  };
  const view = new HydropowerView(api, () => {});
  view.render = payload => { view.rendered = payload; };
  return { view, node: id => document.getElementById(`hydropower-${id}`) };
}
const stations = { features: [], target_date: null };
const runs = { items: [{ dataset_id: 'ch', run_id: 'one' }, { dataset_id: 'other', run_id: 'two' }], skipped: 0 };
const initial = path => Promise.resolve(path.endsWith('/sites') ? stations : runs);
const forecast = { map: { features: [], target_date: '2024-01-02' }, available_dates: ['2024-01-01', '2024-01-02'], dataset_id: 'ch', run_id: 'one', total: 2, matched: 1, with_change: 1 };

test('change colors distinguish missing, falling, stable, and rising discharge', () => {
  assert.equal(dischargeColor(null), '#9ca3af');
  assert.equal(dischargeColor(NaN), '#9ca3af');
  assert.equal(dischargeColor(-50), 'rgb(215,48,39)');
  assert.equal(dischargeColor(0), 'rgb(255,255,191)');
  assert.equal(dischargeColor(50), 'rgb(26,152,80)');
  assert.equal(dischargeColor(100), dischargeColor(50));
});

test('initial load enables controls, loads once, and refresh discovers new runs', async () => {
  let calls = 0;
  const { view, node } = fixture(path => { ++calls; return initial(path); });
  await view.activate();
  assert.equal(view.loading, false);
  assert.equal(node('refresh').disabled, false);
  assert.equal(node('update').disabled, false);
  assert.equal(node('run').value, 'one');
  assert.equal(view.rendered, stations);
  await view.activate();
  assert.equal(calls, 2);
  await view.activate(true);
  assert.equal(calls, 4);
  assert.equal(node('refresh').disabled, false);
});

test('dataset changes select matching runs and clear previous forecast dates', async () => {
  const { view, node } = fixture(initial);
  await view.activate();
  node('date').value = '2024-01-02';
  node('dataset').value = 'other'; node('dataset').listeners.change();
  assert.equal(node('run').value, 'two');
  assert.equal(node('date').value, '');
  assert.equal(view.rendered, stations);
});

test('completed forecast shows counts and a selected date', async () => {
  const { view, node } = fixture(path => path.includes('/runs/one/') ? Promise.resolve(forecast) : initial(path));
  await view.activate(); await view.update();
  assert.equal(view.rendered, forecast.map);
  assert.equal(node('date').value, '2024-01-02');
  assert.match(node('status').textContent, /1\/2 matched plants/);
  assert.equal(node('update').disabled, false);
});

test('changing selection discards a forecast response still in flight', async () => {
  let resolve;
  const { view, node } = fixture(path => path.includes('/runs/one/') ? new Promise(done => { resolve = done; }) : initial(path));
  await view.activate(); const pending = view.update();
  node('dataset').value = 'other'; node('dataset').listeners.change();
  resolve(forecast); await pending;
  assert.equal(node('run').value, 'two');
  assert.equal(node('date').value, '');
  assert.equal(view.rendered, stations);
  assert.equal(node('update').disabled, false);
});

test('unavailable date clears stale results and leaves the view retryable', async () => {
  const { view, node } = fixture(path => path.includes('/runs/one/') ? Promise.reject(new Error('No predictions for this date')) : initial(path));
  await view.activate(); await view.update();
  assert.match(node('status').textContent, /No predictions/);
  assert.equal(view.rendered, stations);
  assert.equal(node('update').disabled, false);
});

test('disabled inventory reports its error without enabling forecast controls', async () => {
  const { view, node } = fixture(() => Promise.reject(new Error('Hydropower monitoring is disabled')));
  await view.activate();
  assert.match(node('status').textContent, /disabled/);
  assert.equal(view.loaded, false);
  assert.equal(node('refresh').disabled, false);
  assert.equal(node('update').disabled, true);
});
