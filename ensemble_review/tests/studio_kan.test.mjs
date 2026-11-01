import test from 'node:test';
import assert from 'node:assert/strict';
import { KanCaseStudyView, splitUsage } from '../complexity_ensemble/studio/kan-case-study.js';

class Node {
  constructor() { this.children = []; this.value = ''; this.listeners = {}; }
  get options() { return this.children; }
  get selectedOptions() { return this.children.filter(o => o.selected); }
  addEventListener(event, fn) { this.listeners[event] = fn; }
  setAttribute(key, value) { this[key] = value; }
  replaceChildren(...children) { this.children = children; this.value = children[0]?.value || ''; }
  append(...children) { this.children.push(...children); }
}
function fixture(api = async () => ({ items: [] })) {
  const nodes = new Map();
  globalThis.document = {
    querySelector(selector) { if (!nodes.has(selector)) nodes.set(selector, new Node()); return nodes.get(selector); },
    createElement() { return new Node(); },
  };
  const view = new KanCaseStudyView(api, async () => {}, () => {}, () => {});
  view.defaults = { source_basins: [], validation_basins: [], target_basins: [], observation_budgets: [3, 6], seeds: [0], teacher_epochs: 2, train_start: '2001-01-01', device: 'auto', selected_models: ['kan_pure', 'kan_tuned'] };
  view.buildFields();
  return { view, node: id => view.fields.get(id) || document.querySelector(`#kan-${id}`) };
}

for (const legacy of [true, false]) {
  test(`saved results ${legacy ? 'with' : 'without'} flow RMSE fields display without those columns`, async () => {
    const metric = {
      basin_id: 'c', seed: 0, budget: 3, method: 'kan_pure', rmse_mm_day: 1.2,
      local_parameters: { capacity_mm: 100, melt_mm_day_c: 2, runoff_logit_offset: 0.1 },
      fit_id: 'fit-1', max_abs_water_balance_mm: 0.001, state_outside_source_range_fraction: 0.2,
      calibration_first: '2001-01-01', calibration_last: '2001-01-03',
      capacity_mm: 100, melt_mm_day_c: 2, runoff_logit_offset: 0.1,
      ...(legacy ? { low_flow_rmse: 0.5, high_flow_rmse: 2.1 } : {}),
    };
    const result = {
      kind: 'kan_case_study', metrics: [metric], equations: legacy ? [
        { seed: 0, logit_expression: 'storage' },
      ] : [
        { seed: 0, logit_expression: 'storage', models: ['kan_pure', 'kan_tuned'] },
        { seed: 0, logit_expression: 'liquid', models: ['kan_complex'] },
      ],
      config: { source_basins: ['a', 'b'], validation_basins: ['v'], target_basins: ['c'], seeds: [0] },
      audit: legacy ? {} : { teacher_training_basins: ['actual-source'], equation_selection_basins: ['actual-validation'] },
    };
    const original = structuredClone(result);
    const { view, node } = fixture(async path => path.endsWith('/result') ? result : { items: [] });
    let displayed;
    view.renderTable = (_table, rows) => { displayed = rows; };
    view.plot = () => {};
    await view.showJob('saved-run');
    assert.deepEqual(displayed, [{
      basin_id: 'c', seed: 0, budget: 3, method: 'kan_pure', rmse_mm_day: 1.2,
    }]);
    assert.deepEqual(result, original);
    assert.equal(view.result, result);
    assert.equal(node('results').hidden, false);
    assert.equal(node('saved').value, 'saved-run');
    assert.match(node('equations').textContent, /F\(storage, liquid, pet\) = storage/);
    if (!legacy) assert.match(node('equations').textContent, /kan_complex\nF\(storage, liquid, pet\) = liquid/);
    assert.deepEqual(node('run-basins').children.map(child => child.textContent), legacy ? [
      'Source training basins: a, b', 'Source validation basins: v',
    ] : [
      'Source training basins: actual-source', 'Source validation basins: actual-validation',
    ]);
  });
}

test('case-study requests use separate cohorts and numeric observation budgets', () => {
  const { view, node } = fixture();
  node('dataset').value = 'ch'; view.basinsDataset = 'ch';
  for (const [cohort, id] of [['source_basins', 'a'], ['validation_basins', 'b'], ['target_basins', 'c']]) {
    view.cohorts.get(cohort).add(id);
  }
  const body = view.request();
  assert.deepEqual(body.config.source_basins, ['a']);
  assert.deepEqual(body.config.target_basins, ['c']);
  assert.deepEqual(body.config.observation_budgets, [3, 6]);
  assert.equal(body.config.teacher_epochs, 2);
  assert.equal(body.dataset_id, 'ch');
});

test('submission waits for the selected datasets catchments', () => {
  const { view, node } = fixture(); node('dataset').value = 'new'; view.basinsDataset = 'old';
  assert.throws(() => view.request(), /Wait for/);
});

test('an older dataset response cannot replace the new catchment cohort', async () => {
  let finish;
  const { view, node } = fixture(() => new Promise(resolve => { finish = resolve; }));
  node('dataset').value = 'old'; const pending = view.loadBasins();
  node('dataset').value = 'new';
  finish({ items: [{ basin_id: 'old-basin', eligible: true }], total: 1 });
  await pending;
  assert.equal(view.cohorts.get('source_basins').size, 0);
  assert.equal(view.basinsDataset, undefined);
});

test('coverage failures remain visible in the dedicated study status', async () => {
  const { view, node } = fixture();
  await view.action(async () => { throw new Error('Target overlaps source'); });
  assert.equal(node('status').textContent, 'Target overlaps source');
});

test('changing date settings discards a coverage scan still in flight', async () => {
  let finish;
  const { view, node } = fixture(() => new Promise(resolve => { finish = resolve; }));
  node('dataset').value = 'ch'; view.basinsDataset = 'ch';
  const pending = view.scanCoverage();
  node('train_start').value = '2002-01-01';
  finish({ items: [], as_of: '2001-01-30' });
  await pending;
  assert.equal(view.coverage, undefined);
});

test('date edits clear obsolete classifications and cohort annotations', () => {
  const { view, node } = fixture();
  view.coverage = { items: [] };
  node('train_start').listeners.change();
  assert.equal(view.coverage, null);
  assert.match(node('coverage-status').textContent, /Scan availability again/);
});

test('coverage scans preserve manual roles and add counts without high/low labels', () => {
  const { view, node } = fixture();
  view.cohorts.get('source_basins').add('c');
  view.basins = [{ basin_id: 'c', name: 'Catchment C', eligible: true }];
  view.coverage = { items: [{ basin_id: 'c', training_observations: 3, weather_missing_days: 0 }] };
  view.renderCohorts();
  assert.equal(view.roleFor('c'), 'source');
  const label = view.cohortLists.get('source_basins').children[0];
  assert.equal(label.children[0].checked, true);
  assert.match(label.children[1].textContent, /train 3/);
  assert.doesNotMatch(label.children[1].textContent, /high.data|low.data/);
});

test('each selected target can override the default budgets independently', () => {
  const { view, node } = fixture();
  node('dataset').value = 'ch'; view.basinsDataset = 'ch';
  for (const id of ['c', 'd']) {
    view.cohorts.get('target_basins').add(id);
  }
  view.targetBudgets.set('c', '3, 6'); view.targetBudgets.set('d', '9');
  let body = view.request();
  assert.deepEqual(body.config.target_observation_budgets, { c: [3, 6], d: [9] });
  assert.deepEqual(body.config.observation_budgets, [3, 6]);
  view.cohorts.get('target_basins').delete('d');
  body = view.request();
  assert.deepEqual(body.config.target_observation_budgets, { c: [3, 6] });
});

function checkbox(view, role, basin) {
  return view.cohortLists.get(role).children.find(label => label.children[0]?.value === basin).children[0];
}
function tick(view, role, basin, checked = true) {
  const input = checkbox(view, role, basin);
  assert.equal(input.type, 'checkbox');
  input.checked = checked; input.listeners.change();
}

test('ticking other role groups preserves source selections until explicitly unticked', () => {
  const { view, node } = fixture();
  node('dataset').value = 'ch'; view.basinsDataset = 'ch';
  view.basins = ['a', 'b', 'c', 'd'].map(basin_id => ({ basin_id, name: basin_id, eligible: true }));
  view.renderCohorts();
  tick(view, 'source_basins', 'a'); tick(view, 'source_basins', 'b');
  tick(view, 'validation_basins', 'c'); tick(view, 'target_basins', 'd');
  assert.deepEqual(view.request().config.source_basins, ['a', 'b']);
  assert.deepEqual(view.request().config.validation_basins, ['c']);
  assert.deepEqual(view.request().config.target_basins, ['d']);
  view.renderCohorts();
  assert.equal(checkbox(view, 'source_basins', 'a').checked, true);
  assert.equal(checkbox(view, 'source_basins', 'b').checked, true);
  assert.equal(checkbox(view, 'validation_basins', 'c').checked, true);
  tick(view, 'source_basins', 'a', false);
  assert.deepEqual(view.request().config.source_basins, ['b']);
  assert.equal(checkbox(view, 'target_basins', 'd').checked, true);
});

test('filtering a checkbox list and refreshing coverage preserve checked catchments', () => {
  const { view } = fixture();
  view.basins = ['a', 'b'].map(basin_id => ({ basin_id, name: basin_id, eligible: true }));
  view.renderCohorts(); tick(view, 'source_basins', 'a');
  const search = view.cohortSearchInputs.get('source_basins');
  search.value = 'b'; search.listeners.input();
  tick(view, 'source_basins', 'b');
  assert.deepEqual([...view.cohorts.get('source_basins')], ['a', 'b']);
  view.invalidateCoverage();
  search.value = ''; search.listeners.input();
  assert.equal(checkbox(view, 'source_basins', 'a').checked, true);
  assert.equal(checkbox(view, 'source_basins', 'b').checked, true);
});

test('Run submits a job and shows its identifier; integer defaults align with their step', async () => {
  const { view, node } = fixture();
  node('dataset').value = 'ch'; view.basinsDataset = 'ch';
  let submitted;
  view.submitJob = async (path, body) => { submitted = { path, body }; return { job_id: 'case-123' }; };
  assert.equal(node('form').noValidate, true);
  const epochs = node('teacher_epochs');
  assert.equal(epochs.min, '1'); assert.equal(epochs.step, '1');
  assert.equal((Number(epochs.value) - Number(epochs.min)) % Number(epochs.step), 0);
  await node('form').listeners.submit({ preventDefault() {} });
  assert.equal(submitted.path, '/case-studies/kan');
  assert.equal(submitted.body.dataset_id, 'ch');
  assert.match(node('status').textContent, /Queued job case-123/);
  assert.equal(node('launch').disabled, false);
});

test('Run reports submission and hidden-field validation errors beside the button', async () => {
  const { view, node } = fixture();
  node('dataset').value = 'ch'; view.basinsDataset = 'ch';
  view.submitJob = async () => { throw new Error('Training worker is unavailable'); };
  await node('form').listeners.submit({ preventDefault() {} });
  assert.match(node('status').textContent, /Training worker is unavailable/);
  assert.equal(node('launch').disabled, false);
  const epochs = node('teacher_epochs'), details = { open: false };
  epochs.checkValidity = () => false; epochs.closest = () => details;
  epochs.validationMessage = 'Enter a whole number';
  await node('form').listeners.submit({ preventDefault() {} });
  assert.match(node('status').textContent, /KAN process-teacher epochs: Enter a whole number/);
  assert.equal(details.open, true);
});

test('split usage distinguishes held-out counts from fitting and flags missing target test data', () => {
  const row = { training_observations: 1096, validation_observations: 0, test_observations: 0 };
  const target = splitUsage(row, 'target', [30, 90, 180]);
  assert.equal(target.train_used, '30, 90, 180');
  assert.equal(target.validation_used, 0); assert.equal(target.test_used, 0);
  assert.match(target.readiness, /Needs 3 test observations; has 0/);
  assert.deepEqual(splitUsage(row, 'source', [30]), { train_used: 1096, validation_used: 0, test_used: 0, readiness: 'Ready' });
  assert.match(splitUsage(row, 'validation', [30]).readiness, /Needs 3 validation observations/);
});

test('job updates and failures are visible inside the case-study tab', () => {
  const { view, node } = fixture();
  view.updateJob({ job_id: 'case-123', status: 'running', progress: .4, message: 'Training teacher' });
  assert.match(node('status').textContent, /40% · Training teacher/);
  view.updateJob({ job_id: 'case-123', status: 'failed', error: { message: 'No valid symbolic candidate' } });
  assert.match(node('status').textContent, /failed.*No valid symbolic candidate/);
});

test('the case study has its own hardware selector and sends its selected device', () => {
  const { view, node } = fixture();
  node('dataset').value = 'ch'; view.basinsDataset = 'ch';
  assert.deepEqual(node('device').options.map(o => o.value), ['auto', 'cpu', 'cuda', 'mps']);
  assert.equal(view.request().config.device, 'auto');
  node('device').value = 'mps';
  assert.equal(view.request().config.device, 'mps');
  node('device').value = 'cpu';
  assert.equal(view.request().config.device, 'cpu');
});

test('all twelve comparison variants have persistent checkboxes and only checked models are submitted', () => {
  const { view, node } = fixture();
  node('dataset').value = 'ch'; view.basinsDataset = 'ch';
  const labels = node('selected_models').children.filter(n => n.children[0]?.type === 'checkbox');
  assert.equal(labels.length, 12);
  assert.deepEqual(view.request().config.selected_models, ['kan_pure', 'kan_tuned']);
  const source = labels.find(label => label.children[0].value === 'pinnmamba_source_target').children[0];
  source.checked = true; source.listeners.change();
  const pure = labels.find(label => label.children[0].value === 'kan_pure').children[0];
  pure.checked = false; pure.listeners.change();
  assert.deepEqual(view.request().config.selected_models, ['kan_tuned', 'pinnmamba_source_target']);
  const complex = labels.find(label => label.children[0].value === 'kan_complex').children[0];
  assert.equal(complex.checked, false);
  complex.checked = true; complex.listeners.change();
  assert.deepEqual(view.request().config.selected_models, ['kan_tuned', 'pinnmamba_source_target', 'kan_complex']);
});

test('coverage usage follows selected comparison scopes', () => {
  const row = { training_observations: 0, validation_observations: 10, test_observations: 15 };
  assert.equal(splitUsage(row, 'target', [30], 3, ['mlp_source_target']).train_used, 0);
  assert.equal(splitUsage(row, 'target', [30], 3, ['mlp_source_target']).readiness, 'Ready');
  assert.equal(splitUsage(row, 'validation', [30], 3, ['mlp_source_target']).validation_used, 0);
  assert.match(splitUsage(row, 'validation', [30], 3, ['kan_pure']).readiness, /Ready/);
  assert.equal(splitUsage(row, 'target', [30], 3, ['kan_complex']).train_used, '30 requested (only 0 available)');
  assert.match(splitUsage(row, 'target', [30], 3, ['kan_complex']).readiness, /30/);
});
