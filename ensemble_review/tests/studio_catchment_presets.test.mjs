import test from 'node:test';
import assert from 'node:assert/strict';
import { selectHydropowerCohort } from '../complexity_ensemble/studio/catchment-presets.js';

const ids = Array.from({ length: 50 }, (_, i) => String(2000 + i));
const preset = { basin_ids: ids, available: true, matched_plants: 100 };
const initial = () => ({ dataset: 'camels-ch', mode: 'bafu_only', version: 1, ready: true, selection: ['old'] });

test('one click selects exactly the server cohort for the chosen dataset/source', async () => {
  let selected, path;
  const result = await selectHydropowerCohort(async p => { path = p; return preset; }, initial, value => { selected = value; });
  assert.equal(path, '/datasets/camels-ch/hydropower-cohort?mode=bafu_only');
  assert.deepEqual(selected, ids);
  assert.equal(result, preset);
});

test('unavailable or partial cohorts preserve the current selection', async () => {
  for (const response of [{ ...preset, available: false, unavailable_basins: { '2000': 'Missing observations' } },
    { ...preset, basin_ids: ids.slice(1) }, { ...preset, basin_ids: ids.map(() => 'same') }]) {
    let called = false;
    await assert.rejects(selectHydropowerCohort(async () => response, initial, () => { called = true; }), /unavailable/);
    assert.equal(called, false);
  }
});

test('dataset, source and manual selection changes discard a pending preset response', async () => {
  for (const change of [{ dataset: 'other' }, { mode: 'none' }, { selection: ['manual'] }, { version: 2 }]) {
    let resolve, selected = false, current = initial();
    const pending = selectHydropowerCohort(() => new Promise(done => { resolve = done; }), () => current, () => { selected = true; });
    current = { ...current, ...change }; resolve(preset);
    assert.equal(await pending, null);
    assert.equal(selected, false);
  }
});

test('an unloaded dataset and an older API give useful errors', async () => {
  let calls = 0;
  await assert.rejects(selectHydropowerCohort(async () => { calls++; }, () => ({ ...initial(), ready: false }), () => {}), /finish loading/);
  assert.equal(calls, 0);
  await assert.rejects(selectHydropowerCohort(async () => { throw new Error('Not Found'); }, initial, () => {}), /Restart the Studio/);
});
