import test from 'node:test';
import assert from 'node:assert/strict';
import { BasinAtlas } from '../complexity_ensemble/studio/charts.js';

class Path {
  constructor() { this.rings = []; }
  moveTo(x, y) { this.rings.push([[x, y]]); }
  lineTo(x, y) { this.rings.at(-1).push([x, y]); }
  closePath() {}
}

function contains(path, x, y) {
  let inside = false;
  for (const ring of path.rings) {
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [xi, yi] = ring[i], [xj, yj] = ring[j];
      if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
    }
  }
  return inside;
}

function fixture() {
  globalThis.Path2D = Path;
  globalThis.ResizeObserver = class { observe() {} };
  globalThis.window = { devicePixelRatio: 2 };
  globalThis.getComputedStyle = () => ({ getPropertyValue: key => key === '--map-fill' ? 'blue' : 'navy' });
  const ctx = {
    fills: [], strokes: [], arcs: [], transforms: [],
    setTransform(...args) { this.transforms.push(args); }, save() {}, restore() {},
    clearRect() { this.fills = []; this.strokes = []; this.arcs = []; },
    fill(path) { if (path) this.fills.push({ path, color: this.fillStyle }); },
    stroke(path) { if (path) this.strokes.push({ path, color: this.strokeStyle }); },
    beginPath() {}, arc(...args) { this.arcs.push(args); },
    isPointInPath(path, x, y, rule) { assert.equal(rule, 'evenodd'); return contains(path, x, y); },
  };
  const canvas = {
    listeners: {}, addEventListener(event, fn) { this.listeners[event] = fn; },
    getBoundingClientRect: () => ({ width: 600, height: 400, left: 15, top: 25 }),
    getContext: () => ctx,
  };
  const selected = [];
  const atlas = new BasinAtlas(canvas, id => selected.push(id));
  const eventAt = entry => ({ clientX: entry.x + 15, clientY: entry.y + 25 });
  return { atlas, ctx, canvas, selected, eventAt };
}

function basin(id, min, max, eligible = true) {
  return { basin_id: id, name: id, eligible, geometry: { type: 'Polygon', coordinates: [[
    [min, min], [max, min], [max, max], [min, max], [min, min],
  ]] } };
}

test('overlapping unselected downstream basins cannot paint over a selected basin', () => {
  for (const reverse of [false, true]) {
    const { atlas, ctx } = fixture();
    const rows = [basin('small', 4, 6), basin('large', 0, 10)];
    atlas.setData(reverse ? rows.reverse() : rows);
    atlas.setSelected(new Set(['small']));
    const small = atlas.paths.find(entry => entry.row.basin_id === 'small');
    const visible = ctx.fills.filter(({ path }) => contains(path, small.x, small.y)).at(-1);
    assert.equal(visible.path, small.path);
    assert.equal(visible.color, 'blue');
    assert.equal(ctx.strokes.at(-1).path, small.path);
  }
});

test('nested selected catchments are painted large first and all retain selected outlines', () => {
  const { atlas, ctx } = fixture();
  atlas.setData([basin('small', 4, 6), basin('large', 0, 10)]);
  atlas.setSelected(new Set(['large', 'small']));
  assert.deepEqual(ctx.fills.map(({ path }) => atlas.paths.find(entry => entry.path === path).row.basin_id), ['large', 'small']);
  assert.ok(ctx.strokes.slice(-2).every(entry => entry.color === 'navy'));
});

test('clicks and hover identify the smallest eligible overlapping basin at high DPI', () => {
  const { atlas, ctx, canvas, selected, eventAt } = fixture();
  atlas.setData([basin('small', 4, 6), basin('large', 0, 10), basin('ineligible', 4.5, 5.5, false)]);
  const small = atlas.paths.find(entry => entry.row.basin_id === 'small');
  canvas.listeners.click(eventAt(small));
  assert.deepEqual(selected, ['small']);
  assert.deepEqual(ctx.transforms.at(-1), [1, 0, 0, 1, 0, 0]);
  canvas.listeners.pointermove(eventAt(small));
  assert.equal(canvas.title, 'small · small');
});

test('six tiny selections each receive a visible clickable marker and clear correctly', () => {
  const { atlas, ctx, canvas, selected, eventAt } = fixture();
  const tiny = Array.from({ length: 6 }, (_, i) => basin(`tiny-${i}`, i + 1, i + 1.01));
  atlas.setData([...tiny, basin('large', 0, 10)]);
  atlas.setSelected(new Set(tiny.map(row => row.basin_id)));
  assert.equal(atlas.markers.length, 6);
  assert.equal(ctx.arcs.length, 6);
  const marker = atlas.markers[0];
  canvas.listeners.click({ ...eventAt(marker), clientX: marker.x + 15 + 4 });
  assert.deepEqual(selected, [marker.row.basin_id]);
  atlas.setSelected(new Set());
  assert.equal(atlas.markers.length, 0);
  assert.equal(ctx.arcs.length, 0);
});

test('multipolygons and holes retain their geometry and performance colors', () => {
  const { atlas, ctx, canvas, eventAt } = fixture();
  const large = basin('large', 0, 10);
  large.geometry.coordinates.push(basin('hole', 4, 6).geometry.coordinates[0]);
  const multi = basin('multi', 1, 2);
  multi.geometry = { type: 'MultiPolygon', coordinates: [multi.geometry.coordinates, basin('part', 8, 9).geometry.coordinates] };
  atlas.setData([large, multi, { basin_id: 'missing' }, { basin_id: 'empty', geometry: { type: 'Polygon', coordinates: [] } }]);
  assert.equal(atlas.paths.length, 2);
  const outer = atlas.paths.find(entry => entry.row.basin_id === 'large');
  assert.equal(atlas.hitTest(eventAt(outer)), undefined);
  atlas.setPerformance(new Map([['multi', { color: 'green', label: 'Good' }]]));
  const entry = atlas.paths.find(entry => entry.row.basin_id === 'multi');
  assert.equal(ctx.fills.find(({ path }) => path === entry.path).color, 'green');
  const [x, y] = entry.path.rings[0][0];
  canvas.listeners.pointermove({ clientX: x + 15 + 1, clientY: y + 25 - 1 });
  assert.equal(canvas.title, 'multi · multi: Good');
  atlas.setData([]);
  assert.equal(atlas.paths.length, 0);
  assert.equal(atlas.markers.length, 0);
});
