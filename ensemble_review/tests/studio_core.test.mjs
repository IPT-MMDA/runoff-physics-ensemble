import test from "node:test";
import assert from "node:assert/strict";
import { allPages, coerce, envelope, meanByDate } from "../complexity_ensemble/studio/core.js";

test("prediction averages are sorted and ignore unavailable measurements", () => {
  assert.deepEqual(meanByDate([
    { target_date: "2020-01-02", discharge: 6 },
    { target_date: "2020-01-01", discharge: null },
    { target_date: "2020-01-02", discharge: 2 },
    { target_date: "2020-01-01", discharge: 3 },
  ], "discharge").map(({ y, label }) => [label, y]), [["2020-01-01", 3], ["2020-01-02", 4]]);
});

test("chart reduction retains a short flood peak and first/last samples", () => {
  const points = Array.from({ length: 10000 }, (_, x) => ({ x, y: x === 4567 ? 100 : 1 }));
  const reduced = envelope(points);
  assert.ok(reduced.length <= 1200);
  assert.ok(reduced.includes(points[4567]));
  assert.equal(reduced[0], points[0]);
  assert.equal(reduced.at(-1), points.at(-1));
  assert.ok(reduced.every((p, i) => i === 0 || p.x > reduced[i - 1].x));
});

test("configuration preserves nullable and structured fields without coercing invalid numbers", () => {
  assert.equal(coerce("", { anyOf: [{ type: "number" }, { type: "null" }] }), null);
  assert.equal(coerce("0.05", { type: "number" }), .05);
  assert.deepEqual(coerce("[0, 2]", { type: "array" }), [0, 2]);
  assert.deepEqual(coerce("0, 2", { type: "array", items: { type: "integer" } }), [0, 2]);
  assert.deepEqual(coerce("", { type: "array", items: { type: "string" } }), []);
  assert.throws(() => coerce("0, typo", { type: "array", items: { type: "integer" } }));
  assert.deepEqual(coerce('{"4009": 2}', { type: "object" }), { "4009": 2 });
  assert.throws(() => coerce("", { type: "integer" }));
  assert.throws(() => coerce("2.5", { type: "integer" }));
  assert.throws(() => coerce("NaN", { type: "number" }));
});

test("pagination loads all predictions and detects a stalled page", async () => {
  const rows = Array.from({ length: 7 }, (_, x) => x), offsets = [];
  assert.deepEqual(await allPages(async (offset, limit) => {
    offsets.push(offset); return { items: rows.slice(offset, offset + limit), total: rows.length };
  }, 3), rows);
  assert.deepEqual(offsets, [0, 3, 6]);
  await assert.rejects(allPages(async () => ({ items: [], total: 2 })), /incomplete page/);
});
