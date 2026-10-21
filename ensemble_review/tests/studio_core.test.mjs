import test from "node:test";
import assert from "node:assert/strict";
import { allPages, basinAvailability, basinCompatibilityLoaded, coerce, envelope, meanByDate, useBenchmarkCoverage } from "../complexity_ensemble/studio/core.js";

test("coverage selection removes unavailable stations from targets and the training pool", () => {
  for (const scope of ["targets_only", "all_basins", "exclude_targets"]) {
    const config = { benchmark: { strict_cohort: true }, basins: {
      target_basins: ["2011", "2012"], training_scope: scope, eligible_basins: null,
    } };
    const report = { effective_targets: ["2011"], training_basins: scope === "exclude_targets" ? ["2013"] : ["2011"] };
    const filtered = useBenchmarkCoverage(config, report);
    assert.deepEqual(filtered.basins.target_basins, ["2011"]);
    assert.deepEqual(new Set(filtered.basins.eligible_basins), new Set([...report.training_basins, "2011"]));
    assert.equal(filtered.basins.training_scope, scope);
    assert.equal(filtered.benchmark.strict_cohort, true);
    assert.deepEqual(config.basins.target_basins, ["2011", "2012"]);
  }
  assert.throws(() => useBenchmarkCoverage({}, { effective_targets: [], training_basins: ["2011"] }), /No usable/);
  assert.throws(() => useBenchmarkCoverage({}, { effective_targets: ["2011"], training_basins: [] }), /No usable/);
});

test("an older API without compatibility metadata is distinct from an incompatible cohort", () => {
  const oldRows = [{ basin_id: "2063", eligible: true }];
  assert.equal(basinCompatibilityLoaded(oldRows, "none"), true);
  assert.equal(basinCompatibilityLoaded(oldRows, "bafu_only"), false);
  assert.equal(basinCompatibilityLoaded(oldRows, "bafu_combined"), false);
  const incompatible = [{ ...oldRows[0], compatibility: { bafu_only: { eligible: false, reasons: ["Missing positive catchment area"] } } }];
  assert.equal(basinCompatibilityLoaded(incompatible, "bafu_only"), true);
});

test("basin selection follows the data source and fails closed for unknown compatibility", () => {
  const row = { eligible: false, compatibility: {
    bafu_only: { eligible: true, reasons: [] },
    bafu_combined: { eligible: false, reasons: ["CAMELS predictors unavailable"] },
  } };
  assert.equal(basinAvailability(row, "none").eligible, false);
  assert.equal(basinAvailability(row, "bafu_only").eligible, true);
  assert.deepEqual(basinAvailability(row, "bafu_combined").reasons, ["CAMELS predictors unavailable"]);
  assert.equal(basinAvailability({ eligible: true }, "bafu_only").eligible, false);
});

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
