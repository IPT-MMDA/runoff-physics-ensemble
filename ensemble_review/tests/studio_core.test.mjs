import test from "node:test";
import assert from "node:assert/strict";
import { allPages, hanzePerformance, floodPerformance, floodMetricDefinitions, floodScoreColor, floodScoreText, basinAvailability, basinCompatibilityLoaded, coerce, configureBafuTask, envelope, meanByDate, useBenchmarkCoverage } from "../complexity_ensemble/studio/core.js";

test("BAFU drought task stays selected and CAMELS settings remain unchanged", () => {
  const config = { benchmark: { mode: "bafu_only", target: "daily_max", threshold_source: "official_return_level" },
    extremes: { mode: "drought", drought_quantile: .15 }, hyperparameters: { physics_weight: .1 } };
  const drought = configureBafuTask(config);
  assert.equal(drought.extremes.mode, "drought");
  assert.equal(drought.benchmark.target, "daily_mean");
  assert.equal(drought.benchmark.threshold_source, "training_quantile");
  assert.equal(config.benchmark.target, "daily_max");
  assert.equal(configureBafuTask({ ...drought, benchmark: { ...drought.benchmark, mode: "bafu_combined" } }).extremes.mode, "drought");
  for (const mode of ["none", "statistical", "event_based", "drought"]) {
    const camels = { ...config, benchmark: { ...config.benchmark, mode: "none" }, extremes: { mode } };
    assert.deepEqual(configureBafuTask(camels), camels);
  }
});

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


test("flood performance averages seeds within catchments before equal-catchment means", () => {
  const row = { event_type: "statistical_high_flow", verifier: "high_flow_one_to_one_v1", predicted_events: 4, matched_events: 3 };
  const result = floodPerformance([
    { ...row, basin_id: "A", seed: 0, event_precision: .75, event_recall: .5, peak_magnitude_mae_mm_day: 2, peak_timing_mae_days: 1 },
    { ...row, basin_id: "A", seed: 1, event_precision: .25, event_recall: 1, predicted_events: 4, matched_events: 1, peak_magnitude_mae_mm_day: 4, peak_timing_mae_days: 3 },
    { ...row, basin_id: "B", seed: 0, event_precision: 1, event_recall: 1, predicted_events: 1, matched_events: 1, peak_magnitude_mae_mm_day: null, peak_timing_mae_days: null },
    { ...row, basin_id: "C", seed: 0, event_precision: null, event_recall: null, predicted_events: 0, matched_events: 0 },
    { ...row, basin_id: "A", lead_day: 1, event_precision: 0 },
    { ...row, basin_id: "D", event_type: "high_flow_day_by_lead", event_precision: 0 },
    { ...row, basin_id: "E", verifier: "old", event_precision: 0 },
  ]);
  assert.deepEqual(result.basins.get("A").event_precision, { value: .5, count: 2 });
  assert.deepEqual(result.basins.get("A").event_far, { value: .5, count: 2 });
  assert.deepEqual(result.basins.get("C").event_far, { value: null, count: 0 });
  assert.deepEqual(result.overall.event_precision, { value: .75, count: 2 });
  assert.deepEqual(result.overall.event_recall, { value: .875, count: 2 });
  assert.deepEqual(result.overall.event_far, { value: .25, count: 2 });
  assert.deepEqual(result.overall.peak_magnitude_mae_mm_day, { value: 3, count: 1 });
  assert.deepEqual(result.overall.peak_timing_mae_days, { value: 2, count: 1 });
  assert.equal(result.basins.size, 3);
  assert.deepEqual(floodPerformance([]).overall.event_recall, { value: null, count: 0 });
});

test("flood map distinguishes missing scores and reverses error colour scales", () => {
  const [precision, , far, peak] = floodMetricDefinitions;
  assert.equal(floodScoreColor(null, precision), "#d8dde3");
  assert.equal(floodScoreColor(1, precision), floodScoreColor(0, far));
  assert.equal(floodScoreColor(0, precision), floodScoreColor(4, peak, 4));
  assert.equal(floodScoreColor(0, far), floodScoreColor(0, peak, 0));
  assert.equal(floodScoreText(.75, precision), "75%");
  assert.equal(floodScoreText(null, peak), "—");
});


test("HANZE scores keep catalogue agreement separate and average only the selected lead", () => {
  const scores = hanzePerformance([
    { basin_id: "A", seed: 0, lead_day: 1, status: "evaluated", catalogue_precision: 1, catalogue_recall: 1 },
    { basin_id: "A", seed: 1, lead_day: 1, status: "evaluated", catalogue_precision: 0, catalogue_recall: 0 },
    { basin_id: "B", seed: 0, lead_day: 1, status: "evaluated", catalogue_precision: 1, catalogue_recall: null },
    { basin_id: "C", seed: 0, lead_day: 1, status: "unavailable", catalogue_precision: 0, catalogue_recall: 0 },
    { basin_id: "A", seed: 0, lead_day: 2, status: "evaluated", catalogue_precision: 0, catalogue_recall: 0 },
  ], "1");
  assert.deepEqual(scores.basins.get("A").catalogue_precision, { value: .5, count: 2 });
  assert.deepEqual(scores.overall.catalogue_precision, { value: .75, count: 2 });
  assert.deepEqual(scores.overall.catalogue_recall, { value: .5, count: 1 });
  assert.equal(scores.basins.has("C"), false);
  assert.deepEqual(hanzePerformance([], 1).overall.catalogue_recall, { value: null, count: 0 });
});


test("HANZE rolling-block averages never mix in sparse fixed-lead scores", () => {
  const scores = hanzePerformance([
    { basin_id: "A", seed: 0, lead_day: "all", status: "evaluated", catalogue_precision: .25, catalogue_recall: 1 },
    { basin_id: "A", seed: 0, lead_day: 1, status: "evaluated", catalogue_precision: null, catalogue_recall: null },
    { basin_id: "A", seed: 0, lead_day: 2, status: "evaluated", catalogue_precision: 0, catalogue_recall: 0 },
  ], "all");
  assert.deepEqual(scores.basins.get("A").catalogue_recall, { value: 1, count: 1 });
  assert.deepEqual(scores.overall.catalogue_precision, { value: .25, count: 1 });
});
