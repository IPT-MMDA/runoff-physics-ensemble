// Pure data transforms shared by the browser and dependency-free Node tests.
export const terminal = status => ["succeeded", "failed", "cancelled"].includes(status);
export const pretty = key => key.replaceAll("_", " ").replace(/\b\w/g, c => c.toUpperCase());
export const finite = value => typeof value === "number" && Number.isFinite(value);
export function configureBafuTask(config) {
  const next = structuredClone(config);
  if (next.benchmark.mode === "none") return next;
  if (next.extremes.mode !== "drought") next.extremes.mode = "statistical";
  if (next.extremes.mode === "drought") {
    next.benchmark.target = "daily_mean";
    next.benchmark.threshold_source = "training_quantile";
  }
  return next;
}
export function useBenchmarkCoverage(config, report) {
  if (!report.effective_targets?.length || !report.training_basins?.length) {
    throw new Error("No usable target or training stations. Adjust the dates or input history and check coverage again.");
  }
  return { ...config, basins: { ...config.basins,
    target_basins: [...report.effective_targets],
    eligible_basins: [...new Set([...report.training_basins, ...report.effective_targets])],
  } };
}
export function basinCompatibilityLoaded(rows, mode = "none") {
  return mode === "none" || rows.every(row => typeof row.compatibility?.[mode]?.eligible === "boolean");
}
export function basinAvailability(row, mode = "none") {
  return mode === "none"
    ? { eligible: row.eligible, reasons: row.eligible ? [] : ["CAMELS predictors unavailable"] }
    : row.compatibility?.[mode] || { eligible: false, reasons: ["BAFU compatibility unavailable"] };
}
export function format(value) {
  if (value == null || (typeof value === "number" && !finite(value))) return "—";
  if (typeof value !== "number") return String(value);
  return new Intl.NumberFormat("en", { maximumFractionDigits: 4 }).format(value);
}
export function meanByDate(rows, key) {
  const groups = new Map();
  for (const row of rows) {
    if (!finite(row[key])) continue;
    const [sum, count] = groups.get(row.target_date) || [0, 0];
    groups.set(row.target_date, [sum + row[key], count + 1]);
  }
  return [...groups].sort(([a], [b]) => a.localeCompare(b)).map(([date, [sum, count]]) => ({ x: Date.parse(date), y: sum / count, label: date }));
}
// Keep local extrema rather than skipping every nth point and hiding flood peaks.
export function envelope(points, max = 1200) {
  if (points.length <= max) return points;
  const output = [points[0]], width = Math.ceil((points.length - 2) / ((max - 2) / 2));
  for (let start = 1; start < points.length - 1; start += width) {
    const end = Math.min(start + width, points.length - 1);
    let low = start, high = start;
    for (let i = start + 1; i < end; i++) {
      if (points[i].y < points[low].y) low = i;
      if (points[i].y > points[high].y) high = i;
    }
    for (const index of [...new Set([low, high])].sort((a, b) => a - b)) output.push(points[index]);
  }
  output.push(points.at(-1));
  return output;
}
export function coerce(raw, schema) {
  const nullable = schema.anyOf?.some(s => s.type === "null");
  const type = schema.type || schema.anyOf?.find(s => s.type !== "null")?.type;
  if (raw === "" && nullable) return null;
  if (type === "boolean") return Boolean(raw);
  if (type === "array") {
    if (raw.trim().startsWith("[")) return JSON.parse(raw);
    if (!raw.trim()) return [];
    return raw.split(",").map(value => coerce(value.trim(), schema.items || { type: "string" }));
  }
  if (type === "object") return JSON.parse(raw);
  if (type === "integer" || type === "number") {
    if (String(raw).trim() === "") throw new Error("A number is required");
    const value = Number(raw);
    if (!finite(value) || (type === "integer" && !Number.isInteger(value))) throw new Error("Enter a valid " + type);
    return value;
  }
  return raw;
}
export async function allPages(fetchPage, pageSize = 1000) {
  const items = [];
  for (;;) {
    const page = await fetchPage(items.length, pageSize);
    items.push(...page.items);
    if (items.length >= page.total) return items;
    if (!page.items.length) throw new Error("The server returned an incomplete page. Please retry.");
  }
}

// Event scores use the one-to-one verifier, never the duplicated per-lead day rows.
export const floodMetricDefinitions = [
  { key: "event_precision", label: "Event precision", percent: true, higher: true },
  { key: "event_recall", label: "Event recall", percent: true, higher: true },
  { key: "event_far", label: "Event FAR", percent: true, higher: false },
  { key: "peak_magnitude_mae_mm_day", label: "Peak magnitude MAE (mm/day)", higher: false },
  { key: "peak_timing_mae_days", label: "Peak timing MAE (days)", higher: false },
];
export function floodPerformance(rows) {
  const groups = new Map();
  for (const row of rows || []) {
    if (row.event_type !== "statistical_high_flow" || row.verifier !== "high_flow_one_to_one_v1" || row.lead_day != null) continue;
    const values = { ...row, event_far: finite(row.predicted_events) && row.predicted_events > 0 && finite(row.matched_events)
      ? (row.predicted_events - row.matched_events) / row.predicted_events : null };
    if (!groups.has(row.basin_id)) groups.set(row.basin_id, []);
    groups.get(row.basin_id).push(values);
  }
  const average = values => Object.fromEntries(floodMetricDefinitions.map(({ key }) => {
    const valid = values.map(row => row[key]).filter(finite);
    return [key, { value: valid.length ? valid.reduce((a, b) => a + b, 0) / valid.length : null, count: valid.length }];
  }));
  const basins = new Map([...groups].map(([id, values]) => [id, average(values)]));
  const overall = average([...basins.values()].map(scores => Object.fromEntries(
    floodMetricDefinitions.map(({ key }) => [key, scores[key].value]))));
  return { basins, overall };
}
export function floodScoreText(value, definition) {
  return !finite(value) ? "—" : definition.percent ? `${format(value * 100)}%` : format(value);
}
export function floodScoreColor(value, definition, maximum = 1) {
  if (!finite(value)) return "#d8dde3";
  const fraction = Math.min(1, Math.max(0, definition.percent ? value : value / (maximum || 1)));
  const quality = definition.higher ? fraction : 1 - fraction;
  return `hsl(${Math.round(quality * 120)}, 55%, 48%)`;
}

export const hanzeMetricDefinitions = [
  { key: "catalogue_precision", label: "HANZE catalogue-linked precision", percent: true, higher: true },
  { key: "catalogue_recall", label: "HANZE catalogue-linked recall", percent: true, higher: true },
];
export function hanzePerformance(rows, lead) {
  const groups = new Map();
  for (const row of rows || []) {
    if (row.status !== "evaluated" || String(row.lead_day) !== String(lead)) continue;
    if (!groups.has(row.basin_id)) groups.set(row.basin_id, []);
    groups.get(row.basin_id).push(row);
  }
  const average = values => Object.fromEntries(hanzeMetricDefinitions.map(({ key }) => {
    const valid = values.map(row => row[key]).filter(finite);
    return [key, { value: valid.length ? valid.reduce((a, b) => a + b, 0) / valid.length : null, count: valid.length }];
  }));
  const basins = new Map([...groups].map(([id, values]) => [id, average(values)]));
  return { basins, overall: average([...basins.values()].map(scores => Object.fromEntries(
    hanzeMetricDefinitions.map(({ key }) => [key, scores[key].value])))) };
}
