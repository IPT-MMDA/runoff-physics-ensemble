// Pure data transforms shared by the browser and dependency-free Node tests.
export const terminal = status => ["succeeded", "failed", "cancelled"].includes(status);
export const pretty = key => key.replaceAll("_", " ").replace(/\b\w/g, c => c.toUpperCase());
export const finite = value => typeof value === "number" && Number.isFinite(value);
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
