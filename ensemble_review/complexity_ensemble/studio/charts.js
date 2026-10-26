import { envelope, finite, format } from "./core.js";
const NS = "http://www.w3.org/2000/svg";
const palette = ["var(--chart-accent, #0099ad)", "#8b97a8", "#b69a66", "#8a80b3", "var(--chart-secondary, #55c9d7)", "#bd8c7a"];
function svgEl(tag, attrs = {}, text) {
  const node = document.createElementNS(NS, tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
  if (text != null) node.textContent = text;
  return node;
}
export function chart(container, series, { scatter = false, date = true, marker = null, xLabel = "", yLabel = "" } = {}) {
  const node = typeof container === "string" ? document.querySelector(container) : container;
  node.replaceChildren();
  series = series.map((s, i) => ({ ...s, color: s.color || palette[i % palette.length], points: s.points.filter(p => finite(p.x) && finite(p.y)) }));
  const points = series.flatMap(s => s.points);
  if (!points.length) { node.textContent = "No data available for this view."; node.classList.add("empty-state"); return; }
  node.classList.remove("empty-state");
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (const p of points) { xmin = Math.min(xmin, p.x); xmax = Math.max(xmax, p.x); ymin = Math.min(ymin, p.y); ymax = Math.max(ymax, p.y); }
  if (xmax === xmin) xmax = xmin + 1;
  const pad = (ymax - ymin || 1) * .08;
  ymin -= pad; ymax += pad;
  const W = 800, H = 245, L = 62, R = 22, T = 20, B = 38;
  const x = value => L + (value - xmin) / (xmax - xmin) * (W - L - R);
  const y = value => H - B - (value - ymin) / (ymax - ymin) * (H - T - B);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": series.map(s => s.name).join(", ") + (xLabel ? `; x: ${xLabel}; y: ${yLabel}` : "") });
  for (let i = 0; i <= 4; i++) {
    const value = ymin + (ymax - ymin) * i / 4;
    svg.append(svgEl("line", { x1: L, x2: W - R, y1: y(value), y2: y(value), stroke: "#dce3eb", "stroke-width": 1 }));
    svg.append(svgEl("text", { x: L - 10, y: y(value) + 4, fill: "#8795a8", "font-size": 10, "text-anchor": "end" }, format(value)));
  }
  for (let i = 0; i <= 4; i++) {
    const value = xmin + (xmax - xmin) * i / 4;
    svg.append(svgEl("text", { x: x(value), y: H - 13, fill: "#8795a8", "font-size": 10, "text-anchor": "middle" }, date ? new Date(value).toISOString().slice(0, 10) : format(value)));
  }
  if (scatter) {
    const lo = Math.max(xmin, ymin), hi = Math.min(xmax, ymax);
    if (lo < hi) svg.append(svgEl("line", { x1: x(lo), x2: x(hi), y1: y(lo), y2: y(hi), stroke: "#afbed2", "stroke-dasharray": "5 5" }));
  }
  for (const s of series) {
    const sampled = envelope(s.points, scatter ? 1000 : 1200);
    if (scatter) {
      for (const p of sampled) svg.append(svgEl("circle", { cx: x(p.x), cy: y(p.y), r: 2.4, fill: s.color, opacity: .5 }));
    } else {
      svg.append(svgEl("path", { d: sampled.map((p, i) => `${i ? "L" : "M"}${x(p.x).toFixed(2)},${y(p.y).toFixed(2)}`).join(" "), fill: "none", stroke: s.color, "stroke-width": 1.8, "stroke-linejoin": "round" }));
    }
  }
  if (finite(marker)) svg.append(svgEl("line", { x1: x(marker), x2: x(marker), y1: T, y2: H - B, stroke: "var(--chart-accent, #0099ad)", "stroke-dasharray": "4 4" }));
  const tooltip = document.createElement("div"); tooltip.className = "chart-tooltip"; tooltip.hidden = true;
  svg.addEventListener("pointermove", event => {
    const box = svg.getBoundingClientRect(), cursor = xmin + ((event.clientX - box.left) / box.width * W - L) / (W - L - R) * (xmax - xmin);
    const lines = [];
    for (const s of series) {
      let lo = 0, hi = s.points.length - 1;
      if (scatter) continue;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (s.points[mid].x < cursor) lo = mid + 1; else hi = mid; }
      const p = s.points[lo];
      if (p) lines.push(`${s.name}: ${format(p.y)}${p.label ? " · " + p.label : ""}`);
    }
    tooltip.textContent = lines.join("\n"); tooltip.hidden = !lines.length;
    tooltip.style.left = Math.min(Math.max(0, event.clientX - box.left + 10), Math.max(0, box.width - 260)) + "px";
    tooltip.style.top = "20px";
  });
  svg.addEventListener("pointerleave", () => { tooltip.hidden = true; });
  const legend = document.createElement("div"); legend.className = "chart-legend";
  for (const s of series) { const label = document.createElement("span"), swatch = document.createElement("i"); swatch.style.background = s.color; label.append(swatch, document.createTextNode(s.name)); legend.append(label); }
  node.append(svg, tooltip, legend);
}

export class BasinAtlas {
  constructor(canvas, onSelect) {
    this.canvas = canvas; this.onSelect = onSelect; this.rows = []; this.paths = []; this.selected = new Set();
    canvas.addEventListener("pointermove", e => {
      if (!this.performance) return;
      const bounds = canvas.getBoundingClientRect(), ctx = canvas.getContext("2d");
      ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0);
      const hit = [...this.paths].reverse().find(({ path }) => ctx.isPointInPath(path, e.clientX - bounds.left, e.clientY - bounds.top, "evenodd"));
      ctx.restore();
      canvas.title = hit ? `${hit.row.basin_id} · ${hit.row.name || ""}: ${this.performance.get(hit.row.basin_id)?.label || "No score"}` : "";
    });
    this.resize = new ResizeObserver(() => this.draw()); this.resize.observe(canvas);
    canvas.addEventListener("click", e => {
      const bounds = canvas.getBoundingClientRect(), ctx = canvas.getContext("2d");
      // Paths use CSS pixels; identity transform avoids device-pixel ambiguity in hit tests.
      ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0);
      const hit = [...this.paths].reverse().find(({ path }) => ctx.isPointInPath(path, e.clientX - bounds.left, e.clientY - bounds.top, "evenodd"));
      ctx.restore();
      if (hit?.row.eligible) this.onSelect(hit.row.basin_id);
    });
  }
  setPerformance(styles) { this.performance = styles; this.draw(); }
  setData(rows) { this.rows = rows; this.layoutKey = null; this.draw(); }
  setSelected(selected) { this.selected = selected; this.draw(); }
  draw() {
    const canvas = this.canvas, { width, height } = canvas.getBoundingClientRect();
    if (!width || !height) return;
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) { canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr); }
    const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
    const layoutKey = `${width}/${height}`;
    if (this.layoutKey !== layoutKey) {
      const rows = this.rows.filter(row => ["Polygon", "MultiPolygon"].includes(row.geometry?.type));
      const polygons = row => row.geometry.type === "Polygon" ? [row.geometry.coordinates] : row.geometry.coordinates;
      const coords = rows.flatMap(row => polygons(row).flat(2));
      this.paths = [];
      this.layoutKey = layoutKey;
      if (!coords.length) return;
      let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
      for (const [lon, lat] of coords) { const x = lon * Math.cos(47 * Math.PI / 180); xmin = Math.min(xmin, x); xmax = Math.max(xmax, x); ymin = Math.min(ymin, lat); ymax = Math.max(ymax, lat); }
      const scale = Math.min((width - 55) / (xmax - xmin || 1), (height - 70) / (ymax - ymin || 1));
      const project = ([lon, lat]) => [(width - (xmax - xmin) * scale) / 2 + (lon * Math.cos(47 * Math.PI / 180) - xmin) * scale, (height - (ymax - ymin) * scale) / 2 + (ymax - lat) * scale];
      for (const row of rows) {
        const path = new Path2D();
        for (const polygon of polygons(row)) for (const ring of polygon) { ring.forEach((point, i) => { const [x, y] = project(point); i ? path.lineTo(x, y) : path.moveTo(x, y); }); path.closePath(); }
        this.paths.push({ path, row });
      }
    }
    const theme = getComputedStyle(canvas);
    const selectedFill = theme.getPropertyValue("--map-fill").trim() || "#00BFFF";
    const selectedStroke = theme.getPropertyValue("--map-stroke").trim() || "#0099ad";
    for (const { path, row } of this.paths) {
      const selected = this.selected.has(row.basin_id);
      ctx.fillStyle = this.performance ? this.performance.get(row.basin_id)?.color || "#d8dde3" : selected ? selectedFill : row.eligible ? "#d8dde3" : "#e1e7ed";
      ctx.strokeStyle = selected ? selectedStroke : "#a8b4c2"; ctx.lineWidth = selected ? (this.performance ? 3 : 1.2) : .65;
      ctx.fill(path, "evenodd"); ctx.stroke(path);
    }
  }
}
