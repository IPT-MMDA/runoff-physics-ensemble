import { finite, format } from "./core.js";

export function dischargeColor(value) {
  if (!finite(value)) return "#9ca3af";
  const yellow = [255, 255, 191], end = value < 0 ? [215, 48, 39] : [26, 152, 80];
  const fraction = Math.min(1, Math.abs(value) / 50);
  return `rgb(${yellow.map((v, i) => Math.round(v + (end[i] - v) * fraction)).join(",")})`;
}

let leafletPromise;
function leaflet() {
  if (window.L) return Promise.resolve(window.L);
  if (!leafletPromise) {
    leafletPromise = new Promise((resolve, reject) => {
      const css = document.createElement("link");
      css.rel = "stylesheet"; css.href = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css";
      const script = document.createElement("script");
      script.src = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js";
      script.onload = () => resolve(window.L);
      script.onerror = () => { script.remove(); css.remove(); leafletPromise = null; reject(new Error("Map library unavailable. Check your browser connection and refresh the view. The plant table remains available.")); };
      document.head.append(css, script);
    });
  }
  return leafletPromise;
}

const $ = id => document.getElementById(id);
function option(value, text) {
  const node = document.createElement("option"); node.value = value; node.textContent = text; return node;
}
function detail(feature) {
  const p = feature.properties, node = document.createElement("div");
  const title = document.createElement("strong"); title.textContent = p.name; node.append(title);
  for (const text of [`Basin: ${p.basin_id || "—"}`, `Predicted: ${format(p.predicted_m3_s)} m³/s`,
    `Previous day: ${format(p.previous_m3_s)} m³/s`, `Change: ${format(p.change_pct)}%`, p.status]) {
    const line = document.createElement("div"); line.textContent = text; node.append(line);
  }
  return node;
}

export class HydropowerView {
  constructor(api, renderTable) {
    this.api = api; this.renderTable = renderTable; this.version = 0;
    $("hydropower-refresh").addEventListener("click", () => this.activate(true));
    $("hydropower-dataset").addEventListener("change", () => { this.populateRuns(); this.clearForecast(); });
    $("hydropower-run").addEventListener("change", () => this.clearForecast());
    $("hydropower-date").addEventListener("change", () => this.clearForecast(false));
    $("hydropower-update").addEventListener("click", () => this.update());
  }
  reset() {
    ++this.version; this.loaded = false; this.loading = false; this.loadingVersion = null; this.stations = null; this.runs = [];
    $("hydropower-dataset").replaceChildren(); $("hydropower-run").replaceChildren();
    $("hydropower-table").replaceChildren(); $("hydropower-summary").textContent = "";
    $("hydropower-update").disabled = true; $("hydropower-refresh").disabled = false; this.layer?.clearLayers();
  }
  async activate(refresh = false) {
    this.map?.invalidateSize();
    if (this.loading || (this.loaded && !refresh)) return;
    const version = ++this.version; this.loading = true; this.loadingVersion = version;
    $("hydropower-refresh").disabled = true; $("hydropower-update").disabled = true;
    $("hydropower-status").textContent = "Loading hydropower plants and available runs…";
    try {
      const [stations, runs] = await Promise.all([this.api("/hydropower/sites"), this.api("/hydropower/runs")]);
      if (version !== this.version) return;
      this.stations = stations; this.runs = runs.items; this.loaded = true;
      const selected = $("hydropower-dataset").value;
      $("hydropower-dataset").replaceChildren(...[...new Set(this.runs.map(r => r.dataset_id))].map(id => option(id, id)));
      if ([...$("hydropower-dataset").options].some(o => o.value === selected)) $("hydropower-dataset").value = selected;
      this.populateRuns(); this.clearForecast(true, false);
      $("hydropower-status").textContent += runs.skipped ? ` ${runs.skipped} runs could not be used because their predictions or registered source dataset are unavailable.` : "";
    } catch (error) {
      if (version !== this.version) return;
      this.loaded = false; this.stations = null; this.layer?.clearLayers();
      $("hydropower-table").replaceChildren(); $("hydropower-summary").textContent = "";
      $("hydropower-status").textContent = error.message;
    } finally {
      if (version === this.loadingVersion) { this.loading = false; $("hydropower-refresh").disabled = false; }
    }
  }
  populateRuns() {
    const selected = $("hydropower-run").value;
    const runs = this.runs.filter(r => r.dataset_id === $("hydropower-dataset").value);
    $("hydropower-run").replaceChildren(...runs.map(r => option(r.run_id, r.run_id)));
    if (!runs.length) $("hydropower-run").append(option("", "No prediction runs available"));
    else if (runs.some(r => r.run_id === selected)) $("hydropower-run").value = selected;
  }
  clearForecast(clearDate = true, invalidate = true) {
    if (invalidate) ++this.version;
    if (clearDate) { $("hydropower-date").value = ""; $("hydropower-date").removeAttribute("min"); $("hydropower-date").removeAttribute("max"); }
    $("hydropower-update").disabled = !this.stations || !$("hydropower-run").value;
    $("hydropower-summary").textContent = "";
    $("hydropower-status").textContent = $("hydropower-run").value
      ? "Choose a date, or leave it blank for the latest prediction, then update the map."
      : "Plant locations are available. Complete an experiment to attach predictions, then refresh runs.";
    if (this.stations) this.render(this.stations);
  }
  async update() {
    const run = $("hydropower-run").value;
    if (!run || !this.stations) return;
    const version = ++this.version, date = $("hydropower-date").value;
    $("hydropower-update").disabled = true; $("hydropower-status").textContent = "Matching plants to predicted catchments…";
    try {
      const result = await this.api(`/runs/${encodeURIComponent(run)}/hydropower${date ? `?target_date=${encodeURIComponent(date)}` : ""}`);
      if (version !== this.version) return;
      $("hydropower-date").value = result.map.target_date;
      $("hydropower-date").min = result.available_dates[0]; $("hydropower-date").max = result.available_dates.at(-1);
      $("hydropower-status").textContent = `Showing ${result.map.target_date} · dataset ${result.dataset_id} · ${result.matched}/${result.total} matched plants · day-over-day change available for ${result.with_change}.`;
      $("hydropower-summary").textContent = `Run ${result.run_id}. Prediction dates: ${result.available_dates[0]} to ${result.available_dates.at(-1)}. Changes compare the previous calendar day and use the mean prediction across ensemble seeds.`;
      this.render(result.map);
    } catch (error) {
      if (version !== this.version) return;
      $("hydropower-status").textContent = error.message; $("hydropower-summary").textContent = "";
      this.render(this.stations);
    } finally {
      if (version === this.version) $("hydropower-update").disabled = false;
    }
  }
  render(payload) {
    const features = [...payload.features].sort((a, b) => Math.abs(b.properties.change_pct ?? 0) - Math.abs(a.properties.change_pct ?? 0));
    $("hydropower-table").replaceChildren();
    this.renderTable($("hydropower-table"), features.map(({ properties: p }) => ({
      plant: p.name, basin: p.basin_id || "—", "predicted (m³/s)": p.predicted_m3_s,
      "previous day (m³/s)": p.previous_m3_s, "change (%)": p.change_pct, status: p.status,
    })));
    this.drawMap(payload, this.version);
  }
  async drawMap(payload, version) {
    try {
      const L = await leaflet();
      if (version !== this.version) return;
      $("hydropower-map-status").textContent = "";
      if (!this.map) {
        this.map = L.map($("hydropower-map"), { preferCanvas: true }).setView([46.8, 8.2], 8);
        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
          maxZoom: 18, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        }).on("tileerror", () => { $("hydropower-map-status").textContent = "Basemap tiles unavailable. Plant markers and the table remain usable."; }).addTo(this.map);
        this.layer = L.featureGroup().addTo(this.map);
      }
      this.layer.clearLayers();
      for (const feature of payload.features) {
        const [lon, lat] = feature.geometry.coordinates;
        L.circleMarker([lat, lon], { radius: 5, color: "#475569", weight: 1,
          fillColor: dischargeColor(feature.properties.change_pct), fillOpacity: .9,
        }).bindPopup(detail(feature)).bindTooltip(() => {
          const label = document.createElement("span"); label.textContent = feature.properties.name; return label;
        }).addTo(this.layer);
      }
      this.map.invalidateSize();
      if (!this.fitted && this.layer.getLayers().length) { this.map.fitBounds(this.layer.getBounds(), { padding: [20, 20] }); this.fitted = true; }
    } catch (error) {
      if (version === this.version) $("hydropower-map-status").textContent = error.message;
    }
  }
}
