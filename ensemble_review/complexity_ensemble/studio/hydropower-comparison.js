import { finite, format, runName, runOptionLabel } from "./core.js";

const $ = name => document.getElementById(`hydro-compare-${name}`);
const fields = ["head_m", "efficiency", "environmental_flow_m3_s", "max_turbine_flow_m3_s", "shortfall_eur_mwh", "surplus_eur_mwh"];
function option(value, text) {
  const node = document.createElement("option"); node.value = value; node.textContent = text; return node;
}

export function errorColor(value, scale) {
  if (!finite(value)) return "#9ca3af";
  const end = value < 0 ? [33, 102, 172] : [178, 24, 43];
  const fraction = Math.min(1, Math.abs(value) / (scale > 0 ? scale : 1));
  return `rgb(${end.map(v => Math.round(247 + (v - 247) * fraction)).join(",")})`;
}

export function comparisonRequest() {
  const scenario = {};
  for (const name of fields) {
    const input = $(name), text = input.value.trim();
    if (!text && name === "max_turbine_flow_m3_s") { scenario[name] = null; continue; }
    const value = Number(text);
    if (!text || !finite(value) || !input.checkValidity()) throw new Error("Review the economic assumptions: enter valid numbers within the indicated ranges.");
    scenario[name] = value;
  }
  const lead = Number($("lead").value);
  if (!Number.isInteger(lead) || lead < 1 || lead > 366) throw new Error("Lead day must be an integer from 1 to 366.");
  if ($("start").value && $("end").value && $("start").value > $("end").value) throw new Error("Start must be on or before end.");
  return { run_a: $("a").value, run_b: $("b").value,
    start: $("start").value || null, end: $("end").value || null,
    target_date: $("date").value || null, lead_day: lead, scenario };
}

export class HydropowerComparisonView {
  constructor(api, renderTable, leaflet) {
    this.api = api; this.renderTable = renderTable; this.leaflet = leaflet;
    this.version = 0; this.maps = {}; this.layers = {};
    for (const name of ["a", "b", "start", "end", "lead", ...fields]) {
      $(name).addEventListener("change", () => this.clear(!fields.includes(name)));
      if (fields.includes(name) || ["start", "end", "lead"].includes(name)) {
        $(name).addEventListener("input", () => this.clear(!fields.includes(name)));
      }
    }
    $("date").addEventListener("change", () => this.clear(false));
    $("update").addEventListener("click", () => this.update());
    $("download").addEventListener("click", () => this.download());
  }
  setRuns(runs) {
    this.runNames = new Map(runs.map(r => [r.run_id, runName(r)]));
    for (const name of ["a", "b"]) {
      const selected = $(name).value;
      $(name).replaceChildren(...runs.map(r => option(r.run_id, runOptionLabel(r))));
      if (runs.some(r => r.run_id === selected)) $(name).value = selected;
      else if (name === "b" && runs.length > 1) $(name).value = runs[1].run_id;
      if (!runs.length) $(name).append(option("", "No available runs"));
    }
    this.clear(true);
  }
  clear(clearDates = false) {
    ++this.version; this.result = null;
    if (clearDates) $("date").replaceChildren(option("", "Latest paired date"));
    $("update").disabled = !$("a").value || !$("b").value;
    $("download").disabled = true;
    for (const name of ["summary", "accuracy", "metrics", "basins", "plants", "audit", "legend", "map-status"]) $(name).replaceChildren();
    $("title-a").textContent = "Run A · error against observations";
    $("title-b").textContent = "Run B · error against observations";
    for (const layer of Object.values(this.layers)) layer.clearLayers();
    $("status").textContent = $("a").value
      ? "Choose runs and assumptions, then compare. Blank period bounds use all paired dates."
      : "Complete an experiment with saved observations to compare runs.";
  }
  resize() { for (const map of Object.values(this.maps)) map.invalidateSize(); }
  async update() {
    const version = ++this.version;
    this.result = null; $("download").disabled = true; $("update").disabled = true;
    $("status").textContent = "Pairing observations and calculating daily scenario costs…";
    try {
      const body = comparisonRequest();
      const result = await this.api("/hydropower/compare", { method: "POST", body });
      if (version !== this.version) return;
      this.result = result;
      $("date").replaceChildren(...result.available_dates.map(day => option(day, day)));
      $("date").value = result.target_date;
      $("status").textContent = `${result.period.start} to ${result.period.end} · ${result.samples} paired basin-days · ${result.basins} abstract plants · lead ${result.lead_day} · map ${result.target_date}.`;
      this.render(result);
      $("download").disabled = false;
      this.drawMaps(result, version);
    } catch (error) {
      if (version !== this.version) return;
      this.clear(false);
      $("status").textContent = /^(not found|request failed \(404\))$/i.test(error.message.trim())
        ? "The running server does not expose the Hydropower comparison endpoint. Restart the Studio/API server to load the updated Python code, then refresh this page and compare again."
        : error.message;
    } finally {
      if (version === this.version) $("update").disabled = !$("a").value || !$("b").value;
    }
  }
  render(result) {
    const e = result.economics;
    const change = result.accuracy_change;
    $("accuracy").textContent = `Using A relative to B: MAE reduction ${format(change.mae_reduction_pct)}% · `
      + `RMSE reduction ${format(change.rmse_reduction_pct)}% · mean basin NSE change ${format(change.mean_basin_nse_gain)} points. `
      + "Negative reductions mean larger errors. Reductions are unavailable when B's error is zero.";
    $("summary").textContent = `Simulated period cost: A €${format(e.cost_a_eur)} · B €${format(e.cost_b_eur)}. `
      + `Savings using A: €${format(e.saving_eur)} (${format(e.saving_pct)}%). `
      + `Cost index: A ${format(e.cost_index_a)} / B ${format(e.cost_index_b)}. `
      + "Negative savings mean A costs more. Percentage and index are unavailable when B costs zero.";
    this.renderTable($("metrics"), ["a", "b"].map(name => {
      const m = result[`metrics_${name}`];
      return { run: name.toUpperCase(), "MAE (m³/s)": m.mae_m3_s, "RMSE (m³/s)": m.rmse_m3_s,
        "bias (m³/s)": m.bias_m3_s, "mean basin NSE": m.mean_basin_nse, "basins with defined NSE": m.nse_basins };
    }));
    $("audit").textContent = JSON.stringify(result.audit, null, 2);
    this.renderTable($("basins"), result.basin_results.map(r => ({ basin: r.basin_id, name: r.name, days: r.days,
      "NSE A": r.metrics_a.mean_basin_nse, "NSE B": r.metrics_b.mean_basin_nse,
      "cost A (€)": r.cost_a_eur, "cost B (€)": r.cost_b_eur, "saving (€)": r.saving_eur })));
    this.renderTable($("plants"), result.map.features.map(({ properties: p }) => ({ plant: p.name,
      basin: p.basin_id || "—", "observed (m³/s)": p.observed, "A (m³/s)": p.a, "B (m³/s)": p.b,
      "error A (m³/s)": p.error_a_m3_s, "error B (m³/s)": p.error_b_m3_s, status: p.status })));
    const scale = format(result.map.error_scale_m3_s);
    $("legend").textContent = `Shared error scale (prediction − observation): blue −${scale} m³/s · white 0 · red +${scale} m³/s. Gray: unavailable. Scale uses both maps on the selected date.`;
    $("title-a").textContent = `Run A · ${result.run_a_name || this.runNames?.get(result.run_a) || result.run_a}`;
    $("title-b").textContent = `Run B · ${result.run_b_name || this.runNames?.get(result.run_b) || result.run_b}`;
  }
  async drawMaps(result, version) {
    try {
      const L = await this.leaflet();
      if (version !== this.version) return;
      for (const name of ["a", "b"]) {
        if (!this.maps[name]) {
          const map = L.map($(`map-${name}`), { preferCanvas: true }).setView([46.8, 8.2], 8);
          L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
            maxZoom: 18, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
          }).on("tileerror", () => { $("map-status").textContent = "Basemap tiles unavailable. Markers and tables remain usable."; }).addTo(map);
          this.maps[name] = map; this.layers[name] = L.featureGroup().addTo(map);
          map.on("moveend", () => {
            const other = this.maps[name === "a" ? "b" : "a"];
            if (other && (!other.getCenter().equals(map.getCenter()) || other.getZoom() !== map.getZoom())) {
              other.setView(map.getCenter(), map.getZoom(), { animate: false });
            }
          });
        }
        const layer = this.layers[name]; layer.clearLayers();
        for (const feature of result.map.features) {
          const p = feature.properties, [lon, lat] = feature.geometry.coordinates;
          const popup = document.createElement("div");
          for (const text of [p.name, `Basin: ${p.basin_id || "—"}`, `Observed: ${format(p.observed)} m³/s`,
            `Run ${name.toUpperCase()}: ${format(p[name])} m³/s`, `Error: ${format(p[`error_${name}_m3_s`])} m³/s`,
            `Observed-flow energy: ${format(p.energy_observed_mwh)} MWh/day`,
            `Predicted energy: ${format(p[`energy_${name}_mwh`])} MWh/day`,
            `Scenario cost: €${format(p[`cost_${name}_eur`])}/day`, p.status]) {
            const line = document.createElement("div"); line.textContent = text; popup.append(line);
          }
          L.circleMarker([lat, lon], { radius: 5, color: "#475569", weight: 1, fillOpacity: .9,
            fillColor: errorColor(p[`error_${name}_m3_s`], result.map.error_scale_m3_s),
          }).bindPopup(popup).addTo(layer);
        }
        this.maps[name].invalidateSize();
      }
    } catch (error) {
      if (version === this.version) $("map-status").textContent = error.message;
    }
  }
  download() {
    if (!this.result) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(this.result, null, 2)], { type: "application/json" }));
    const link = document.createElement("a"); link.href = url; link.download = "hydropower_comparison.json";
    link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}
