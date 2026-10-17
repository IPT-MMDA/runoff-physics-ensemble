import { allPages, coerce, finite, format, meanByDate, pretty, terminal } from "./core.js";
import { BasinAtlas, chart } from "./charts.js?v=fixed-sky-blue-1";

const $ = selector => document.querySelector(selector);
const state = { token: sessionStorage.getItem("runoff.token") || "", datasets: [], basins: [], selected: new Set(), datasetCache: new Map(), predictions: new Map(), runs: [], totalRuns: 0, connected: false, bindings: [], run: null, job: null, pollTimer: null, replay: null, playTimer: null, datasetVersion: 0, resultVersion: 0, pageVersion: 0 };
const clone = value => structuredClone(value);
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2).toLowerCase(), value);
    else if (key in node) node[key] = value;
    else node.setAttribute(key, value);
  }
  node.append(...children.filter(c => c != null)); return node;
}
function notice(message, error = false, inline = false) {
  const node = $("#notice"); node.textContent = message; node.classList.toggle("error", error); node.classList.toggle("status-line", inline && !error); node.hidden = !message;
}
function action(fn) { return async event => { try { await fn(event); } catch (error) { notice(error.message, true); } }; }
function bind(selector, event, fn) { $(selector).addEventListener(event, action(fn)); }
async function api(path, { method = "GET", body, download = false, idempotency = false } = {}) {
  const headers = { Accept: download ? "*/*" : "application/json" };
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (idempotency) headers["Idempotency-Key"] = crypto.randomUUID();
  const response = await fetch(`/api/v1${path}`, { method, headers, body: body === undefined ? undefined : JSON.stringify(body), signal: AbortSignal.timeout(120000) });
  if (!response.ok) {
    const problem = await response.json().catch(() => ({}));
    if (response.status === 401) { setConnection(false, "Authentication required"); if (!$("#connection-dialog").open) $("#connection-dialog").showModal(); }
    throw new Error([problem.message || `Request failed (${response.status})`, ...(problem.details || []).map(d => `${(d.location || []).join(".")}: ${d.message}`)].join("\n"));
  }
  return download ? response.blob() : response.json();
}
function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob), anchor = el("a", { href: url, download: filename });
  document.body.append(anchor); anchor.click(); anchor.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function setConnection(connected, label) {
  state.connected = connected; $("#connection-label").textContent = label;
  $("#connection").classList.toggle("ready", connected); $("#connection").classList.toggle("error", !connected);
  updateLaunch();
}
function updateLaunch() { $("#launch").disabled = !state.connected || !state.selected.size || !state.config || state.submitting || Boolean(state.job && !terminal(state.job.status)); }
function option(value, label = value) { return el("option", { value, text: label }); }
function setOptions(node, values, previous = node.value) { node.replaceChildren(...values.map(v => option(v.value ?? v, v.label ?? v))); if ([...node.options].some(o => o.value === previous)) node.value = previous; }
function resolve(schema) { return schema?.$ref ? state.bootstrap.schema.$defs[schema.$ref.split("/").at(-1)] : schema; }
function schemaFor(section) { return resolve(resolve(state.bootstrap.schema.properties.config).properties[section]); }
const choices = {
  complex_expert: ["pinnmamba", "mlp"], simple_expert: ["rbf", "fourier"],
  approach: ["soft_routing", "hard_routing", "distillation", "no_routing", "static_50_50", "ood_fallback", "stacking"],
  device: ["cpu", "cuda", "mps", "auto"], mode: [], optimization_mode: ["none", "accuracy", "robustness", "balanced", "distillation"],
  distillation_teacher: ["soft_routing", "hard_routing", "distillation", "no_routing", "static_50_50", "ood_fallback", "stacking"], definition: ["automatic_q95", "absolute", "quantile"], name: ["linear_reservoir"],
};
const hints = {
  seeds: "Comma-separated random seeds, e.g. 0, 1, 2.", mlp_widths: "Hidden layer widths, separated by commas.",
  routing_context_length: "Historical days used to estimate catchment complexity.", sequence_length: "Days of input history for each prediction.",
  complexity_method: "Morse routing or a history-based complexity estimator.", physics_weight: "Weight of the physics constraint in training.",
  training_noise: "Observation noise applied during training.", horizon_days: "Forecast length for rolling multi-day hindcasts.",
};
function field(parent, key, value, schema, path, bindings, customOptions) {
  schema = resolve(schema) || {}; const base = schema.anyOf?.find(s => s.type !== "null") || schema;
  const type = base.type || (Array.isArray(value) ? "array" : typeof value);
  const label = el("label", { class: "field" }, el("span", { text: pretty(key) }));
  let input; const opts = customOptions || base.enum || choices[key];
  if (opts?.length) { input = el("select"); input.append(...opts.map(v => option(v, pretty(v)))); input.value = value ?? ""; }
  else if (type === "boolean") { input = el("input", { type: "checkbox", checked: Boolean(value) }); label.classList.add("check"); }
  else if (type === "array") { input = el("input", { type: "text", value: (value || []).join(", "), spellcheck: false }); label.classList.add("field-full"); }
  else if (type === "object") { input = el("textarea", { value: JSON.stringify(value ?? {}), rows: 2, spellcheck: false }); label.classList.add("field-full"); }
  else { input = el("input", { type: ["integer", "number"].includes(type) ? "number" : key.includes("date") ? "date" : "text", value: value ?? "" }); if (input.type === "number") input.step = type === "integer" ? "1" : "any"; if (schema.anyOf?.some(s => s.type === "null")) input.placeholder = "Automatic"; }
  input.id = "field-" + path.join("-");
  input.addEventListener("change", () => { if (bindings === state.bindings || bindings === state.flagBindings) persistDraft(); });
  label.append(input); if (hints[key]) label.append(el("span", { class: "field-hint", text: hints[key] }));
  parent.append(label); bindings.push({ path, input, schema }); return input;
}
function section(title, number, open = false) {
  const details = el("details", { open });
  details.append(el("summary", {}, el("span", { class: "section-number", text: number }), title));
  const fields = el("div", { class: "fields-grid" }); details.append(fields); $("#configuration").append(details); return fields;
}
function renderConfig() {
  state.bindings = []; state.flagBindings = []; $("#configuration").replaceChildren();
  const essential = section("Model & learning", "01", true);
  for (const [group, keys] of [["strategy", ["approach", "epochs"]], ["model", ["complex_expert", "simple_expert"]], ["hyperparameters", ["sequence_length", "batch_size", "learning_rate", "device"]]]) {
    for (const key of keys) field(essential, key, state.config[group][key], schemaFor(group).properties[key], [group, key], state.bindings, key === "device" ? state.capabilities.device_choices : undefined);
  }
  const dates = section("Date splits", "02");
  for (const [group, range] of Object.entries(state.config.dates)) for (const [key, value] of Object.entries(range)) {
    const input = field(dates, `${group} ${key}`, value, { type: "string" }, ["dates", group, key], state.bindings); input.type = "date";
  }
  const remaining = {
    strategy: ["approach", "epochs"], model: ["complex_expert", "simple_expert"],
    hyperparameters: ["sequence_length", "batch_size", "learning_rate", "device"],
  };
  let n = 3;
  for (const [group, title] of [["hyperparameters", "Routing & hyperparameters"], ["model", "Model architecture"], ["strategy", "Training phases"], ["physics", "Physics & KAN"], ["extremes", "Extreme events"], ["forecast", "Forecasting"]]) {
    const grid = section(title, String(n++).padStart(2, "0"));
    for (const [key, schema] of Object.entries(schemaFor(group).properties)) {
      if (remaining[group]?.includes(key)) continue;
      const options = key === "mode" ? group === "forecast" ? ["one_day", "rolling"] : ["none", "statistical", "event_based"] : undefined;
      field(grid, key, state.config[group][key] ?? schema.default, schema, [group, key], state.bindings, options);
    }
  }
  const extras = section("Additional data & features", "09");
  const dataset = state.datasets.find(d => d.dataset_id === $("#dataset").value);
  for (const kind of ["estreams", "camels_chem", "hyd_responses"]) {
    const input = field(extras, `use_${kind}`, state.flags?.[`use_${kind}`] || false, { type: "boolean" }, [`use_${kind}`], state.flagBindings);
    input.disabled = !dataset?.[`${kind}_configured`]; if (input.disabled) input.checked = false;
  }
  const rootProperties = resolve(state.bootstrap.schema.properties.config).properties;
  for (const key of ["estreams_features", "camels_chem_features", "camels_state_features", "hyd_responses_features", "hyd_responses_static"]) {
    field(extras, key, state.config[key] ?? rootProperties[key].default ?? [], rootProperties[key], [key], state.bindings);
  }
  $("#scope").value = state.config.basins.training_scope;
}
function readBindings(base, bindings) {
  for (const { path, input, schema } of bindings) {
    let target = base; for (const key of path.slice(0, -1)) target = target[key] ??= {};
    try { target[path.at(-1)] = coerce(input.type === "checkbox" ? input.checked : input.value, schema); }
    catch (error) { throw new Error(`${pretty(path.join(" / "))}: ${error.message}`); }
  }
  return base;
}
function requestBody() {
  if (!state.config) throw new Error("Connect to a server first.");
  const config = readBindings(clone(state.config), state.bindings);
  config.basins.target_basins = [...state.selected]; config.basins.training_scope = $("#scope").value;
  return { dataset_id: $("#dataset").value, config, ...readBindings({}, state.flagBindings), save_to_leaderboard: $("#save-leaderboard").checked };
}
function persistDraft() { try { localStorage.setItem("runoff.draft.v1", JSON.stringify(requestBody())); } catch { /* An incomplete field stays editable. */ } }
const atlas = new BasinAtlas($("#basin-map"), id => { state.selected.has(id) ? state.selected.delete(id) : state.selected.add(id); selectionChanged(); renderBasinList(); });
function selectionChanged() { $("#basin-count").textContent = `${state.selected.size} selected`; atlas.setSelected(state.selected); updateLaunch(); persistDraft(); }
function renderBasinList() {
  const query = $("#basin-search").value.toLowerCase();
  const rows = state.basins.filter(r => r.eligible && `${r.basin_id} ${r.name || ""}`.toLowerCase().includes(query));
  $("#basin-list").replaceChildren(...rows.map(row => {
    const input = el("input", { type: "checkbox", checked: state.selected.has(row.basin_id), value: row.basin_id, onChange: () => { input.checked ? state.selected.add(row.basin_id) : state.selected.delete(row.basin_id); selectionChanged(); } });
    return el("label", { class: "basin-choice", title: `${row.basin_id} · ${row.name || ""}` }, input, el("span", { text: `${row.basin_id}${row.name ? " · " + row.name : ""}` }));
  }));
  if (!rows.length) $("#basin-list").append(el("p", { class: "subtle", text: state.basins.length ? "No matching eligible catchments." : "No catchments loaded." }));
}
async function loadDataset({ retain = false } = {}) {
  const id = $("#dataset").value, version = ++state.datasetVersion;
  if (!id) { state.basins = []; state.selected.clear(); atlas.setData([]); renderBasinList(); selectionChanged(); return; }
  $("#map-empty").hidden = false; $("#map-empty").textContent = "Loading catchment atlas…";
  $("#launch").disabled = true;
  if (!retain) state.selected.clear();
  try {
    const rows = state.datasetCache.get(id) || await allPages((offset, limit) => api(`/datasets/${encodeURIComponent(id)}/basins?offset=${offset}&limit=${limit}`));
    if (version !== state.datasetVersion) return;
    state.datasetCache.set(id, rows); state.basins = rows;
    const eligible = new Set(rows.filter(r => r.eligible).map(r => r.basin_id));
    state.selected = new Set([...state.selected].filter(id => eligible.has(id)));
    atlas.setData(rows); renderBasinList(); selectionChanged();
    $("#map-empty").hidden = rows.some(r => r.geometry);
    $("#map-empty").textContent = "Map geometry is unavailable.\nSelect catchments from the list below.";
  } catch (error) {
    if (version !== state.datasetVersion) return;
    state.basins = []; state.selected.clear(); atlas.setData([]); renderBasinList(); selectionChanged();
    $("#map-empty").textContent = "The dataset could not be loaded.\nCheck its registration and data files."; throw error;
  } finally {
    if (version === state.datasetVersion && state.config) { state.config = readBindings(clone(state.config), state.bindings); state.flags = readBindings({}, state.flagBindings); state.flagBindings = []; renderConfig(); updateLaunch(); }
  }
}
async function connect() {
  notice(""); setConnection(false, "Connecting");
  const [bootstrap, datasets, capabilities, health] = await Promise.all([api("/ui/bootstrap"), api("/datasets"), api("/capabilities"), api("/health")]);
  state.bootstrap = bootstrap; state.capabilities = capabilities; state.datasets = datasets.items;
  state.config = clone(bootstrap.config); state.flags = {}; state.flagBindings = [];
  let draft;
  try { draft = JSON.parse(localStorage.getItem("runoff.draft.v1")); } catch { /* Ignore an obsolete draft. */ }
  if (draft?.config?.dates && draft?.config?.basins) {
    // Merge each section to pick up new server defaults after an upgrade.
    for (const [key, value] of Object.entries(draft.config)) state.config[key] = value && typeof value === "object" && !Array.isArray(value) ? { ...state.config[key], ...value } : value;
    state.flags = Object.fromEntries(["use_estreams", "use_camels_chem", "use_hyd_responses"].map(k => [k, draft[k] || false]));
  }
  setOptions($("#dataset"), state.datasets.map(d => ({ value: d.dataset_id, label: d.dataset_id + (d.available ? "" : " · unavailable") })), draft?.dataset_id);
  if (!state.datasets.length) $("#dataset").append(option("", "No datasets registered"));
  state.selected = new Set(draft?.config?.basins?.target_basins || []);
  $("#save-leaderboard").disabled = !bootstrap.leaderboard_enabled;
  $("#save-leaderboard").checked = bootstrap.leaderboard_enabled && Boolean(draft?.save_to_leaderboard);
  $("#promote-run").disabled = !bootstrap.leaderboard_enabled;
  renderConfig(); setConnection(true, health.worker_ready ? "Worker ready" : "Worker offline");
  if (!health.worker_ready) notice("The workspace is connected. Start the training worker to process submitted jobs.");
  await loadDataset({ retain: true }).catch(error => notice(error.message, true));
  await loadRuns();
  try { const saved = JSON.parse(sessionStorage.getItem("runoff.job")); if (saved?.job_id && !state.job) { state.job = saved; schedulePoll(0); } } catch { /* No resumable job. */ }
}
async function healthCheck() {
  if (!state.config || document.hidden) return;
  try { const health = await api("/health"); setConnection(true, health.worker_ready ? "Worker ready" : "Worker offline"); }
  catch { setConnection(false, "Connection lost"); }
}
function ensureNoJob() { if (state.submitting || (state.job && !terminal(state.job.status))) throw new Error("A job is already being tracked. Wait for it to finish or cancel it before submitting another."); }
async function submitJob(path, body) {
  ensureNoJob(); state.submitting = true; updateLaunch();
  try { trackJob(await api(path, { method: "POST", body, idempotency: true })); }
  finally { state.submitting = false; updateLaunch(); }
}
function trackJob(job) {
  state.job = job; sessionStorage.setItem("runoff.job", JSON.stringify(job)); renderJob(); schedulePoll(0); updateLaunch();
}
function renderJob() {
  if (!state.job) return;
  const job = state.job; $("#job-panel").hidden = false;
  $("#job-kind").textContent = `${pretty(job.kind).toUpperCase()} / ${job.status.toUpperCase()}`;
  $("#job-title").textContent = ({ queued: "In the queue", running: "Following the learning", succeeded: "Ready to explore", cancelled: "Job cancelled", failed: "Job needs attention" })[job.status] || "Connecting to job";
  $("#job-message").textContent = job.error?.message || job.message || "Waiting for a worker update.";
  $("#job-id").textContent = job.job_id;
  $("#progress").value = job.progress || 0; $("#job-percent").textContent = `${Math.round((job.progress || 0) * 100)}%`;
  $("#cancel-job").hidden = terminal(job.status); $("#cancel-job").disabled = job.cancel_requested;
  $("#cancel-job").textContent = job.cancel_requested ? "Cancellation requested" : "Cancel job";
}
function schedulePoll(ms = document.hidden ? 5000 : 1000) { clearTimeout(state.pollTimer); state.pollTimer = setTimeout(pollJob, ms); }
async function pollJob() {
  if (!state.job) return;
  try {
    const job = await api(`/jobs/${encodeURIComponent(state.job.job_id)}`); state.job = job; renderJob();
    sessionStorage.setItem("runoff.job", JSON.stringify(job));
    if (!terminal(job.status)) { schedulePoll(); return; }
    sessionStorage.removeItem("runoff.job"); updateLaunch();
    if (job.status === "failed") { notice(job.error?.message || "Job failed. Inspect the server logs for details.", true); return; }
    if (job.kind === "experiment" && job.result?.run_id) { await loadRuns(); location.hash = "runs"; await openRun(job.result.run_id); }
    if (job.kind === "scenario" && job.status === "succeeded") {
      const result = await api(`/jobs/${job.job_id}/result`); location.hash = "impact";
      renderData($("#scenario-output"), result.summary);
      chart("#scenario-chart", [ { name: "Baseline", points: (result.daily || []).map(r => ({ x: Date.parse(r.date), y: r.baseline_mm_day })) }, { name: "Proposed site", points: (result.daily || []).map(r => ({ x: Date.parse(r.date), y: r.scenario_mm_day })) } ]);
      if (result.events?.length) renderTable($("#scenario-output"), result.events);
    }
    if (job.kind === "gif" && job.status === "succeeded") { location.hash = "replay"; downloadBlob(await api(`/jobs/${job.job_id}/result`, { download: true }), "rainfall-replay.gif"); notice("Rainfall replay GIF is ready."); }
  } catch (error) { notice(`Could not update the job: ${error.message}. Progress will reconnect automatically.`, true); schedulePoll(5000); }
}
function metricCards(node, values) {
  node.replaceChildren(...Object.entries(values).map(([key, value]) => el("div", { class: "metric" }, el("span", { text: pretty(key) }), el("strong", { text: format(value) }))));
}
function renderTable(node, rows) {
  if (!rows?.length) { node.append(el("p", { class: "subtle", text: "No records available." })); return; }
  const keys = [...new Set(rows.flatMap(r => Object.keys(r)))];
  const table = el("table", { class: "data-table" });
  table.append(el("thead", {}, el("tr", {}, ...keys.map(k => el("th", { text: pretty(k) })))));
  table.append(el("tbody", {}, ...rows.map(row => el("tr", {}, ...keys.map(k => el("td", { text: row[k] && typeof row[k] === "object" ? JSON.stringify(row[k]) : format(row[k]) }))))));
  node.append(table);
}
function renderData(node, data) { node.replaceChildren(el("pre", { text: JSON.stringify(data, null, 2) })); }
function renderRuns() {
  const query = $("#run-search").value.toLowerCase();
  const rows = state.runs.filter(r => `${r.run_id} ${r.config.strategy?.approach} ${r.config.dataset_id || ""}`.toLowerCase().includes(query));
  $("#run-count").textContent = `${state.runs.length} of ${state.totalRuns} runs`;
  $("#runs-list").replaceChildren(...rows.map(run => {
    const metrics = run.aggregate_metrics, approach = pretty(run.config.strategy?.approach || "Experiment");
    return el("button", { class: "run-card" + (state.run?.run_id === run.run_id ? " selected" : ""), onClick: action(() => openRun(run.run_id)) },
      el("div", { class: "run-card-head" }, el("span", { class: "eyebrow", text: new Date(run.created_at).toLocaleDateString() }), el("span", { class: "tag", text: run.cancelled ? "Partial" : "Completed" })),
      el("h3", { text: approach }), el("code", { text: run.run_id }),
      el("p", { text: `${run.config.basins?.target_basins?.length || 0} catchments · ${format(run.runtime_seconds)} s · ${run.config.hyperparameters?.device || "cpu"}` }),
      el("div", { class: "card-metrics" }, ...Object.entries(metrics).slice(0, 2).map(([k, v]) => el("span", {}, pretty(k), el("strong", { text: format(v) })))));
  }));
  if (!rows.length) $("#runs-list").append(el("div", { class: "empty-state", text: query ? "No runs match your search." : "Your research starts here. Run an experiment to build your library." }));
  $("#more-runs").hidden = state.runs.length >= state.totalRuns;
  for (const picker of document.querySelectorAll(".run-picker")) {
    setOptions(picker, state.runs.map(r => ({ value: r.run_id, label: `${pretty(r.config.strategy?.approach || "Experiment")} · ${r.run_id}` })));
    if (!picker.options.length) picker.append(option("", "No runs available"));
  }
  populateBasins("replay"); populateBasins("impact");
}
async function loadRuns(append = false) {
  const page = await api(`/ui/runs?limit=100&offset=${append ? state.runs.length : 0}`);
  state.runs = append ? [...state.runs, ...page.items] : page.items; state.totalRuns = page.total; renderRuns();
  if (page.invalid_entries) notice(`${page.invalid_entries} saved run entries could not be read. Other runs remain available.`);
}
async function openRun(id) {
  const version = ++state.resultVersion;
  const run = await api(`/runs/${encodeURIComponent(id)}`); if (version !== state.resultVersion) return;
  state.run = run; $("#run-detail").hidden = false; $("#detail-id").textContent = run.run_id;
  const basinIds = run.config.basins?.target_basins || Object.keys(run.per_basin_metrics);
  setOptions($("#result-basin"), basinIds);
  setOptions($("#result-lead"), Array.from({ length: run.config.forecast?.mode === "rolling" ? run.config.forecast.horizon_days : 1 }, (_, i) => ({ value: String(i + 1), label: `Day ${i + 1}` })), "1");
  metricCards($("#metric-cards"), run.aggregate_metrics);
  $("#run-warnings").textContent = [...(run.cancelled ? ["Partial results from a cancelled experiment."] : []), ...run.failures].join("\n"); $("#run-warnings").hidden = !$("#run-warnings").textContent;
  $("#promote-run").disabled = !state.bootstrap.leaderboard_enabled || run.cancelled;
  chart("#losses", Object.entries(run.loss_traces).map(([name, values]) => ({ name: pretty(name), points: values.map((y, x) => ({ x: x + 1, y })) })), { date: false });
  renderData($("#run-data"), { per_basin_metrics: run.per_basin_metrics, forecast_metrics: run.forecast_metrics, routing: run.routing_diagnostics, config: run.config });
  renderTable($("#run-data"), run.extreme_events); renderRuns(); await renderPredictions();
  $("#run-detail").scrollIntoView({ behavior: "smooth", block: "start" });
}
async function renderPredictions() {
  if (!state.run) return;
  const run = state.run, basin = $("#result-basin").value, lead = $("#result-lead").value || "1", version = ++state.resultVersion;
  const cacheKey = `${run.run_id}/${basin}/${lead}`;
  for (const id of ["hydrograph", "scatter", "routing"]) { $("#" + id).replaceChildren(el("p", { class: "empty-state", text: "Loading catchment predictions…" })); }
  const rows = state.predictions.get(cacheKey) || await allPages((offset, limit) => api(`/runs/${encodeURIComponent(run.run_id)}/predictions?basin_id=${encodeURIComponent(basin)}&lead_day=${lead}&offset=${offset}&limit=${limit}`), 10000);
  if (version !== state.resultVersion) return;
  // A small LRU bounds browser memory when browsing many catchments.
  if (state.predictions.size >= 8) state.predictions.delete(state.predictions.keys().next().value);
  state.predictions.set(cacheKey, rows);
  metricCards($("#metric-cards"), run.per_basin_metrics[basin] || run.aggregate_metrics);
  chart("#hydrograph", [{ name: "Predicted", points: meanByDate(rows, "predicted_mm_day") }, { name: "Observed", points: meanByDate(rows, "observed_mm_day") }]);
  chart("#routing", [{ name: "Complex expert weight", points: meanByDate(rows, "complex_weight") }]);
  chart("#scatter", [{ name: "Observed / predicted", points: rows.map(r => ({ x: r.observed_mm_day, y: r.predicted_mm_day })).sort((a, b) => a.x - b.x) }], { scatter: true, date: false, xLabel: "observed discharge", yLabel: "predicted discharge" });
}
function populateBasins(prefix) {
  const run = state.runs.find(r => r.run_id === $(`#${prefix}-run`).value);
  setOptions($(`#${prefix}-basin`), run?.config.basins?.target_basins || []);
}
function replayRequest() {
  const episode = $("#replay-episode").value, basin = $("#replay-basin").value;
  if (!episode || !basin) throw new Error("Select a run, catchment and rainfall episode first.");
  return { basin_id: basin, episode, event_days: Number($("#replay-days").value) };
}
function stopPlayback() { clearInterval(state.playTimer); state.playTimer = null; $("#replay-play").textContent = "Play"; }
function renderReplayDay() {
  if (!state.replay) return;
  const { data, days } = state.replay, day = Number($("#replay-day").value), index = data.context_days + day - 1;
  $("#replay-date").textContent = data.dates[index];
  metricCards($("#replay-metrics"), { rainfall_mm: data.precipitation[index], temperature_c: data.temperature[index], observed_mm_day: data.observed[index], predicted_mm_day: data.predicted[index] });
  const toPoints = key => data[key].map((y, i) => ({ x: Date.parse(data.dates[i]), y, label: data.dates[i] }));
  const marker = Date.parse(data.dates[index]);
  chart("#replay-weather", [{ name: "Precipitation", points: toPoints("precipitation") }, { name: "PET", points: toPoints("pet") }], { marker });
  chart("#replay-flow", [{ name: "Predicted", points: toPoints("predicted") }, { name: "Observed", points: toPoints("observed") }], { marker });
  renderData($("#replay-snapshot"), days[day - 1]);
}
function navigate() {
  const route = location.hash.slice(1) || "experiments", target = ["experiments", "runs", "compare", "replay", "impact"].includes(route) ? route : "experiments";
  document.querySelectorAll(".view").forEach(node => { node.hidden = node.id !== target; });
  document.querySelectorAll(".header nav a").forEach(node => { node.classList.toggle("active", node.hash === `#${target}`); if (node.hash === `#${target}`) node.setAttribute("aria-current", "page"); else node.removeAttribute("aria-current"); });
  if (target !== "replay") stopPlayback();
  if (target === "experiments") atlas.draw();
}

// The progress panel follows every workflow and remains visible across tabs.
$("footer").before($("#job-panel"));
bind("#connection", "click", () => { $("#api-token").value = state.token; $("#connection-dialog").showModal(); });
bind("#close-connection", "click", () => $("#connection-dialog").close());
bind("#connection-form", "submit", async event => { event.preventDefault(); state.token = $("#api-token").value.trim(); sessionStorage.setItem("runoff.token", state.token); state.datasetCache.clear(); state.predictions.clear(); await connect(); $("#connection-dialog").close(); });
bind("#dataset", "change", () => loadDataset());
bind("#basin-search", "input", renderBasinList);
bind("#scope", "change", () => { if (state.config) state.config.basins.training_scope = $("#scope").value; persistDraft(); });
bind("#save-leaderboard", "change", persistDraft);
bind("#select-all", "click", () => { state.selected = new Set(state.basins.filter(r => r.eligible).map(r => r.basin_id)); selectionChanged(); renderBasinList(); });
bind("#clear-basins", "click", () => { state.selected.clear(); selectionChanged(); renderBasinList(); });
bind("#validate-dataset", "click", async () => { const result = await api(`/datasets/${encodeURIComponent($("#dataset").value)}/validation`); notice(`Dataset verified. ${result.eligible_basins} eligible catchments.`, false, true); });
bind("#validate-config", "click", async () => { const result = await api("/experiments/validate", { method: "POST", body: requestBody() }); notice(result.valid ? `Setup checked. ${result.data_check}` : "Review the setup."); });
bind("#experiment-form", "submit", async event => {
  event.preventDefault(); const body = requestBody(); persistDraft();
  await submitJob("/experiments", body); $("#job-panel").scrollIntoView({ behavior: "smooth" }); notice("");
});
bind("#cancel-job", "click", async () => { state.job = await api(`/jobs/${state.job.job_id}/cancel`, { method: "POST" }); renderJob(); schedulePoll(0); });
bind("#export-config", "click", () => downloadBlob(new Blob([JSON.stringify(requestBody(), null, 2)], { type: "application/json" }), "runoff-experiment.json"));
bind("#import-config", "change", async event => {
  const file = event.target.files[0]; if (!file) return;
  try {
    const body = JSON.parse(await file.text());
    if (!body.dataset_id || !body.config?.basins || !body.config?.dates) throw new Error("Import an exported experiment request with dataset_id, config.basins and config.dates.");
    if (!state.datasets.some(d => d.dataset_id === body.dataset_id)) throw new Error("This setup uses a dataset that is not registered on this server.");
    // Server validation catches unknown fields, types and invalid values before replacing the draft.
    await api("/experiments/validate", { method: "POST", body });
    state.config = clone(state.bootstrap.config); for (const [key, value] of Object.entries(body.config)) state.config[key] = value && typeof value === "object" && !Array.isArray(value) ? { ...state.config[key], ...value } : value;
    state.flags = body; state.flagBindings = []; $("#dataset").value = body.dataset_id;
    $("#save-leaderboard").checked = state.bootstrap.leaderboard_enabled && Boolean(body.save_to_leaderboard);
    state.selected = new Set(body.config.basins.target_basins); renderConfig(); await loadDataset({ retain: true }); persistDraft(); notice("Experiment setup imported.");
  } finally { event.target.value = ""; }
});
bind("#reset-config", "click", () => { state.config = clone(state.bootstrap.config); state.flags = {}; state.flagBindings = []; renderConfig(); persistDraft(); notice("Configuration reset to server defaults. Catchment selection is preserved."); });
bind("#refresh-runs", "click", () => loadRuns()); bind("#more-runs", "click", () => loadRuns(true)); bind("#run-search", "input", renderRuns);
bind("#result-basin", "change", renderPredictions);
bind("#result-lead", "change", renderPredictions);
document.querySelectorAll("[data-download]").forEach(button => button.addEventListener("click", action(async () => {
  if (!state.run) return; const name = button.dataset.download;
  downloadBlob(await api(`/runs/${encodeURIComponent(state.run.run_id)}/artifacts/${name}`, { download: true }), name === "archive" ? "regional_experiment.zip" : `${name}.${name === "config" ? "json" : "csv"}`);
})));
bind("#promote-run", "click", async () => { await api(`/runs/${encodeURIComponent(state.run.run_id)}/save`, { method: "POST" }); notice("Run saved to the shared leaderboard."); });
bind("#compare-button", "click", async () => {
  const a = $("#compare-a").value, b = $("#compare-b").value; if (!a || !b || a === b) throw new Error("Choose two different experiments.");
  const result = await api("/comparisons", { method: "POST", body: { run_a: a, run_b: b } });
  const node = $("#comparison-output"); node.classList.remove("empty-state"); node.replaceChildren(el("h4", { text: "Aggregate metrics · candidate minus baseline" }));
  renderTable(node, Object.entries(result.aggregate_delta_b_minus_a).map(([metric, delta]) => ({ metric: pretty(metric), baseline: result.run_a.aggregate_metrics[metric], candidate: result.run_b.aggregate_metrics[metric], delta })));
  node.append(el("h4", { text: "Per-catchment differences" })); renderTable(node, Object.entries(result.per_basin_delta_b_minus_a).map(([basin, values]) => ({ basin, ...values })));
});
bind("#replay-run", "change", () => { populateBasins("replay"); $("#replay-episode").replaceChildren(option("", "Find episodes to begin")); $("#replay-output").hidden = true; stopPlayback(); });
bind("#replay-basin", "change", () => { $("#replay-episode").replaceChildren(option("", "Find episodes to begin")); $("#replay-output").hidden = true; stopPlayback(); });
bind("#replay-days", "change", () => { $("#replay-episode").replaceChildren(option("", "Find episodes for this duration")); $("#replay-output").hidden = true; stopPlayback(); });
bind("#impact-run", "change", () => populateBasins("impact"));
bind("#find-episodes", "click", async () => {
  const id = $("#replay-run").value, basin = $("#replay-basin").value; if (!id || !basin) throw new Error("Select an experiment and catchment first.");
  const result = await api(`/runs/${encodeURIComponent(id)}/replay/episodes?basin_id=${encodeURIComponent(basin)}&event_days=${Number($("#replay-days").value)}`);
  setOptions($("#replay-episode"), result.items.map(r => ({ value: r.episode, label: r.label })));
  if (!result.items.length) { $("#replay-episode").append(option("", "No episodes with sufficient coverage")); notice("No rainfall episodes have enough context and prediction coverage for this duration."); }
});
bind("#load-replay", "click", async () => {
  stopPlayback(); state.replay = await api(`/runs/${encodeURIComponent($("#replay-run").value)}/replay`, { method: "POST", body: replayRequest() });
  $("#replay-day").max = state.replay.days.length; $("#replay-day").value = 1; $("#replay-output").hidden = false; renderReplayDay();
});
bind("#replay-day", "input", renderReplayDay);
bind("#replay-play", "click", () => {
  if (state.playTimer) { stopPlayback(); return; } if (!state.replay) return;
  $("#replay-play").textContent = "Pause";
  if (Number($("#replay-day").value) >= state.replay.days.length) $("#replay-day").value = 1;
  state.playTimer = setInterval(() => { const slider = $("#replay-day"); if (Number(slider.value) >= state.replay.days.length) { stopPlayback(); return; } slider.value = Number(slider.value) + 1; renderReplayDay(); }, 900);
});
bind("#export-gif", "click", async () => { if (!state.capabilities.optional_methods.gif) throw new Error("Install the animations extra on the server to export GIFs."); await submitJob(`/runs/${encodeURIComponent($("#replay-run").value)}/replay/gif`, replayRequest()); });
state.scenarioBindings = [];
const scenarioDefaults = { site_fraction: .001, original_paved_fraction: .1, proposed_paved_fraction: .8, pervious_runoff_coefficient: .25, paved_runoff_coefficient: .9, connected_fraction: 1, storage_capacity_mm: 0, release_mm_day: 0, start_date: null };
for (const [key, value] of Object.entries(scenarioDefaults)) field($("#scenario-fields"), key, value, key === "start_date" ? { anyOf: [{ type: "string" }, { type: "null" }] } : { type: "number" }, [key], state.scenarioBindings);
bind("#scenario-form", "submit", async event => {
  event.preventDefault(); ensureNoJob(); const id = $("#impact-run").value, basin = $("#impact-basin").value;
  if (!id || !basin) throw new Error("Select an experiment and catchment first.");
  await submitJob(`/runs/${encodeURIComponent(id)}/scenarios`, { basin_id: basin, site: readBindings({}, state.scenarioBindings) }); $("#job-panel").scrollIntoView({ behavior: "smooth" });
});
window.addEventListener("hashchange", navigate);
document.addEventListener("visibilitychange", () => { if (document.hidden) stopPlayback(); else if (state.job && !terminal(state.job.status)) schedulePoll(0); });
window.addEventListener("online", () => { if (!state.connected) action(connect)(); });
navigate(); action(connect)(); setInterval(healthCheck, 15000);
