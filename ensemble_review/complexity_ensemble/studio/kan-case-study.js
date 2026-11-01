import { allPages } from "./core.js";
import { chart } from "./charts.js";

const $ = selector => document.querySelector(selector);
const labels = {
  source_basins: "Source catchments · your selection", validation_basins: "Source-validation catchments · your selection",
  target_basins: "Target catchments · your selection", observation_budgets: "Default target observation budgets",
  seeds: "Random seeds", warmup_days: "Forcing-only spin-up (days)",
  minimum_source_days: "Required source observations (optional stricter minimum)", teacher_epochs: "Teacher epochs",
  adaptation_epochs: "Local calibration epochs", teacher_width: "Teacher hidden width",
  learning_rate: "Learning rate", kan_candidates: "KAN candidate seeds", kan_steps: "KAN fitting steps",
  kan_grid: "KAN spline grid", kan_hidden: "KAN hidden nodes", max_formula_ops: "Maximum symbolic operations",
  max_distillation_samples_per_basin: "Distillation states per basin", sparsity_weight: "KAN sparsity penalty",
  device: "Compute device",
  teacher_epochs: "KAN process-teacher epochs", teacher_width: "KAN process-teacher MLP width",
  adaptation_epochs: "KAN local calibration epochs",
  neural_epochs: "Neural comparison epochs", neural_batch_size: "Neural batch size",
  sequence_length: "Neural weather history (days)", mlp_widths: "MLP hidden widths",
};
function node(tag, text, className) {
  const n = document.createElement(tag); if (text != null) n.textContent = text;
  if (className) n.className = className; return n;
}

export function splitUsage(row, role, budgets, minimumSourceDays = 3, selectedModels = null) {
  const train = row.training_observations || 0, validation = row.validation_observations || 0, test = row.test_observations || 0;
  let train_used = 0, validation_used = 0, test_used = 0;
  const issues = [];
  const needsSources = !selectedModels || selectedModels.some(m => m.startsWith("kan_") || m.includes("_source_") || m.includes("_mixed_"));
  const needsValidation = !selectedModels || selectedModels.some(m => m.startsWith("kan_"));
  const needsTargets = !selectedModels || selectedModels.some(m => ["kan_tuned", "kan_complex"].includes(m) || m.includes("_target_target") || m.includes("_mixed_"));
  if ((role === "source" && !needsSources) || (role === "validation" && !needsValidation)) return { train_used: 0, validation_used: 0, test_used: 0, readiness: "Not used by selected models" };
  if (role === "source") {
    train_used = train;
    if (train < minimumSourceDays) issues.push(`Needs ${minimumSourceDays} training observations; has ${train}`);
  } else if (role === "validation") {
    validation_used = validation;
    if (validation < 3) issues.push(`Needs 3 validation observations; has ${validation}`);
  } else if (role === "target") {
    if (!budgets.length || budgets.some(n => !Number.isInteger(n) || n < 3) || budgets.some((n, i) => i && n <= budgets[i - 1])) {
      return { train_used: "—", validation_used: 0, test_used: test, readiness: "Enter increasing integer target budgets of at least 3" };
    }
    train_used = needsTargets ? budgets.map(n => n <= train ? n : `${n} requested (only ${train} available)`).join(", ") : 0;
    test_used = test;
    if (needsTargets && train < Math.max(...budgets)) issues.push(`Largest training budget is ${Math.max(...budgets)}; has ${train}`);
    if (test < 3) issues.push(`Needs 3 test observations; has ${test}`);
  } else return { train_used: "—", validation_used: "—", test_used: "—", readiness: role === "conflict" ? "Choose one role" : "Assign a role" };
  if (row.weather_missing_days) issues.push(`Missing ${row.weather_missing_days} weather days`);
  if (row.error) issues.push(row.error);
  return { train_used, validation_used, test_used, readiness: issues.join("; ") || "Ready" };
}

export class KanCaseStudyView {
  constructor(api, submitJob, renderTable, downloadBlob) {
    Object.assign(this, { api, submitJob, renderTable, downloadBlob });
    this.fields = new Map();
    this.targetBudgets = new Map();
    this.models = new Set();
    this.cohorts = new Map(["source_basins", "validation_basins", "target_basins"].map(key => [key, new Set()]));
    this.cohortLists = new Map();
    this.cohortSearchInputs = new Map();
    $("#kan-scan").addEventListener("click", () => this.action(() => this.scanCoverage()));
    $("#kan-coverage-search").addEventListener("input", () => this.renderCoverage());
    $("#kan-coverage-filter").addEventListener("change", () => this.renderCoverage());
    // Validate explicitly so an invalid field inside a collapsed settings panel
    // cannot silently suppress the browser's submit event.
    $("#kan-form").noValidate = true;
    $("#kan-form").addEventListener("submit", event => { event.preventDefault(); return this.action(() => this.launch()); });
    $("#kan-validate").addEventListener("click", () => this.action(async () => {
      $("#kan-status").textContent = "Checking cohorts and daily coverage…";
      const result = await this.api("/case-studies/kan/validate", { method: "POST", body: this.request() });
      $("#kan-status").textContent = "Coverage complete. No training was started.";
      $("#kan-audit").textContent = JSON.stringify(result, null, 2);
    }));
    $("#kan-dataset").addEventListener("change", () => { this.invalidateCoverage(); this.action(() => this.loadBasins()); });
    $("#kan-load").addEventListener("click", () => this.action(() => this.showJob($("#kan-saved").value)));
    $("#kan-download").addEventListener("click", () => this.action(async () => {
      this.downloadBlob(await this.api(`/case-studies/kan/jobs/${this.jobId}/archive`, { download: true }), "kan_case_study.zip");
    }));
    $("#kan-plot-basin").addEventListener("change", () => this.plot());
    $("#kan-plot-seed").addEventListener("change", () => this.plot());
    $("#kan-plot-scenario").addEventListener("change", () => this.plot());
  }

  async action(fn) {
    try { await fn(); } catch (error) { $("#kan-status").textContent = error.message; }
  }

  async launch() {
    if (this.submitting) return;
    const button = $("#kan-launch");
    this.submitting = true; button.disabled = true; button.textContent = "Submitting…";
    $("#kan-status").textContent = "Submitting case study…";
    try {
      const job = await this.submitJob("/case-studies/kan", this.request());
      $("#kan-status").textContent = `Queued${job?.job_id ? ` job ${job.job_id}` : ""}. Waiting for the training worker; follow progress in the job panel.`;
      $("#job-panel")?.scrollIntoView?.({ behavior: "smooth", block: "nearest" });
      return job;
    } finally {
      this.submitting = false; button.disabled = false; button.textContent = "Run case study ↗";
    }
  }

  updateJob(job) {
    const detail = job.error?.message || job.message || "Waiting for the training worker";
    $("#kan-status").textContent = `Job ${job.job_id} · ${job.status} · ${Math.round((job.progress || 0) * 100)}% · ${detail}`;
  }

  async connect(datasets) {
    const response = await this.api("/case-studies/kan/defaults");
    this.defaults = response.config; this.comparisonModels = response.comparison_models;
    const select = $("#kan-dataset"), previous = select.value;
    select.replaceChildren(...datasets.map(d => {
      const option = node("option", d.dataset_id); option.value = d.dataset_id; return option;
    }));
    if ([...select.options].some(o => o.value === previous)) select.value = previous;
    if (!this.fields.size) this.buildFields();
    if (select.value) await this.loadBasins();
    await this.loadSaved();
  }

  buildFields() {
    for (const [key, value] of Object.entries(this.defaults)) {
      if (key === "target_observation_budgets") continue;
      if (key === "selected_models") {
        const group = node("fieldset", null, "kan-cohort-group");
        group.append(node("legend", "Models to include in this comparison"));
        this.models = new Set(value);
        const available = this.comparisonModels || ["pinnmamba", "mlp", "lstm"].flatMap(a => ["source", "target", "mixed"].map(s => `${a}_${s}_target`)).concat(["kan_pure", "kan_tuned", "kan_complex"]);
        for (const method of available) {
          const label = node("label", null, "kan-cohort-choice"), checkbox = node("input");
          checkbox.type = "checkbox"; checkbox.value = method; checkbox.checked = this.models.has(method);
          checkbox.addEventListener("change", () => { if (checkbox.checked) this.models.add(method); else this.models.delete(method); this.renderCoverage(); });
          label.append(checkbox, node("span", method)); group.append(label);
        }
        this.fields.set(key, group); $("#kan-models").append(group); continue;
      }
      if (key.endsWith("_basins")) {
        const group = node("fieldset", null, "kan-cohort-group");
        group.id = `kan-${key}`; group.append(node("legend", labels[key]));
        const search = node("input"); search.type = "search";
        search.placeholder = "Filter by catchment name or ID";
        search.setAttribute("aria-label", `Filter ${labels[key]}`);
        const list = node("div", null, "kan-cohort-list");
        search.addEventListener("input", () => this.renderCohortList(key));
        group.append(search, list); $("#kan-cohorts").append(group);
        this.fields.set(key, group); this.cohortLists.set(key, list); this.cohortSearchInputs.set(key, search);
        continue;
      }
      const label = node("label", null, "field");
      label.append(node("span", labels[key] || key.replaceAll("_", " ")));
      let input;
      if (key === "device") {
        input = node("select");
        for (const device of ["auto", "cpu", "cuda", "mps"]) {
          const option = node("option", device === "mps" ? "MPS · Apple GPU" : device === "cuda" ? "CUDA · NVIDIA GPU" : device === "auto" ? "Auto" : "CPU");
          option.value = device; input.append(option);
        }
        input.value = value;
      } else {
        input = node("input"); input.type = key.endsWith("_start") || key.endsWith("_end") ? "date" : typeof value === "number" ? "number" : "text";
        input.value = Array.isArray(value) ? value.join(", ") : value;
      }
      if (input.type === "number") {
        input.step = Number.isInteger(value) ? "1" : "any";
        input.min = key === "minimum_source_days" ? "3" : Number.isInteger(value) ? "1" : key === "sparsity_weight" ? "0" : "0.000001";
      }
      input.id = `kan-${key}`; label.append(input); this.fields.set(key, input);
      if (key === "observation_budgets") input.addEventListener("change", () => this.renderTargetBudgets());
      if (key.endsWith("_start") || key.endsWith("_end") || ["observation_budgets", "minimum_source_days", "warmup_days"].includes(key)) input.addEventListener("change", () => this.invalidateCoverage());
      const group = key.endsWith("_basins") ? "#kan-cohorts" : key.endsWith("_start") || key.endsWith("_end") ? "#kan-dates" : ["observation_budgets", "seeds", "device"].includes(key) ? "#kan-budgets" : "#kan-settings";
      if (key === "device") label.append(node("span", "Independent of the main experiment. Auto chooses CUDA, MPS, then CPU. On MPS, KAN spline fitting uses CPU. Symbolic extraction and scoring use CPU.", "field-hint"));
      $(group).append(label);
    }
  }

  async loadBasins() {
    const dataset = $("#kan-dataset").value;
    const rows = await allPages((offset, limit) => this.api(`/datasets/${encodeURIComponent(dataset)}/basins?offset=${offset}&limit=${limit}`), 1000);
    if (dataset !== $("#kan-dataset").value) return;
    this.basins = rows;
    const eligible = new Set(rows.filter(r => r.eligible).map(r => r.basin_id));
    for (const selected of this.cohorts.values()) for (const basin of selected) if (!eligible.has(basin)) selected.delete(basin);
    this.renderCohorts();
    this.basinsDataset = dataset;
  }

  renderCohorts() {
    for (const key of this.cohorts.keys()) this.renderCohortList(key);
    this.renderRoles(); this.renderTargetBudgets();
  }

  renderCohortList(key) {
    const coverage = new Map((this.coverage?.items || []).map(r => [r.basin_id, r]));
    const query = this.cohortSearchInputs.get(key).value.trim().toLowerCase();
    const list = this.cohortLists.get(key), selected = this.cohorts.get(key);
    list.replaceChildren(...(this.basins || []).filter(r => r.eligible && `${r.basin_id} ${r.name || ""}`.toLowerCase().includes(query)).map(row => {
        const record = coverage.get(row.basin_id);
        const hint = record ? ` · train ${record.training_observations} / val ${record.validation_observations ?? "?"} / test ${record.test_observations ?? "?"}${record.weather_missing_days ? " · weather gaps" : ""}` : "";
        const label = node("label", null, "kan-cohort-choice");
        const checkbox = node("input"); checkbox.type = "checkbox";
        checkbox.value = row.basin_id; checkbox.name = key; checkbox.checked = selected.has(row.basin_id);
        checkbox.addEventListener("change", () => {
          if (checkbox.checked) selected.add(row.basin_id); else selected.delete(row.basin_id);
          this.renderRoles(); this.renderCoverage();
          if (key === "target_basins") this.renderTargetBudgets();
        });
        label.append(checkbox, node("span", `${row.basin_id} · ${row.name || row.basin_id}${hint}`));
        return label;
    }));
    if (!list.children.length) list.append(node("p", "No catchments match this filter.", "field-hint"));
  }

  roleFor(basin) {
    const roles = ["source", "validation", "target"].filter(role =>
      this.cohorts.get(`${role}_basins`).has(basin));
    return roles.length > 1 ? "conflict" : roles[0] || "unassigned";
  }

  renderRoles() {
    const count = key => this.cohorts.get(key).size;
    $("#kan-role-summary").textContent = `Your selection: ${count("source_basins")} source · ${count("validation_basins")} source-validation · ${count("target_basins")} target. Roles are assigned by you; observation counts do not assign or restrict them. Each catchment must have one role.`;
  }

  renderTargetBudgets() {
    const parent = $("#kan-target-budgets"); parent.replaceChildren();
    for (const basin of this.cohorts.get("target_basins")) {
      const label = node("label", null, "field");
      label.append(node("span", `${basin} · observation budgets`));
      const input = node("input"); input.type = "text";
      input.value = this.targetBudgets.get(basin) || "";
      input.placeholder = `Use defaults: ${this.fields.get("observation_budgets")?.value || ""}`;
      input.addEventListener("input", () => this.targetBudgets.set(basin, input.value));
      input.addEventListener("change", () => {
        this.targetBudgets.set(basin, input.value); this.invalidateCoverage();
      });
      label.append(input); parent.append(label);
    }
  }

  coverageKey() {
    return JSON.stringify([$("#kan-dataset").value, ...["train_start", "train_end", "validation_start", "validation_end", "test_start", "test_end", "observation_budgets", "minimum_source_days", "warmup_days"].map(k => this.fields.get(k)?.value), [...this.targetBudgets]]);
  }

  invalidateCoverage() {
    this.coverage = null;
    $("#kan-coverage-table").replaceChildren();
    $("#kan-coverage-status").textContent = "Setup changed. Scan availability again for these dates and thresholds.";
    if (this.basins) this.renderCohorts();
  }

  async scanCoverage() {
    const body = this.request(), snapshot = this.coverageKey();
    $("#kan-coverage-status").textContent = "Scanning catchment records. No cohorts or training are required…";
    const result = await this.api("/case-studies/kan/coverage", { method: "POST", body });
    if (snapshot !== this.coverageKey()) return;
    this.coverage = result;
    const dates = result.dates;
    $("#kan-coverage-status").textContent = dates
      ? `${result.items.length} catchments · Train ${dates.train.start}–${dates.train.end} · Validation ${dates.validation.start}–${dates.validation.end} · Test ${dates.test.start}–${dates.test.end}. Available = valid observed labels with weather. Used = planned per-run labels for your assigned role; target budgets are separate runs. Test labels are scoring-only.`
      : "Refresh the server to obtain train, validation and test availability for the selected dates.";
    this.renderCoverage(); this.renderCohorts();
  }

  renderCoverage() {
    const parent = $("#kan-coverage-table"); parent.replaceChildren();
    if (!this.coverage) return;
    const query = $("#kan-coverage-search").value.trim().toLowerCase(), role = $("#kan-coverage-filter").value;
    const rows = this.coverage.items.filter(r => (!role || this.roleFor(r.basin_id) === role)
      && `${r.basin_id} ${r.name}`.toLowerCase().includes(query));
    this.renderTable(parent, rows.map(r => {
      const role = this.roleFor(r.basin_id), raw = this.targetBudgets.get(r.basin_id)?.trim() || this.fields.get("observation_budgets")?.value || "";
      const budgets = raw.split(",").filter(v => v.trim()).map(Number);
      const usage = splitUsage(r, role, budgets, Number(this.fields.get("minimum_source_days")?.value || 3), this.fields.has("selected_models") ? [...this.models] : null);
      return { basin: r.basin_id, name: r.name, assigned_role: role,
      train_available: r.training_observations, train_used: usage.train_used,
      validation_available: r.validation_observations ?? "Refresh coverage", validation_used: usage.validation_used,
      test_available: r.test_observations ?? "Refresh coverage", test_used: usage.test_used, readiness: usage.readiness,
      train_coverage_percent: r.coverage_percent,
      first: r.first_observation, last: r.last_observation, longest_gap_days: r.longest_missing_spell_days,
      longest_observed_run_days: r.longest_observed_spell_days, calendar_months: r.calendar_months_represented,
      observations_by_month: Object.entries(r.observations_by_month || {}).map(([m, n]) => `${m}:${n}`).join(" "),
      weather_missing_days: r.weather_missing_days, error: r.error };
    }));
    if (!rows.length) parent.textContent = "No catchments match this filter.";
  }

  request() {
    if (this.basinsDataset !== $("#kan-dataset").value) throw new Error("Wait for the selected dataset's catchments to load.");
    const config = {};
    for (const [key, input] of this.fields) {
      if (key === "selected_models") { config[key] = [...this.models]; continue; }
      if (!key.endsWith("_basins") && input.checkValidity && !input.checkValidity()) {
        const details = input.closest?.("details"); if (details) details.open = true;
        input.reportValidity?.();
        throw new Error(`${labels[key] || key.replaceAll("_", " ")}: ${input.validationMessage || "Enter a valid value"}`);
      }
      config[key] = key.endsWith("_basins") ? [...this.cohorts.get(key)]
        : Array.isArray(this.defaults[key]) ? input.value.split(",").filter(v => v.trim()).map(Number)
        : typeof this.defaults[key] === "number" ? Number(input.value) : input.value;
    }
    config.target_observation_budgets = Object.fromEntries((config.target_basins || []).flatMap(basin => {
      const raw = this.targetBudgets.get(basin)?.trim();
      return raw ? [[basin, raw.split(",").filter(v => v.trim()).map(Number)]] : [];
    }));
    return { dataset_id: $("#kan-dataset").value, config };
  }

  async loadSaved() {
    const result = await this.api("/case-studies/kan");
    $("#kan-saved").replaceChildren(...result.items.filter(j => j.status === "succeeded").map(job => {
      const option = node("option", `${job.created_at} · ${job.job_id.slice(0, 8)}`); option.value = job.job_id; return option;
    }));
  }

  async showJob(jobId) {
    if (!jobId) throw new Error("Choose a completed case study.");
    const result = await this.api(`/jobs/${encodeURIComponent(jobId)}/result`);
    if (result.kind !== "kan_case_study") throw new Error("This is not a KAN case study.");
    this.jobId = jobId; this.result = result;
    $("#kan-results").hidden = false;
    const runBasins = $("#kan-run-basins");
    runBasins.replaceChildren(...[
      ["Source training basins", result.audit?.teacher_training_basins ?? result.config.source_basins],
      ["Source validation basins", result.audit?.equation_selection_basins ?? result.config.validation_basins],
    ].map(([label, basins]) => node("div", `${label}: ${basins == null ? "Not recorded" : basins.length ? basins.join(", ") : "None"}`)));
    $("#kan-equations").textContent = result.equations.map(e => `Seed ${e.seed}${e.models ? ` · ${e.models.join(", ")}` : ""}\nF(storage, liquid, pet) = ${e.logit_expression}`).join("\n\n");
    $("#kan-audit").textContent = JSON.stringify({ audit: result.audit, config: result.config, discovery: result.equations }, null, 2);
    const table = $("#kan-metrics"); table.replaceChildren();
    // Hide these columns at display time so older saved runs remain readable.
    const hiddenColumns = new Set([
      "low_flow_rmse", "high_flow_rmse", "local_parameters", "fit_id",
      "max_abs_water_balance_mm", "state_outside_source_range_fraction",
      "calibration_first", "calibration_last", "capacity_mm", "melt_mm_day_c", "runoff_logit_offset",
    ]);
    this.renderTable(table, result.metrics.map(row => Object.fromEntries(
      Object.entries(row).filter(([key]) => !hiddenColumns.has(key)),
    )));
    for (const [id, values] of [["#kan-plot-basin", result.config.target_basins], ["#kan-plot-seed", result.config.seeds]]) {
      $(id).replaceChildren(...values.map(value => { const option = node("option", String(value)); option.value = value; return option; }));
    }
    const selector = $("#kan-plot-scenario"), all = node("option", "All budget scenarios"); all.value = "";
    selector.replaceChildren(all, ...(result.audit.budget_scenarios || []).map(s => {
      const option = node("option", `Scenario ${s.scenario} · ${Object.entries(s.target_budgets).map(([b, n]) => `${b}: ${n}`).join(", ")}`);
      option.value = String(s.scenario); return option;
    }));
    this.plot(); await this.loadSaved(); $("#kan-saved").value = jobId;
    const hardware = result.hardware;
    $("#kan-status").textContent = "Case study complete. Metrics use untouched target test observations."
      + (hardware ? ` Compute: ${hardware.resolved}; KAN fitting: ${hardware.kan_fitting}; symbolic extraction: ${hardware.symbolic_extraction}.` : "");
  }

  plot() {
    if (!this.result) return;
    const basin = $("#kan-plot-basin").value, seed = Number($("#kan-plot-seed").value);
    const scenario = $("#kan-plot-scenario").value;
    const rows = this.result.metrics.filter(r => r.basin_id === basin && r.seed === seed && (!scenario || r.scenario === Number(scenario)));
    const methods = [...new Set(rows.map(r => r.method))];
    const colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#4F46E5", "#8B4513", "#E69F00", "#008B8B", "#6B7280", "#111827", "#DC2626"];
    chart("#kan-learning-curve", methods.map((method, i) => ({ name: method, color: colors[i % colors.length], points: rows.filter(r => r.method === method).map(r => ({ x: r.budget, y: r.rmse_mm_day, label: `scenario ${r.scenario || 1}` })) })),
      { date: false, markers: true, xLabel: "Target observation budget", yLabel: "Test RMSE (mm/day)" });
  }
}
