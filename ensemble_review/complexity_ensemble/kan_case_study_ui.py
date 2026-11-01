"""Independent native UI for the process-oriented KAN transfer experiment."""
from __future__ import annotations

from dataclasses import asdict
import uuid

from .kan_case_study import (CaseStudyConfig, COMPARISON_MODELS, StudyCancelled, config_from_dict,
                             prepare_data, run_case_study, save_case_study)
from .kan_coverage import coverage_inventory, planned_usage
from .devices import DEVICE_CHOICES


def build_case_study_tab(gr, catalog, choices, output_dir, *, executor=None):
    from .regional import CancellationToken
    import pandas as pd
    import plotly.express as px

    with gr.Tab("KAN low-data case study"):
        gr.Markdown("""### Symbolic runoff transfer to low-data catchments

Train a neural runoff relationship on source catchments, discover a symbolic
approximation with KAN, and freeze its structure before local calibration.
Choose three **disjoint** cohorts. Source-validation catchments select the equation;
target observations are used only for the selected calibration budgets and final scoring.

The model simulates snow and water storage and calibrates three local parameters:
water capacity, degree-day melt factor and runoff offset. Daily weather drives a
continuous hindcast; observed discharge is never assimilated. The equation is a
teacher approximation, not an established physical law. Complete daily CAMELS-CH
weather records including spin-up are required. Missing discharge labels are
supported; budgets use the first N available observations.
""")
        fields, controls = [], []
        defaults = asdict(CaseStudyConfig())
        gr.Markdown("**Assign catchment roles yourself.** Observation counts are guidance; "
                    "you decide which catchments are sources, source-validation or targets. "
                    "Any selected target can use a restricted local observation budget.")
        for name, label in (("source_basins", "Source catchments · your selection"),
                            ("validation_basins", "Source-validation catchments · your selection"),
                            ("target_basins", "Target catchments · your selection")):
            controls.append(gr.CheckboxGroup(choices, label=label))
            fields.append(name)
        with gr.Row():
            for name in ("train_start", "train_end", "validation_start", "validation_end", "test_start", "test_end"):
                controls.append(gr.Textbox(defaults[name], label=name.replace("_", " ").title()))
                fields.append(name)
        with gr.Row():
            for name in ("observation_budgets", "seeds"):
                controls.append(gr.Textbox(",".join(map(str, defaults[name])), label="Default target observation budgets" if name == "observation_budgets" else "Seeds"))
                fields.append(name)
            controls.append(gr.Radio(list(DEVICE_CHOICES), value=defaults["device"], label="Compute device"))
            fields.append("device")
        gr.Markdown("Teacher training, target calibration and simulation use this device, independently of the main experiment. "
                    "PyKAN fits on CUDA when selected; MPS uses CPU for spline fitting. "
                    "Symbolic extraction and scoring run on CPU. Auto chooses CUDA, then MPS, then CPU.")
        model_selector = gr.CheckboxGroup(list(COMPARISON_MODELS), value=list(defaults["selected_models"]),
                                          label="Models to include in this comparison")
        controls.append(model_selector); fields.append("selected_models")
        gr.Markdown("`source_target`: fit source labels, predict targets. `target_target`: fit only budgeted target labels. "
                    "`mixed_target`: fit sources and budgeted targets. Each scope uses one shared model for the selected cohort. "
                    "`kan_pure`: frozen source-derived symbolic equation. `kan_tuned`: same equation with three local coefficients calibrated. "
                    "`kan_complex`: independent MLP teacher (at least 24 hidden units, 50 epochs), at least six candidates, "
                    "and x³/abs symbolic functions, with the same KAN size and three local calibrated coefficients. "
                    "Budgets are paired by list position across targets; shorter lists retain their last budget. All models use weather-only inputs.")
        gr.Markdown("**Per-target observation budgets:** enter comma-separated counts for each "
                    "target, or leave its budget blank to use the defaults above.")
        target_budgets = gr.Dataframe(headers=["Target catchment", "Observation budgets"],
            datatype=["str", "str"], type="array", value=[], interactive=True,
            label="Your target calibration budgets")
        def target_budget_rows(targets, rows):
            previous = {str(row[0]): row[1] for row in (rows or []) if len(row) >= 2}
            return [[basin, previous.get(basin, "")] for basin in (targets or [])]
        controls[2].change(target_budget_rows, inputs=[controls[2], target_budgets], outputs=target_budgets)
        with gr.Accordion("Teacher, symbolic discovery and calibration settings", open=False):
            for name, value in defaults.items():
                if name in fields or name == "target_observation_budgets":
                    continue
                if name == "mlp_widths":
                    controls.append(gr.Textbox(",".join(map(str, value)), label="MLP hidden widths"))
                else:
                    controls.append(gr.Number(value, precision=0 if type(value) is int else None,
                                              label="Required source observations (optional stricter minimum)" if name == "minimum_source_days" else name.replace("_", " ").title()))
                fields.append(name)
        gr.Markdown("""Scan train, validation and test availability for **your manual cohort choices**.
Available counts follow each selected date range. Used counts show the planned
labels for the assigned role; multiple target budgets are separate runs. Target
test labels are scoring-only and target validation labels are unused.
Compare gaps and calendar-month coverage as well as counts. The platform does not label catchments
as high-data or low-data. It checks cohort separation, available weather and
enough observations for the budgets you choose.
""")
        scan = gr.Button("Scan catchment availability before choosing cohorts")
        coverage_state = gr.State(None)
        coverage_status = gr.Markdown()
        with gr.Row():
            coverage_search = gr.Textbox(label="Find catchment by name or ID")
            coverage_filter = gr.Dropdown(["all", "source", "validation", "target", "unassigned", "conflict"],
                                           value="all", label="Your assigned role")
        coverage_table = gr.Dataframe(label="Catchment availability for the selected training dates", interactive=False)
        with gr.Row():
            validate = gr.Button("Check cohorts and daily coverage")
            run = gr.Button("Run KAN case study", variant="primary")
            cancel = gr.Button("Cancel case study")
        token_state = gr.State(None)
        status = gr.Markdown()
        metrics = gr.Dataframe(label="Test performance by local observation budget", interactive=False)
        curve = gr.Plot(label="Local-data learning curves")
        equations = gr.Code(label="Executed symbolic runoff expressions", language=None)
        audit = gr.JSON(label="Protocol and source-only selection audit")
        archive = gr.File(label="Download equations, coefficients, metrics and daily fluxes")

        def config(values, *, require_cohorts=True):
            value = dict(zip(fields, values))
            for key in ("observation_budgets", "seeds", "mlp_widths"):
                value[key] = [int(v.strip()) for v in value[key].split(",") if v.strip()]
            for key, default in defaults.items():
                if type(default) is int:
                    raw = value[key]
                    if raw is None or int(raw) != raw:
                        raise ValueError(f"{key} must be an integer")
                    value[key] = int(raw)
            value["target_observation_budgets"] = {}
            for row in values[len(fields)] or []:
                if len(row) < 2 or not str(row[0] or "").strip():
                    continue
                basin, raw = str(row[0]).strip(), str(row[1] or "").strip()
                if raw:
                    if basin in value["target_observation_budgets"]:
                        raise ValueError("Each target must have a single budget row")
                    value["target_observation_budgets"][basin] = [int(v.strip()) for v in raw.split(",") if v.strip()]
            return config_from_dict(value, require_cohorts=require_cohorts)

        def coverage_rows(report, search="", role="all", sources=(), validations=(), targets=(), selected_models=None):
            if not report:
                return pd.DataFrame()
            rows = []
            for row in report["items"]:
                roles = [r for r, cohort in (("source", sources), ("validation", validations), ("target", targets)) if row["basin_id"] in (cohort or [])]
                assigned = "conflict" if len(roles) > 1 else roles[0] if roles else "unassigned"
                if role != "all" and assigned != role:
                    continue
                if search.lower() not in f"{row['basin_id']} {row['name']}".lower():
                    continue
                budgets = report["target_observation_budgets"].get(row["basin_id"], report["default_target_budgets"])
                usage = planned_usage(row, assigned, budgets, report["minimum_source_days"],
                                      selected_models if selected_models is not None else report.get("selected_models"))
                rows.append({"basin_id": row["basin_id"], "name": row["name"], "assigned_role": assigned,
                    "train_available": row["training_observations"], "train_used": usage["train_used"],
                    "validation_available": row["validation_observations"], "validation_used": usage["validation_used"],
                    "test_available": row["test_observations"], "test_used": usage["test_used"], "readiness": usage["readiness"],
                    "train_calendar_days": row["training_calendar_days"],
                    "validation_calendar_days": row["validation_calendar_days"], "test_calendar_days": row["test_calendar_days"],
                    "train_coverage_percent": row["coverage_percent"], "weather_missing_days": row["weather_missing_days"],
                    "observations_by_month": " ".join(
                    f"{month}:{count}" for month, count in row["observations_by_month"].items())})
            return pd.DataFrame(rows)

        def scan_records(*values):
            try:
                cfg = config(values, require_cohorts=False)
                report = (executor.request("POST", "/case-studies/kan/coverage",
                    {"dataset_id": executor.dataset_id, "config": asdict(cfg)}) if executor else
                    coverage_inventory(cfg, catalog))
                names = [(f"{r['basin_id']} — {r['name']} · train {r['training_observations']} / val {r['validation_observations']} / test {r['test_observations']}"
                          + (" · weather gaps" if r['weather_missing_days'] else ""), r['basin_id']) for r in report["items"]]
                ranges = report["dates"]
                return (report, f"Train: {ranges['train']['start']} through {ranges['train']['end']}. "
                        f"Validation: {ranges['validation']['start']} through {ranges['validation']['end']}. "
                        f"Test: {ranges['test']['start']} through {ranges['test']['end']}. "
                        "Available/used counts are per split and assigned role; test labels are scoring-only.",
                        coverage_rows(report, values[-2], values[-1], *values[:3], cfg.selected_models), *[gr.update(choices=names) for _ in range(3)])
            except (ValueError, OSError, RuntimeError) as exc:
                return None, f"Availability scan failed: {exc}", pd.DataFrame(), *[gr.update() for _ in range(3)]

        coverage_outputs = [coverage_state, coverage_status, coverage_table, *controls[:3]]
        scan.click(scan_records, inputs=[*controls, target_budgets, coverage_search, coverage_filter], outputs=coverage_outputs)
        for component in (coverage_search, coverage_filter, *controls[:3], model_selector):
            component.change(coverage_rows, inputs=[coverage_state, coverage_search, coverage_filter, *controls[:3], model_selector], outputs=coverage_table)
        for key, control in zip(fields, controls):
            if key.endswith("_start") or key.endswith("_end") or key in {"observation_budgets", "minimum_source_days", "warmup_days"}:
                control.change(lambda: (None, "Setup changed. Scan availability again for these dates and thresholds.",
                                         pd.DataFrame(), *[gr.update(choices=choices) for _ in range(3)]),
                               outputs=coverage_outputs)

        def check(*values):
            try:
                cfg = config(values)
                if executor:
                    executor.request("POST", "/case-studies/kan/validate",
                                     {"dataset_id": executor.dataset_id, "config": asdict(cfg)})
                else:
                    prepare_data(cfg, catalog)
                return "Cohorts are disjoint and daily coverage is complete. No training was started."
            except (ValueError, OSError, RuntimeError) as exc:
                return f"Coverage check failed: {exc}"

        def execute(*values, progress=gr.Progress()):
            token = CancellationToken()
            yield token, "Starting case study…", None, None, "", None, None
            try:
                cfg = config(values)
                callback = lambda f, message: progress(f, desc=message)
                if executor:
                    job = executor.request("POST", "/case-studies/kan",
                                           {"dataset_id": executor.dataset_id, "config": asdict(cfg)}, key=uuid.uuid4().hex)
                    job = executor.wait(job, callback, token)
                    if job["status"] == "cancelled":
                        raise StudyCancelled()
                    result = executor.request("GET", f"/jobs/{job['job_id']}/result")
                else:
                    result = run_case_study(cfg, catalog=catalog, progress=callback, cancellation_token=token)
                path = save_case_study(result, output_dir / uuid.uuid4().hex)
                frame = pd.DataFrame([{k: v for k, v in row.items()
                                       if k not in {"local_parameters", "low_flow_rmse", "high_flow_rmse", "fit_id",
                                                    "max_abs_water_balance_mm", "state_outside_source_range_fraction",
                                                    "calibration_first", "calibration_last", "capacity_mm",
                                                    "melt_mm_day_c", "runoff_logit_offset"}}
                                      for row in result["metrics"]])
                figure = px.line(frame, x="budget", y="rmse_mm_day", color="method", markers=True,
                                 facet_col="basin_id", line_group="seed", labels={"budget": "Local discharge observations"})
                formulas = "\n\n".join(f"Seed {e['seed']} ({', '.join(e.get('models', ['kan_pure', 'kan_tuned']))}): "
                                       f"F(storage, liquid, pet) = {e['logit_expression']}" for e in result["equations"])
                hardware = result.get("hardware", {})
                description = (f" Compute: {hardware['resolved']}; KAN fitting: {hardware['kan_fitting']}; symbolic extraction: {hardware['symbolic_extraction']}."
                               if hardware else "")
                yield token, "Case study complete." + description, frame, figure, formulas, result["audit"], str(path)
            except StudyCancelled:
                yield token, "Case study cancelled.", None, None, "", None, None
            except (ValueError, RuntimeError, OSError) as exc:
                yield token, f"Case study failed: {exc}", None, None, "", None, None

        target_budgets.change(lambda: (None, "Budget changed. Scan availability again to refresh budget readiness.", pd.DataFrame()),
                              outputs=[coverage_state, coverage_status, coverage_table])
        validate.click(check, inputs=[*controls, target_budgets], outputs=status)
        run.click(execute, inputs=[*controls, target_budgets], outputs=[token_state, status, metrics, curve, equations, audit, archive])
        cancel.click(lambda token: (token.cancel() if token else None, "Cancellation requested.")[1],
                     inputs=token_state, outputs=status, queue=False)
