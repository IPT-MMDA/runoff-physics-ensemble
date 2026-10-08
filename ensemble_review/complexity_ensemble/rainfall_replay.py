"""Data-backed rainfall–runoff replay for the Gradio application."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import html
from pathlib import Path
import tempfile

from .hydrology_data import HydrologySeries


@dataclass(frozen=True)
class Replay:
    basin_id: str
    dates: tuple[date, ...]
    precipitation: tuple[float, ...]
    pet: tuple[float, ...]
    temperature: tuple[float, ...]
    observed: tuple[float, ...]
    predicted: tuple[float, ...]
    context_days: int
    storage: tuple[float, ...]
    illustrative_evap: tuple[float, ...]
    illustrative_runoff: tuple[float, ...]
    previous_input_flow: tuple[float, ...]
    lead_days: tuple[int, ...]
    issue_dates: tuple[date | None, ...]
    extra_features: tuple[str, ...]
    static_feature_count: int
    forecast_mode: str


def illustrative_bucket(precipitation: tuple[float, ...],
                        pet: tuple[float, ...], *, release_fraction: float = 0.30
                        ) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Uncalibrated teaching model, in basin-average millimetres per day."""
    storage = 0.0
    stores, evaporation, runoff = [], [], []
    for rain, demand in zip(precipitation, pet):
        available = storage + max(rain, 0.0)
        actual_evap = min(max(demand, 0.0), available)
        available -= actual_evap
        release = release_fraction * available
        storage = available - release
        stores.append(storage)
        evaporation.append(actual_evap)
        runoff.append(release)
    return tuple(stores), tuple(evaporation), tuple(runoff)


def event_choices(series: HydrologySeries, predictions: list[object], *,
                  context_days: int = 30, event_days: int = 14,
                  limit: int = 8) -> list[tuple[str, str]]:
    """Choose the largest distinct rainfall episodes with prediction coverage."""
    prediction_dates = {row.target_date for row in predictions
                        if row.basin_id == series.basin_id}
    index = {day: i for i, day in enumerate(series.dates)}
    rainfall = series.forcings[:, series.forcing_names.index("precipitation")]
    candidates = []
    for day in prediction_dates:
        i = index.get(day)
        if i is None or i < context_days or i + event_days > len(series.dates):
            continue
        dates = series.dates[i - context_days:i + event_days]
        if dates[-1] - dates[0] != timedelta(days=len(dates) - 1):
            continue
        if any(item not in prediction_dates for item in dates[context_days:]):
            continue
        score = float(rainfall[i - 3:i + 4].sum())
        candidates.append((score, day))
    selected: list[date] = []
    for _, day in sorted(candidates, reverse=True):
        if all(abs((day - other).days) >= event_days for other in selected):
            selected.append(day)
        if len(selected) == limit:
            break
    return [(f"{day.isoformat()} · rainfall episode", day.isoformat())
            for day in selected]


def make_replay(series: HydrologySeries, predictions: list[object],
                start: date | str, *, context_days: int = 30,
                event_days: int = 14, extra_features: tuple[str, ...] = (),
                static_feature_count: int = 0,
                forecast_mode: str = "one_day") -> Replay:
    start = date.fromisoformat(start) if isinstance(start, str) else start
    by_date: dict[date, list[float]] = {}
    for row in predictions:
        if row.basin_id == series.basin_id:
            by_date.setdefault(row.target_date, []).append(row.predicted_mm_day)
    indices = {day: i for i, day in enumerate(series.dates)}
    index = indices.get(start)
    if index is None or index < context_days or index + event_days > len(series.dates):
        raise ValueError("The selected episode has insufficient daily data")
    sl = slice(index - context_days, index + event_days)
    days = series.dates[sl]
    if days[-1] - days[0] != timedelta(days=len(days) - 1):
        raise ValueError("The selected episode contains missing dates")
    if any(day not in by_date for day in days[context_days:]):
        raise ValueError("The selected episode has missing predictions")
    forcing = series.forcings[sl]
    def column(name: str) -> tuple[float, ...]:
        return tuple(float(x) for x in forcing[:, series.forcing_names.index(name)])
    rain, pet = column("precipitation"), column("pet")
    storage, evaporation, runoff = illustrative_bucket(rain, pet)
    prior_predictions = {
        (getattr(row, "seed", 0), getattr(row, "issue_date", None), row.target_date):
        row.predicted_mm_day for row in predictions if row.basin_id == series.basin_id
    }
    previous_input_flow, lead_days, issue_dates = [], [], []
    for offset, day in enumerate(days):
        rows = [row for row in predictions if row.basin_id == series.basin_id
                and row.target_date == day]
        lead = int(getattr(rows[0], "lead_day", 1)) if rows else 1
        issue = getattr(rows[0], "issue_date", None) if rows else None
        lead_days.append(lead)
        issue_dates.append(issue)
        if lead > 1:
            feedback = [prior_predictions[(getattr(row, "seed", 0), issue,
                                           day - timedelta(days=1))]
                        for row in rows
                        if (getattr(row, "seed", 0), issue, day - timedelta(days=1))
                        in prior_predictions]
            previous_input_flow.append(sum(feedback) / len(feedback) if feedback else float("nan"))
        else:
            previous_input_flow.append(float(series.discharge[index - context_days + offset - 1])
                                       if offset else float("nan"))
    return Replay(
        series.basin_id, days, rain, pet, column("temperature"),
        tuple(float(x) for x in series.discharge[sl]),
        tuple(float("nan") if i < context_days else
              sum(by_date[day]) / len(by_date[day]) for i, day in enumerate(days)),
        context_days,
        storage, evaporation, runoff,
        tuple(previous_input_flow), tuple(lead_days), tuple(issue_dates),
        extra_features, static_feature_count, forecast_mode,
    )


def replay_figures(replay: Replay, day_number: int):
    """Four independent views controlled by Gradio's native day slider."""
    import plotly.graph_objects as go

    count = len(replay.dates) - replay.context_days
    if not 1 <= day_number <= count:
        raise ValueError(f"day number must be between 1 and {count}")
    i = replay.context_days + day_number - 1
    day = replay.dates[i]

    def finish(fig, title, *, marker_day=None):
        fig.update_layout(
            title=title, height=325, margin=dict(l=65, r=25, t=65, b=85),
            legend=dict(orientation="h", x=0, y=-0.32, xanchor="left", yanchor="top"),
        )
        fig.add_vline(x=(marker_day or day).isoformat(),
                      line_dash="dash", line_color="#475569")
        fig.update_xaxes(title="Date", tickformat="%d %b", nticks=7)
        return fig

    weather = go.Figure()
    weather.add_bar(x=replay.dates, y=replay.precipitation, name="Rainfall",
                    marker_color="#3b82f6")
    weather.add_scatter(x=replay.dates, y=replay.pet, name="PET",
                        line=dict(color="#e39a35"))
    weather.update_yaxes(title="mm/day")
    finish(weather, "1 · Historical weather · dashed = last input day",
           marker_day=replay.dates[i - 1])

    discharge = go.Figure()
    discharge.add_scatter(x=replay.dates, y=replay.observed, name="Observed",
                          line=dict(color="#123b5d"))
    discharge.add_scatter(x=replay.dates[replay.context_days:],
                          y=replay.predicted[replay.context_days:],
                          name="Predicted (seed mean)", line=dict(color="#e66c42"))
    discharge.add_scatter(x=[day], y=[replay.predicted[i]], mode="markers",
                          marker=dict(color="#e66c42", size=12), showlegend=False)
    discharge.update_yaxes(title="Discharge (mm/day)")
    finish(discharge, "2 · Next-day discharge at the gauge")

    bucket = go.Figure()
    bucket.add_scatter(x=replay.dates, y=replay.storage, name="Storage",
                       line=dict(color="#5c8158"))
    bucket.add_scatter(x=replay.dates, y=replay.illustrative_evap,
                       name="Evaporation", line=dict(color="#e39a35"))
    bucket.add_scatter(x=replay.dates, y=replay.illustrative_runoff,
                       name="Stream release", line=dict(color="#1682b3"))
    bucket.update_yaxes(title="Water depth (mm)")
    finish(bucket, "3 · Separate teaching bucket (uncalibrated)")

    return weather, discharge, bucket, forecast_diagram(replay, day_number)


def forecast_snapshot(replay: Replay, day_number: int) -> dict[str, object]:
    """Summarize only the dates available when the target is predicted."""
    count = len(replay.dates) - replay.context_days
    if not 1 <= day_number <= count:
        raise ValueError(f"day number must be between 1 and {count}")
    i = replay.context_days + day_number - 1
    first = i - replay.context_days
    return {
        "target_date": replay.dates[i],
        "start_date": replay.dates[first],
        "end_date": replay.dates[i - 1],
        "rain_total": sum(replay.precipitation[first:i]),
        "pet_total": sum(replay.pet[first:i]),
        "temperature_mean": sum(replay.temperature[first:i]) / replay.context_days,
        "previous_flow": replay.previous_input_flow[i],
        "previous_flow_kind": "Predicted feedback" if replay.lead_days[i] > 1 else "Observed",
        "predicted": replay.predicted[i],
        "observed": replay.observed[i],
        "lead_day": replay.lead_days[i],
        "issue_date": replay.issue_dates[i],
    }


def forecast_diagram(replay: Replay, day_number: int) -> str:
    """A dated, data-backed handoff from input window to target discharge."""
    item = forecast_snapshot(replay, day_number)
    target = item["target_date"].isoformat()
    start = item["start_date"].isoformat()
    end = item["end_date"].isoformat()
    previous = item["previous_flow"]
    previous_text = f"{previous:.2f} mm/day" if previous == previous else "unavailable"
    mode = (f"Rolling hindcast · lead {item['lead_day']}"
            if replay.forecast_mode == "rolling" else "One-day hindcast")
    issue = item["issue_date"]
    issue_text = f"Issued {issue.isoformat()}" if issue else "Observed flow through prior day"
    extra = ", ".join(replay.extra_features) if replay.extra_features else "none"
    weather_note = ("Rolling mode uses historical weather after the issue date."
                    if replay.forecast_mode == "rolling" else
                    "The input window ends before the target day.")
    card = "border:1px solid #cbdde8;border-radius:16px;padding:20px;box-shadow:0 3px 12px #17324d14;"
    return f'''<section style="background:#f5f9fc;border:1px solid #cbdde8;border-radius:16px;padding:20px;color:#17324d;font-family:system-ui,sans-serif;overflow-x:auto"
      role="region" aria-label="Input window {start} through {end} produces forecast for {target}">
      <div style="display:flex;justify-content:space-between;gap:12px;align-items:baseline;margin-bottom:16px">
        <strong style="font-size:20px">Basin {html.escape(replay.basin_id)} · forecast for {target}</strong>
        <span style="color:#52718a;white-space:nowrap">{html.escape(mode)}</span>
      </div>
      <div style="display:grid;grid-template-columns:minmax(270px,1.4fr) minmax(150px,.7fr) minmax(245px,1fr);gap:16px;min-width:740px;align-items:stretch">
        <div style="{card}background:linear-gradient(135deg,#eaf5fc,#fff)">
          <strong style="color:#1d5d82">1 · INPUT WINDOW · {replay.context_days} DAYS</strong>
          <div style="font-size:13px;color:#52718a;margin:6px 0 16px">{start} through {end}</div>
          <div style="display:grid;grid-template-columns:1fr auto;gap:10px;font-size:14px">
            <span>Rainfall total</span><strong style="color:#235d9d">{item['rain_total']:.1f} mm</strong>
            <span>PET total</span><strong style="color:#b66f1d">{item['pet_total']:.1f} mm</strong>
            <span>Mean temperature</span><strong>{item['temperature_mean']:.1f} °C</strong>
            <span>{html.escape(item['previous_flow_kind'])} flow on {end}</span><strong>{previous_text}</strong>
          </div>
          <div style="font-size:12px;color:#52718a;margin-top:17px">Also uses the daily discharge sequence, seasonal encoding and configured basin predictors.</div>
        </div>
        <div style="{card}background:#eaf0ff;display:flex;flex-direction:column;justify-content:center;text-align:center;gap:10px">
          <div style="font-size:28px;color:#657da8" aria-hidden="true">→</div>
          <strong style="color:#334e91">2 · TRAINED ENSEMBLE</strong>
          <small style="color:#4b618e">{html.escape(mode)}<br>{html.escape(issue_text)}</small>
          <div style="font-size:28px;color:#657da8" aria-hidden="true">→</div>
        </div>
        <div style="{card}background:linear-gradient(135deg,#fff0e7,#fff)">
          <strong style="color:#b85d3c">3 · PREDICTION TARGET</strong>
          <div style="font-size:13px;color:#755e52;margin:6px 0 16px">Gauge discharge on {target}</div>
          <div style="display:grid;grid-template-columns:1fr auto;gap:12px;align-items:baseline">
            <span>Predicted · seed mean</span><strong style="font-size:22px;color:#d36a43">{item['predicted']:.2f}</strong>
            <span>Observed · evaluation only</span><strong style="font-size:18px">{item['observed']:.2f}</strong>
          </div>
          <div style="text-align:right;font-size:12px;color:#755e52">mm/day</div>
          <div style="font-size:12px;color:#755e52;margin-top:16px">Target-day weather is not used for this prediction.</div>
        </div>
      </div>
      <div style="font-size:12px;color:#52718a;margin-top:14px">
        Additional selected features: {html.escape(extra)} · Static basin descriptors: {replay.static_feature_count}.
        {html.escape(weather_note)}
      </div>
    </section>'''


def export_gif(replay: Replay) -> Path:
    """Export a compact chart replay; Pillow is needed only for this action."""
    from .visualization import _pyplot
    plt = _pyplot()
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
        from matplotlib.patches import FancyBboxPatch
    except ImportError as error:
        raise RuntimeError("GIF export requires matplotlib and Pillow") from error
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    weather, flow = axes[0]
    bucket, scene = axes[1]
    first = replay.context_days
    x = list(range(len(replay.dates)))
    weather.bar(x, replay.precipitation, color="#3b82f6", label="Rainfall")
    weather.plot(x, replay.pet, color="#e39a35", label="PET")
    weather.set_title("1 · Historical weather · dashed = last input day")
    weather.set_ylabel("mm/day")
    flow.plot(x, replay.observed, color="#123b5d", label="Observed")
    flow.plot(x[first:], replay.predicted[first:], color="#e66c42", label="Predicted")
    flow.set_title("2 · Next-day discharge at the gauge")
    flow.set_ylabel("Discharge (mm/day)")
    bucket.plot(x, replay.storage, color="#5c8158", label="Storage")
    bucket.plot(x, replay.illustrative_evap, color="#e39a35", label="Evaporation")
    bucket.plot(x, replay.illustrative_runoff, color="#1682b3", label="Stream release")
    bucket.set_title("3 · Separate teaching bucket (uncalibrated)")
    bucket.set_ylabel("Water depth (mm)")
    ticks = list(range(0, len(x), 10))
    for axis in (weather, flow, bucket):
        axis.set_xlim(-0.5, len(x) - 0.5)
        axis.set_xticks(ticks, [replay.dates[i].strftime("%d %b") for i in ticks])
        axis.set_xlabel("Date")
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25),
                    ncol=2 if axis is not bucket else 3, fontsize=8,
                    frameon=False)
    def draw_handoff(day_number):
        item = forecast_snapshot(replay, day_number)
        scene.clear()
        scene.set(xlim=(0, 1), ylim=(0, 1))
        scene.axis("off")
        scene.set_title("4 · Inputs → next-day prediction")
        for y, height, color in ((0.61, 0.36, "#eaf5fc"),
                                 (0.40, 0.16, "#eaf0ff"),
                                 (0.04, 0.31, "#fff0e7")):
            scene.add_patch(FancyBboxPatch((0.03, y), 0.94, height,
                            boxstyle="round,pad=0.015", facecolor=color,
                            edgecolor="#cbdde8"))
        scene.text(0.07, 0.91, f"INPUT · {replay.context_days} preceding days",
                   weight="bold", color="#1d5d82", fontsize=10)
        scene.text(0.07, 0.84,
                   f"{item['start_date'].isoformat()} to {item['end_date'].isoformat()}",
                   fontsize=9)
        scene.text(0.07, 0.76,
                   f"Rain {item['rain_total']:.1f} mm  ·  PET {item['pet_total']:.1f} mm"
                   f"  ·  Mean temp {item['temperature_mean']:.1f} °C", fontsize=9)
        previous = item["previous_flow"]
        previous_text = f"{previous:.2f}" if previous == previous else "unavailable"
        scene.text(0.07, 0.67,
                   f"{item['previous_flow_kind']} prior flow: {previous_text} mm/day",
                   fontsize=9)
        scene.text(0.50, 0.48, "↓   TRAINED ENSEMBLE   ↓",
                   ha="center", va="center", weight="bold", color="#334e91", fontsize=10)
        scene.text(0.07, 0.27,
                   f"TARGET · gauge discharge on {item['target_date'].isoformat()}",
                   weight="bold", color="#b85d3c", fontsize=10)
        scene.text(0.07, 0.17,
                   f"Predicted {item['predicted']:.2f} mm/day  ·  "
                   f"Observed {item['observed']:.2f} mm/day", fontsize=10)
        scene.text(0.07, 0.08, "Target-day weather is outside the input window.",
                   color="#755e52", fontsize=8)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.91, bottom=0.10,
                        hspace=0.60, wspace=0.27)
    cursors = [axis.axvline(first - 1 if axis is weather else first,
                           color="#475569", linestyle="--")
               for axis in (weather, flow, bucket)]
    def update(offset):
        day = first + offset
        for axis, cursor in zip((weather, flow, bucket), cursors):
            cursor_day = day - 1 if axis is weather else day
            cursor.set_xdata([cursor_day, cursor_day])
        draw_handoff(offset + 1)
        fig.suptitle(f"Basin {replay.basin_id} · {replay.dates[day].isoformat()}")
        return cursors
    animation = FuncAnimation(fig, update, frames=len(x) - first, interval=450)
    with tempfile.NamedTemporaryFile(suffix=".gif", prefix="rainfall-replay-", delete=False) as target:
        path = Path(target.name)
    try:
        animation.save(path, writer=PillowWriter(fps=2))
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        plt.close(fig)
    return path
