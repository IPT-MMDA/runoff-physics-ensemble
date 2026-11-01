# Daily hydropower forecast comparison

Open Studio at `/#hydropower`, choose the shared catchment dataset, then choose Run A
and Run B. B is the economic baseline. Blank period bounds use all paired dates;
lead day defaults to 1. Click **Compare runs**. Change the map date and compare again
to inspect another day without changing the evaluation period. The existing
single-run flow monitoring remains below the comparison.

The comparison reads saved predictions and does not train models or require a
worker. Runs must contain `observed_mm_day`; monitoring-only archives without
observations cannot be used. Download comparison JSON to retain run IDs, selected
scenario, paired period, metrics, per-basin costs, map values and exclusion audit.

After updating the Python source, restart Studio/API before using the new
comparison. UI assets refresh immediately, but an already running Python process
retains its old routes. A plain `Not Found` from the comparison endpoint can mean
the backend has not loaded the update even when both runs appear in the selectors.

## Pairing and maps

- Both runs must resolve to the same registered dataset and the same benchmark
  mode, observation target, day timezone and release-state selection.
  Daily-maximum targets are rejected because the energy scenario requires daily means.
- Match by basin, target date, forecast issue date and selected lead day. Average
  predictions across seeds only inside this key. Duplicate seed/key rows and
  inconsistent seed observations are rejected. Different seed counts are allowed.
- Missing or nonfinite observations/predictions and negative observations are
  excluded. Different observations across runs are excluded, using relative and
  absolute tolerances of 1e-6 mm/day. The audit reports the counts. Negative finite
  predictions remain in accuracy metrics but energy conversion bounds flow at zero.
- Legacy one-day rows without issue metadata infer issue=target−1 day and lead=1;
  the audit counts these. Rolling runs require explicit, consistent issue/lead
  metadata. Unselected leads do not mix into the comparison.
- Full-resolution CAMELS-CH polygons are used for spatial matching. Each marker
  receives one stable basin assignment for the entire paired period: the smallest
  covering catchment among basins having paired data, geometry and a valid area.
  Both runs use that assignment. Basins without assigned inventory points are
  excluded from metrics and economics and counted in the audit.
- Basin runoff becomes gauge flow via `Q_m3_s = runoff_mm_day × area_km2 / 86.4`.
  This is a basin gauge estimate, not a verified turbine intake measurement.
- Maps show `prediction − observation` in m³/s for a common date. Blue means
  underprediction; red means overprediction; white means zero error; gray means no
  paired value on that date. A shared symmetric scale uses the maximum absolute
  error across both maps on the selected date. Pan and zoom are synchronized.
- Metrics use all paired basin-days in the requested period, once per basin/day.
  MAE/RMSE/bias are in m³/s. Mean basin NSE is the arithmetic mean of separately
  calculated basin scores; constant series have undefined NSE. Basin coverage can
  vary by date; per-basin day counts make the evaluated exposure visible.
  Relative MAE/RMSE reductions use B as baseline; NSE differences are score points.

## Energy and costs

The scenario represents one abstract run-of-river plant per uniquely assigned
basin, with identical parameters for A and B. Several plant markers can share the
same basin values; these copies are never counted again in totals. Nested basins
can share water, so this abstract cohort is not a physical fleet energy balance.

Let H be net head in metres, η combined turbine/generator efficiency, and Q the
daily mean basin flow in m³/s. Environmental flow and maximum turbine flow are
scenario inputs:

```
Q_turbine = min(Q_max, max(0, Q − Q_environmental))  # omit min when unlimited
P_MW     = 0.00981 × η × H × Q_turbine
E_MWh    = 24 × P_MW = 0.23544 × η × H × Q_turbine
cost     = shortfall_penalty × max(E_predicted − E_observed_flow, 0)
         + surplus_penalty × max(E_observed_flow − E_predicted, 0)
```

The constants derive from water density 1000 kg/m³ and gravity 9.81 m/s². Daily
mean flow is held constant over a 24-hour day, equivalent to 96 quarter-hours.
With constant head/efficiency and no nonlinear limits this preserves energy.
Flow caps, minimum flows, variable efficiency and real intraday operation make
daily averaging approximate. No intraday generation, reservoir dispatch, variable
head, outages or actual market settlement is reconstructed.

Defaults are H=100 m, η=0.9, zero environmental flow, unlimited turbine flow,
€50/MWh shortfall and €20/MWh surplus. These are illustrative assumptions, not
estimated properties of the mapped plants or empirically calibrated average
tariffs. The local inventory's `FallHeight` is not substituted silently: it is not
necessarily net head, and the underlying flow is still a catchment proxy.

Sum costs over the identical paired period. Report saving=`cost_B − cost_A`,
saving percentage=`100 × saving / cost_B`, and A's index=`100 × cost_A / cost_B`
with B=100. Percentage/index are null when B costs zero. Negative savings remain
visible. Totals are not annualized. There is no direct NSE-to-money conversion;
the price-weighted energy errors determine the scenario costs independently.

## API

`POST /api/v1/hydropower/compare` uses the normal API authentication and respects
the hydropower enabled setting. Example request:

```json
{
  "run_a": "RUN_A_ID",
  "run_b": "RUN_B_ID",
  "start": "2020-01-01",
  "end": "2020-12-31",
  "target_date": null,
  "lead_day": 1,
  "scenario": {
    "head_m": 100,
    "efficiency": 0.9,
    "environmental_flow_m3_s": 0,
    "max_turbine_flow_m3_s": null,
    "shortfall_eur_mwh": 50,
    "surplus_eur_mwh": 20
  }
}
```

The response includes `metrics_a/b`, `economics`, `basin_results`, `scenario`,
`accuracy_change`, `audit`, `available_dates`, `target_date` and a GeoJSON `map` with both runs' errors.
Empty cohorts, incompatible observation definitions and invalid date selections
return explanatory validation errors. Missing map assets leave the tables usable.

## Scientific context

- [NREL, Electrical Systems of Pumped Storage Hydropower](https://www.nrel.gov/docs/fy21osti/74721.pdf)
  documents the water-power equation. The equation is physical; our daily flow,
  head, efficiency and tariff assumptions define the simplified scenario.
- [Wergeland et al. (2026)](https://doi.org/10.1016/j.renene.2026.126147)
  evaluate forecast-related imbalance costs for run-of-river hydropower. Their
  reported 21% fee reduction is context, not a coefficient used here.
- [Anghileri et al. (2019)](https://doi.org/10.1029/2019WR025280)
  evaluate forecast value in the Swiss Verzasca system via operational simulation.
  Their reservoir-revenue experiment differs from this abstract daily cost model.
- [Swissgrid imbalance data](https://www.swissgrid.ch/en/home/customers/topics/bgm/balance-energy.html)
  are a possible future pricing source. The current component deliberately uses
  editable positive penalties; actual signed settlement requires matching market
  rules, prices and energy profiles.
