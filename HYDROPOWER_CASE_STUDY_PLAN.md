# Hydropower run comparison

Implement a daily, abstract run-of-river case study inside Studio Hydropower.

1. Add a read-only comparison API. Require the same registered dataset and observation definition. Pair predictions by basin, target, issue and selected lead; average seeds only within those keys. Exclude missing/nonfinite or inconsistent observations with counts. Legacy one-day runs may infer issue/lead; rolling runs require explicit metadata.
2. Use full-resolution catchment geometry and a stable common basin assignment for both runs. Show two maps of prediction minus observation using one symmetric scale. Keep existing monitoring available.
3. Calculate MAE/RMSE and mean basin NSE on the identical paired period. Convert daily mean flow to daily energy with E=0.23544*efficiency*head*turbine_flow, applying optional environmental flow and turbine-flow limits. Model asymmetric positive shortfall/surplus penalties. All defaults are illustrative, editable assumptions.
4. Summarize one abstract plant per uniquely mapped basin, never one copy per inventory marker. Report cost A/B, savings versus B, and B=100 index; zero denominators are unavailable. Keep negative savings visible. Explain flat daily profiles, gauge-flow proxies and the absence of actual generation/settlement data. Link scientific and market references.
5. Verify numerical formulas, paired cohorts, incompatible observations, issue/lead alignment, exclusions, zero costs, API authentication/validation and browser state/races. Exercise a real saved-run comparison.

Acceptance: selecting A/B and a period renders both errors against shared observations, offers a map date within that period, and produces reproducible scenario economics without claiming a universal NSE-to-money conversion. No training or worker is required.

Implemented in Studio and `POST /api/v1/hydropower/compare`. Validation: 28 Python
unit/API tests and all 49 Studio Node tests passed. A real HTTP-adapter comparison
using saved runs `20261021T124615Z-0bae752b7d-dce609` and
`20261021T120158Z-02ac9faeba-119525` evaluated 40,728 basin-days across 112 basins in
2020, with 700 paired plant markers. The exported example is at
`artifacts/hydropower_case_study/example_comparison.json`. Browser visual review
could not run because this session forbids binding a local server socket; DOM,
map rendering and synchronization were exercised with Node fixtures.
