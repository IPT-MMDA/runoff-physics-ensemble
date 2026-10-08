"""Short, contextual explanations for the data sources."""


_DATASET_HELP = {
    "camels": (
        "CAMELS-CH daily hydrology",
        "Provides the required daily hydrology inputs, observed discharge, and basin "
        "attributes. The model predicts daily discharge for the selected basins.",
        "This is the baseline for every run. Recent observed discharge is also used "
        "as an input; this dataset cannot be turned off.",
    ),
    "estreams": (
        "EStreams vegetation and snow",
        "Adds the selected leaf area index, vegetation index, and snow-cover "
        "measurements as predictors. Monthly values are used only from their "
        "observation date onward; missing values are marked for the model.",
        "The target remains daily discharge. Extra predictors may change the "
        "forecasts and metrics, but improvement depends on data coverage and is "
        "not guaranteed.",
    ),
    "camels-chem": (
        "CAMELS-CH-Chem catchment pressures",
        "Adds selected agriculture, livestock, atmospheric deposition, and rain "
        "isotope measurements as predictors. Annual measurements become "
        "available only after their source year ends.",
        "The target remains daily discharge; the app does not predict water "
        "chemistry. Forecasts and metrics may change, depending on coverage, "
        "without a guaranteed improvement.",
    ),
}


def dataset_info(gr, dataset: str):
    """Render an information button backed by Gradio's own popup."""
    title, provides, impact = _DATASET_HELP[dataset]
    button = gr.Button(
        "ⓘ",
        size="sm",
        variant="secondary",
        min_width=40,
        scale=0,
    )
    button.click(
        lambda: gr.Info(f"{provides}\n\n{impact}", title=title, duration=None),
        inputs=None,
        outputs=None,
        show_progress="hidden",
    )
    return button
