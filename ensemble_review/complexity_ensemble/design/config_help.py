"""Contextual help for experiment configuration controls."""


_CONFIG_HELP = {
    "dates": (
        "Train, validation, and test dates",
        "Choose three ordered, non-overlapping date ranges. Training fits the models; "
        "validation is used for model selection or blend weights where applicable; "
        "test dates are held out for the reported evaluation.",
    ),
    "approach": (
        "Training and prediction approach",
        "Soft routing blends the two experts using a complexity score; hard routing "
        "selects one. Distillation trains a complex teacher, transfers its predictions "
        "to the simple expert, then jointly fine-tunes. No routing uses only the "
        "complex expert. Static 50/50 averages independently trained experts. "
        "OOD fallback uses the simple expert for inputs flagged as out of distribution. "
        "Stacking learns a fixed blend weight from validation data and requires "
        "prediction targets in the training scope.",
    ),
    "complex_expert": (
        "Complex expert",
        "Mamba processes the input history as a sequence. MLP uses a feed-forward "
        "network on the flattened history. Both predict daily discharge and can "
        "use basin attributes.",
    ),
    "simple_expert": (
        "Simple expert",
        "RBF uses radial basis functions; Fourier uses sinusoidal features. Both "
        "are smaller alternatives to the complex expert and predict daily discharge.",
    ),
    "physics_backbone": (
        "Physics backbone",
        "A linear reservoir water balance relates the change in discharge to "
        "effective precipitation and previous discharge. Its residual contributes "
        "to the physics loss during training.",
    ),
    "physics_optimization": (
        "Physics optimization",
        "The default uses the linear reservoir alone. KAN adds a learned correction "
        "to the reservoir balance, selected for accuracy, robustness, a balance of "
        "both, or agreement with a teacher model. KAN modes require target basins "
        "in the training scope and the optional pykan dependency.",
    ),
    "physics_teacher": (
        "Physics-distillation teacher",
        "This approach supplies target predictions when Physics optimization is "
        "set to KAN — physics distillation. The choice has no effect for other "
        "physics modes.",
    ),
}


def config_info(gr, setting: str):
    """Render the same information popup used for dataset help."""
    title, explanation = _CONFIG_HELP[setting]
    button = gr.Button("ⓘ", size="sm", variant="secondary", min_width=40, scale=0)
    button.click(
        lambda: gr.Info(explanation, title=title, duration=None),
        inputs=None,
        outputs=None,
        show_progress="hidden",
    )
    return button
