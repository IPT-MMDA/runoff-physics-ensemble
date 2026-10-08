"""Learned-Morse hydrology comparison with disjoint expert training data."""

from pathlib import Path

from complexity_ensemble.hydrology_hard_routing_consolidation_comparison import (
    compare_hydrology_models_with_hard_routing_and_consolidation,
)
from examples.compare_learned_morse_hydrology_consolidation import main


if __name__ == "__main__":
    main(
        compare_hydrology_models_with_hard_routing_and_consolidation,
        description=(
            "Compare selectable hydrology routing models with hard expert-data "
            "assignment and isolated boundary consolidation"
        ),
        default_output_dir=Path(
            "artifacts/hydrology/learned_morse_hard_routing_consolidation"
        ),
        stem_suffix="_hard_routing",
        result_label="hard-routing consolidation",
    )
