from __future__ import annotations

import os
from pathlib import Path
import tempfile


def _pyplot():
    cache = Path(tempfile.gettempdir()) / "complexity-ensemble-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib.pyplot as plt

    return plt


def _finish_figure(figure: object, path: str | Path) -> None:
    plt = _pyplot()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
