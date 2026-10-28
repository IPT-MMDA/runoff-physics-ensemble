"""Launch this checkout's Gradio app, avoiding a stale globally installed copy."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = PROJECT_ROOT / "ensemble_review"
if not PACKAGE_ROOT.is_dir():
    raise SystemExit(f"Cannot find project package at {PACKAGE_ROOT}")
# Put this archive's source first so a previously installed package cannot shadow it.
sys.path.insert(0, str(PACKAGE_ROOT))

from complexity_ensemble.app import main  # noqa: E402

if __name__ == "__main__":
    main()
