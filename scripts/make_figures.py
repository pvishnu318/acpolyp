"""Regenerate every figure and LaTeX table from the raw run logs (300-dpi vector PDFs + PNGs, booktabs tables).

    python scripts/make_figures.py

Runs, in order: EDA figures, contour-quality figures, the aggregate analysis (tables with bold-best and +/- SD,
ablation / calibration / error / learning-curve figures), the website export and the paper macros.
Interpretability and robustness figures come from scripts/interpretability.py and scripts/robustness2.py (GPU).
"""

from __future__ import annotations

import runpy
from pathlib import Path

HERE = Path(__file__).parent
STEPS = ["eda.py", "ac_figures.py", "analyze.py", "export_results.py", "make_paper_assets.py"]

if __name__ == "__main__":
    for s in STEPS:
        print(f"== {s}")
        runpy.run_path(str(HERE / s), run_name="__main__")
