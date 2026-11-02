"""Shared matplotlib style: validated categorical palette, thin marks, 300-dpi vector + raster output."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..paths import FIGURES, ensure  # noqa: E402

# categorical slots 1-3 (+7) of the validated reference palette, in fixed order (never cycled)
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
SERIES = [BLUE, ORANGE, AQUA, VIOLET]
MARKERS = ["o", "s", "^", "D"]  # secondary encoding: identity never relies on colour alone
LINESTYLES = ["-", "--", "-.", ":"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
SEQ = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
]  # sequential blue


def apply() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, name: str, directory: Path = FIGURES) -> None:
    ensure(directory)
    fig.savefig(directory / f"{name}.pdf")
    fig.savefig(directory / f"{name}.png")
    plt.close(fig)


apply()
