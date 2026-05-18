from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

CUSTOM_COLORS = [
    "#AB47BC",
    "#FFA726",
    "#29B6F6",
    "#EC407A",
    "#66BB6A",
    "#EF5350",
    "#81D4FA",
    "#FFCC80",
]

HEATMAP_COLORS = ["#FCE4EC", "#F8BBD0", "#E1BEE7", "#90CAF9", "#1E88E5"]
CMAP_SUNSET = LinearSegmentedColormap.from_list("SunsetGradient", HEATMAP_COLORS)


def apply_theme() -> None:
    """Apply the palette and Matplotlib style from Experimentplan/testplot.py."""
    sns.set_palette(CUSTOM_COLORS)
    sns.set_style(
        "whitegrid", {"grid.linestyle": "--", "axes.edgecolor": ".85", "grid.color": ".9"}
    )
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
