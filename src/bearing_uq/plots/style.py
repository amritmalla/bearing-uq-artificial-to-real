"""Shared figure style: validated 3-colour palette, marker shapes for grayscale print, recessive axes."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference palette slots 1-3 (validated all-pairs for colour-vision deficiency, light mode).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"

MODEL_LABEL = {"random_forest": "Random Forest", "svm": "SVM", "xgboost": "XGBoost"}
MODEL_COLOR = {"random_forest": BLUE, "svm": ORANGE, "xgboost": AQUA}
MODEL_MARKER = {"random_forest": "o", "svm": "s", "xgboost": "^"}

DOMAIN_COLOR = {"in_domain": BLUE, "real": ORANGE}
DOMAIN_STYLE = {"in_domain": "-", "real": "--"}
DOMAIN_LABEL = {"in_domain": "Unseen artificial bearings", "real": "Real damage"}

CLASS_LABEL = {"healthy": "Healthy", "inner_race": "Inner race", "outer_race": "Outer race"}
CLASS_COLOR = {"healthy": BLUE, "inner_race": ORANGE, "outer_race": AQUA}
CLASS_MARKER = {"healthy": "o", "inner_race": "s", "outer_race": "^"}

FIG_DIR = Path("results/figures")


def apply() -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.edgecolor": INK_MUTED, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "lines.linewidth": 1.6, "lines.markersize": 5,
        "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def target_line(ax, y: float, label: str) -> None:
    """Dotted reference line, labelled at the left edge where curves start near zero."""
    ax.axhline(y, color=INK_MUTED, linewidth=1, linestyle=":", zorder=1)
    ax.annotate(label, xy=(0, y), xycoords=("axes fraction", "data"), xytext=(3, 3),
                textcoords="offset points", ha="left", va="bottom", fontsize=7.5, color=INK_2)


def save(fig, name: str) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = [FIG_DIR / f"{name}.{ext}" for ext in ("png", "pdf")]
    for p in paths:
        # no creation timestamp, so re-running gives byte-identical PDFs
        fig.savefig(p, metadata={"CreationDate": None} if p.suffix == ".pdf" else None)
    plt.close(fig)
    return paths
