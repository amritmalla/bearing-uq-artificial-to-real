"""Figure 1: risk-coverage curves, unseen artificial bearings vs real damage."""

import matplotlib.pyplot as plt
import numpy as np

from bearing_uq.plots import style
from bearing_uq.plots.curves import risk_on_grid

GRID = np.linspace(0.02, 1.0, 99)


def draw(in_domain: dict, pred: dict):
    """in_domain: {model: (probs, y)}; pred: saved real-damage predictions (all rotations)."""
    raw = pred["calibrators"].index("raw")
    fig, axes = plt.subplots(1, len(pred["models"]), figsize=(7.0, 2.5), sharey=True)
    for ax, (mi, model) in zip(axes, enumerate(pred["models"])):
        probs, y = in_domain[model]
        ax.plot(GRID, risk_on_grid(probs.max(1), probs.argmax(1) == y, GRID),
                color=style.DOMAIN_COLOR["in_domain"], linestyle=style.DOMAIN_STYLE["in_domain"],
                label=style.DOMAIN_LABEL["in_domain"])

        curves = np.array([risk_on_grid(p.max(1), p.argmax(1) == pred["y"], GRID)
                           for p in pred["probs"][:, mi, raw]])
        lo, mid, hi = np.percentile(curves, [25, 50, 75], axis=0)
        ax.fill_between(GRID, lo, hi, color=style.DOMAIN_COLOR["real"], alpha=0.18, linewidth=0)
        ax.plot(GRID, mid, color=style.DOMAIN_COLOR["real"], linestyle=style.DOMAIN_STYLE["real"],
                label=style.DOMAIN_LABEL["real"] + " (median, IQR)")

        ax.axhline(0.05, color=style.INK_MUTED, linewidth=1, linestyle=":", zorder=1, label="5% error target")
        ax.set_title(style.MODEL_LABEL[model])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.5)
        ax.set_xlabel("Share of cases automated")
    axes[0].set_ylabel("Error among automated cases")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig
