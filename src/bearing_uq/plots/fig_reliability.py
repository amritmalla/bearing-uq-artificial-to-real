"""Figure 2: reliability diagrams, unseen artificial bearings vs real damage (raw probabilities)."""

import matplotlib.pyplot as plt
import numpy as np

from bearing_uq import metrics as M
from bearing_uq.plots import style
from bearing_uq.plots.curves import reliability_bins


def draw(in_domain: dict, pred: dict):
    raw = pred["calibrators"].index("raw")
    fig, axes = plt.subplots(1, len(pred["models"]), figsize=(7.0, 2.6), sharey=True)
    for ax, (mi, model) in zip(axes, enumerate(pred["models"])):
        ax.plot([0, 1], [0, 1], color=style.INK_MUTED, linewidth=1, linestyle=":", zorder=1)

        probs, y = in_domain[model]
        real_p = pred["probs"][:, mi, raw].reshape(-1, probs.shape[1])  # pool all rotations
        real_y = np.tile(pred["y"], pred["probs"].shape[0])
        ece_text = []
        for key, (p, yy) in {"in_domain": (probs, y), "real": (real_p, real_y)}.items():
            b = reliability_bins(p, yy)
            ax.plot(b[:, 0], b[:, 1], color=style.DOMAIN_COLOR[key], linestyle=style.DOMAIN_STYLE[key],
                    marker="o" if key == "in_domain" else "s", markersize=4, label=style.DOMAIN_LABEL[key])
            ece_text.append(f"{'Artificial' if key == 'in_domain' else 'Real'}: {M.ece(p, yy):.2f}")

        ax.text(0.97, 0.04, "ECE\n" + "\n".join(ece_text), transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.5, color=style.INK_2)
        ax.set_title(style.MODEL_LABEL[model])
        ax.set_xlim(0.3, 1.0)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Predicted confidence")
    axes[0].set_ylabel("Observed accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig
