"""Figure 4: per-bearing mean confidence vs accuracy on real damage (averaged over rotations)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bearing_uq import bearings as B
from bearing_uq.plots import style

LABEL_BELOW_ACCURACY = 0.5


def per_bearing(pred: dict) -> pd.DataFrame:
    raw = pred["calibrators"].index("raw")
    rows = []
    for mi, model in enumerate(pred["models"]):
        p = pred["probs"][:, mi, raw]                      # (R, N, K)
        correct = p.argmax(2) == pred["y"][None, :]
        conf = p.max(2)
        for code in np.unique(pred["bearing"]):
            m = pred["bearing"] == code
            rows.append({"model": model, "bearing": code, "label": B.get(code).label,
                         "accuracy": correct[:, m].mean(), "mean_confidence": conf[:, m].mean()})
    return pd.DataFrame(rows)


def draw(pred: dict):
    data = per_bearing(pred)
    fig, axes = plt.subplots(1, len(pred["models"]), figsize=(7.0, 2.7), sharey=True)
    for ax, model in zip(axes, pred["models"]):
        ax.plot([0, 1], [0, 1], color=style.INK_MUTED, linewidth=1, linestyle=":", zorder=1)
        d = data[data.model == model]
        for label in B.CLASSES:
            g = d[d.label == label]
            ax.scatter(g.mean_confidence, g.accuracy, s=28, color=style.CLASS_COLOR[label],
                       marker=style.CLASS_MARKER[label], edgecolor="white", linewidth=0.8,
                       label=style.CLASS_LABEL[label], zorder=3)
        low = d[d.accuracy < LABEL_BELOW_ACCURACY].sort_values("accuracy")
        for i, row in enumerate(low.itertuples()):
            # alternate the vertical offset so labels of nearby points don't touch
            ax.annotate(row.bearing, (row.mean_confidence, row.accuracy), xytext=(5, -7 if i % 2 == 0 else 3),
                        textcoords="offset points", fontsize=7, color=style.INK_2)
        ax.set_title(style.MODEL_LABEL[model])
        ax.set_xlim(0.4, 1.03)
        ax.set_ylim(-0.1, 1.03)
        ax.set_xlabel("Mean confidence")
    axes[0].set_ylabel("Accuracy (per bearing)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig, data
