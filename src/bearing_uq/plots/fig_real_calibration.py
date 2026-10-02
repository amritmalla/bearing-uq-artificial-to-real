"""Figure 3: effect of calibrating on a few real-damage bearings (temperature scaling)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bearing_uq.plots import style

SETTINGS = [("Artificial\nbearings", "source", None), ("1 real\nbearing/class", "target", 1),
            ("2 real\nbearings/class", "target", 2)]
OFFSETS = {"random_forest": -0.12, "svm": 0.0, "xgboost": 0.12}


def _values(df: pd.DataFrame, model: str, cal_set: str, n, metric: str) -> pd.Series:
    rows = df[(df.model == model) & (df.calibration == "temperature") & (df.calibration_set == cal_set)]
    if n is not None:
        rows = rows[rows.n_per_class == n]
    return rows[metric].dropna()


def draw(df: pd.DataFrame):
    metrics = [("auto_error@5%", "Error among automated cases"), ("auto_rate@5%", "Share of cases automated")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    x = np.arange(len(SETTINGS))
    for ax, (metric, ylabel) in zip(axes, metrics):
        for model in OFFSETS:
            vals = [_values(df, model, s, n, metric) for _, s, n in SETTINGS]
            mean = np.array([v.mean() for v in vals])
            lo = mean - np.array([v.quantile(0.025) for v in vals])
            hi = np.array([v.quantile(0.975) for v in vals]) - mean
            ax.errorbar(x + OFFSETS[model], mean, yerr=[lo, hi], color=style.MODEL_COLOR[model],
                        marker=style.MODEL_MARKER[model], linewidth=1.2, elinewidth=1, capsize=2,
                        label=style.MODEL_LABEL[model])
        ax.set_xticks(x, [s[0] for s in SETTINGS])
        ax.set_xlim(-0.4, len(SETTINGS) - 0.6)
        ax.set_ylim(0, 1.0 if metric.startswith("auto_rate") else 0.5)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Calibration set (threshold chosen for 5% error)")
        ax.grid(axis="x", visible=False)
    style.target_line(axes[0], 0.05, "5% target")
    axes[0].set_title("(a) Error on real damage")
    axes[1].set_title("(b) Automation on real damage")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig
