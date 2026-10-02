"""Summarise the 105 calibration rotations (mean, std, min, max) into results/rotations_summary_*.csv.

Usage (from the project root, after run_rotations.py):
    python scripts/summarise_rotations.py
"""

from pathlib import Path

import pandas as pd

OUT = Path("results")
MAIN_COLS = ["accuracy", "macro_f1", "ece", "brier", "aurc",
             "auto_rate@5%", "auto_error@5%", "oracle_auto_rate@5%",
             "auto_rate@1%", "auto_error@1%", "oracle_auto_rate@1%"]


def main() -> None:
    m = pd.read_csv(OUT / "rotations_main.csv")
    c = pd.read_csv(OUT / "rotations_conformal.csv")
    r = pd.read_csv(OUT / "rotations_recall.csv")
    stats = ["mean", "std", "min", "max"]

    target = m[m.domain == "target_real"]
    main_summary = target.groupby(["model", "calibration"])[MAIN_COLS].agg(stats)
    automated = target[target["auto_rate@5%"] > 0]
    main_summary["share_rotations_error_above_5%"] = (
        automated.groupby(["model", "calibration"])["auto_error@5%"].apply(lambda s: (s > 0.05).mean()))

    conformal_summary = c.groupby(["model", "method"])[
        ["coverage", "mean_set_size", "singleton_rate", "singleton_error"]].agg(stats)
    recall_summary = r.groupby("model")[
        ["recall_healthy", "recall_inner_race", "recall_outer_race"]].agg(stats)

    main_summary.round(4).to_csv(OUT / "rotations_summary_main.csv")
    conformal_summary.round(4).to_csv(OUT / "rotations_summary_conformal.csv")
    recall_summary.round(4).to_csv(OUT / "rotations_summary_recall.csv")
    print(f"Summarised {m.calib_set.nunique()} rotations into {OUT}/rotations_summary_*.csv")


if __name__ == "__main__":
    main()
