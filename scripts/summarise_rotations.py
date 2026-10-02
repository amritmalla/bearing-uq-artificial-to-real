"""Summarise the 105 calibration rotations (mean, std, min, max) into results/rotations_summary_*.csv.

Usage (from the project root, after run_rotations.py):
    python scripts/summarise_rotations.py [--features all|fault_only|fault_only_sk] [--condition N15_M07_F10]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bearing_uq import config  # noqa: E402

OUT = Path("results")
MAIN_COLS = ["accuracy", "macro_f1", "ece", "brier", "aurc",
             "auto_rate@5%", "auto_error@5%", "oracle_auto_rate@5%",
             "auto_rate@1%", "auto_error@1%", "oracle_auto_rate@1%"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", choices=["all", "fault_only", "fault_only_sk"], default="all")
    parser.add_argument("--condition", choices=config.CONDITIONS, default=config.OPERATING_CONDITION)
    args = parser.parse_args()
    sfx = config.result_suffix(args.features, args.condition)
    m = pd.read_csv(OUT / f"rotations_main{sfx}.csv")
    c = pd.read_csv(OUT / f"rotations_conformal{sfx}.csv")
    r = pd.read_csv(OUT / f"rotations_recall{sfx}.csv")
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

    main_summary.round(4).to_csv(OUT / f"rotations_summary_main{sfx}.csv")
    conformal_summary.round(4).to_csv(OUT / f"rotations_summary_conformal{sfx}.csv")
    recall_summary.round(4).to_csv(OUT / f"rotations_summary_recall{sfx}.csv")
    print(f"Summarised {m.calib_set.nunique()} rotations into {OUT}/rotations_summary_*{sfx}.csv")


if __name__ == "__main__":
    main()
