"""Calibrate on 1 or 2 real-damage bearings per class vs. on source bearings (same test bearings).

Each draw picks a random source rotation and random real calibration bearings.
Usage (from the project root):
    python scripts/run_target_calibration.py [--features fault_only|all] [--draws 50] [--max-seconds N]
Resumable: draws already in the results file are skipped. Writes a summary when all draws are done.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bearing_uq import config, splits  # noqa: E402
from bearing_uq.dataset import FEATURE_SETS, load_features  # noqa: E402
from bearing_uq.target_calibration import run_draw  # noqa: E402

N_PER_CLASS = (1, 2)
METRICS = ["accuracy", "ece", "auto_rate@5%", "auto_error@5%", "coverage", "mean_set_size"]


def summarise(rows: pd.DataFrame) -> pd.DataFrame:
    long = rows.melt(id_vars=["n_per_class", "model", "calibration_set", "calibration"],
                     value_vars=[m for m in METRICS if m in rows], var_name="metric").dropna(subset=["value"])
    g = long.groupby(["n_per_class", "model", "calibration_set", "calibration", "metric"])["value"]
    return pd.DataFrame({"mean": g.mean(), "p2.5": g.quantile(0.025), "p97.5": g.quantile(0.975)}).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", choices=FEATURE_SETS, default="fault_only")
    parser.add_argument("--draws", type=int, default=50)
    parser.add_argument("--max-seconds", type=float, default=float("inf"))
    args = parser.parse_args()

    start = time.monotonic()
    out = Path("results") / f"target_calibration_{args.features}.csv"
    done = set() if not out.exists() else set(map(tuple, pd.read_csv(out)[["n_per_class", "seed"]].drop_duplicates().values))
    df = load_features(config.FEATURES_DIR / f"features_{config.OPERATING_CONDITION}.csv")
    rotations = list(splits.calibration_rotations())

    for n in N_PER_CLASS:
        for d in range(args.draws):
            seed = 1000 * n + d
            if (n, seed) in done:
                continue
            if time.monotonic() - start > args.max_seconds:
                print("time budget reached; run again to continue")
                return
            calib, split = rotations[np.random.default_rng(seed).integers(len(rotations))]
            frame = run_draw(df, calib, split, n, seed, FEATURE_SETS[args.features])
            frame.to_csv(out, mode="a", header=not out.exists(), index=False)
            print(f"n={n} draw {d + 1}/{args.draws}", flush=True)

    summary = summarise(pd.read_csv(out))
    summary.round(4).to_csv(out.with_name(f"target_calibration_summary_{args.features}.csv"), index=False)
    print("all draws done; summary written")


if __name__ == "__main__":
    main()
