"""Calibrate on 1 or 2 real-damage bearings per class vs. on source bearings (same test bearings).

Each draw picks a random source rotation and random real calibration bearings.
Usage (from the project root):
    python scripts/run_target_calibration.py [--features fault_only|all|fault_only_sk|fault_only_fk] [--condition N15_M07_F10] [--draws 50] [--max-seconds N]
Resumable: draws already in the results file are skipped. Writes a summary when all draws are done.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bearing_uq import config, splits  # noqa: E402
from bearing_uq.dataset import FEATURE_SETS, load_features  # noqa: E402
from bearing_uq.target_calibration import N_PER_CLASS, draw_rotation, run_draw, summarise  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", choices=FEATURE_SETS, default="fault_only")
    parser.add_argument("--condition", choices=config.CONDITIONS, default=config.OPERATING_CONDITION)
    parser.add_argument("--draws", type=int, default=50)
    parser.add_argument("--max-seconds", type=float, default=float("inf"))
    args = parser.parse_args()

    start = time.monotonic()
    name = args.features + config.result_suffix("all", args.condition)
    out = Path("results") / f"target_calibration_{name}.csv"
    done = set() if not out.exists() else set(map(tuple, pd.read_csv(out)[["n_per_class", "seed"]].drop_duplicates().values))
    df = load_features(config.features_path(args.condition))
    rotations = list(splits.calibration_rotations())

    for n in N_PER_CLASS:
        for d in range(args.draws):
            seed = 1000 * n + d
            if (n, seed) in done:
                continue
            if time.monotonic() - start > args.max_seconds:
                print("time budget reached; run again to continue")
                return
            calib, split = draw_rotation(rotations, seed)
            frame = run_draw(df, calib, split, n, seed, FEATURE_SETS[args.features])
            frame.to_csv(out, mode="a", header=not out.exists(), index=False)
            print(f"n={n} draw {d + 1}/{args.draws}", flush=True)

    summary = summarise(pd.read_csv(out))
    summary.round(4).to_csv(out.with_name(f"target_calibration_summary_{name}.csv"), index=False)
    print("all draws done; summary written")


if __name__ == "__main__":
    main()
