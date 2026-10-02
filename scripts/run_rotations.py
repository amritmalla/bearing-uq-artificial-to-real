"""Repeat the artificial-to-real experiment for every choice of calibration bearings.

One source bearing per class is held out for calibration: 3 healthy x 5 inner x 7 outer = 105 runs.
Progress is saved after each run, so an interrupted job resumes where it stopped.

Usage (from the project root):
    python scripts/run_rotations.py [--features all|fault_only] [--max-seconds N]
With --max-seconds, the script stops cleanly after that time; run it again to continue.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bearing_uq import config, splits  # noqa: E402
from bearing_uq.dataset import FEATURE_SETS, load_features  # noqa: E402
from bearing_uq.experiment import run  # noqa: E402

OUT = Path("results")


def result_files(feature_set: str) -> dict[str, Path]:
    suffix = "" if feature_set == "all" else f"_{feature_set}"
    return {name: OUT / f"rotations_{name}{suffix}.csv" for name in ("main", "conformal", "recall")}


def _done(files) -> set[str]:
    return set(pd.read_csv(files["main"])["calib_set"]) if files["main"].exists() else set()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", choices=FEATURE_SETS, default="all")
    parser.add_argument("--max-seconds", type=float, default=float("inf"))
    args = parser.parse_args()

    start = time.monotonic()
    OUT.mkdir(exist_ok=True)
    files = result_files(args.features)
    df = load_features(config.FEATURES_DIR / f"features_{config.OPERATING_CONDITION}.csv")
    done = _done(files)
    rotations = list(splits.calibration_rotations())
    for i, (calib, split) in enumerate(rotations, start=1):
        key = "+".join(calib)
        if key in done:
            continue
        if time.monotonic() - start > args.max_seconds:
            print("time budget reached; run again to continue")
            return
        frames = run(df, split, features=FEATURE_SETS[args.features])[:3]
        for name, frame in zip(("main", "conformal", "recall"), frames):
            path = files[name]
            frame.assign(calib_set=key).to_csv(path, mode="a", header=not path.exists(), index=False)
        print(f"{i}/{len(rotations)} {key}", flush=True)
    print("all rotations done")


if __name__ == "__main__":
    main()
