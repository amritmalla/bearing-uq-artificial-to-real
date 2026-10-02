"""Repeat the artificial-to-real experiment for every choice of calibration bearings.

One source bearing per class is held out for calibration: 3 healthy x 5 inner x 7 outer = 105 runs.
Progress is saved after each run, so an interrupted job resumes where it stopped.

Usage (from the project root):
    python scripts/run_rotations.py [max_seconds]
With max_seconds, the script stops cleanly after that time; run it again to continue.
"""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bearing_uq import config, splits  # noqa: E402
from bearing_uq.dataset import load_features  # noqa: E402
from bearing_uq.experiment import run  # noqa: E402

OUT = Path("results")
FILES = {"main": OUT / "rotations_main.csv", "conformal": OUT / "rotations_conformal.csv",
         "recall": OUT / "rotations_recall.csv"}


def _done() -> set[str]:
    if not FILES["main"].exists():
        return set()
    return set(pd.read_csv(FILES["main"])["calib_set"])


def _append(name: str, frame: pd.DataFrame) -> None:
    path = FILES[name]
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> None:
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else float("inf")
    start = time.monotonic()
    OUT.mkdir(exist_ok=True)
    df = load_features(config.FEATURES_DIR / f"features_{config.OPERATING_CONDITION}.csv")
    done = _done()
    rotations = list(splits.calibration_rotations())
    for i, (calib, split) in enumerate(rotations, start=1):
        key = "+".join(calib)
        if key in done:
            continue
        if time.monotonic() - start > budget:
            print("time budget reached; run again to continue")
            return
        main_df, conformal_df, recall_df, _ = run(df, split)
        for name, frame in (("main", main_df), ("conformal", conformal_df), ("recall", recall_df)):
            _append(name, frame.assign(calib_set=key))
        print(f"{i}/{len(rotations)} {key}", flush=True)
    print("all rotations done")


if __name__ == "__main__":
    main()
