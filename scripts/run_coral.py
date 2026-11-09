"""CORAL-aligned target features vs the unaligned baseline (setup in src/bearing_uq/coral.py).

Usage (from the project root):
    python scripts/run_coral.py [--draws 2000] [--max-seconds N]

Saves predictions to data/predictions/fault_only_{rerun,coral}/ (resumable), then writes
results/bootstrap_fault_only_{rerun,coral}.csv and results/coral_{rotations,per_bearing}.csv. The rotation and
per-bearing files contain both the baseline rerun and CORAL rows, computed the same way in the same environment.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bearing_uq import bootstrap, config, splits  # noqa: E402
from bearing_uq.compare import per_bearing, per_rotation, summarise  # noqa: E402
from bearing_uq.coral import align_held_out_bearings  # noqa: E402
from bearing_uq.dataset import FEATURE_SETS, load_features  # noqa: E402
from bearing_uq.predictions import load_all, predict_rotation, save  # noqa: E402

PRED = Path("data/predictions")
OUT = Path("results")


VARIANTS = {"fault_only_rerun": None, "fault_only_coral": align_held_out_bearings}


def save_variants(max_seconds: float) -> bool:
    """Baseline rerun and CORAL in the same environment, so CORAL is the only difference between them.

    Run with the pinned versions in requirements.txt. Random forest and SVM then reproduce the saved
    predictions (data/predictions/fault_only) exactly, but XGBoost's floating-point results depend on the build
    and the number of threads, so the saved XGBoost run cannot be repeated bit for bit and is not reused here."""
    start = time.monotonic()
    df = load_features(config.features_path())
    rotations = list(splits.calibration_rotations())
    for i, (calib, split) in enumerate(rotations, start=1):
        for folder, transform in VARIANTS.items():
            path = PRED / folder / f"{'+'.join(calib)}.npz"
            if path.exists():
                continue
            if time.monotonic() - start > max_seconds:
                print("time budget reached; run again to continue")
                return False
            save(path, predict_rotation(df, split, FEATURE_SETS["fault_only"], test_transform=transform))
        print(f"{i}/{len(rotations)} {'+'.join(calib)}", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--max-seconds", type=float, default=float("inf"))
    args = parser.parse_args()
    if not save_variants(args.max_seconds):
        return

    base, coral = load_all(PRED / "fault_only_rerun"), load_all(PRED / "fault_only_coral")
    if base["rotations"] != coral["rotations"] or not np.array_equal(base["bearing"], coral["bearing"]):
        raise ValueError("Baseline and CORAL predictions are not aligned")

    for name, pred in (("fault_only_rerun", base), ("fault_only_coral", coral)):
        bootstrap.summarise(bootstrap.run(pred, args.draws)).round(4).to_csv(
            OUT / f"bootstrap_{name}.csv", index=False)
    rot = pd.concat([per_rotation(base, "baseline"), per_rotation(coral, "coral")])
    rot.round(4).to_csv(OUT / "coral_rotations.csv", index=False)
    pd.concat([per_bearing(base, "baseline"), per_bearing(coral, "coral")]).round(4).to_csv(
        OUT / "coral_per_bearing.csv", index=False)

    summary = summarise(rot)
    cols = ["accuracy", "ece", "auto_rate@5%", "auto_error@5%", "exceeded_5%_share", "auto_rate@1%",
            "auto_error@1%", "coverage_marginal", "missed_share_of_accepted_errors@5%"]
    print(summary[cols].round(3).to_string())


if __name__ == "__main__":
    main()
