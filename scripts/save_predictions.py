"""Save per-rotation test predictions to data/predictions/<features>/ for the bootstrap.

Usage (from the project root):
    python scripts/save_predictions.py [--features fault_only|all|fault_only_sk] [--condition N15_M07_F10] [--max-seconds N]
Resumable: rotations already saved are skipped.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bearing_uq import config, splits  # noqa: E402
from bearing_uq.dataset import FEATURE_SETS, load_features  # noqa: E402
from bearing_uq.predictions import predict_rotation, save  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", choices=FEATURE_SETS, default="fault_only")
    parser.add_argument("--condition", choices=config.CONDITIONS, default=config.OPERATING_CONDITION)
    parser.add_argument("--max-seconds", type=float, default=float("inf"))
    args = parser.parse_args()

    start = time.monotonic()
    out = Path("data/predictions") / (args.features + config.result_suffix("all", args.condition))
    df = load_features(config.features_path(args.condition))
    rotations = list(splits.calibration_rotations())
    for i, (calib, split) in enumerate(rotations, start=1):
        path = out / f"{'+'.join(calib)}.npz"
        if path.exists():
            continue
        if time.monotonic() - start > args.max_seconds:
            print("time budget reached; run again to continue")
            return
        save(path, predict_rotation(df, split, FEATURE_SETS[args.features]))
        print(f"{i}/{len(rotations)} {path.stem}", flush=True)
    print("all rotations saved")


if __name__ == "__main__":
    main()
