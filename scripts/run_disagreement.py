"""Confidence threshold alone vs confidence threshold plus feature-model/CNN agreement (rule in disagreement.py).

Usage (from the project root, after save_predictions.py for fault_only and cnn):
    python scripts/run_disagreement.py [--draws 2000] [--calibrator temperature|raw]

Writes results/disagreement_{rotations,bootstrap,per_bearing}[_raw].csv.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bearing_uq import disagreement as G  # noqa: E402
from bearing_uq.predictions import load_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--calibrator", choices=["temperature", "raw"], default="temperature")
    args = parser.parse_args()

    feat = load_all(Path("data/predictions/fault_only"))
    cnn = load_all(Path("data/predictions/cnn"))
    suffix = "" if args.calibrator == "temperature" else f"_{args.calibrator}"
    out = Path("results")

    rot = G.summarise_rotations(G.per_rotation(feat, cnn, args.calibrator))
    rot.round(4).to_csv(out / f"disagreement_rotations{suffix}.csv", index=False)
    G.per_bearing(feat, cnn, 0.05, args.calibrator).round(4).to_csv(
        out / f"disagreement_per_bearing{suffix}.csv", index=False)
    boot = G.summarise_draws(G.bootstrap_draws(feat, cnn, args.draws, calibrator=args.calibrator))
    boot.round(4).to_csv(out / f"disagreement_bootstrap{suffix}.csv", index=False)

    print(rot.round(3).to_string(index=False))
    print(f"\n{len(feat['rotations'])} rotations, {args.draws} draws -> {out}/disagreement_*{suffix}.csv")


if __name__ == "__main__":
    main()
