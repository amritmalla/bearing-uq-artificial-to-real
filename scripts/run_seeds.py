"""Training-seed variability of the feature-based classifiers, for the baseline and for CORAL.

Seed 0 is the run from scripts/run_coral.py (data/predictions/fault_only_{rerun,coral}); this script adds seeds 1 and
2 in the same environment. Only the classifiers' random_state changes (random forest and XGBoost sampling; for the
SVM, the internal cross-validation of Platt scaling). Bearings, rotations, calibration and thresholds are unchanged.

Usage (from the project root):
    python scripts/run_seeds.py [--draws 2000] [--max-seconds N]

Writes results/seeds_{per_seed,rotation_spread}.csv and results/bootstrap_fault_only_{rerun,coral}_seeds.csv
(hierarchical bootstrap with the seed as an extra level: each draw picks a rotation and a seed).
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bearing_uq import bootstrap, config, splits  # noqa: E402
from bearing_uq.compare import per_rotation, stack_runs, summarise  # noqa: E402
from bearing_uq.coral import align_held_out_bearings  # noqa: E402
from bearing_uq.dataset import FEATURE_SETS, load_features  # noqa: E402
from bearing_uq.predictions import load_all, predict_rotation, save  # noqa: E402

PRED = Path("data/predictions")
OUT = Path("results")
SEEDS = (0, 1, 2)
VARIANTS = {"fault_only_rerun": None, "fault_only_coral": align_held_out_bearings}
METRICS = ["accuracy", "ece", "auto_rate@5%", "auto_error@5%", "exceeded_5%_share", "auto_rate@1%",
           "auto_error@1%", "coverage_marginal"]


def folder(variant: str, seed: int) -> Path:
    return PRED / (variant if seed == 0 else f"{variant}_seed{seed}")


def save_seeds(max_seconds: float) -> bool:
    start = time.monotonic()
    df = load_features(config.features_path())
    rotations = list(splits.calibration_rotations())
    for seed in SEEDS[1:]:
        for i, (calib, split) in enumerate(rotations, start=1):
            for variant, transform in VARIANTS.items():
                path = folder(variant, seed) / f"{'+'.join(calib)}.npz"
                if path.exists():
                    continue
                if time.monotonic() - start > max_seconds:
                    print("time budget reached; run again to continue")
                    return False
                save(path, predict_rotation(df, split, FEATURE_SETS["fault_only"], seed=seed,
                                            test_transform=transform))
            print(f"seed {seed}: {i}/{len(rotations)} {'+'.join(calib)}", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--max-seconds", type=float, default=float("inf"))
    args = parser.parse_args()
    if not save_seeds(args.max_seconds):
        return

    per_seed, spread = [], []
    for variant in VARIANTS:
        preds = {f"seed{s}": load_all(folder(variant, s)) for s in SEEDS}
        rot = pd.concat([per_rotation(p, variant).assign(seed=s) for s, p in zip(SEEDS, preds.values())])
        per_seed.append(summarise(rot, by=("variant", "model", "seed")).reset_index())
        # Within each rotation, how much do the seeds disagree? (range across seeds, averaged over rotations)
        rng = rot.groupby(["variant", "model", "rotation"])[METRICS[:4] + ["coverage_marginal"]].agg(
            lambda s: s.max() - s.min())
        spread.append(rng.groupby(["variant", "model"]).agg(["mean", "max"]))
        bootstrap.summarise(bootstrap.run(stack_runs(preds), args.draws)).round(4).to_csv(
            OUT / f"bootstrap_{variant}_seeds.csv", index=False)

    per_seed = pd.concat(per_seed)
    per_seed.round(4).to_csv(OUT / "seeds_per_seed.csv", index=False)
    pd.concat(spread).round(4).to_csv(OUT / "seeds_rotation_spread.csv")
    print(per_seed.set_index(["variant", "model", "seed"])[METRICS].round(3).to_string())


if __name__ == "__main__":
    main()
