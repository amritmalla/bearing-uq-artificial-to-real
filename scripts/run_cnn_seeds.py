"""Training-seed variability of the 1D-CNN, and of the feature-model/CNN agreement rule (disagreement.py).

Seed 0 is the original CNN run (data/predictions/cnn). Seeds 1 and 2 come from step 17 of notebooks/colab_cnn.ipynb;
unzip them into data/ and run  python scripts/run_cnn_analysis.py --variant seed1  (and seed2) first, which writes
data/predictions/cnn_seed{1,2}/. The feature-based predictions (data/predictions/fault_only) are the same for every
CNN seed, so only the CNN's training seed varies.

Usage (from the project root):
    python scripts/run_cnn_seeds.py [--draws 2000]

Writes
  results/cnn_seeds_per_seed.csv              CNN target metrics per seed (means over the 105 rotations)
  results/disagreement_seeds_per_seed.csv     confidence-only vs agreement rule, per CNN seed
  results/disagreement_bootstrap_seeds.csv    hierarchical bootstrap with the CNN seed as an extra level
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bearing_uq import disagreement as G  # noqa: E402
from bearing_uq.compare import per_rotation, stack_runs, summarise  # noqa: E402
from bearing_uq.predictions import load_all  # noqa: E402

PRED = Path("data/predictions")
OUT = Path("results")
CNN_RUNS = {0: "cnn", 1: "cnn_seed1", 2: "cnn_seed2"}
CNN_METRICS = ["accuracy", "ece", "auto_rate@5%", "auto_error@5%", "exceeded_5%_share", "coverage_marginal",
               "missed_share_of_accepted_errors@5%"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()

    missing = [name for name in CNN_RUNS.values() if not (PRED / name).is_dir()]
    if missing:
        sys.exit(f"missing {', '.join(str(PRED / m) for m in missing)}; run run_cnn_analysis.py --variant seedN")
    feat = load_all(PRED / "fault_only")
    # run_cnn_analysis.py names the model after its folder (cnn, cnn_seed1, ...); give all runs one name.
    cnns = {seed: {**load_all(PRED / name), "models": ["cnn"]} for seed, name in CNN_RUNS.items()}

    rot = pd.concat([per_rotation(c, "cnn").assign(seed=s) for s, c in cnns.items()])
    cnn_summary = summarise(rot, by=("model", "seed")).reset_index()
    cnn_summary.round(4).to_csv(OUT / "cnn_seeds_per_seed.csv", index=False)

    rules = pd.concat([G.summarise_rotations(G.per_rotation(feat, c)).assign(cnn_seed=s) for s, c in cnns.items()])
    rules.round(4).to_csv(OUT / "disagreement_seeds_per_seed.csv", index=False)

    # Each draw picks a rotation and a CNN seed: the same feature predictions are repeated once per seed so that
    # rotation r of the stacked feature run pairs with rotation r of the stacked CNN runs.
    names = [f"seed{s}" for s in cnns]
    feat_stacked = stack_runs({n: feat for n in names})
    cnn_stacked = stack_runs(dict(zip(names, cnns.values())))
    boot = G.summarise_draws(G.bootstrap_draws(feat_stacked, cnn_stacked, args.draws))
    boot.round(4).to_csv(OUT / "disagreement_bootstrap_seeds.csv", index=False)

    print(cnn_summary.set_index(["model", "seed"])[CNN_METRICS].round(3).to_string())
    print()
    five = rules[rules["target"] == 0.05].set_index(["model", "rule", "cnn_seed"])
    print(five[["automation_rate", "automated_error", "target_met_share"]].round(3).to_string())
    print(f"\n{args.draws} draws -> {OUT}/disagreement_bootstrap_seeds.csv")


if __name__ == "__main__":
    main()
