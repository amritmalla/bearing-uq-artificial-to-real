"""Analyse the 1D-CNN's saved probabilities (from notebooks/colab_cnn.ipynb) like the other models.

Expects data/cnn/lobo.csv and data/cnn/rotations/<rotation>.csv. Writes, with the suffix _cnn:
  results/in_domain_summary_cnn.csv, results/in_domain_per_bearing_cnn.csv   (leave one bearing out)
  results/rotations_{main,conformal,recall}_cnn.csv                          (105 rotations)
  data/predictions/cnn/<rotation>.npz                                         (for run_bootstrap.py --features cnn)
  results/target_calibration_cnn.csv and its summary                          (same draws as the other models)
Then run:  summarise_rotations.py --features cnn,  run_bootstrap.py --features cnn,
           error_breakdown.py --predictions cnn

Usage (from the project root):
    python scripts/run_cnn_analysis.py [--draws 50] [--max-seconds N]
Resumable: finished rotations and draws are skipped.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bearing_uq import config, splits  # noqa: E402
from bearing_uq import dataset as D  # noqa: E402
from bearing_uq.experiment import run  # noqa: E402
from bearing_uq.in_domain import summarise as summarise_lobo  # noqa: E402
from bearing_uq.precomputed import KEY_FEATURES, PrecomputedModel, probabilities, with_row_key  # noqa: E402
from bearing_uq.predictions import predict_rotation, save  # noqa: E402
from bearing_uq.target_calibration import draw_rotation, draw_seeds, run_draw, summarise  # noqa: E402

CNN = Path("data/cnn")
RES = Path("results")
PRED = Path("data/predictions/cnn")
NAME = "cnn"


class Budget:
    def __init__(self, seconds):
        self.end = time.monotonic() + seconds

    def exceeded(self):
        if time.monotonic() > self.end:
            print("time budget reached; run again to continue")
            return True
        return False


def model_for(key: str, df: pd.DataFrame) -> dict:
    return {NAME: PrecomputedModel(probabilities(CNN / "rotations" / f"{key}.csv", df))}


def in_domain(df: pd.DataFrame) -> None:
    lobo = probabilities(CNN / "lobo.csv", df)
    data = df[df["bearing"].isin(splits.source_bearings())].reset_index(drop=True)
    probs = PrecomputedModel(lobo).predict_proba(D.design_matrix(data, KEY_FEATURES))
    summary, per_bearing = summarise_lobo(data, D.targets(data), {NAME: probs})
    summary.to_csv(RES / f"in_domain_summary_{NAME}.csv", index=False)
    per_bearing.to_csv(RES / f"in_domain_per_bearing_{NAME}.csv", index=False)
    print(summary.round(3).to_string(index=False))


def rotations(df: pd.DataFrame, budget: Budget) -> bool:
    files = {n: RES / f"rotations_{n}_{NAME}.csv" for n in ("main", "conformal", "recall")}
    done = set(pd.read_csv(files["main"])["calib_set"]) if files["main"].exists() else set()
    missing = []
    for calib, split in splits.calibration_rotations():
        key = "+".join(calib)
        if key in done and (PRED / f"{key}.npz").exists():
            continue
        if not (CNN / "rotations" / f"{key}.csv").exists():
            missing.append(key)
            continue
        if budget.exceeded():
            return False
        models = model_for(key, df)
        if key not in done:
            for name, frame in zip(files, run(df, split, features=KEY_FEATURES, models=models)[:3]):
                frame.assign(calib_set=key).to_csv(files[name], mode="a", header=not files[name].exists(),
                                                   index=False)
        save(PRED / f"{key}.npz", predict_rotation(df, split, KEY_FEATURES, models=models))
    if missing:
        print(f"{len(missing)} of 105 rotation files are missing from {CNN / 'rotations'}, e.g. {missing[0]}")
        return False
    print("rotations done")
    return True


def real_calibration(df: pd.DataFrame, draws: int, budget: Budget) -> bool:
    out = RES / f"target_calibration_{NAME}.csv"
    done = set(map(tuple, pd.read_csv(out)[["n_per_class", "seed"]].drop_duplicates().values)) \
        if out.exists() else set()
    rots = list(splits.calibration_rotations())
    for n, seed in draw_seeds(draws):
        if (n, seed) in done:
            continue
        if budget.exceeded():
            return False
        calib, split = draw_rotation(rots, seed)
        frame = run_draw(df, calib, split, n, seed, KEY_FEATURES, models=model_for("+".join(calib), df))
        frame.to_csv(out, mode="a", header=not out.exists(), index=False)
    summarise(pd.read_csv(out)).round(4).to_csv(RES / f"target_calibration_summary_{NAME}.csv", index=False)
    print("real-bearing calibration done")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=50)
    parser.add_argument("--max-seconds", type=float, default=float("inf"))
    args = parser.parse_args()
    budget = Budget(args.max_seconds)
    df = with_row_key(D.load_features(config.features_path()))
    in_domain(df)
    if rotations(df, budget):
        real_calibration(df, args.draws, budget)


if __name__ == "__main__":
    main()
