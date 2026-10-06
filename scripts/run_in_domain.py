"""In-domain reference: leave-one-bearing-out over the source (artificial + source healthy) bearings.

Usage (from the project root):
    python scripts/run_in_domain.py [--features all|fault_only|fault_only_sk|fault_only_fk] [--condition N15_M07_F10]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bearing_uq import config, splits  # noqa: E402
from bearing_uq.dataset import FEATURE_SETS, load_features  # noqa: E402
from bearing_uq.in_domain import leave_one_bearing_out  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", choices=FEATURE_SETS, default="all")
    parser.add_argument("--condition", choices=config.CONDITIONS, default=config.OPERATING_CONDITION)
    args = parser.parse_args()
    feature_set = args.features
    sfx = config.result_suffix(feature_set, args.condition)

    df = load_features(config.features_path(args.condition))
    summary, per_bearing = leave_one_bearing_out(df, splits.source_bearings(),
                                                 features=FEATURE_SETS[feature_set])
    Path("results").mkdir(exist_ok=True)
    summary.to_csv(f"results/in_domain_summary{sfx}.csv", index=False)
    per_bearing.to_csv(f"results/in_domain_per_bearing{sfx}.csv", index=False)
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False), "\n")
    print(per_bearing.pivot(index=["label", "bearing"], columns="model", values="accuracy").round(2).to_string())


if __name__ == "__main__":
    main()
