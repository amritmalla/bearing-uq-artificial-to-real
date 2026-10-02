"""In-domain reference: leave-one-bearing-out over the source (artificial + source healthy) bearings.

Usage (from the project root):
    python scripts/run_in_domain.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bearing_uq import config, splits  # noqa: E402
from bearing_uq.dataset import load_features  # noqa: E402
from bearing_uq.in_domain import leave_one_bearing_out  # noqa: E402


def main() -> None:
    df = load_features(config.FEATURES_DIR / f"features_{config.OPERATING_CONDITION}.csv")
    summary, per_bearing = leave_one_bearing_out(df, splits.source_bearings())
    Path("results").mkdir(exist_ok=True)
    summary.to_csv("results/in_domain_summary.csv", index=False)
    per_bearing.to_csv("results/in_domain_per_bearing.csv", index=False)
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False), "\n")
    print(per_bearing.pivot(index=["label", "bearing"], columns="model", values="accuracy").round(2).to_string())


if __name__ == "__main__":
    main()
