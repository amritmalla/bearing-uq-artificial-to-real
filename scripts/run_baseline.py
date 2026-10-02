"""Run the main artificial-to-real experiment on the saved feature table.

Usage (from the project root):
    python scripts/run_baseline.py
Results are written to results/.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bearing_uq import config, splits  # noqa: E402
from bearing_uq.dataset import load_features  # noqa: E402
from bearing_uq.experiment import run  # noqa: E402

RESULTS_DIR = Path("results")


def main() -> None:
    df = load_features(config.FEATURES_DIR / f"features_{config.OPERATING_CONDITION}.csv")
    main_df, conformal_df, recall_df, sizes = run(df, splits.artificial_to_real())

    RESULTS_DIR.mkdir(exist_ok=True)
    main_df.to_csv(RESULTS_DIR / "main_results.csv", index=False)
    conformal_df.to_csv(RESULTS_DIR / "conformal_results.csv", index=False)
    recall_df.to_csv(RESULTS_DIR / "per_class_recall.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("Windows per split:", sizes, "\n")
    print(main_df.round(3).to_string(index=False), "\n")
    print(conformal_df.round(3).to_string(index=False), "\n")
    print(recall_df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
