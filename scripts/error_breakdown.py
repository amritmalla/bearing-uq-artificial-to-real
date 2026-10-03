"""Missed-fault shares and per-bearing accuracy on real damage, from saved predictions.

Usage (from the project root, after save_predictions.py or run_cnn_analysis.py):
    python scripts/error_breakdown.py --predictions fault_only [fault_only_N09_M07_F10 ... cnn]
Writes results/missed_faults_<name>.csv and results/per_bearing_real_<name>.csv for each name.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bearing_uq.errors import missed_fault_shares  # noqa: E402
from bearing_uq.plots.fig_per_bearing import per_bearing  # noqa: E402
from bearing_uq.predictions import load_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="+", default=["fault_only"])
    for name in parser.parse_args().predictions:
        pred = load_all(Path("data/predictions") / name)
        missed = missed_fault_shares(pred)
        missed.round(4).to_csv(f"results/missed_faults_{name}.csv", index=False)
        per_bearing(pred).round(4).to_csv(f"results/per_bearing_real_{name}.csv", index=False)
        print(name, "\n", missed.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
