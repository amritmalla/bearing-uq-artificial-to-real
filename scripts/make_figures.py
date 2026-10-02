"""Draw the paper figures from saved results into results/figures/ (PNG + PDF).

Usage (from the project root, after save_predictions.py and run_target_calibration.py):
    python scripts/make_figures.py
Uses the main feature set (fault-frequency features only).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bearing_uq import config, splits  # noqa: E402
from bearing_uq.dataset import FEATURE_SETS, load_features  # noqa: E402
from bearing_uq.in_domain import lobo_probabilities  # noqa: E402
from bearing_uq.plots import fig_per_bearing, fig_real_calibration, fig_reliability, fig_risk_coverage, style  # noqa: E402
from bearing_uq.predictions import load_all  # noqa: E402

FEATURES = "fault_only"


def main() -> None:
    style.apply()
    df = load_features(config.FEATURES_DIR / f"features_{config.OPERATING_CONDITION}.csv")
    pred = load_all(Path("data/predictions") / FEATURES)
    _, y_in, probs_in = lobo_probabilities(df, splits.source_bearings(), features=FEATURE_SETS[FEATURES])
    in_domain = {m: (p, y_in) for m, p in probs_in.items()}

    saved = []
    saved += style.save(fig_risk_coverage.draw(in_domain, pred), "fig1_risk_coverage")
    saved += style.save(fig_reliability.draw(in_domain, pred), "fig2_reliability")
    tc = pd.read_csv(f"results/target_calibration_{FEATURES}.csv")
    saved += style.save(fig_real_calibration.draw(tc), "fig4_real_calibration")
    fig, per_bearing = fig_per_bearing.draw(pred)
    saved += style.save(fig, "fig3_per_bearing")
    per_bearing.round(4).to_csv("results/per_bearing_real_fault_only.csv", index=False)
    print("\n".join(str(p) for p in saved))


if __name__ == "__main__":
    main()
