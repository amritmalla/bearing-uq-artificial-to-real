"""Sanity check: accuracy with a random window split (same bearings in training and test) vs by bearing.

Uses the same random split of the 15 source bearings' windows as the CNN notebook (cnn.protocol.random_window_split).
Fits Random Forest, SVM and XGBoost on the fault-frequency features of the training windows and reads the CNN's
predictions from data/cnn/random_split.csv. Writes results/random_split_check.csv.

Usage (from the project root, after colab_cnn.ipynb has produced data/cnn/random_split.csv):
    python scripts/random_split_check.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bearing_uq import config, splits  # noqa: E402
from bearing_uq import dataset as D  # noqa: E402
from bearing_uq import metrics as M  # noqa: E402
from bearing_uq.cnn.protocol import random_window_split  # noqa: E402
from bearing_uq.models import fit, make_models  # noqa: E402
from bearing_uq.precomputed import PrecomputedModel, probabilities, with_row_key  # noqa: E402

KEYS = ["bearing", "recording", "window"]


def main() -> None:
    df = with_row_key(D.load_features(config.features_path()))
    source = df[df["bearing"].isin(splits.source_bearings())].reset_index(drop=True)
    is_test = random_window_split(source)
    train, test = source[~is_test], source[is_test]
    y_test = D.targets(test)

    probs = {}
    for name, model in make_models(0).items():
        fit(model, D.design_matrix(train, D.FAULT_FEATURES), D.targets(train))
        probs[name] = model.predict_proba(D.design_matrix(test, D.FAULT_FEATURES))

    cnn_file = Path("data/cnn/random_split.csv")
    saved = pd.read_csv(cnn_file)
    if set(map(tuple, saved[KEYS].values)) != set(map(tuple, test[KEYS].values)):
        raise ValueError("The CNN's random split differs from this one")
    probs["cnn"] = PrecomputedModel(probabilities(cnn_file, df)).predict_proba(D.design_matrix(test, ["row_key"]))

    rows = [{"model": name, "split": "random windows", "accuracy": M.accuracy(p, y_test), "ece": M.ece(p, y_test),
             "n_test_windows": len(y_test)} for name, p in probs.items()]
    for name, fs in [("random_forest", "fault_only"), ("svm", "fault_only"), ("xgboost", "fault_only"), ("cnn", "cnn")]:
        ind = pd.read_csv(f"results/in_domain_summary_{fs}.csv").set_index("model").loc[name]
        rows.append({"model": name, "split": "by bearing (leave one out)", "accuracy": ind["accuracy"],
                     "ece": ind["ece"]})
    out = pd.DataFrame(rows)
    out.round(4).to_csv("results/random_split_check.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
