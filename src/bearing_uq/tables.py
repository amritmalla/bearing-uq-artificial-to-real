"""Formatting helpers for the paper tables."""

import pandas as pd

MODEL_LABEL = {"random_forest": "Random Forest", "svm": "SVM", "xgboost": "XGBoost"}


def ci(mean: float, low: float, high: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"
