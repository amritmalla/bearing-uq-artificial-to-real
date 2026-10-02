"""Load the feature table and turn it into model inputs."""

from pathlib import Path

import numpy as np
import pandas as pd

from bearing_uq import bearings as B

FEATURES = ["rms", "kurtosis", "skewness", "crest_factor", "peak_to_peak",
            "bpfo_h1", "bpfo_h2", "bpfi_h1", "bpfi_h2"]

# Positive, heavy-tailed features are log-transformed; skewness can be negative.
LOG_FEATURES = [f for f in FEATURES if f != "skewness"]

LABELS = list(B.CLASSES)  # class index = position in this list


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(FEATURES + ["bearing", "label"]) - set(df.columns)
    if missing:
        raise ValueError(f"Feature table is missing columns: {sorted(missing)}")
    return df


def design_matrix(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURES].copy()
    X[LOG_FEATURES] = np.log(X[LOG_FEATURES].clip(lower=1e-12))
    return X.to_numpy(dtype=float)


def targets(df: pd.DataFrame) -> np.ndarray:
    return df["label"].map({name: i for i, name in enumerate(LABELS)}).to_numpy()


def subset(df: pd.DataFrame, split: dict[str, str], name: str) -> pd.DataFrame:
    """Rows whose bearing is assigned to the named split."""
    return df[df["bearing"].map(split) == name].reset_index(drop=True)
