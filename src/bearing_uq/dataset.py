"""Load the feature table and turn it into model inputs."""

from pathlib import Path

import numpy as np
import pandas as pd

from bearing_uq import bearings as B

FEATURES = ["rms", "kurtosis", "skewness", "crest_factor", "peak_to_peak",
            "bpfo_h1", "bpfo_h2", "bpfi_h1", "bpfi_h2"]

# Self-normalised envelope-spectrum features (ratio to the spectrum's noise floor).
FAULT_FEATURES = ["bpfo_h1", "bpfo_h2", "bpfi_h1", "bpfi_h2"]

FEATURE_SETS = {"all": FEATURES, "fault_only": FAULT_FEATURES}

# Positive, heavy-tailed features are log-transformed; skewness can be negative.
LOG_FEATURES = [f for f in FEATURES if f != "skewness"]

LABELS = list(B.CLASSES)  # class index = position in this list


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(FEATURES + ["bearing", "label"]) - set(df.columns)
    if missing:
        raise ValueError(f"Feature table is missing columns: {sorted(missing)}")
    return df


def design_matrix(df: pd.DataFrame, features: list[str] = FEATURES) -> np.ndarray:
    X = df[list(features)].copy()
    logged = [f for f in features if f in LOG_FEATURES]
    X[logged] = np.log(X[logged].clip(lower=1e-12))
    return X.to_numpy(dtype=float)


def targets(df: pd.DataFrame) -> np.ndarray:
    return df["label"].map({name: i for i, name in enumerate(LABELS)}).to_numpy()


def subset(df: pd.DataFrame, split: dict[str, str], name: str) -> pd.DataFrame:
    """Rows whose bearing is assigned to the named split."""
    return df[df["bearing"].map(split) == name].reset_index(drop=True)
