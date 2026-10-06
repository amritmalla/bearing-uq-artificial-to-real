"""Load the feature table and turn it into model inputs."""

from pathlib import Path

import numpy as np
import pandas as pd

from bearing_uq import bearings as B

FEATURES = ["rms", "kurtosis", "skewness", "crest_factor", "peak_to_peak",
            "bpfo_h1", "bpfo_h2", "bpfi_h1", "bpfi_h2"]

# Self-normalised envelope-spectrum features (ratio to the spectrum's noise floor).
FAULT_FEATURES = ["bpfo_h1", "bpfo_h2", "bpfi_h1", "bpfi_h2"]

# Same four features from a spectral-kurtosis-selected demodulation band (per window).
FAULT_FEATURES_SK = [f"{f}_sk" for f in FAULT_FEATURES]

# Same four features from the fast-kurtogram band (per window; main condition only, see colab_kurtogram.ipynb).
FAULT_FEATURES_FK = [f"{f}_fk" for f in FAULT_FEATURES]

FEATURE_SETS = {"all": FEATURES, "fault_only": FAULT_FEATURES, "fault_only_sk": FAULT_FEATURES_SK,
                "fault_only_fk": FAULT_FEATURES_FK}

# Positive, heavy-tailed features are log-transformed; skewness can be negative.
LOG_FEATURES = [f for f in FEATURES if f != "skewness"] + FAULT_FEATURES_SK + FAULT_FEATURES_FK

LABELS = list(B.CLASSES)  # class index = position in this list


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(FEATURES + ["bearing", "label"]) - set(df.columns)  # *_sk and *_fk columns are optional
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
