"""Add the fast-kurtogram features (from notebooks/colab_kurtogram.ipynb) to the main feature table.

Reads data/features/kurtogram_N15_M07_F10.csv and adds its columns (*_fk, fk_band_low_hz, fk_band_high_hz,
fk_level) to data/features/features_N15_M07_F10.csv, matched by bearing, recording and window. Every window of
the feature table must have kurtogram features. Existing kurtogram columns are replaced; nothing else changes (the
tables are handled as text, so the existing values are written back exactly as they were).

Usage (from the project root):
    python scripts/add_kurtogram_features.py
Then, as for the spectral-kurtosis band:
    python scripts/run_in_domain.py --features fault_only_fk
    python scripts/run_rotations.py --features fault_only_fk
    python scripts/summarise_rotations.py --features fault_only_fk
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bearing_uq import config  # noqa: E402

KEYS = ["bearing", "recording", "window"]


def merge(features: pd.DataFrame, kurt: pd.DataFrame) -> pd.DataFrame:
    new_cols = [c for c in kurt.columns if c not in KEYS]
    base = features.drop(columns=[c for c in new_cols if c in features.columns])
    merged = base.merge(kurt, on=KEYS, how="left", validate="one_to_one")
    missing = merged[new_cols[0]].isna().sum()
    if missing:
        raise ValueError(f"{missing} windows of the feature table have no kurtogram features")
    return merged


def main() -> None:
    path = config.features_path()
    kurt = pd.read_csv(config.FEATURES_DIR / f"kurtogram_{config.OPERATING_CONDITION}.csv", dtype=str)
    merged = merge(pd.read_csv(path, dtype=str), kurt)
    merged.to_csv(path, index=False)
    print(f"added {len(kurt.columns) - len(KEYS)} kurtogram columns to {path} ({len(merged)} windows)")


if __name__ == "__main__":
    main()
