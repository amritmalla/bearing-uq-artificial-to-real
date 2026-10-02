"""Build the per-window feature table for one operating condition.

Usage (from the project root, after downloading the data into data/raw/):
    python scripts/build_features.py [--condition N15_M07_F10]

Writes data/features/features_<condition>.csv with time-domain features, fixed-band
fault features and spectral-kurtosis-band fault features (suffix _sk).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bearing_uq import bearings as B, config  # noqa: E402
from bearing_uq.data.files import missing_files, recording_paths  # noqa: E402
from bearing_uq.data.loader import LoaderError, load_vibration  # noqa: E402
from bearing_uq.data.windows import segment  # noqa: E402
from bearing_uq.features.extract import default_fault_freqs, window_features  # noqa: E402


def bearing_rows(bearing: B.Bearing, condition: str) -> list[dict]:
    window = int(config.WINDOW_SECONDS * config.SAMPLING_RATE)
    fault_freqs = default_fault_freqs(config.shaft_speed_hz(condition))
    rows = []
    for rec_no, path in enumerate(recording_paths(bearing.code, condition), start=1):
        try:
            signal = load_vibration(path)
        except (LoaderError, FileNotFoundError) as exc:
            print(f"  skipped: {exc}")
            continue
        for win_no, x in enumerate(segment(signal, window, config.WINDOW_OVERLAP)):
            rows.append({"bearing": bearing.code, "label": bearing.label,
                         "origin": bearing.origin, "recording": rec_no, "window": win_no,
                         **window_features(x, fault_freqs=fault_freqs, adaptive=True)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=config.CONDITIONS, default=config.OPERATING_CONDITION)
    condition = parser.parse_args().condition

    missing = missing_files([b.code for b in B.BEARINGS], condition)
    if missing:
        print(f"Warning: {len(missing)} expected files are missing, e.g. {missing[0]}")

    rows = []
    for bearing in B.BEARINGS:
        print(f"{bearing.code} ({bearing.label}, {bearing.origin})")
        rows.extend(bearing_rows(bearing, condition))

    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out = config.features_path(condition)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {len(rows)} windows to {out}")


if __name__ == "__main__":
    main()
