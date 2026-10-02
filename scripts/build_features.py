"""Build the per-window feature table for all study bearings.

Usage (from the project root, after downloading the data into data/raw/):
    python scripts/build_features.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bearing_uq import bearings as B, config  # noqa: E402
from bearing_uq.data.files import missing_files, recording_paths  # noqa: E402
from bearing_uq.data.loader import LoaderError, load_vibration  # noqa: E402
from bearing_uq.data.windows import segment  # noqa: E402
from bearing_uq.features.extract import window_features  # noqa: E402


def bearing_rows(bearing: B.Bearing) -> list[dict]:
    window = int(config.WINDOW_SECONDS * config.SAMPLING_RATE)
    rows = []
    for rec_no, path in enumerate(recording_paths(bearing.code), start=1):
        try:
            signal = load_vibration(path)
        except (LoaderError, FileNotFoundError) as exc:
            print(f"  skipped: {exc}")
            continue
        for win_no, x in enumerate(segment(signal, window, config.WINDOW_OVERLAP)):
            rows.append({"bearing": bearing.code, "label": bearing.label,
                         "origin": bearing.origin, "recording": rec_no,
                         "window": win_no, **window_features(x)})
    return rows


def main() -> None:
    missing = missing_files([b.code for b in B.BEARINGS])
    if missing:
        print(f"Warning: {len(missing)} expected files are missing, e.g. {missing[0]}")

    rows = []
    for bearing in B.BEARINGS:
        print(f"{bearing.code} ({bearing.label}, {bearing.origin})")
        rows.extend(bearing_rows(bearing))

    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out = config.FEATURES_DIR / f"features_{config.OPERATING_CONDITION}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {len(rows)} windows to {out}")


if __name__ == "__main__":
    main()
