"""Build notebooks/colab_kurtogram.ipynb: fast-kurtogram band features for the main operating condition.

The notebook is self-contained: the needed project modules become code cells, with their ``from bearing_uq ...``
import lines removed. It reuses the 1 s windows cached in Drive by colab_cnn.ipynb (and caches any that are
missing). Rebuild after changing any of the modules listed in MODULES.

Usage (from the project root):
    python scripts/build_kurtogram_notebook.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_cnn_notebook import DOWNLOAD, MOUNT, code, md  # noqa: E402

SRC = Path("src/bearing_uq")
OUT = Path("notebooks/colab_kurtogram.ipynb")
MODULES = [("Bearings (same as the project's splits)", "cnn/protocol.py"),
           ("Loading and caching windows (same as colab_cnn.ipynb)", "cnn/data.py"),
           ("Bearing geometry and fault frequencies", "geometry.py"),
           ("Envelope spectrum", "features/envelope.py"),
           ("Fault-frequency features", "features/fault_bands.py"),
           ("Fast kurtogram", "features/kurtogram.py")]

INTRO = """# Bearing fault diagnosis — fast-kurtogram demodulation band (Colab)

Self-contained: no project files needed. A CPU runtime is enough.

For every 1-second window at the main operating condition (1500 rpm, 0.7 Nm, 1000 N), selects the demodulation
band with the fast kurtogram (Antoni, 2007), which searches over band centres *and* widths, and computes the four
fault-frequency features in that band (suffix `_fk`).

It uses the windows cached in Drive by `colab_cnn.ipynb` (`data/raw_windows_N15_M07_F10/`); bearings not cached
yet are downloaded first (~160 MB each). Results are saved per bearing, so the notebook can be re-run after a
disconnection.

At the end, download `data/features/kurtogram_N15_M07_F10.csv` from Drive into the project's `data/features/`
folder and run `python scripts/add_kurtogram_features.py`.

Data licence: CC BY-NC 4.0 — cite Lessmeier et al. (2016) and the Paderborn KAt-DataCenter."""

SETUP = """!which unrar > /dev/null || apt-get -qq install -y unrar > /dev/null

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

DRIVE_DATA = Path("/content/drive/MyDrive/load_bearing_research_paper/data")
CACHE_DIR = DRIVE_DATA / "raw_windows_N15_M07_F10"       # cached 1 s windows, one .npz per bearing
OUT_DIR = DRIVE_DATA / "features" / "kurtogram_parts"    # one file per bearing
OUT_CSV = DRIVE_DATA / "features" / "kurtogram_N15_M07_F10.csv"
FS = 64_000
SHAFT_HZ = 1500 / 60"""

COMPUTE = """FAULT_FREQS = {"bpfo": bpfo(SHAFT_HZ), "bpfi": bpfi(SHAFT_HZ)}
print({k: round(v, 2) for k, v in FAULT_FREQS.items()})

for b in BEARINGS.itertuples():
    part = OUT_DIR / f"{b.bearing}.csv"
    if part.exists():
        continue
    d = np.load(CACHE_DIR / f"{b.bearing}.npz")
    rows = [{"bearing": b.bearing, "recording": int(r), "window": int(w),
             **kurtogram_features(x.astype(float), FS, FAULT_FREQS)}
            for x, r, w in zip(d["x"], d["recording"], d["window"])]
    pd.DataFrame(rows).to_csv(part, index=False)
    print(f"{b.bearing}: {len(rows)} windows")"""

COMBINE = """table = pd.concat([pd.read_csv(OUT_DIR / f"{c}.csv") for c in BEARINGS.bearing], ignore_index=True)
table.to_csv(OUT_CSV, index=False)
print(f"wrote {OUT_CSV} ({len(table)} windows)")"""

CHECK = """t = table.merge(BEARINGS, on="bearing")
print(t.groupby(["origin", "label"])[["bpfo_h1_fk", "bpfi_h1_fk"]].median().round(2))
print()
print("selected band width (Hz):")
print((t.fk_band_high_hz - t.fk_band_low_hz).describe().round(0))
print()
print("selected band centre (kHz), median per bearing:")
print(((t.fk_band_low_hz + t.fk_band_high_hz) / 2000).groupby(t.bearing).median().round(1).to_string())"""


def module_cell(rel: str) -> str:
    lines = (SRC / rel).read_text().splitlines()
    return "\n".join(l for l in lines if not l.startswith("from bearing_uq")).strip()


def main() -> None:
    cells = [md(INTRO), md("## 1. Setup"), code(SETUP), md("## 2. Mount Google Drive"), code(MOUNT)]
    for i, (title, rel) in enumerate(MODULES, start=3):
        cells += [md(f"## {i}. {title}"), code(module_cell(rel))]
    n = len(MODULES) + 3
    cells += [md(f"## {n}. Cache any windows not cached yet (skipped if colab_cnn.ipynb already did it)"),
              code(DOWNLOAD),
              md(f"## {n + 1}. Kurtogram band and features for every window (a few minutes)"), code(COMPUTE),
              md(f"## {n + 2}. Combine into one table"), code(COMBINE),
              md(f"## {n + 3}. Sanity check\n\nMedian features per group, and the bands the kurtogram chose."),
              code(CHECK)]
    nb = {"cells": cells, "metadata": {"colab": {"provenance": []},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 0}
    OUT.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
