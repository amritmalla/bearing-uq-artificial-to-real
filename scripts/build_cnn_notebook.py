"""Build notebooks/colab_cnn.ipynb from the modules in src/bearing_uq/cnn/.

The notebook is self-contained (no project files needed in Colab): each module becomes one code cell, with its
``from bearing_uq.cnn ...`` import lines removed. Rebuild after changing any CNN module.

Usage (from the project root):
    python scripts/build_cnn_notebook.py
"""

import json
from pathlib import Path

CNN = Path("src/bearing_uq/cnn")
OUT = Path("notebooks/colab_cnn.ipynb")

INTRO = """# Bearing fault diagnosis — 1D-CNN baseline (Colab)

Self-contained: no project files needed. **Use a GPU runtime** (Runtime → Change runtime type → T4 GPU); on a
CPU runtime the training takes many hours.

What it does, for the main operating condition (1500 rpm, 0.7 Nm, 1000 N):
1. Downloads the 29 bearings again (~4.6 GB, 20–40 min) and caches their 1-second vibration windows in Drive
   (`data/raw_windows_N15_M07_F10/`, ~0.6 GB). Skipped for bearings already cached.
2. Trains a WDCNN (Zhang et al., 2017) on raw vibration with the project's bearing-level splits:
   15 leave-one-bearing-out models (in-domain reference) and one model per calibration rotation (105).
3. Saves window probabilities to Drive (`data/cnn/`). Every model's output is saved as soon as it finishes, so if
   the session drops, run all cells again and it continues where it stopped.

At the end, download `cnn_results.zip` from Drive (`load_bearing_research_paper/data/`) and unzip it into the
project's `data/` folder, so that `data/cnn/lobo.csv` and `data/cnn/rotations/` exist.

Data licence: CC BY-NC 4.0 — cite Lessmeier et al. (2016) and the Paderborn KAt-DataCenter."""

SETUP = """!which unrar > /dev/null || apt-get -qq install -y unrar > /dev/null

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DRIVE_DATA = Path("/content/drive/MyDrive/load_bearing_research_paper/data")
CACHE_DIR = DRIVE_DATA / "raw_windows_N15_M07_F10"   # cached 1 s windows, one .npz per bearing
OUT_DIR = DRIVE_DATA / "cnn"                          # CNN probabilities
FEATURES_CSV = DRIVE_DATA / "features" / "features_N15_M07_F10.csv"  # from colab_run.ipynb, used as a check

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "- switch to a GPU runtime!")"""

MOUNT = """from google.colab import drive
drive.mount("/content/drive")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)"""

DOWNLOAD = """for b in BEARINGS.itertuples():
    if (CACHE_DIR / f"{b.bearing}.npz").exists():
        print(f"{b.bearing}: already cached")
        continue
    try:
        print(f"{b.bearing}: {cache_bearing(b.bearing, CACHE_DIR)} windows")
    except Exception as e:
        print(f"{b.bearing}: FAILED - {e}  (run this cell again)")"""

LOAD = """x, meta = load_cache(CACHE_DIR, BEARINGS)
print(f"{len(meta)} windows from {meta.bearing.nunique()} bearings")

# The windows must be the same as in the feature table used by the other models.
if FEATURES_CSV.exists():
    keys = ["bearing", "recording", "window"]
    feat = pd.read_csv(FEATURES_CSV, usecols=keys)
    same = feat.merge(meta[keys], how="outer", indicator=True)["_merge"].value_counts()
    print("windows only in features:", same.get("left_only", 0), "| only here:", same.get("right_only", 0))
else:
    print("feature table not found in Drive; skipping the check")

data = Data(x, meta, DEVICE)
del x"""

LOBO = "run_lobo(data, OUT_DIR)"
ROTATIONS = "run_rotations(data, OUT_DIR)"

FINISH = """done = len(list((OUT_DIR / "rotations").glob("*.csv")))
print(f"{done}/105 rotations, lobo.csv present: {(OUT_DIR / 'lobo.csv').exists()}")
if done == 105:
    shutil.make_archive(str(DRIVE_DATA / "cnn_results"), "zip", root_dir=DRIVE_DATA, base_dir="cnn")
    print("wrote", DRIVE_DATA / "cnn_results.zip")"""


def module_cell(name: str) -> str:
    lines = (CNN / f"{name}.py").read_text().splitlines()
    return "\n".join(l for l in lines if not l.startswith("from bearing_uq.cnn")).strip()


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


def main() -> None:
    cells = [
        md(INTRO),
        md("## 1. Setup"), code(SETUP),
        md("## 2. Mount Google Drive"), code(MOUNT),
        md("## 3. Bearings and splits (same as the project's `splits.py`)"), code(module_cell("protocol")),
        md("## 4. Loading and caching windows"), code(module_cell("data")),
        md("## 5. Model: WDCNN"), code(module_cell("model")),
        md("## 6. Training and prediction"), code(module_cell("train")),
        md("## 7. Experiment runner"), code(module_cell("runner")),
        md("## 8. Download and cache the windows (20–40 min, once)"), code(DOWNLOAD),
        md("## 9. Load the windows onto the GPU"), code(LOAD),
        md("## 10. In-domain reference: leave one source bearing out (15 models)"), code(LOBO),
        md("## 11. Artificial → real: one model per calibration rotation (105 models)"), code(ROTATIONS),
        md("## 12. Package the results"), code(FINISH),
    ]
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 0}
    OUT.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
