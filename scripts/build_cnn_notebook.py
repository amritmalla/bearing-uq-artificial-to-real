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
3. As a sanity check, trains one model on a random split of windows (the same bearings in training and test).
4. AdaBN (domain adaptation): for each rotation, trains the model again and re-estimates its batch-norm statistics
   on unlabelled windows of real-damage and target healthy bearings — for each target bearing, on the *other* target
   bearings only.
5. Saves window probabilities to Drive (`data/cnn/`). Every model's output is saved as soon as it finishes, so if
   the session drops, run all cells again and it continues where it stopped. Steps already done in an earlier run
   (files present in Drive) are skipped.

At the end, download `cnn_results.zip` and `adabn_results.zip` from Drive (`load_bearing_research_paper/data/`) and
unzip them into the project's `data/` folder, so that `data/cnn/lobo.csv`, `data/cnn/rotations/` and
`data/cnn/adabn/rotations/` exist.

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
RANDOM_SPLIT = "run_random_split(data, OUT_DIR)"
ADABN = "run_adabn(data, OUT_DIR)"

ADABN_CHECK = """# The unadapted predictions should repeat the first run (same seed and settings); a small difference is possible
# on a different GPU or PyTorch version.
keys = ["bearing", "recording", "window"]
diffs = []
for f in sorted((OUT_DIR / "adabn" / "source_bn" / "rotations").glob("*.csv")):
    old = OUT_DIR / "rotations" / f.name
    if old.exists():
        m = pd.read_csv(f).merge(pd.read_csv(old), on=keys, suffixes=("", "_old"))
        diffs.append(max((m[c] - m[c + "_old"]).abs().max() for c in PROB_COLS))
print(f"{len(diffs)} rotations compared; largest probability difference from the first run: {max(diffs):.2e}"
      if diffs else "no earlier rotation files to compare")"""

FINISH = """done = len(list((OUT_DIR / "rotations").glob("*.csv")))
print(f"{done}/105 rotations, lobo.csv present: {(OUT_DIR / 'lobo.csv').exists()}")
if done == 105:
    shutil.make_archive(str(DRIVE_DATA / "cnn_results"), "zip", root_dir=DRIVE_DATA, base_dir="cnn")
    print("wrote", DRIVE_DATA / "cnn_results.zip")
done = len(list((OUT_DIR / "adabn" / "rotations").glob("*.csv")))
print(f"AdaBN: {done}/105 rotations")
if done == 105:
    shutil.make_archive(str(DRIVE_DATA / "adabn_results"), "zip", root_dir=DRIVE_DATA, base_dir="cnn/adabn")
    print("wrote", DRIVE_DATA / "adabn_results.zip")"""


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
        md("## 8. AdaBN (domain adaptation by batch-norm statistics)"), code(module_cell("adabn")),
        md("## 9. Download and cache the windows (20–40 min, once)"), code(DOWNLOAD),
        md("## 10. Load the windows onto the GPU"), code(LOAD),
        md("## 11. In-domain reference: leave one source bearing out (15 models)"), code(LOBO),
        md("## 12. Artificial → real: one model per calibration rotation (105 models)"), code(ROTATIONS),
        md("## 13. Sanity check: random window split (1 model)\n\nTrains on a random 80 % of the source bearings' "
           "windows and tests on the other 20 %, so the same bearings appear in training and test. High accuracy "
           "here shows the network works; the bearing-level results show how much of that is bearing identity."),
        code(RANDOM_SPLIT),
        md("## 14. AdaBN: one model per rotation, adapted per target bearing (105 models)\n\nEach rotation's "
           "model is trained again with the same seed and settings as in step 12. For each target bearing, a copy "
           "of the model gets batch-norm statistics re-estimated on the unlabelled windows of the *other* target "
           "bearings, so no test bearing is used to adapt its own prediction. Calibration bearings keep the "
           "source statistics. About as long as step 12."),
        code(ADABN),
        md("## 15. Check: the unadapted models repeat the first run"), code(ADABN_CHECK),
        md("## 16. Package the results"), code(FINISH),
    ]
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 0}
    OUT.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
