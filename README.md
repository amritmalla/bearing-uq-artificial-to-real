# Bearing Fault Diagnosis Under Artificial-to-Real Damage Shift

Research code for a study of calibration and selective automation in bearing fault diagnosis,
using the Paderborn University bearing dataset.

- Proposal: [`docs/proposal.md`](docs/proposal.md)
- Project status and open decisions: [`docs/status.md`](docs/status.md)

## Setup

```bash
pip install -r requirements.txt
pytest
```

## Run in Google Colab

Upload `notebooks/colab_run.ipynb` to Colab and run the cells in order. The notebook is
self-contained (no other project files needed): it downloads the data to Colab's temporary disk
and saves the feature table to `MyDrive/load_bearing_research_paper/data/features/` in Google Drive.

## Data

The download script fetches the study bearings and keeps only the `N15_M07_F10` recordings:

```bash
BEARING_RAW_DIR=data/raw python scripts/download_paderborn.py
```

It needs `unrar`. Alternatively, download the Paderborn bearing data manually from the
[KAt-DataCenter](https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter/data-sets-and-download)
and extract one folder per bearing into `data/raw/`, e.g. `data/raw/KA01/N15_M07_F10_KA01_1.mat`.
The data is licensed CC BY-NC 4.0; cite Lessmeier et al. (2016) and the KAt-DataCenter.

Then build the feature table:

```bash
python scripts/build_features.py
```

## Layout

```
src/bearing_uq/
  config.py            settings (paths, sampling rate, operating condition, windows)
  bearings.py          bearing codes, labels and damage origin
  geometry.py          6203 geometry and fault frequencies
  splits.py            bearing-level train / calib / test splits
  data/files.py        recording file paths
  data/download.py     download archives, extract one operating condition
  data/loader.py       read vibration signal from .mat
  data/windows.py      fixed-length windowing
  features/time_domain.py   RMS, kurtosis, crest factor, ...
  features/envelope.py      envelope spectrum
  features/fault_bands.py   energy at BPFO / BPFI harmonics
  features/extract.py       all features for one window
scripts/download_paderborn.py   download and extract the study data
scripts/build_features.py   build the feature table
notebooks/colab_run.ipynb   run the pipeline in Google Colab
tests/                      unit tests on synthetic signals
```
