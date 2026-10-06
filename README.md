# Calibration, Conformal Prediction and Selective Automation for Bearing Fault Diagnosis: From Artificial to Real Damage

Code and results for a study of whether calibrated confidence in bearing fault diagnosis survives the shift from
artificially made faults to real damage, on the Paderborn University bearing dataset. Random Forest, SVM and
XGBoost on envelope-spectrum features, and a 1D-CNN on raw vibration (also with AdaBN domain adaptation), are
trained and calibrated on artificially damaged bearings and tested on bearings with real damage, with bearing-level
splits, 105 calibration-bearing rotations and bootstrap intervals over bearings.

- Paper (LaTeX, IJPHM format): [`paper/`](paper/)
- Every number cited in the paper and the result file it comes from: [`docs/paper/numbers.md`](docs/paper/numbers.md)

## Setup

```bash
pip install -r requirements.txt
pytest
```

The 1D-CNN is trained in Google Colab (PyTorch is preinstalled there); its tests are skipped when PyTorch is not
installed locally.

## Data

The Paderborn bearing data are **not included** in this repository. They are available from the
[KAt-DataCenter](https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter/data-sets-and-download) of Paderborn
University under the CC BY-NC 4.0 licence; cite Lessmeier et al. (2016) and the KAt-DataCenter. The notebooks below
download them automatically. The derived results in `results/` are provided for reproducibility under the same
non-commercial terms.

## Reproducing the paper

All commands run from the project root. Scripts with `--max-seconds` stop cleanly after that time and resume where
they stopped when run again.

**1. Feature tables** (`data/features/features_<condition>.csv`, all four operating conditions). Either run
`notebooks/colab_run.ipynb` in Colab (self-contained; downloads the data, writes the tables to Google Drive) and
copy the four CSV files into `data/features/`, or locally:

```bash
BEARING_RAW_DIR=data/raw python scripts/download_paderborn.py     # needs unrar
python scripts/build_features.py --condition N15_M07_F10          # repeat for each condition
```

**2. Feature-based models.** Main setting `fault_only`; `all` is the feature ablation, `fault_only_fk` the adaptive
(fast-kurtogram) band and `fault_only_sk` an earlier fixed-width spectral-kurtosis band. Add
`--condition <condition>` for the other operating conditions.

The kurtogram features (main condition only) come from `notebooks/colab_kurtogram.ipynb` (Colab, CPU); copy
`kurtogram_N15_M07_F10.csv` into `data/features/` and run `python scripts/add_kurtogram_features.py`. The local
pipeline (`build_features.py`) computes them directly.

```bash
python scripts/run_rotations.py --features fault_only             # 105 calibration rotations
python scripts/summarise_rotations.py --features fault_only
python scripts/run_in_domain.py --features fault_only             # leave-one-bearing-out reference
python scripts/save_predictions.py --features fault_only          # per-rotation predictions
python scripts/run_bootstrap.py --features fault_only             # 95 % intervals over bearings
python scripts/run_target_calibration.py --features fault_only    # calibrating on labelled real bearings
python scripts/error_breakdown.py --predictions fault_only         # missed faults, per-bearing accuracy
```

**3. 1D-CNN.** Run `notebooks/colab_cnn.ipynb` in Colab on a GPU runtime and unzip `cnn_results.zip` into `data/`
(giving `data/cnn/lobo.csv`, `data/cnn/rotations/`, `data/cnn/random_split.csv` and, for AdaBN,
`data/cnn/adabn/rotations/`). Then:

```bash
python scripts/run_cnn_analysis.py              # same calibration, conformal and selective-automation analysis
python scripts/summarise_rotations.py --features cnn
python scripts/run_bootstrap.py --features cnn
python scripts/error_breakdown.py --predictions cnn
python scripts/random_split_check.py            # sanity check: random window split vs split by bearing
python scripts/run_cnn_analysis.py --variant adabn   # the same with AdaBN; then the three scripts above
                                                     # with --features / --predictions cnn_adabn
```

**4. Figures, tables and paper assets.**

```bash
python scripts/make_figures.py                  # results/figures/
python scripts/make_tables.py                   # results/tables/
python scripts/build_paper_assets.py            # LaTeX tables and figure PDFs -> paper/
```

Upload the `paper/` folder to Overleaf (main file `paper/main.tex`, class `paper/ijphm.cls`).

## Layout

```
src/bearing_uq/
  config.py            settings (paths, sampling rate, operating conditions, windows)
  bearings.py          bearing codes, labels and damage origin
  geometry.py          6203 geometry and fault frequencies
  splits.py            bearing-level train / calib / test splits and the 105 rotations
  dataset.py           feature table -> model inputs; feature sets
  models.py            Random Forest, SVM, XGBoost
  calibration.py       temperature scaling, isotonic calibration
  conformal.py         split conformal prediction (marginal, class-conditional)
  metrics.py           accuracy, ECE, Brier, risk-coverage, automation rate
  experiment.py        main artificial-to-real experiment
  in_domain.py         leave-one-bearing-out reference on source bearings
  predictions.py       save / load per-rotation test predictions
  bootstrap.py         hierarchical bootstrap over rotations and bearings
  target_calibration.py  calibrating on labelled real bearings
  precomputed.py       serves saved probabilities (the CNN's) through the scikit-learn interface
  errors.py            missed-fault shares
  tables.py            table formatting
  plots/               figure style, curve helpers, one module per figure
  data/                file paths, download, .mat loader, windowing
  features/            time-domain, envelope spectrum, fault-band, spectral-kurtosis and fast-kurtogram features
  cnn/                 1D-CNN (WDCNN-style): splits, data cache, model, training, experiment runner, AdaBN
scripts/                one script per step (see "Reproducing the paper")
  build_cnn_notebook.py  rebuilds notebooks/colab_cnn.ipynb from src/bearing_uq/cnn/
  build_kurtogram_notebook.py  rebuilds notebooks/colab_kurtogram.ipynb
  run_baseline.py        single calibration split (first exploratory run)
notebooks/colab_run.ipynb   feature tables for all four operating conditions (Colab)
notebooks/colab_cnn.ipynb   1D-CNN training and AdaBN (Colab, GPU)
notebooks/colab_kurtogram.ipynb  fast-kurtogram band features (Colab)
results/                    result tables (CSV), figures and paper tables
paper/                      LaTeX manuscript (IJPHM format)
docs/paper/numbers.md       every number cited in the paper, with its source file
tests/                      unit tests on synthetic data
```

## Licence

Code: MIT licence (see [`LICENSE`](LICENSE)). Data: Paderborn KAt-DataCenter, CC BY-NC 4.0 (not included).
