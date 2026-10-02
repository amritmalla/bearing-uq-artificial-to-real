# Project Status

_Last updated: 2026-10-02_

## Current phase

Week 1 — data pipeline. Real data downloaded and feature table built (2,309 windows, 29 bearings). Healthy-bearing split decided; ready for week 2 (models).

## Checklist

### Week 1 — Data pipeline
- [x] Proposal written (`docs/proposal.md`)
- [x] Bearing catalogue and 6203 fault-frequency calculation
- [x] `.mat` loader (tested on a synthetic file shaped like the Paderborn format)
- [x] Windowing, time-domain and envelope-spectrum features
- [x] Bearing-level split builder
- [x] Download script and Colab notebook (`notebooks/colab_run.ipynb`)
- [x] Download Paderborn data for `N15_M07_F10` (via Colab notebook)
- [ ] Verify bearing codes against the dataset fact sheets
- [x] Verify loader on real `.mat` files
- [x] Build feature table for all bearings (`data/features/features_N15_M07_F10.csv`)

### Week 2 — Models and uncertainty
- [ ] RF, XGBoost, SVM
- [ ] Temperature scaling / isotonic calibration
- [ ] Split conformal prediction (marginal + class-conditional)
- [ ] Metrics: ECE, Brier, risk–coverage, AURC, automation rate

### Week 3 — Experiments
- [ ] Artificial → real evaluation
- [ ] Target-calibration experiment (0 / 1 / 2 real bearings per class)
- [ ] Bootstrap confidence intervals over bearings

### Week 4 — Analysis
- [ ] Figures and tables

### Weeks 5–6 — Writing
- [ ] Draft
- [ ] Supervisor / co-author review
- [ ] arXiv preprint
- [ ] Journal submission

## Open decisions (need your input)

1. **Calibration bearings within source.** Proposed: KA09, KI08 (healthy calibration bearing K002 is decided).
2. **Window length.** Proposed: 1 s (64,000 samples), non-overlapping.
3. **Faculty supervisor / co-author.** Not yet identified.

## Decided

- **Healthy-bearing split** (2026-10-02): source K001, K002, K004 (K002 for calibration); target K003, K005, K006.
  Reason: K004 and K005 have much lower kurtosis than the other healthy bearings, so each side gets one.

## Log

- 2026-10-02 — Topic chosen (Option B). Proposal and data-pipeline skeleton created.
- 2026-10-02 — Project moved to local folder; added download script and Colab notebook.
- 2026-10-02 — Colab notebook made self-contained (no requirements.txt or project files needed).
- 2026-10-02 — First run on real data. Fault-frequency features separate fault types for artificial damage;
  the signal is weaker and less consistent for real damage (e.g. KA15, KA22, KI17 look close to healthy).
  Found high kurtosis in 4 of 6 healthy bearings (see open decision 1).
