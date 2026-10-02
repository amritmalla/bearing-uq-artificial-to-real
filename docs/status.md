# Project Status

_Last updated: 2026-10-02_

## Current phase

Week 3 — robustness runs done: 105 calibration-bearing rotations and an in-domain reference.
Main findings hold across all rotations. Framing of the in-domain comparison needs a decision (open decision 1).

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
- [x] RF, XGBoost, SVM (`models.py`)
- [x] Temperature scaling / isotonic calibration (`calibration.py`)
- [x] Split conformal prediction, marginal + class-conditional (`conformal.py`)
- [x] Metrics: ECE, Brier, risk–coverage, AURC, automation rate (`metrics.py`)
- [x] First artificial→real run (`scripts/run_baseline.py` → `results/`)

### Week 3 — Experiments
- [x] Artificial → real evaluation repeated over all 105 calibration-bearing choices (`scripts/run_rotations.py`, `summarise_rotations.py`)
- [x] In-domain reference: leave-one-bearing-out within the source bearings (`scripts/run_in_domain.py`)
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

1. **How to present the in-domain comparison.** Unseen-bearing variation is as large as the artificial→real shift
   (see 2026-10-02 log). Proposed: keep the artificial→real framing, compare per damaged class (outer race degrades,
   inner race does not), report the healthy class separately, and lead with the trustworthiness findings, which hold
   in every rotation.
2. **Feature set (optional experiment).** Amplitude-based features (RMS, kurtosis, crest factor, peak-to-peak) vary
   strongly between individual bearings. Proposed: also run with only the self-normalised fault-frequency features
   and compare.
3. **Faculty supervisor / co-author.** Not yet identified.

## Decided

- **Healthy-bearing split** (2026-10-02): source K001, K002, K004 (K002 for calibration); target K003, K005, K006.
  Reason: K004 and K005 have much lower kurtosis than the other healthy bearings, so each side gets one.
- **Calibration bearings** (2026-10-02): K002, KA09, KI08 (one per class, held out from training).
- **Window length** (2026-10-02): 1 s (64,000 samples), non-overlapping.

## Log

- 2026-10-02 — Topic chosen (Option B). Proposal and data-pipeline skeleton created.
- 2026-10-02 — Project moved to local folder; added download script and Colab notebook.
- 2026-10-02 — Colab notebook made self-contained (no requirements.txt or project files needed).
- 2026-10-02 — First run on real data. Fault-frequency features separate fault types for artificial damage;
  the signal is weaker and less consistent for real damage (e.g. KA15, KA22, KI17 look close to healthy).
  Found high kurtosis in 4 of 6 healthy bearings (see open decision 1).
- 2026-10-02 — Week 2 modules built and tested (31 tests). First artificial→real run, one calibration split
  (K002, KA09, KI08). Preliminary findings:
  - Test accuracy on real damage 0.70–0.74. Most real bearings are classified well; four are systematically
    wrong: KA15 (→ inner race), KA22 (→ healthy), KI16 (→ outer race), partly KI17.
  - Models are often confidently wrong on these, e.g. SVM 0.96 mean confidence on KI16 at 1 % accuracy,
    XGBoost 0.91 on KA15 at 4 %. Thresholds chosen on artificial data to give 5 % error gave 12–17 % error on real damage.
  - Conformal coverage fell below the nominal 90 % (marginal 0.73–0.88); class-conditional was worse (0.64–0.71).
  - The "source holdout" reference is unreliable: with one bearing per class it is dominated by K002
    (healthy), which RF and SVM label as inner race. Calibration fitted on such a small set depends heavily on
    which bearings are held out, so results must be repeated over many calibration-bearing choices.
- 2026-10-02 — Robustness runs. Over 105 rotations of the calibration bearings (mean ± std):
  - Accuracy on real damage 0.65–0.69 (std 0.04–0.10). The first single split was on the optimistic side.
  - Thresholds chosen on artificial data for 5 % error give 12–31 % error on real damage; the 5 % target is
    exceeded in 59–96 % of rotations, for every model and calibration method.
  - Even the best possible threshold on real damage automates few cases at 5 % error: RF 8 %, SVM 19 %, XGBoost 35 %.
  - Temperature scaling lowers ECE for SVM and XGBoost (0.21→0.16, 0.22→0.16) but raises it for RF (0.10→0.16).
    Isotonic calibration makes ECE worse for every model.
  - Conformal coverage (nominal 90 %): marginal 0.80–0.89 on average, worst rotation 0.63;
    class-conditional 0.64–0.65, consistently worse.
  - In-domain reference (leave one source bearing out): accuracy RF 0.62, SVM 0.49, XGBoost 0.77. Held-out
    healthy bearings are almost always misclassified (only 2 healthy bearings left for training). Per damaged
    class (RF): outer race 0.83 in-domain → 0.58 on real damage; inner race 0.70 in-domain → 0.80 on real damage.
