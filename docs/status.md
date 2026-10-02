# Project Status

_Last updated: 2026-10-02_

## Current phase

Week 3 — robustness runs done: 105 calibration-bearing rotations and an in-domain reference.
Main findings hold across all rotations. Weeks 5–6 — writing. Paper moved to LaTeX for Overleaf (`paper/`); Methods and Results written, compiles cleanly.
Next: Introduction, Related work, Discussion, Abstract.

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
- [x] Target-calibration experiment (1 / 2 real bearings per class vs source calibration; `scripts/run_target_calibration.py`)
- [x] Bootstrap confidence intervals over bearings (`scripts/save_predictions.py`, `scripts/run_bootstrap.py`)

### Week 4 — Analysis
- [x] Figures: risk–coverage, reliability, real-bearing calibration, per-bearing confidence vs accuracy
- [x] Tables: main results with 95 % intervals, feature ablation, real-bearing calibration
- [x] Draft captions (`docs/figures.md`)

### Weeks 5–6 — Writing
- [x] Outline (`docs/paper/outline.md`)
- [x] Methods draft (`docs/paper/03_methods.md`)
- [x] Results draft (`docs/paper/04_results.md`), numbers traced in `docs/paper/numbers.md`
- [x] LaTeX project for Overleaf (`paper/`): sections, tables generated from results, bibliography
- [ ] Introduction, Related work, Discussion, Limitations, Conclusion, Abstract
- [ ] Verify related-work references
- [ ] Supervisor / co-author review
- [ ] arXiv preprint
- [ ] Journal submission

## Open decisions (need your input)

1. **Faculty supervisor / co-author.** Not yet identified.
2. **Target journal.** Affects length and format (see `docs/proposal.md` §11).

## Decided

- **Healthy-bearing split** (2026-10-02): source K001, K002, K004 (K002 for calibration); target K003, K005, K006.
  Reason: K004 and K005 have much lower kurtosis than the other healthy bearings, so each side gets one.
- **Calibration bearings** (2026-10-02): K002, KA09, KI08 (one per class, held out from training).
- **Window length** (2026-10-02): 1 s (64,000 samples), non-overlapping.
- **Framing** (2026-10-02): keep the artificial→real framing; compare per damaged class; report the healthy class
  separately; lead with the trustworthiness findings.
- **Main feature set** (2026-10-02): fault-frequency features only (`--features fault_only`). All nine features
  are reported as an ablation showing that amplitude features overfit to individual bearings.

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
- 2026-10-02 — Fault-frequency-only experiment (bpfo_h1/h2, bpfi_h1/h2; RMS, kurtosis, crest factor, peak-to-peak,
  skewness dropped). 105 rotations + in-domain reference:
  - In-domain accuracy rises to 0.88–0.92 (from 0.49–0.77) and in-domain ECE falls to 0.01–0.09; held-out healthy
    bearings are now recognised (0.89, from 0–0.28). The amplitude features were overfitting to individual bearings.
  - On real damage accuracy is unchanged or better: 0.69 for all three models (all features: 0.65–0.69).
  - The artificial→real gap is now clear: ~0.90 in-domain → ~0.69 on real damage; RF ECE 0.013 → 0.177.
  - Thresholds chosen on artificial data for 5 % error give 24–27 % error on real damage and exceed 5 % in
    every rotation for every model.
  - Ranking quality improves (AURC RF 0.18 → 0.14, SVM 0.21 → 0.14); best possible automation at 5 % error rises
    for RF (8 % → 29 %) and SVM (19 % → 26 %), falls for XGBoost (35 % → 26 %).
  - Conformal marginal coverage falls further below nominal 90 %: 0.65–0.68 (all features 0.80–0.89).
  - Recall on real damage: healthy 0.82–0.85 (from 0.66), inner race 0.73 (from 0.75–0.80), outer race 0.56–0.57.
- 2026-10-02 — Bootstrap (fault_only; 2,000 draws resampling the calibration rotation and the test bearings within
  each class). Mean [95 % interval], temperature scaling:
  - Accuracy on real damage ≈ 0.69 [0.50–0.86] for all models: wide, because there are only 3–6 real bearings per class.
  - Automated error at the 5 % target: RF 0.27 [0.10–0.46], SVM 0.24 [0.08–0.45], XGBoost 0.26 [0.09–0.45]
    — the lower bound is above 5 % for every model.
  - Conformal marginal coverage (nominal 0.90): 0.65–0.68, upper bound 0.84–0.87 — below nominal for every model.
- 2026-10-02 — Real-damage calibration (fault_only; 50 draws each for 1 and 2 real bearings per class; all
  calibration sets scored on the same remaining test bearings). Temperature scaling, mean over models:
  - Automated error at the 5 % target: source calibration 0.25–0.28; 1 real bearing/class 0.16–0.19;
    2 real bearings/class 0.10–0.11. The 5 % target is met in 0–1 % of draws with source calibration,
    36 % with 1 and 51 % with 2 real bearings per class.
  - The price is less automation: 80–89 % of cases automated with source calibration vs 35–39 % with 2 real bearings.
  - ECE: 0.19–0.23 (source) → 0.13–0.17 (2 real bearings). Conformal coverage: 0.64–0.67 → 0.84–0.85 (nominal 0.90).
  - Mixing source and real calibration bearings ("combined") is in between.
- 2026-10-02 — Figures and tables generated (`scripts/make_figures.py`, `scripts/make_tables.py`). Silent real
  bearings with the main feature set: KA15, KA22 (outer race), KI16, KI17 (inner race), all at 69–94 % mean confidence.
- 2026-10-02 — Paper moved to LaTeX for Overleaf. Results reordered (thresholds before reliability; ablation before
  real-bearing calibration) and figures 3/4 renamed so file names match paper numbering.
