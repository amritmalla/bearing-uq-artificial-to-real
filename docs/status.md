# Project Status

_Last updated: 2026-10-02_

## Current phase

Week 1 — data pipeline. Code skeleton written and tested on synthetic signals; real data not yet downloaded.

## Checklist

### Week 1 — Data pipeline
- [x] Proposal written (`docs/proposal.md`)
- [x] Bearing catalogue and 6203 fault-frequency calculation
- [x] `.mat` loader (tested on a synthetic file shaped like the Paderborn format)
- [x] Windowing, time-domain and envelope-spectrum features
- [x] Bearing-level split builder
- [ ] Download Paderborn data for `N15_M07_F10` (needs to run on your computer; ~32 bearings)
- [ ] Verify bearing codes against the dataset fact sheets
- [ ] Verify loader on real `.mat` files
- [ ] Build feature table for all bearings

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

1. **Healthy-bearing split.** Proposed: K001–K003 in source, K004–K006 in target.
2. **Calibration bearings within source.** Proposed: hold out one bearing per class (e.g. KA09, KI08, K003).
3. **Window length.** Proposed: 1 s (64,000 samples), non-overlapping.
4. **Faculty supervisor / co-author.** Not yet identified.

## Log

- 2026-10-02 — Topic chosen (Option B). Proposal and data-pipeline skeleton created.
