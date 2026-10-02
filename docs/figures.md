# Figures and Tables

All figures and tables use the main setting (fault-frequency features only) unless noted, and are regenerated with:

```bash
python scripts/save_predictions.py --features fault_only   # if data/predictions/ is missing
python scripts/make_figures.py
python scripts/make_tables.py
```

Files: `results/figures/` (PNG at 300 dpi + PDF), `results/tables/` (CSV + Markdown).

## Figure 1 — `fig1_risk_coverage`

**Draft caption.** Error among automatically decided cases as a function of the share of cases automated (most
confident first), for unseen artificial-damage bearings (leave one bearing out; solid) and real-damage bearings
(dashed; median and interquartile range over 105 calibration-bearing rotations). Dotted line: 5 % error target.
With the best possible threshold, the models automate 81–91 % of unseen artificial-bearing cases below 5 % error,
but only 25–30 % of real-damage cases (median over rotations).

## Figure 2 — `fig2_reliability`

**Draft caption.** Reliability diagrams (raw probabilities) for unseen artificial bearings and for real damage
(pooled over 105 rotations). Bins with fewer than 20 windows are omitted. Dotted line: perfect calibration.
Models that are close to calibrated on unseen artificial bearings are overconfident on real damage (ECE 0.18–0.25).

**Note for the text:** the SVM's in-domain curve dips at 0.75 confidence because 80 of that bin's 131 windows come
from one held-out artificial bearing (KI01), all misclassified — an illustration of how few bearings each bin rests on.

## Figure 3 — `fig3_per_bearing`

**Draft caption.** Mean confidence and accuracy of each real-damage test bearing, averaged over 105 rotations (raw
probabilities). Dotted line: confidence equals accuracy. Four bearings (KA15, KA22 outer race; KI16, KI17 inner
race) are mostly misclassified while the models remain 69–94 % confident — errors that calibration fitted on
artificial damage cannot detect. Data: `results/per_bearing_real_fault_only.csv`.

## Figure 4 — `fig4_real_calibration`

**Draft caption.** Calibrating on a few labelled real-damage bearings (temperature scaling; confidence threshold
chosen on the calibration set for 5 % error). (a) Error among automated real-damage cases; (b) share of real-damage
cases automated. Points: mean over draws; bars: 2.5th–97.5th percentile over draws (50 draws per real-bearing
setting; artificial values pooled over all 100). Calibrating on real bearings lowers the error from about 0.27
to 0.10–0.11 but does not reliably reach the target, and the system automates fewer cases.

## Table 1 — `table1_main`

Main results per model: in-domain (unseen artificial bearings) vs real damage, with 95 % bootstrap intervals.

## Table 2 — `table2_feature_ablation`

All nine features vs fault-frequency features only (means over 105 rotations).

## Table 3 — `table3_real_calibration`

Calibrating on artificial bearings vs 1 or 2 real bearings per class.
