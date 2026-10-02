# 4. Results

> **Superseded:** the LaTeX version in `paper/sections/` is now the source of truth; edit that instead.

*Draft. Main setting: fault-frequency features. Unless stated otherwise, values are bootstrap means with 95 %
intervals in brackets (Section 3.9), and calibrated results use temperature scaling. Sources for every number are
in `numbers.md`.*

## 4.1 The artificial-to-real gap

On unseen artificial bearings, all three classifiers generalise well: leave-one-bearing-out accuracy is 0.91
(Random Forest), 0.88 (SVM) and 0.91 (XGBoost), and their probabilities are close to calibrated (ECE 0.01, 0.09 and
0.06; Table 1). On real damage, accuracy falls to about 0.69–0.70 for every model (e.g. Random Forest 0.69 [0.52,
0.86]), and ECE rises to 0.17–0.20 after temperature scaling. The wide intervals reflect how strongly individual
real bearings differ (Section 4.4).

The reliability diagrams in Fig. 2 show the nature of the change: on real damage, observed accuracy falls below the
predicted confidence across almost the whole range — the models become systematically overconfident.

Recall on real damage differs by class: 0.82–0.85 for healthy bearings, 0.73 for inner-race and 0.56–0.57 for
outer-race damage (means over rotations). Outer-race damage, the easiest class on artificial bearings
(0.97–0.99 in-domain), is the hardest on real ones.

## 4.2 Confidence thresholds chosen on artificial damage do not transfer

Fig. 1 shows the error among automatically decided cases as more cases are automated. On unseen artificial
bearings, the best possible threshold keeps error below 5 % while automating 81–91 % of cases. On real damage, the
same 5 % is reached at only 25–30 % of cases (median over rotations).

In deployment the threshold must be chosen without real-damage labels. A threshold chosen on the calibration set
for a 5 % error target automates 82–87 % of real-damage cases, with an error of 0.24–0.27 among them (Table 1). The
lower ends of the 95 % intervals (0.07–0.10) are still above the target, and the target is exceeded in all 105
rotations for every model. A stricter 1 % target does not solve the problem: about half of the cases are still
automated (0.48–0.53), with an error of 0.13–0.15.

Post-hoc calibration on artificial bearings does not help reliably. Temperature scaling lowers ECE on real damage
for SVM (0.19 → 0.17) and XGBoost (0.25 → 0.20) but raises it slightly for Random Forest (0.18 → 0.19); isotonic
regression makes it worse for all three (0.23–0.24).

## 4.3 Conformal prediction under-covers

Split conformal prediction with a nominal coverage of 0.90 reaches 0.65–0.68 on real damage (Table 1), and even the
upper ends of the intervals (0.84–0.87) stay below nominal. Mean prediction-set sizes are only 0.90–0.96 classes:
83–89 % of sets contain a single class, and 25–27 % of those single-class sets are wrong. Class-conditional
(Mondrian) conformal prediction is no better (coverage 0.63–0.64).

## 4.4 Silent bearings

Fig. 4 shows that the failures concentrate on a few bearings. Two outer-race bearings, KA15 and KA22, are almost
never recognised (accuracy 0.02–0.07), and two inner-race bearings, KI16 and KI17, are recognised in only 29–44 % of
windows. The models remain confident on all four: mean confidence is 0.69–0.94, and 0.83–0.93 for the two outer-race
bearings. The remaining ten test bearings are recognised in at least 68 % of windows. Confidently misclassified
bearings are rare among unseen artificial bearings — none for Random Forest and XGBoost, one for the SVM (KI01,
accuracy 0.00 at mean confidence 0.76) — so calibration fitted on artificial damage has little chance to anticipate
them.

## 4.5 Calibrating on labelled real bearings

Table 3 and Fig. 3 compare calibration sets on identical test bearings. With source (artificial) calibration, the
error at the 5 % target is 0.26–0.28 and the target is met in at most 1 % of draws. Calibrating on one real bearing
per class lowers the error to 0.16–0.19 and meets the target in 29–42 % of draws; two real bearings per class lower it
to 0.10–0.11 and meet it in 49–52 % of draws.

The improvement comes mainly from caution: the share of cases automated falls from 0.81–0.88 to 0.51–0.52 (one
bearing per class) and 0.35–0.38 (two). Calibration error improves moderately (ECE 0.19–0.22 → 0.13–0.17 with two
bearings per class), and conformal coverage rises from 0.64–0.67 to 0.84–0.85, still below nominal. Variation
across draws is large (Table 3), as it depends on which real bearings are available for calibration.

## 4.6 Feature ablation

Adding the five amplitude-based time-domain features (Table 2) lowers in-domain accuracy from 0.88–0.91 to
0.49–0.77, while accuracy on real damage barely changes (0.65–0.69 vs 0.69). The amplitude features describe
individual bearings rather than damage: unseen healthy bearings in particular are rarely recognised with them, in
line with the kurtosis differences between healthy bearings noted in Section 3.3. With all features, the
artificial→real gap is therefore hidden by a poor in-domain reference, but the trustworthiness failure remains: the
5 % target is exceeded in 59–92 % of rotations, with errors of 0.12–0.17. Conformal coverage is higher with all
features (0.80–0.89), because the prediction sets are larger (1.3–1.9 classes on average, against 0.9–1.0 with
fault-frequency features only).
