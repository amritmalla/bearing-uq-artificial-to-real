# Research Proposal

## Working title

**Trustworthy Bearing Fault Diagnosis Under Artificial-to-Real Damage Shift: Calibration and Selective Automation**

## 1. Motivation

Bearing fault classifiers are usually trained on faults that were made artificially in a lab
(EDM, drilling, engraving), but in a factory they must diagnose damage that developed naturally.
Accuracy is known to drop under this shift. Less is known about whether the models' **confidence**
stays trustworthy, which is what decides whether a diagnosis can be acted on automatically or must
be sent to a human expert.

## 2. Research question

When a bearing fault classifier trained on artificial damage is applied to real damage, which methods
can automatically diagnose faults while keeping the error rate below a target, and what share of
cases must they escalate to a human?

Sub-questions:

1. How much do accuracy and calibration degrade from artificial to real damage?
2. Do post-hoc calibration (temperature scaling / isotonic) and conformal prediction, fitted on
   artificial-damage data, still hold on real damage?
3. How many labelled real-damage bearings (0, 1, 2 per class) are needed to restore trustworthy
   confidence?

## 3. Position relative to prior work

| Work | What it covers | What it does not cover |
|---|---|---|
| González-García et al., *Electronics* 2026 (shapelets) | Calibration and conformal prediction for bearings, recording-level splits, CWRU and MFPT, cross-severity shift | Artificial→real shift; automation rate; small target calibration set |
| Calibrated Adaptive Teacher (PMC 2024) | Calibration in deep domain adaptation on Paderborn | Uses motor current and operating-condition shift, not artificial→real; deep models only |
| Uncertainty-aware Bayesian DL fault diagnosis (2024) | Uncertainty from domain shift, Paderborn and IMS | Deep Bayesian models; no automation/selective framing |
| Lessmeier et al. 2016 (Paderborn dataset) | Shows artificial→real generalisation gap | Accuracy only |

**Contribution:** the first evaluation of calibration and selective automation under the
artificial→real damage shift, using physics-based features and standard models, including how much
real-damage calibration data is needed.

## 4. Data

- **Dataset:** Paderborn University KAt bearing dataset (bearing type 6203), vibration channel,
  64 kHz, 20 recordings × 4 s per bearing per operating condition.
- **Licence:** CC BY-NC 4.0; citation of Lessmeier et al. and the KAt-DataCenter is required.
- **Operating condition:** one condition only, `N15_M07_F10` (1500 rpm, 0.7 Nm, 1000 N).
- **Classes:** healthy, inner-race fault, outer-race fault. Combined-damage bearings (KB*) excluded.

| Group | Bearing codes |
|---|---|
| Healthy | K001–K006 |
| Artificial outer race | KA01, KA03, KA05, KA06, KA07, KA08, KA09 |
| Artificial inner race | KI01, KI03, KI05, KI07, KI08 |
| Real outer race | KA04, KA15, KA16, KA22, KA30 |
| Real inner race | KI04, KI14, KI16, KI17, KI18, KI21 |

Codes must be checked against the dataset's own fact sheets after download.

## 5. Splits (bearing level — a bearing never appears in two splits)

- **Source (train + calibration):** artificial-damage bearings + healthy K001, K002, K004.
  Calibration set = held-out source bearings (K002, KA09, KI08).
- **Target (test):** real-damage bearings + healthy K003, K005, K006.
- **Why this healthy split:** K001, K002, K003 and K006 have median kurtosis 14–16, K004 and K005 only 4–5
  (an impulsive component unrelated to bearing damage). Each side gets both kinds so the healthy class
  does not itself shift between source and target.
- **Target-calibration experiment:** move 1 or 2 real bearings per class from test into the
  calibration set; evaluate on the remaining real bearings.

## 6. Features

Computed per window (default 1 s windows):

- Time domain: RMS, kurtosis, crest factor, skewness, peak-to-peak.
- Envelope spectrum (band-pass + Hilbert): energy at BPFO and BPFI and their first harmonics,
  normalised by the envelope spectrum's median level.

Fault frequencies for 6203 at 25 Hz shaft speed: BPFO ≈ 76.4 Hz, BPFI ≈ 123.6 Hz.

## 7. Methods

- Classifiers: Random Forest, XGBoost, SVM.
- Uncertainty: raw probabilities; temperature scaling or isotonic calibration; split conformal
  prediction (marginal and class-conditional).

## 8. Evaluation

- Accuracy, macro-F1.
- Expected calibration error (ECE), Brier score.
- Risk–coverage curves and area under them (AURC).
- **Automation rate** at a fixed error target (e.g. 1%, 5%).
- Conformal empirical coverage and set size.
- Uncertainty: per-bearing results and bootstrap confidence intervals over bearings.

## 9. Timeline (part-time)

| Week | Work |
|---|---|
| 1 | Download data, loader, bearing-level splits, features |
| 2 | Classifiers, calibration, conformal prediction |
| 3 | Artificial→real experiments; target-calibration experiment |
| 4 | Risk–coverage analysis, figures |
| 5–6 | Writing, supervisor review, arXiv preprint, journal submission |

## 10. Risks

| Risk | Mitigation |
|---|---|
| Accuracy on real damage collapses | Fault-frequency features; report the collapse as a finding |
| Few real-damage bearings | Bootstrap over bearings; state limitation |
| Overlap with the 2026 *Electronics* paper | Cite it; differences are the shift type, automation framing and target-calibration experiment |

## 11. Target venues

*Sensors*, *Machines*, *Electronics* (MDPI); stretch: *Mechanical Systems and Signal Processing*.

## References

- Lessmeier, C. et al. (2016). Condition monitoring of bearing damage in electromechanical drive
  systems by using motor current signals of electric motors: a benchmark data set for data-driven
  classification. PHM Society European Conference.
- KAt-DataCenter, Chair of Design and Drive Technology, Paderborn University.
  https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter
- González-García, L. et al. (2026). Shapelet-Based Bearing Fault Diagnosis Under Interpretability
  Constraints: A Recording-Level Evaluation. *Electronics* 15(14), 3035.
  https://doi.org/10.3390/electronics15143035
- Calibrated Adaptive Teacher for Domain-Adaptive Intelligent Fault Diagnosis.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11644747/
- Guo, C. et al. (2017). On calibration of modern neural networks. ICML.
- Angelopoulos, A. N., Bates, S. (2023). A gentle introduction to conformal prediction.
  *Foundations and Trends in Machine Learning* 16, 494–591.
