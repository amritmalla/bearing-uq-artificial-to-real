# Numbers Cited in the Draft

Every number in the paper text should appear here with its source, so the text can be re-checked after any re-run.
Main setting: fault-frequency features, temperature scaling, unless noted. "Rotations" = 105 calibration-bearing choices.

## Data and splits
| Quantity | Value | Source |
|---|---|---|
| Windows (1 s) | 2,309; 78–80 per bearing | `data/features/features_N15_M07_F10.csv` |
| Bearings | 6 healthy; 12 artificial (7 outer, 5 inner); 11 real (5 outer, 6 inner) | `src/bearing_uq/bearings.py` |
| Healthy median kurtosis | K001 13.6, K002 14.9, K003 15.8, K006 14.8; K004 5.1, K005 4.2 | feature table |
| Windows per split | train 954–957, calibration 237–240, test 1,115 (14 bearings, same in every rotation) | `splits.py` |

## In-domain (leave one source bearing out, raw probabilities)
| Model | Accuracy | ECE | Per-class accuracy (healthy / inner / outer) | Best automation at 5 % |
|---|---|---|---|---|
| Random Forest | 0.91 | 0.01 | 0.89 / 0.85 / 0.97 | 0.91 |
| SVM | 0.88 | 0.09 | 0.89 / 0.73 / 0.99 | 0.81 |
| XGBoost | 0.91 | 0.06 | 0.90 / 0.84 / 0.97 | 0.91 |
Source: `results/in_domain_summary_fault_only.csv`, `results/in_domain_per_bearing_fault_only.csv`.

## Real damage — bootstrap mean [95 % interval] (`results/bootstrap_fault_only.csv`, `results/tables/table1_main.csv`)
| Model | Accuracy | ECE (temp.) | ECE (raw) | AURC | Error at 5 % target | Automated at 5 % | Best automation at 5 % | Conformal coverage |
|---|---|---|---|---|---|---|---|---|
| RF | 0.69 [0.52, 0.86] | 0.19 [0.05, 0.35] | 0.18 [0.05, 0.32] | 0.15 [0.04, 0.34] | 0.27 [0.10, 0.46] | 0.87 [0.66, 1.00] | 0.32 [0.00, 0.71] | 0.65 [0.43, 0.84] |
| SVM | 0.69 [0.50, 0.86] | 0.17 [0.04, 0.35] | 0.19 [0.06, 0.33] | 0.15 [0.03, 0.34] | 0.24 [0.07, 0.45] | 0.82 [0.47, 1.00] | 0.31 [0.03, 0.75] | 0.68 [0.45, 0.87] |
| XGB | 0.70 [0.52, 0.85] | 0.20 [0.06, 0.35] | 0.25 [0.10, 0.41] | 0.15 [0.04, 0.33] | 0.26 [0.09, 0.44] | 0.86 [0.69, 1.00] | 0.30 [0.00, 0.72] | 0.67 [0.45, 0.85] |

## Real damage — means over rotations (`results/rotations_*_fault_only.csv`)
| Quantity | RF | SVM | XGB |
|---|---|---|---|
| Macro-F1 | 0.68 | 0.68 | 0.68 |
| ECE raw / temperature / isotonic | 0.18 / 0.19 / 0.24 | 0.19 / 0.17 / 0.23 | 0.25 / 0.20 / 0.23 |
| Recall healthy / inner / outer | 0.84 / 0.73 / 0.56 | 0.82 / 0.73 / 0.57 | 0.85 / 0.73 / 0.56 |
| Rotations exceeding the 5 % target | 100 % | 100 % | 100 % |
| 1 % target: automated / error | 0.53 / 0.15 | 0.53 / 0.14 | 0.48 / 0.13 |
| Best automation at 5 %, median [IQR] over rotations | 0.30 [0.27, 0.32] | 0.25 [0.23, 0.31] | 0.28 [0.23, 0.33] |
| Conformal marginal: coverage / set size / singleton error | 0.65 / 0.90 / 0.27 | 0.68 / 0.96 / 0.25 | 0.66 / 0.93 / 0.27 |
| Conformal class-conditional: coverage | 0.63 | 0.63 | 0.64 |

## Silent bearings (real damage, raw, mean over rotations; `results/per_bearing_real_fault_only.csv`)
| Bearing | Accuracy (RF / SVM / XGB) | Mean confidence (RF / SVM / XGB) |
|---|---|---|
| KA15 (outer) | 0.02 / 0.03 / 0.03 | 0.84 / 0.84 / 0.92 |
| KA22 (outer) | 0.07 / 0.07 / 0.07 | 0.84 / 0.83 / 0.93 |
| KI16 (inner) | 0.38 / 0.44 / 0.40 | 0.69 / 0.76 / 0.86 |
| KI17 (inner) | 0.34 / 0.41 / 0.29 | 0.82 / 0.84 / 0.94 |

What the silent bearings are predicted as (share of windows, RF / SVM / XGB; raw, all rotations):
- KA15: healthy 0.80 / 0.80 / 0.79, inner race 0.17 / 0.16 / 0.18
- KA22: healthy 0.85 / 0.83 / 0.85
- KI16: inner 0.38 / 0.44 / 0.40, outer 0.38 / 0.44 / 0.42, healthy 0.24 / 0.12 / 0.18
- KI17: healthy 0.65 / 0.58 / 0.70, inner 0.34 / 0.41 / 0.29

Median fault-frequency features (feature table):
- Silent bearings: KA15 bpfo_h1 2.21; KA22 2.16; KI16 bpfi_h1 2.83 (bpfo_h1 3.17); KI17 bpfi_h1 2.20
- Healthy bearings: bpfo_h1 1.45–2.64, bpfi_h1 1.64–2.12
- Artificial outer race bpfo_h1 5.24–41.91; artificial inner race bpfi_h1 3.84–15.28
- Other real outer race bpfo_h1: KA04 16.78, KA16 24.36, KA30 4.42; other real inner bpfi_h1: KI04 8.47, KI14 4.29, KI18 18.46, KI21 3.38

Missed faults (damaged predicted healthy) as share of errors on real damage, temperature scaling, mean over rotations:
all errors RF 0.67 / SVM 0.62 / XGB 0.68; automated errors at the 5 % target RF 0.74 / SVM 0.72 / XGB 0.74.

Other 10 real-test bearings: accuracy ≥ 0.68 (RF 0.68, SVM 0.70, XGB 0.73 minimum).
Unseen artificial bearings with accuracy < 0.5 (in-domain): only SVM on KI01 (0.00 at mean confidence 0.76).
Conformal marginal singleton rate (fault_only): RF 0.87, SVM 0.83, XGB 0.89.

## Real-bearing calibration (`results/tables/table3_real_calibration.csv`; mean over draws)
| Calibrated on | Error at 5 % (RF / SVM / XGB) | Draws meeting 5 % | Automated | ECE | Conformal coverage |
|---|---|---|---|---|---|
| Artificial bearings | 0.28 / 0.26 / 0.28 | 0 % / 1 % / 0 % | 0.87 / 0.81 / 0.88 | 0.21 / 0.19 / 0.22 | 0.64 / 0.67 / 0.65 |
| 1 real bearing/class | 0.19 / 0.18 / 0.16 | 29 % / 35 % / 42 % | 0.51 / 0.52 / 0.52 | 0.20 / 0.18 / 0.19 | 0.78 / 0.77 / 0.79 |
| 2 real bearings/class | 0.11 / 0.10 / 0.11 | 51 % / 52 % / 49 % | 0.38 / 0.37 / 0.35 | 0.17 / 0.13 / 0.16 | 0.85 / 0.85 / 0.84 |

## Feature ablation (`results/tables/table2_feature_ablation.csv`; means over rotations)
| Features | In-domain accuracy | Real accuracy | Error at 5 % | Runs exceeding 5 % | Conformal coverage |
|---|---|---|---|---|---|
| All 9 | 0.62 / 0.49 / 0.77 | 0.69 / 0.65 / 0.68 | 0.15 / 0.12 / 0.17 | 92 % / 59 % / 91 % | 0.85 / 0.89 / 0.80 |
| Fault-frequency only | 0.91 / 0.88 / 0.91 | 0.69 / 0.69 / 0.69 | 0.27 / 0.24 / 0.26 | 100 % (all) | 0.65 / 0.68 / 0.66 |

Conformal marginal mean set size: all features 1.63 / 1.88 / 1.34; fault-frequency only 0.90 / 0.96 / 0.93.
In-domain healthy accuracy with all features: 0.00 / 0.00 / 0.28 (`results/in_domain_per_bearing.csv`).

## Other operating conditions (`results/tables/table4_conditions.csv`; fault_only, temperature scaling)
Same bearing splits in every condition. Real-damage values: bootstrap mean (2,000 draws). Order RF / SVM / XGB.
| Condition | In-domain acc. | Real acc. | Error at 5 % | Runs > 5 % | Conformal cov. | Missed-fault share |
|---|---|---|---|---|---|---|
| N15_M07_F10 (main) | 0.91 / 0.88 / 0.91 | 0.69 / 0.69 / 0.70 | 0.27 / 0.24 / 0.26 | 100 % (all) | 0.65 / 0.68 / 0.67 | 0.67 / 0.62 / 0.68 |
| N15_M01_F10 (0.1 Nm) | 0.94 / 0.81 / 0.94 | 0.71 / 0.71 / 0.71 | 0.27 / 0.25 / 0.26 | 100 % (all) | 0.64 / 0.68 / 0.67 | 0.77 / 0.69 / 0.73 |
| N15_M07_F04 (400 N) | 0.76 / 0.70 / 0.74 | 0.66 / 0.65 / 0.67 | 0.17 / 0.21 / 0.15 | 98 / 99 / 88 % | 0.80 / 0.80 / 0.82 | 0.26 / 0.20 / 0.27 |
| N09_M07_F10 (900 rpm) | 0.69 / 0.57 / 0.68 | 0.61 / 0.61 / 0.61 | 0.11 / 0.10 / 0.12 | 79 / 68 / 83 % | 0.85 / 0.86 / 0.85 | 0.50 / 0.45 / 0.48 |

Silent bearings across conditions (`results/per_bearing_real_fault_only_all_conditions.csv`): KA15 and KA22
misclassified at all four conditions (mean accuracy over models ≤ 0.41); KI16, KI17 at N15_M07_F10, N15_M01_F10 and
N09_M07_F10. K006 classified as damaged in 92 % of windows at N15_M07_F04.
Windows per condition: 2,309 (main), 2,308 (900 rpm), 2,305 (0.1 Nm), 2,307 (400 N).
Fault frequencies at 900 rpm: BPFO 45.8 Hz, BPFI 74.2 Hz.

## Adaptive band (`fault_only_sk`; main condition)
- Spectral-kurtosis band centre, median over windows per bearing: 18.0–22.75 kHz for 27 of 29 bearings (healthy
  included); KI01 10.75 kHz, KI18 11.0 kHz (`data/features/features_N15_M07_F10.csv`, `sk_band_centre_hz`).
- KA01 bpfo_h1 41.91 (fixed band) → 1.41 (SK band). Only KI01 and KI18 keep a clear peak (bpfi_h1_sk 13.95, 13.84).
- In-domain accuracy 0.38 / 0.32 / 0.36 (`results/in_domain_summary_fault_only_sk.csv`); real accuracy (temperature,
  mean over rotations) 0.37 / 0.39 / 0.37 (`results/rotations_summary_main_fault_only_sk.csv`).

## 1D-CNN baseline (main condition; raw vibration; `results/*_cnn.csv`)
- In-domain (leave one source bearing out, raw): accuracy 0.465, ECE 0.448 (`in_domain_summary_cnn.csv`).
  Held-out healthy K001/K002/K004 accuracy 0.00 at mean confidence 0.96/1.00/0.94; artificial inner race: only KI01
  recognised (KI03, KI05, KI07, KI08 accuracy 0.00); artificial outer race: all except KA03 (0.00) at >= 0.98.
- Real damage, bootstrap (temperature): accuracy 0.43 [0.16, 0.71]; ECE raw 0.44 -> temperature 0.19 [0.04, 0.40];
  error at 5 % target 0.38 [0.00, 1.00]; automated 0.13 [0.00, 0.80]; best possible automation 0.10 [0.00, 0.38];
  marginal conformal coverage 0.94 [0.62, 1.00], mean set size 2.48; class-conditional coverage 0.54
  (`bootstrap_cnn.csv`). 5 % target exceeded in 86 % of rotations (`rotations_summary_main_cnn.csv`).
- Missed faults: 3.2 % of errors (2.2 % of automated errors) (`missed_faults_cnn.csv`).
- Per real bearing (raw, mean over rotations; `per_bearing_real_cnn.csv`): KA22 0.02, KI16 0.27, KI17 0.11, KA15 0.48,
  KI14 0.93, KI18 1.00.
- Real-bearing calibration (temperature, error at 5 % target, mean over draws): n=1 source 0.24 / target 0.48 /
  combined 0.22; n=2 source 0.45 / target 0.41 / combined 0.35; target met in 7-49 % of draws
  (`target_calibration_summary_cnn.csv`, `target_calibration_cnn.csv`).

## Software
scikit-learn 1.7.2, XGBoost 3.2.0, SciPy 1.15.3, NumPy 2.2.6, pandas 2.3.3 (Python 3.10).
