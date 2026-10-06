| Model | In-domain accuracy | In-domain ECE | Real accuracy | Real ECE | AURC | Error at 5% target | Automated at 5% target | Best possible automation at 5% | Conformal coverage (nominal 0.90) |
|---|---|---|---|---|---|---|---|---|---|
| Random Forest | 0.91 | 0.01 | 0.69 [0.52, 0.86] | 0.19 [0.05, 0.35] | 0.15 [0.04, 0.34] | 0.27 [0.10, 0.46] | 0.87 [0.66, 1.00] | 0.32 [0.00, 0.71] | 0.65 [0.43, 0.84] |
| SVM | 0.88 | 0.09 | 0.69 [0.50, 0.86] | 0.17 [0.04, 0.35] | 0.15 [0.03, 0.34] | 0.24 [0.07, 0.45] | 0.82 [0.47, 1.00] | 0.31 [0.03, 0.75] | 0.68 [0.45, 0.87] |
| XGBoost | 0.91 | 0.06 | 0.70 [0.52, 0.85] | 0.20 [0.06, 0.35] | 0.15 [0.04, 0.33] | 0.26 [0.09, 0.44] | 0.86 [0.69, 1.00] | 0.30 [0.00, 0.72] | 0.67 [0.45, 0.85] |
| 1D-CNN | 0.46 | 0.45 | 0.43 [0.16, 0.71] | 0.19 [0.04, 0.40] | 0.47 [0.17, 0.86] | 0.38 [0.00, 1.00] | 0.13 [0.00, 0.80] | 0.10 [0.00, 0.38] | 0.94 [0.62, 1.00] |
| 1D-CNN + AdaBN | -- | -- | 0.51 [0.24, 0.77] | 0.20 [0.05, 0.41] | 0.44 [0.14, 0.81] | 0.42 [0.00, 0.96] | 0.17 [0.00, 0.79] | 0.09 [0.00, 0.38] | 0.89 [0.57, 1.00] |

Fault-frequency features; temperature scaling; real-damage values: mean [95 % interval] from 2,000 bootstrap draws over calibration rotations and test bearings. In-domain: leave one source bearing out, raw probabilities.
