| Model | Calibrated on | Error at 5% target | Draws meeting 5% target | Automated | ECE | Conformal coverage |
|---|---|---|---|---|---|---|
| Random Forest | Artificial bearings | 0.28 [0.13, 0.42] | 0% | 0.87 [0.65, 1.00] | 0.21 [0.10, 0.33] | 0.64 [0.49, 0.80] |
| Random Forest | 1 real bearing/class | 0.19 [0.00, 0.36] | 29% | 0.51 [0.00, 0.99] | 0.20 [0.05, 0.32] | 0.78 [0.52, 0.98] |
| Random Forest | 2 real bearings/class | 0.11 [0.01, 0.38] | 51% | 0.38 [0.16, 0.77] | 0.17 [0.02, 0.33] | 0.85 [0.57, 0.99] |
| SVM | Artificial bearings | 0.26 [0.12, 0.39] | 1% | 0.81 [0.45, 1.00] | 0.19 [0.04, 0.33] | 0.67 [0.51, 0.84] |
| SVM | 1 real bearing/class | 0.18 [0.00, 0.45] | 35% | 0.52 [0.00, 0.99] | 0.18 [0.04, 0.34] | 0.77 [0.45, 0.98] |
| SVM | 2 real bearings/class | 0.10 [0.00, 0.36] | 52% | 0.37 [0.18, 0.76] | 0.13 [0.04, 0.29] | 0.85 [0.60, 1.00] |
| XGBoost | Artificial bearings | 0.28 [0.15, 0.41] | 0% | 0.88 [0.70, 1.00] | 0.22 [0.11, 0.32] | 0.65 [0.51, 0.80] |
| XGBoost | 1 real bearing/class | 0.16 [0.00, 0.37] | 42% | 0.52 [0.08, 0.99] | 0.19 [0.07, 0.33] | 0.79 [0.54, 0.97] |
| XGBoost | 2 real bearings/class | 0.11 [0.00, 0.37] | 49% | 0.35 [0.03, 0.76] | 0.16 [0.06, 0.31] | 0.84 [0.58, 0.99] |

Fault-frequency features; temperature scaling; mean [2.5th, 97.5th percentile] over 50 draws per setting. Artificial-bearing values pooled over the 100 draws. Each draw's settings are scored on the same test bearings. Draws meeting the target counts only draws that automated at least one case.
