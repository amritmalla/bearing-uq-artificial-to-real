# 3. Methods

*Draft. Numbers are listed with their sources in `numbers.md`.*

## 3.1 Dataset

We use the Paderborn University KAt bearing dataset [Lessmeier et al., 2016], which records deep-groove ball
bearings of type 6203 on an electromechanical drive test rig. Its distinguishing feature for this study is that it
contains both **artificially damaged** bearings (damage introduced by electric discharge machining, drilling or
electric engraving) and bearings with **real damage** that developed during accelerated lifetime tests. We use the
vibration signal (sampled at 64 kHz) at a single operating condition, `N15_M07_F10`: 1,500 rpm shaft speed, 0.7 Nm
load torque and 1,000 N radial force. Each bearing has 20 recordings of 4 s.

We consider three classes — healthy, inner-race fault and outer-race fault — and use 29 bearings: 6 healthy,
12 with artificial damage (7 outer race, 5 inner race) and 11 with real damage (5 outer race, 6 inner race).
Bearings with combined inner- and outer-race damage are excluded. Each recording is cut into non-overlapping 1 s
windows (64,000 samples), giving 2,309 windows, or 78–80 per bearing.

## 3.2 Features

**Fault-frequency features (main setting).** For each window we compute the envelope spectrum: a fourth-order
Butterworth band-pass filter (2–10 kHz) applied forwards and backwards, the magnitude of the analytic signal
(Hilbert transform) with its mean removed, and the magnitude of its discrete Fourier transform. The characteristic
fault frequencies of the 6203 bearing (8 balls, ball diameter *d* = 6.75 mm, pitch diameter *D* = 28.55 mm,
contact angle 0°) at shaft frequency *f*ᵣ = 25 Hz are

BPFO = (*n*/2)(1 − *d*/*D*) *f*ᵣ ≈ 76.4 Hz,  BPFI = (*n*/2)(1 + *d*/*D*) *f*ᵣ ≈ 123.6 Hz.

For each of BPFO and BPFI and their second harmonics, the feature is the largest envelope-spectrum amplitude within
±2 % of the target frequency, divided by the median envelope-spectrum amplitude between 10 and 500 Hz. These four
ratios are self-normalised: they compare a peak with the spectrum's own noise floor rather than measuring absolute
vibration level. All four are log-transformed before modelling.

**All features (ablation).** The ablation adds five time-domain statistics of the raw window: root mean square,
kurtosis, skewness, crest factor and peak-to-peak amplitude (log-transformed except skewness).

## 3.3 Bearing-level splits

Every bearing belongs to exactly one split, so windows from one bearing never appear in both training and
evaluation. The **source domain** contains the 12 artificially damaged bearings and healthy bearings K001, K002 and
K004; the **target domain** contains the 11 real-damage bearings and healthy bearings K003, K005 and K006.

The healthy bearings were assigned with care. Their median kurtosis differs strongly: 13.6–15.8 for K001, K002,
K003 and K006, but 4.2–5.1 for K004 and K005, which indicates an impulsive component unrelated to bearing damage. We
place one low-kurtosis healthy bearing on each side, so that the healthy class itself does not shift between
domains.

From the source domain, one bearing per class is held out as the **calibration set**; the remaining source bearings
form the **training set**. Because results depend strongly on which bearings are held out, we repeat the whole
experiment for every possible choice: 3 healthy × 5 inner-race × 7 outer-race = **105 rotations**. Training sets
contain 954–957 windows and calibration sets 237–240. The **test set** — all 14 target-domain bearings, 1,115 windows —
is the same in every rotation.

## 3.4 Classifiers

We compare three classifiers widely used in condition monitoring:

- **Random Forest:** 500 trees, minimum 2 samples per leaf, class-balanced weights.
- **Support vector machine (SVM):** RBF kernel, *C* = 1, γ = "scale", standardised inputs, class-balanced weights;
  probabilities from scikit-learn's internal cross-validated Platt scaling.
- **XGBoost:** 300 trees, maximum depth 4, learning rate 0.1, row and column subsampling 0.8, class-balanced
  sample weights.

All models use a fixed random seed (0) and are fitted on the training set only.

## 3.5 Calibration and conformal prediction

Post-hoc calibrators are fitted on the calibration set's predicted probabilities:

- **Temperature scaling:** probabilities are rescaled as softmax(log *p* / *T*), with a single *T* chosen to
  minimise the negative log-likelihood.
- **Isotonic regression:** one-vs-rest isotonic regression per class, followed by renormalisation.

The uncalibrated ("raw") probabilities are reported as a third option.

**Split conformal prediction** [Angelopoulos and Bates, 2023] uses the non-conformity score 1 − *p̂*(true class) on
the calibration set. The threshold is the ⌈(*n*+1)(1−α)⌉/*n* empirical quantile of the scores, with α = 0.1
(nominal coverage 0.90). A class enters a test window's prediction set when its score is at or below the threshold,
so sets may be empty. We use a single threshold (**marginal**) and one threshold per class (**class-conditional**,
Mondrian).

## 3.6 Selective automation

A deployed system acts automatically only when its confidence (the largest class probability) reaches a threshold,
and passes the remaining cases to a human. For an error target ε (1 % or 5 %), we choose the lowest threshold whose
automatically decided calibration windows have an error rate of at most ε; ties in confidence are kept together. We
then apply this threshold to the test set and report the **automation rate** (share of test windows decided
automatically) and the **automated error** (error rate among them). As a reference, the **best possible automation**
is the automation rate obtained when the threshold is chosen on the test set itself.

## 3.7 Metrics

- **Accuracy and macro-F1.**
- **Expected calibration error (ECE):** top-label, 10 equal-width confidence bins.
- **Brier score.**
- **Area under the risk–coverage curve (AURC)** [Geifman and El-Yaniv, 2017]: mean error among the *k* most confident
  windows over all *k*.
- **Conformal empirical coverage and mean set size.**

## 3.8 In-domain reference

To separate the artificial→real shift from the difficulty of generalising to any unseen bearing, we also evaluate
each classifier on the source domain by **leave-one-bearing-out**. Each of the 15 source bearings is predicted by a
model trained on the other 14, and metrics are computed on the pooled predictions (raw probabilities).

## 3.9 Uncertainty of the estimates

With only 3–6 test bearings per class, bearing-to-bearing variation dominates the uncertainty. We therefore use a
hierarchical bootstrap with 2,000 draws. Each draw picks one of the 105 rotations at random, then resamples the
test bearings with replacement within each class, keeping the number of bearings per class (3 healthy, 6 inner race,
5 outer race). All metrics are recomputed on the windows of the resampled bearings, and we report the mean and the
2.5th–97.5th percentile interval.

## 3.10 Calibrating on labelled real bearings

To test whether a small amount of labelled target data restores trustworthy confidence, we move *n* = 1 or 2
target-domain bearings per class from the test set into a separate calibration set. The draws use 50 random
selections per *n*, each combined with a randomly chosen rotation. The classifier is always trained on that
rotation's source training set. We compare three calibration sets, all evaluated on the **same remaining test
bearings**:

- **Source:** the rotation's held-out source bearings.
- **Target:** only the *n* target bearings per class.
- **Combined:** both.

Calibrators, conformal thresholds and automation thresholds are fitted on each calibration set in turn.

## 3.11 Implementation

The pipeline is implemented in Python using NumPy, SciPy, pandas, scikit-learn 1.7.2 and XGBoost 3.2.0. Code,
configuration and all result tables are available at [repository link to be added]; every figure and table can be
regenerated from the feature table with the scripts listed in the repository README.
