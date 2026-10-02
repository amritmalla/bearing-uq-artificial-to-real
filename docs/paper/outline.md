# Paper Outline

## Working title

**Confident but Wrong: Calibration and Selective Automation of Bearing Fault Diagnosis Under Artificial-to-Real Damage Shift**

Alternatives:
- Can Fault Classifiers Trained on Artificial Damage Decide When to Ask a Human? Evidence from the Paderborn Bearing Dataset
- Trustworthy Confidence in Bearing Fault Diagnosis Does Not Transfer from Artificial to Real Damage

## One-sentence message

Bearing fault classifiers trained on artificial damage can look well calibrated on unseen artificial bearings, yet
their confidence thresholds, calibration and conformal guarantees all fail on real damage — and a few labelled real
bearings reduce, but do not remove, the failure.

## Contributions (for the end of the introduction)

1. The first evaluation of calibration and selective automation under the artificial→real damage shift, with
   bearing-level splits, 105 calibration-bearing rotations and bootstrap intervals over bearings.
2. Evidence that confidence thresholds chosen on artificial damage for a 5 % error target give 24–27 % error on real
   damage in every rotation, and that split conformal coverage falls to 0.65–0.68 against a nominal 0.90.
3. Identification of "silent" real-damage bearings that are misclassified with 69–94 % mean confidence.
4. A practical test of calibrating on one or two labelled real bearings per class: error at the 5 % target falls
   from about 0.27 to 0.10–0.11, at the price of automating fewer cases.
5. A feature ablation showing that amplitude features (RMS, kurtosis, crest factor, peak-to-peak) overfit to
   individual bearings — including healthy ones — and hide the artificial→real gap.

## Sections and target length (journal article, ~6,000–7,000 words)

| # | Section | Words | Content | Figures / tables |
|---|---|---|---|---|
| — | Abstract | 200–250 | Problem, method, three headline numbers, recommendation | — |
| 1 | Introduction | 800 | Why automate diagnosis; why confidence must be trustworthy; artificial vs real damage in practice; research questions; contributions | — |
| 2 | Related work | 700 | Bearing benchmarks and evaluation pitfalls; domain shift in fault diagnosis; calibration and conformal prediction in fault diagnosis; selective classification | — |
| 3 | Methods | 1,300 | Draft: `03_methods.md` | Table of bearings/splits (optional) |
| 4 | Results | 1,300 | Draft: `04_results.md` | Fig. 1–4, Tables 1–3 |
| 5 | Discussion | 900 | Why confidence fails (silent bearings, fault signature weaker or absent); what calibration can and cannot fix; recommendations for practice; relation to the 2026 *Electronics* findings (class-conditional conformal) | — |
| 6 | Limitations | 300 | One operating condition; few bearings per class; one test rig; fixed demodulation band; window-level conformal exchangeability | — |
| 7 | Conclusion | 200 | Message + recommendation | — |

## Related work to cite (verify every entry before submission)

- Lessmeier, C. et al. (2016). Paderborn KAt benchmark dataset. PHM Society European Conference. **Dataset citation required by licence.**
- Smith, W. A., Randall, R. B. (2015). CWRU benchmark study. *Mechanical Systems and Signal Processing* 64, 100–131.
- González-García, L. et al. (2026). Shapelet-based bearing fault diagnosis under interpretability constraints. *Electronics* 15(14), 3035. — closest prior work.
- Calibrated Adaptive Teacher for domain-adaptive intelligent fault diagnosis (PMC 11644747) — **authors and venue to verify**.
- Uncertainty-aware fault diagnosis under calibration (Bayesian DL; 2024) — **authors and venue to verify**.
- Guo, C. et al. (2017). On calibration of modern neural networks. ICML.
- Ovadia, Y. et al. (2019). Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. NeurIPS.
- Angelopoulos, A. N., Bates, S. (2023). A gentle introduction to conformal prediction. *Foundations and Trends in ML* 16.
- Tibshirani, R. J. et al. (2019). Conformal prediction under covariate shift. NeurIPS.
- Geifman, Y., El-Yaniv, R. (2017). Selective classification for deep neural networks. NeurIPS.
- Platt, J. (1999) — Platt scaling; Zadrozny, B., Elkan, C. (2002) — isotonic calibration (KDD).

## Discussion points to develop

- **Why thresholds fail:** for the silent bearings the fault-frequency signature is weak or absent, so the model
  sees "healthy-like" or "other-fault-like" evidence and is confidently wrong. No post-hoc calibration fitted on
  artificial data can see this.
- **Calibration ≠ transfer:** in-domain ECE 0.01–0.09 says nothing about real-damage ECE 0.17–0.20.
- **Conformal prediction:** exchangeability between calibration and test is broken by the shift; class-conditional
  conformal did not help here (0.63–0.64), unlike the class-proportion mismatch case in the 2026 *Electronics* paper.
- **Practical recommendation:** calibrate and set thresholds on labelled real bearings; expect lower automation;
  report results per bearing.
- **Feature lesson:** absolute-amplitude features describe individual bearings and rigs, not damage; self-normalised
  fault-frequency features transfer better between bearings.

## Still to do before writing the remaining sections

- Verify related-work entries (authors, venues, years).
- Decide target journal (affects length and format).
- Faculty supervisor / co-author review.
