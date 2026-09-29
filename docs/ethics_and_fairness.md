# Ethics, Responsible AI & Algorithmic Fairness Audit Report
### DropoutGuard — AI-Powered Academic Dropout Prediction & Intervention System
**Target Context**: Smart India Hackathon 2026 (PSID 7-L) & SDG 4: Quality Education  
**Evaluation Scope**: Quantitative algorithmic fairness, subgroup False-Negative-Rate (FNR) parity, within-group calibration (ECE), temporal presentation shift, and mitigation benchmarking across real-data cohorts and simulated benchmarks.

---

## 1. Executive Summary & Audit Mandate
In educational early-warning systems, the primary ethical risk is **unequal intervention access driven by disparate False Negative Rates (FNR)**. A False Negative represents an at-risk student who is missed by the AI system and thus denied proactive mentoring, financial counseling, or academic tutoring.

All evaluations in this report adhere to the following principles:
- **Audit Frame Isolation**: Audited demographic and sensitive attributes are strictly quarantined in an audit frame and never passed to the model's feature matrix $X$.
- **Objective Metric Reporting**: Numbers and markdown tables represent empirical measurements computed on verified splits. No subjective claims regarding algorithmic equity or compliance are made.
- **Small-Sample Guardrails**: Subgroups with $n < 50$ are flagged as `insufficient_sample` with disparity gaps and confidence intervals suppressed.
- **Empirical Uncertainty**: All reported disparity gaps are accompanied by 95% bootstrap confidence intervals (1,000 resamples).

---

## 2. Real Higher Education Benchmark: UCI Dataset 697 Audit
- **Dataset**: UCI "Predict Students' Dropout and Academic Success" (Portuguese Higher Education, $N = 3,630$, Enrolled excluded).
- **Feature Set**: `END_OF_SEM1` ($25$ model features, strictly omitting protected columns `gender`, `scholarship_holder`, `debtor`, `displaced`, `age_at_enrollment`).
- **Inference Mode**: 5-Fold Stratified Cross-Validation out-of-fold risk probabilities.
- **Selection Rate Threshold**: Top 20% predicted risk cohort.

| Attribute | Subgroup | Sample Size ($N$) | Base Rate | Selection Rate (Top 20%) | Recall (TPR) | Miss Rate (FNR) | False Alarm (FPR) | Precision | ECE | FNR Gap vs Ref | 95% Bootstrap CI | Small Group Flag |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **gender** | Female | 2381 | 30.2% | 14.0% | 45.7% | 54.3% | 0.2% | 98.8% | 0.018 | 9.94% | [4.78%, 15.34%] | sufficient |
| **gender** | Male *(Ref)* | 1249 | 56.1% | 31.5% | 55.6% | 44.4% | 0.5% | 99.2% | 0.038 | 0.00% | — | sufficient |
| **scholarship_holder** | No Scholarship *(Ref)* | 2661 | 48.4% | 25.5% | 52.3% | 47.7% | 0.4% | 99.1% | 0.021 | 0.00% | — | sufficient |
| **scholarship_holder** | Scholarship | 969 | 13.8% | 4.9% | 34.3% | 65.7% | 0.1% | 97.9% | 0.055 | 17.96% | [9.82%, 25.75%] | sufficient |
| **debtor** | Debtor | 413 | 75.5% | 44.8% | 58.7% | 41.3% | 2.0% | 98.9% | 0.068 | -10.32% | [-16.67%, -4.35%] | sufficient |
| **debtor** | Non-Debtor *(Ref)* | 3217 | 34.5% | 16.8% | 48.3% | 51.7% | 0.2% | 99.1% | 0.010 | 0.00% | — | sufficient |
| **displaced** | Displaced | 1993 | 33.6% | 13.4% | 39.2% | 60.8% | 0.3% | 98.5% | 0.015 | 21.61% | [16.50%, 26.76%] | sufficient |
| **displaced** | Non-Displaced *(Ref)* | 1637 | 45.9% | 28.1% | 60.8% | 39.2% | 0.3% | 99.4% | 0.024 | 0.00% | — | sufficient |
| **age_group** | 21-25 | 646 | 45.7% | 23.8% | 51.5% | 48.5% | 0.6% | 98.7% | 0.014 | -14.45% | [-22.11%, -7.78%] | sufficient |
| **age_group** | <=20 *(Ref)* | 2080 | 26.1% | 9.8% | 37.1% | 62.9% | 0.1% | 99.0% | 0.014 | 0.00% | — | sufficient |
| **age_group** | >25 | 904 | 64.6% | 40.8% | 62.7% | 37.3% | 0.9% | 99.2% | 0.032 | -25.59% | [-31.29%, -20.27%] | sufficient |

---

## 3. Algorithmic Fairness Mitigations Comparative Benchmark
Evaluates four mitigation approaches on an identical 70/30 stratified train/test split of the UCI cohort ($N = 3,630$, sensitive attribute: `gender`):
1. **None**: Unmitigated baseline model ($L_2$-regularized Logistic Regression).
2. **Sample Reweighing**: Inversely proportional joint class/group weights $w_i = \frac{N}{K \cdot \text{count}(s, y)}$.
3. **Group-Specific Thresholds**: Post-processing threshold optimization equalizing subgroup False Negative Rates to target cohort FNR.
4. **Fairlearn ExponentiatedGradient**: In-processing constrained optimization enforcing `TruePositiveRateParity`.

| Strategy | Sensitive Attribute | ROC-AUC | PR-AUC | Recall (Sensitivity) | Precision | Brier Score | Max FNR Disparity Gap |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **None (Unmitigated)** | gender | 0.939 | 0.935 | 82.2% | 87.9% | 0.0834 | **4.24%** |
| **Sample Reweighing** | gender | 0.941 | 0.935 | 85.7% | 84.3% | 0.0881 | **5.20%** |
| **Group-Specific Thresholds (FNR Parity)** | gender | 0.939 | 0.935 | 81.7% | 86.8% | 0.0834 | **2.29%** |
| **Fairlearn ExponentiatedGradient** | gender | 0.894 | 0.847 | 70.9% | 90.7% | 0.1055 | **2.90%** |

---

## 4. Time-Based Learning Analytics Benchmark: OULAD Snapshot Day 56
- **Dataset**: Open University Learning Analytics Dataset (OULAD, Day $t = 56$).
- **Cohort**: Model trained on 2013 presentations (2013B + 2013J); audited on the **unseen 2014 temporal holdout cohort** (2014B + 2014J).
- **Feature Set**: 24 cumulative behavioural and engagement features up to Day 56 (all demographic attributes strictly excluded from $X$).
- **Selection Rate Threshold**: Top 20% predicted risk cohort.

| Attribute | Subgroup | Sample Size ($N$) | Base Rate | Selection Rate (Top 20%) | Recall (TPR) | Miss Rate (FNR) | False Alarm (FPR) | Precision | ECE | FNR Gap vs Ref | 95% Bootstrap CI | Sample Flag |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **gender** | F | 6458 | 15.0% | 18.0% | 34.7% | 65.3% | 15.0% | 29.0% | 0.033 | 5.41% | [1.75%, 9.11%] | sufficient |
| **gender** | M *(Ref)* | 8634 | 17.3% | 21.5% | 40.2% | 59.9% | 17.6% | 32.3% | 0.053 | 0.00% | — | sufficient |
| **age_band** | 0-35 *(Ref)* | 10489 | 16.6% | 21.0% | 39.6% | 60.4% | 17.3% | 31.3% | 0.045 | 0.00% | — | sufficient |
| **age_band** | 35-55 | 4491 | 15.8% | 17.8% | 34.1% | 65.9% | 14.7% | 30.4% | 0.044 | 5.47% | [1.56%, 9.49%] | sufficient |
| **age_band** | 55<= | 112 | 13.4% | 17.9% | 40.0% | 60.0% | 14.4% | 30.0% | 0.046 | -0.40% | [-25.32%, 27.67%] | sufficient |
| **disability** | N *(Ref)* | 13681 | 15.6% | 19.4% | 36.9% | 63.1% | 16.2% | 29.7% | 0.039 | 0.00% | — | sufficient |
| **disability** | Y | 1411 | 23.2% | 25.9% | 45.4% | 54.6% | 20.0% | 40.7% | 0.101 | -8.54% | [-14.05%, -2.46%] | sufficient |
| **imd_band** | 0-10% | 1420 | 18.1% | 23.7% | 40.9% | 59.1% | 19.9% | 31.2% | 0.055 | 2.08% | [-7.90%, 11.38%] | sufficient |
| **imd_band** | 10-20% | 1525 | 17.8% | 22.8% | 39.3% | 60.7% | 19.1% | 30.8% | 0.057 | 3.60% | [-6.01%, 13.26%] | sufficient |
| **imd_band** | 20-30% | 1636 | 19.7% | 21.3% | 41.5% | 58.5% | 16.4% | 38.4% | 0.076 | 1.45% | [-7.58%, 10.82%] | sufficient |
| **imd_band** | 30-40% | 1621 | 16.4% | 21.1% | 33.1% | 66.9% | 18.8% | 25.7% | 0.044 | 9.86% | [0.51%, 19.51%] | sufficient |
| **imd_band** | 40-50% | 1475 | 16.5% | 19.9% | 37.3% | 62.7% | 16.5% | 30.9% | 0.050 | 5.64% | [-3.68%, 15.41%] | sufficient |
| **imd_band** | 50-60% | 1461 | 15.6% | 20.0% | 41.7% | 58.3% | 16.0% | 32.5% | 0.037 | 1.27% | [-7.81%, 11.20%] | sufficient |
| **imd_band** | 60-70% | 1408 | 16.8% | 18.0% | 36.9% | 63.1% | 14.2% | 34.4% | 0.054 | 6.08% | [-3.83%, 15.97%] | sufficient |
| **imd_band** | 70-80% | 1372 | 14.4% | 19.1% | 35.9% | 64.1% | 16.3% | 27.1% | 0.032 | 7.08% | [-3.62%, 16.25%] | sufficient |
| **imd_band** | 80-90% | 1349 | 14.5% | 17.1% | 31.6% | 68.4% | 14.7% | 26.8% | 0.032 | 11.31% | [1.57%, 20.99%] | sufficient |
| **imd_band** | 90-100% *(Ref)* | 1229 | 13.3% | 17.7% | 42.9% | 57.1% | 13.8% | 32.3% | 0.026 | 0.00% | — | sufficient |
| **imd_band** | Missing | 596 | 14.1% | 16.1% | 33.3% | 66.7% | 13.3% | 29.2% | 0.034 | 9.61% | [-3.01%, 21.58%] | sufficient |
| **highest_education** | A Level or Equivalent *(Ref)* | 6682 | 15.2% | 19.4% | 37.7% | 62.3% | 16.1% | 29.6% | 0.034 | 0.00% | — | sufficient |
| **highest_education** | HE Qualification | 2424 | 15.7% | 18.2% | 36.8% | 63.2% | 14.7% | 31.8% | 0.046 | 0.93% | [-4.29%, 6.33%] | sufficient |
| **highest_education** | Lower Than A Level | 5655 | 18.0% | 21.4% | 39.0% | 61.0% | 17.5% | 32.7% | 0.058 | -1.30% | [-5.75%, 2.95%] | sufficient |
| **highest_education** | No Formal quals | 136 | 16.9% | 30.9% | 34.8% | 65.2% | 30.1% | 19.1% | 0.056 | 2.90% | [-17.50%, 22.32%] | sufficient |
| **highest_education** | Post Graduate Qualification | 195 | 14.4% | 14.4% | 35.7% | 64.3% | 10.8% | 35.7% | 0.041 | 1.97% | [-16.85%, 20.51%] | sufficient |
| **region** | East Anglian Region | 1466 | 14.5% | 17.6% | 37.7% | 62.3% | 14.2% | 31.0% | 0.028 | -4.72% | [-14.15%, 4.18%] | sufficient |
| **region** | East Midlands Region | 1073 | 17.9% | 18.7% | 39.6% | 60.4% | 14.2% | 37.8% | 0.061 | -6.56% | [-15.68%, 2.70%] | sufficient |
| **region** | Ireland | 590 | 15.8% | 15.8% | 29.0% | 71.0% | 13.3% | 29.0% | 0.050 | 3.99% | [-7.39%, 14.63%] | sufficient |
| **region** | London Region | 1409 | 16.4% | 21.6% | 36.4% | 63.6% | 18.8% | 27.5% | 0.041 | -3.34% | [-12.19%, 5.08%] | sufficient |
| **region** | North Region | 869 | 17.2% | 21.2% | 36.2% | 63.8% | 18.1% | 29.3% | 0.065 | -3.22% | [-12.59%, 6.67%] | sufficient |
| **region** | North Western Region | 1257 | 16.6% | 22.2% | 41.8% | 58.2% | 18.3% | 31.2% | 0.043 | -8.81% | [-18.33%, 0.28%] | sufficient |
| **region** | Scotland | 1908 | 18.4% | 20.0% | 39.3% | 60.7% | 15.7% | 36.1% | 0.067 | -6.30% | [-13.55%, 2.56%] | sufficient |
| **region** | South East Region | 964 | 16.0% | 19.3% | 33.1% | 66.9% | 16.7% | 27.4% | 0.043 | -0.10% | [-8.69%, 10.19%] | sufficient |
| **region** | South Region *(Ref)* | 1427 | 15.1% | 17.4% | 33.0% | 67.0% | 14.6% | 28.6% | 0.035 | 0.00% | — | sufficient |
| **region** | South West Region | 1121 | 17.1% | 19.2% | 40.1% | 59.9% | 14.8% | 35.8% | 0.053 | -7.08% | [-15.61%, 2.33%] | sufficient |
| **region** | Wales | 1039 | 14.7% | 24.2% | 41.8% | 58.2% | 21.1% | 25.5% | 0.025 | -8.81% | [-18.20%, 1.76%] | sufficient |
| **region** | West Midlands Region | 1098 | 16.4% | 21.9% | 40.0% | 60.0% | 18.4% | 29.9% | 0.044 | -6.98% | [-15.97%, 3.16%] | sufficient |
| **region** | Yorkshire Region | 871 | 15.7% | 20.2% | 41.6% | 58.4% | 16.2% | 32.4% | 0.040 | -8.59% | [-18.05%, 2.41%] | sufficient |
| **imd_x_gender** | 0-10%_F | 721 | 16.2% | 21.5% | 38.5% | 61.5% | 18.2% | 29.0% | 0.041 | 5.79% | [-6.83%, 18.26%] | sufficient |
| **imd_x_gender** | 0-10%_M | 699 | 20.0% | 25.9% | 42.9% | 57.1% | 21.6% | 33.1% | 0.073 | 1.39% | [-11.36%, 13.61%] | sufficient |
| **imd_x_gender** | 10-20%_F | 714 | 17.1% | 20.0% | 29.5% | 70.5% | 18.1% | 25.2% | 0.050 | 14.74% | [2.47%, 26.79%] | sufficient |
| **imd_x_gender** | 10-20%_M | 811 | 18.5% | 25.1% | 47.3% | 52.7% | 20.1% | 34.8% | 0.062 | -3.08% | [-14.85%, 8.68%] | sufficient |
| **imd_x_gender** | 20-30%_F | 803 | 19.8% | 19.2% | 36.5% | 63.5% | 14.9% | 37.7% | 0.081 | 7.77% | [-3.85%, 20.00%] | sufficient |
| **imd_x_gender** | 20-30%_M | 833 | 19.7% | 23.4% | 46.3% | 53.7% | 17.8% | 39.0% | 0.074 | -2.09% | [-13.58%, 9.82%] | sufficient |
| **imd_x_gender** | 30-40%_F | 700 | 15.7% | 19.1% | 32.7% | 67.3% | 16.6% | 26.9% | 0.044 | 11.52% | [-0.21%, 23.86%] | sufficient |
| **imd_x_gender** | 30-40%_M | 921 | 16.9% | 22.6% | 33.3% | 66.7% | 20.4% | 25.0% | 0.048 | 10.92% | [-1.19%, 22.69%] | sufficient |
| **imd_x_gender** | 40-50%_F | 687 | 15.6% | 16.9% | 34.6% | 65.4% | 13.6% | 31.9% | 0.043 | 9.67% | [-3.38%, 21.80%] | sufficient |
| **imd_x_gender** | 40-50%_M | 788 | 17.4% | 22.6% | 39.4% | 60.6% | 19.1% | 30.3% | 0.057 | 4.83% | [-7.90%, 17.83%] | sufficient |
| **imd_x_gender** | 50-60%_F | 653 | 11.8% | 14.5% | 35.1% | 64.9% | 11.8% | 28.4% | 0.018 | 9.19% | [-5.05%, 23.06%] | sufficient |
| **imd_x_gender** | 50-60%_M | 808 | 18.7% | 24.4% | 45.0% | 55.0% | 19.6% | 34.5% | 0.064 | -0.78% | [-12.38%, 11.35%] | sufficient |
| **imd_x_gender** | 60-70%_F | 579 | 14.2% | 16.6% | 37.8% | 62.2% | 13.1% | 32.3% | 0.027 | 6.45% | [-6.77%, 21.07%] | sufficient |
| **imd_x_gender** | 60-70%_M | 829 | 18.6% | 18.9% | 36.4% | 63.6% | 15.0% | 35.7% | 0.073 | 7.89% | [-4.25%, 21.19%] | sufficient |
| **imd_x_gender** | 70-80%_F | 504 | 12.1% | 17.9% | 29.5% | 70.5% | 16.2% | 20.0% | 0.014 | 14.74% | [-0.86%, 28.62%] | sufficient |
| **imd_x_gender** | 70-80%_M | 868 | 15.8% | 19.8% | 38.7% | 61.3% | 16.3% | 30.8% | 0.048 | 5.56% | [-6.40%, 17.40%] | sufficient |
| **imd_x_gender** | 80-90%_F | 511 | 12.9% | 14.5% | 30.3% | 69.7% | 12.1% | 27.0% | 0.021 | 13.95% | [0.21%, 28.57%] | sufficient |
| **imd_x_gender** | 80-90%_M | 838 | 15.5% | 18.7% | 32.3% | 67.7% | 16.2% | 26.8% | 0.040 | 11.94% | [-0.44%, 23.95%] | sufficient |
| **imd_x_gender** | 90-100%_F | 437 | 11.4% | 16.0% | 40.0% | 60.0% | 12.9% | 28.6% | 0.009 | 4.25% | [-11.26%, 19.64%] | sufficient |
| **imd_x_gender** | 90-100%_M *(Ref)* | 792 | 14.3% | 18.6% | 44.2% | 55.8% | 14.3% | 34.0% | 0.044 | 0.00% | — | sufficient |
| **imd_x_gender** | Missing_F | 149 | 12.8% | 22.1% | 47.4% | 52.6% | 18.5% | 27.3% | 0.033 | -3.12% | [-26.67%, 19.61%] | sufficient |
| **imd_x_gender** | Missing_M | 447 | 14.5% | 14.1% | 29.2% | 70.8% | 11.5% | 30.2% | 0.040 | 15.02% | [0.75%, 29.17%] | sufficient |

---

## 5. OULAD Presentation Shift Check: 2013 In-Sample vs. 2014 Temporal Holdout
Evaluates temporal stability of subgroup error rates and disparity gaps across academic years (2013 calendar presentations vs. 2014 calendar presentations at snapshot $t = 56$).

| Attribute | Group | $N$ (2013) | $N$ (2014) | Base Rate 2013 | Base Rate 2014 | FNR 2013 | FNR 2014 | FNR Shift (2014 - 2013) | FNR Gap 2013 | FNR Gap 2014 | Gap Shift (2014 - 2013) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **gender** | F | 5497 | 6458 | 13.8% | 15.0% | 55.4% | 65.3% | 9.84% | -9.98% | 5.41% | 15.39% |
| **gender** | M *(Ref)* | 5933 | 8634 | 14.7% | 17.3% | 65.4% | 59.9% | -5.55% | 0.00% | 0.00% | 0.00% |
| **age_band** | 0-35 *(Ref)* | 7964 | 10489 | 14.5% | 16.6% | 61.9% | 60.4% | -1.45% | 0.00% | 0.00% | 0.00% |
| **age_band** | 35-55 | 3393 | 4491 | 13.6% | 15.8% | 58.4% | 65.9% | 7.43% | -3.41% | 5.47% | 8.88% |
| **age_band** | 55<= | 73 | 112 | 11.0% | 13.4% | 37.5% | 60.0% | 22.50% | -24.35% | -0.40% | 23.95% |
| **disability** | N *(Ref)* | 10356 | 13681 | 13.4% | 15.6% | 63.2% | 63.1% | -0.07% | 0.00% | 0.00% | 0.00% |
| **disability** | Y | 1074 | 1411 | 22.2% | 23.2% | 46.6% | 54.6% | 7.93% | -16.54% | -8.54% | 8.00% |
| **imd_band** | 0-10% | 1128 | 1420 | 18.8% | 18.1% | 50.0% | 59.1% | 9.14% | -11.48% | 2.08% | 13.56% |
| **imd_band** | 10-20% | 1201 | 1525 | 15.4% | 17.8% | 62.7% | 60.7% | -2.04% | 1.22% | 3.60% | 2.38% |
| **imd_band** | 20-30% | 1229 | 1636 | 17.2% | 19.7% | 57.1% | 58.5% | 1.43% | -4.40% | 1.45% | 5.85% |
| **imd_band** | 30-40% | 1261 | 1621 | 13.6% | 16.4% | 53.5% | 66.9% | 13.43% | -7.99% | 9.86% | 17.85% |
| **imd_band** | 40-50% | 1136 | 1475 | 13.5% | 16.5% | 62.1% | 62.7% | 0.61% | 0.61% | 5.64% | 5.03% |
| **imd_band** | 50-60% | 1129 | 1461 | 12.3% | 15.6% | 65.5% | 58.3% | -7.14% | 3.99% | 1.27% | -2.72% |
| **imd_band** | 60-70% | 1011 | 1408 | 13.6% | 16.8% | 57.7% | 63.1% | 5.48% | -3.82% | 6.08% | 9.90% |
| **imd_band** | 70-80% | 1047 | 1372 | 13.4% | 14.4% | 72.9% | 64.1% | -8.72% | 11.38% | 7.08% | -4.30% |
| **imd_band** | 80-90% | 950 | 1349 | 12.1% | 14.5% | 69.6% | 68.4% | -1.20% | 8.09% | 11.31% | 3.22% |
| **imd_band** | 90-100% *(Ref)* | 936 | 1229 | 13.0% | 13.3% | 61.5% | 57.1% | -4.42% | 0.00% | 0.00% | 0.00% |
| **imd_band** | Missing | 402 | 596 | 9.7% | 14.1% | 79.5% | 66.7% | -12.82% | 18.01% | 9.61% | -8.40% |
| **highest_education** | A Level or Equivalent *(Ref)* | 4984 | 6682 | 12.8% | 15.2% | 59.8% | 62.3% | 2.57% | 0.00% | 0.00% | 0.00% |
| **highest_education** | HE Qualification | 1607 | 2424 | 12.8% | 15.7% | 62.0% | 63.2% | 1.30% | 2.20% | 0.93% | -1.27% |
| **highest_education** | Lower Than A Level | 4648 | 5655 | 16.2% | 18.0% | 61.4% | 61.0% | -0.36% | 1.63% | -1.30% | -2.93% |
| **highest_education** | No Formal quals | 109 | 136 | 22.0% | 16.9% | 54.2% | 65.2% | 11.05% | -5.58% | 2.90% | 8.48% |
| **highest_education** | Post Graduate Qualification | 82 | 195 | 12.2% | 14.4% | 70.0% | 64.3% | -5.71% | 10.25% | 1.97% | -8.28% |
| **region** | East Anglian Region | 1244 | 1466 | 13.2% | 14.5% | 65.8% | 62.3% | -3.59% | -0.82% | -4.72% | -3.90% |
| **region** | East Midlands Region | 778 | 1073 | 14.9% | 17.9% | 60.3% | 60.4% | 0.08% | -6.33% | -6.56% | -0.23% |
| **region** | Ireland | 480 | 590 | 13.8% | 15.8% | 60.6% | 71.0% | 10.36% | -6.06% | 3.99% | 10.05% |
| **region** | London Region | 1085 | 1409 | 14.6% | 16.4% | 47.8% | 63.6% | 15.84% | -18.87% | -3.34% | 15.53% |
| **region** | North Region | 611 | 869 | 13.8% | 17.2% | 71.4% | 63.8% | -7.67% | 4.76% | -3.22% | -7.98% |
| **region** | North Western Region | 966 | 1257 | 14.8% | 16.6% | 56.6% | 58.2% | 1.53% | -10.03% | -8.81% | 1.22% |
| **region** | Scotland | 1152 | 1908 | 14.8% | 18.4% | 61.8% | 60.7% | -1.08% | -4.91% | -6.30% | -1.39% |
| **region** | South East Region | 736 | 964 | 11.4% | 16.0% | 65.5% | 66.9% | 1.40% | -1.19% | -0.10% | 1.09% |
| **region** | South Region *(Ref)* | 1098 | 1427 | 13.4% | 15.1% | 66.7% | 67.0% | 0.31% | 0.00% | 0.00% | 0.00% |
| **region** | South West Region | 871 | 1121 | 14.1% | 17.1% | 54.5% | 59.9% | 5.43% | -12.20% | -7.08% | 5.12% |
| **region** | Wales | 803 | 1039 | 15.8% | 14.7% | 63.0% | 58.2% | -4.82% | -3.68% | -8.81% | -5.13% |
| **region** | West Midlands Region | 903 | 1098 | 16.9% | 16.4% | 64.7% | 60.0% | -4.71% | -1.96% | -6.98% | -5.02% |
| **region** | Yorkshire Region | 703 | 871 | 12.8% | 15.7% | 54.4% | 58.4% | 3.95% | -12.23% | -8.59% | 3.64% |

---

## 6. Income Feature Ablation Benchmark (Simulated Cohort)
Evaluates model performance and disaggregated False Negative Rates across household income brackets when `income_slab_idx` and `financial_stress_index` are included versus completely ablated from the 37-feature simulated pipeline.

| Feature Configuration | Predictor Features | ROC-AUC | PR-AUC | Recall | Precision | Brier Score | FNR (<2 LPA) | FNR (>8 LPA) | FNR Disparity Gap |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **With Income Features (37 features)** | 37 | 0.929 | 0.908 | 77.1% | 87.7% | 0.0914 | 45.0% | 44.1% | **0.88%** |
| **Without Income Features (35 features)** | 35 | 0.929 | 0.908 | 77.4% | 87.7% | 0.0913 | 43.6% | 44.1% | **-0.47%** |

#### Disaggregated Performance Across Income Slabs (Simulated Cohort)

| Income Bracket | $N$ Students | Actual Dropouts | Base Rate | FNR (With Income Features) | FNR (Without Income Features) | FNR Difference (Without - With) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **<2 LPA** | 432 | 149 | 34.5% | 44.97% | 43.62% | -1.35% |
| **2-5 LPA** | 774 | 281 | 36.3% | 43.06% | 43.77% | 0.71% |
| **5-8 LPA** | 509 | 180 | 35.4% | 45.56% | 45.56% | 0.00% |
| **>8 LPA** | 285 | 93 | 32.6% | 44.09% | 44.09% | 0.00% |

---

## 7. Generator Sanity Check (Simulated Indian Cohort)
Verification of synthetic data generator properties across $N = 2,000$ simulated students.

> [!NOTE]
> **Generator Sanity Check Disclaimer**:
> The simulated Indian cohort is generated from known mathematical equations (`ml/simulation/estimate_parameters.py` and `ml/data_pipeline/generate_synthetic_indian.py`). The numbers below verify internal generator calibration and absence of unintentional statistical distortions; they are **NOT evidence of real-world predictive validity**.

| Demographic Slice | Single Held-Out Test Split ($N=300$) | 5-Fold Cross-Validation ($N=2,000$ Out-of-Fold) | Empirical Difference |
| :--- | :--- | :--- | :--- |
| **Gender Disparity Gap** (Female vs Male FNR) | **7.93%** | **0.80%** | **7.13%** |
| **Economic Proxy Gap** (<5 LPA vs $\ge$5 LPA) | **14.60%** | **4.94%** | **9.66%** |
| **First-Generation Gap** (First-Gen vs Non-First-Gen) | **20.67%** | **0.90%** | **19.77%** |

---

## 8. Four Core Pillars of Responsible AI Governance in DropoutGuard

1. **Human-in-the-Loop Decision Support**:
   The system never executes autonomous punitive actions (e.g. debarment or scholarship cancellation). All outputs serve exclusively as confidential decision-support recommendations for designated faculty mentors.
2. **Deficit Framing Avoidance**:
   Risk assessments avoid pejorative labels. Interventions are framed as proactive resource allocations (e.g., "Peer Tutoring Referral" or "Financial Aid Desk Check-in") rather than student deficits.
3. **SHAP-Verifiable Interpretability**:
   Every risk probability is accompanied by signed SHAP local drivers, enabling mentors to verify the causal rationale before taking action.
4. **Data Minimization, Need Signals & Confidentiality**:
   Household income is used as an active model input via `income_slab_idx` and `financial_stress_index` exclusively as an objective need signal to prioritize and route emergency financial aid and fee-waiver interventions to economically vulnerable students. Only the duplicate raw string label (`family_income_slab`) and sensitive social categories are excluded from model training to prevent categorical bias. All socio-economic indicators are confidential, encrypted at rest, and never used for punitive academic actions.
