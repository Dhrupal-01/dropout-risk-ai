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
- **Feature Set**: `END_OF_SEM1` ($28$ model features). Protected attributes excluded from the model: `gender`, `age_at_enrollment`. Audit groups that are also model features (documented): `scholarship_holder`, `debtor`, `displaced`.
- **Inference Mode**: 5-Fold Stratified Cross-Validation out-of-fold risk probabilities.
- **Selection Rate Threshold**: Top 20% predicted risk cohort.

| Attribute | Subgroup | Sample Size ($N$) | Base Rate | Selection Rate (Top 20%) | Recall (TPR) | Miss Rate (FNR) | False Alarm (FPR) | Precision | ECE | FNR Gap vs Ref | 95% Bootstrap CI | Small Group Flag |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **gender** | Female | 2381 | 30.2% | 14.4% | 47.1% | 52.9% | 0.3% | 98.6% | 0.020 | 6.84% | [1.63%, 12.27%] | sufficient |
| **gender** | Male *(Ref)* | 1249 | 56.1% | 30.6% | 53.9% | 46.1% | 0.7% | 99.0% | 0.035 | 0.00% | — | sufficient |
| **scholarship_holder** | No Scholarship *(Ref)* | 2661 | 48.4% | 26.0% | 53.1% | 46.9% | 0.6% | 98.8% | 0.014 | 0.00% | — | sufficient |
| **scholarship_holder** | Scholarship | 969 | 13.8% | 3.6% | 25.4% | 74.6% | 0.1% | 97.1% | 0.027 | 27.70% | [20.36%, 35.27%] | sufficient |
| **debtor** | Debtor | 413 | 75.5% | 52.1% | 67.6% | 32.4% | 4.0% | 98.1% | 0.030 | -22.00% | [-27.86%, -16.15%] | sufficient |
| **debtor** | Non-Debtor *(Ref)* | 3217 | 34.5% | 15.9% | 45.6% | 54.4% | 0.2% | 99.0% | 0.011 | 0.00% | — | sufficient |
| **displaced** | Displaced | 1993 | 33.6% | 13.8% | 40.4% | 59.6% | 0.4% | 98.2% | 0.014 | 19.08% | [13.86%, 23.88%] | sufficient |
| **displaced** | Non-Displaced *(Ref)* | 1637 | 45.9% | 27.6% | 59.4% | 40.6% | 0.4% | 99.1% | 0.017 | 0.00% | — | sufficient |
| **age_group** | 21-25 | 646 | 45.7% | 22.9% | 49.5% | 50.5% | 0.6% | 98.7% | 0.018 | -12.04% | [-19.49%, -5.14%] | sufficient |
| **age_group** | <=20 *(Ref)* | 2080 | 26.1% | 9.9% | 37.5% | 62.5% | 0.2% | 98.5% | 0.018 | 0.00% | — | sufficient |
| **age_group** | >25 | 904 | 64.6% | 41.1% | 63.0% | 37.0% | 1.2% | 98.9% | 0.028 | -25.56% | [-31.21%, -20.20%] | sufficient |

---

## 3. Algorithmic Fairness Mitigations Comparative Benchmark
Evaluates four mitigation approaches on an identical 70/30 stratified train/test split of the UCI cohort ($N = 3,630$, sensitive attribute: `gender`):
1. **None**: Unmitigated baseline model ($L_2$-regularized Logistic Regression).
2. **Sample Reweighing**: Inversely proportional joint class/group weights $w_i = \frac{N}{K \cdot \text{count}(s, y)}$.
3. **Group-Specific Thresholds**: Post-processing threshold optimization equalizing subgroup False Negative Rates to target cohort FNR.
4. **Fairlearn ExponentiatedGradient**: In-processing constrained optimization enforcing `TruePositiveRateParity`.

| Strategy | Sensitive Attribute | ROC-AUC | PR-AUC | Recall (Sensitivity) | Precision | Brier Score | Max FNR Disparity Gap |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **None (Unmitigated)** | gender | 0.942 | 0.938 | 84.7% | 90.5% | 0.0805 | **7.21%** |
| **Sample Reweighing** | gender | 0.942 | 0.938 | 86.2% | 86.0% | 0.0859 | **8.90%** |
| **Group-Specific Thresholds (FNR Parity)** | gender | 0.942 | 0.938 | 81.7% | 87.9% | 0.0805 | **1.48%** |
| **Fairlearn ExponentiatedGradient** | gender | 0.893 | 0.835 | 77.7% | 86.2% | 0.0970 | **4.36%** |

---

## 4. Time-Based Learning Analytics Benchmark: OULAD Snapshot Day 56
- **Dataset**: Open University Learning Analytics Dataset (OULAD, Day $t = 56$).
- **Cohort**: Model trained on 2013 presentations (2013B + 2013J); audited on the **unseen 2014 temporal holdout cohort** (2014B + 2014J).
- **Feature Set**: 24 cumulative behavioural and engagement features up to Day 56 (all demographic attributes strictly excluded from $X$).
- **Selection Rate Threshold**: Top 20% predicted risk cohort.

| Attribute | Subgroup | Sample Size ($N$) | Base Rate | Selection Rate (Top 20%) | Recall (TPR) | Miss Rate (FNR) | False Alarm (FPR) | Precision | ECE | FNR Gap vs Ref | 95% Bootstrap CI | Sample Flag |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **gender** | F | 6458 | 15.0% | 18.2% | 34.6% | 65.4% | 15.3% | 28.5% | 0.034 | 5.17% | [1.37%, 8.85%] | sufficient |
| **gender** | M *(Ref)* | 8634 | 17.3% | 21.3% | 39.8% | 60.2% | 17.4% | 32.4% | 0.055 | 0.00% | — | sufficient |
| **age_band** | 0-35 *(Ref)* | 10489 | 16.6% | 21.1% | 39.4% | 60.6% | 17.5% | 31.0% | 0.045 | 0.00% | — | sufficient |
| **age_band** | 35-55 | 4491 | 15.8% | 17.5% | 33.9% | 66.1% | 14.5% | 30.6% | 0.047 | 5.58% | [1.32%, 9.77%] | sufficient |
| **age_band** | 55<= | 112 | 13.4% | 15.2% | 33.3% | 66.7% | 12.4% | 29.4% | 0.043 | 6.10% | [-17.50%, 31.75%] | sufficient |
| **disability** | N *(Ref)* | 13681 | 15.6% | 19.3% | 36.3% | 63.7% | 16.2% | 29.4% | 0.040 | 0.00% | — | sufficient |
| **disability** | Y | 1411 | 23.2% | 26.4% | 47.3% | 52.7% | 20.1% | 41.5% | 0.100 | -10.93% | [-16.73%, -4.84%] | sufficient |
| **imd_band** | 0-10% | 1420 | 18.1% | 24.5% | 44.0% | 56.0% | 20.2% | 32.5% | 0.054 | -7.16% | [-16.31%, 2.25%] | sufficient |
| **imd_band** | 10-20% | 1525 | 17.8% | 22.2% | 39.0% | 61.0% | 18.6% | 31.3% | 0.056 | -2.16% | [-12.01%, 6.83%] | sufficient |
| **imd_band** | 20-30% | 1636 | 19.7% | 21.9% | 41.5% | 58.5% | 17.1% | 37.4% | 0.077 | -4.68% | [-13.41%, 4.55%] | sufficient |
| **imd_band** | 30-40% | 1621 | 16.4% | 21.0% | 31.9% | 68.0% | 18.8% | 25.0% | 0.046 | 4.86% | [-4.57%, 14.22%] | sufficient |
| **imd_band** | 40-50% | 1475 | 16.5% | 19.8% | 36.9% | 63.1% | 16.4% | 30.8% | 0.048 | -0.08% | [-9.62%, 9.49%] | sufficient |
| **imd_band** | 50-60% | 1461 | 15.6% | 19.8% | 41.7% | 58.3% | 15.7% | 32.9% | 0.042 | -4.86% | [-14.40%, 4.85%] | sufficient |
| **imd_band** | 60-70% | 1408 | 16.8% | 18.5% | 36.4% | 63.6% | 14.8% | 33.1% | 0.055 | 0.37% | [-9.34%, 10.40%] | sufficient |
| **imd_band** | 70-80% | 1372 | 14.4% | 19.2% | 35.4% | 64.6% | 16.5% | 26.5% | 0.032 | 1.46% | [-9.17%, 10.76%] | sufficient |
| **imd_band** | 80-90% | 1349 | 14.5% | 17.1% | 32.6% | 67.3% | 14.5% | 27.7% | 0.035 | 4.16% | [-5.43%, 14.34%] | sufficient |
| **imd_band** | 90-100% *(Ref)* | 1229 | 13.3% | 16.9% | 36.8% | 63.2% | 13.9% | 28.8% | 0.036 | 0.00% | — | sufficient |
| **imd_band** | Missing | 596 | 14.1% | 15.1% | 34.5% | 65.5% | 11.9% | 32.2% | 0.041 | 2.29% | [-11.05%, 14.31%] | sufficient |
| **highest_education** | A Level or Equivalent *(Ref)* | 6682 | 15.2% | 17.8% | 35.6% | 64.4% | 14.5% | 30.6% | 0.039 | 0.00% | — | sufficient |
| **highest_education** | HE Qualification | 2424 | 15.7% | 12.7% | 26.5% | 73.5% | 10.1% | 32.9% | 0.061 | 9.11% | [4.31%, 14.42%] | sufficient |
| **highest_education** | Lower Than A Level | 5655 | 18.0% | 25.8% | 44.6% | 55.4% | 21.7% | 31.1% | 0.049 | -8.97% | [-13.55%, -4.58%] | sufficient |
| **highest_education** | No Formal quals | 136 | 16.9% | 39.7% | 43.5% | 56.5% | 38.9% | 18.5% | 0.061 | -7.86% | [-28.17%, 12.62%] | sufficient |
| **highest_education** | Post Graduate Qualification | 195 | 14.4% | 7.2% | 17.9% | 82.1% | 5.4% | 35.7% | 0.066 | 17.76% | [1.83%, 31.33%] | sufficient |
| **region** | East Anglian Region | 1466 | 14.5% | 18.1% | 39.1% | 60.9% | 14.6% | 31.2% | 0.028 | -4.73% | [-14.09%, 4.27%] | sufficient |
| **region** | East Midlands Region | 1073 | 17.9% | 18.7% | 39.6% | 60.4% | 14.2% | 37.8% | 0.063 | -5.16% | [-14.50%, 4.03%] | sufficient |
| **region** | Ireland | 590 | 15.8% | 16.1% | 30.1% | 69.9% | 13.5% | 29.5% | 0.052 | 4.31% | [-7.01%, 15.17%] | sufficient |
| **region** | London Region | 1409 | 16.4% | 21.0% | 33.3% | 66.7% | 18.6% | 26.0% | 0.045 | 1.09% | [-7.74%, 9.63%] | sufficient |
| **region** | North Region | 869 | 17.2% | 20.8% | 35.6% | 64.4% | 17.8% | 29.3% | 0.064 | -1.15% | [-10.65%, 8.58%] | sufficient |
| **region** | North Western Region | 1257 | 16.6% | 22.0% | 40.9% | 59.1% | 18.3% | 30.7% | 0.044 | -6.45% | [-16.24%, 1.83%] | sufficient |
| **region** | Scotland | 1908 | 18.4% | 18.9% | 37.0% | 63.0% | 14.8% | 36.1% | 0.073 | -2.62% | [-10.20%, 5.92%] | sufficient |
| **region** | South East Region | 964 | 16.0% | 20.0% | 35.1% | 64.9% | 17.2% | 28.0% | 0.043 | -0.64% | [-9.46%, 9.96%] | sufficient |
| **region** | South Region *(Ref)* | 1427 | 15.1% | 18.1% | 34.4% | 65.6% | 15.3% | 28.6% | 0.036 | 0.00% | — | sufficient |
| **region** | South West Region | 1121 | 17.1% | 20.2% | 41.7% | 58.3% | 15.8% | 35.2% | 0.054 | -7.25% | [-16.72%, 2.08%] | sufficient |
| **region** | Wales | 1039 | 14.7% | 24.4% | 40.5% | 59.5% | 21.7% | 24.4% | 0.031 | -6.10% | [-15.15%, 4.22%] | sufficient |
| **region** | West Midlands Region | 1098 | 16.4% | 21.8% | 40.0% | 60.0% | 18.2% | 30.1% | 0.044 | -5.58% | [-14.79%, 4.26%] | sufficient |
| **region** | Yorkshire Region | 871 | 15.7% | 19.6% | 42.3% | 57.7% | 15.4% | 33.9% | 0.041 | -7.92% | [-17.43%, 3.35%] | sufficient |
| **imd_x_gender** | 0-10%_F | 721 | 16.2% | 22.2% | 39.3% | 60.7% | 18.9% | 28.7% | 0.040 | -1.27% | [-13.01%, 10.95%] | sufficient |
| **imd_x_gender** | 0-10%_M | 699 | 20.0% | 26.9% | 47.9% | 52.1% | 21.6% | 35.6% | 0.073 | -9.81% | [-22.05%, 2.02%] | sufficient |
| **imd_x_gender** | 10-20%_F | 714 | 17.1% | 20.2% | 29.5% | 70.5% | 18.2% | 25.0% | 0.049 | 8.54% | [-4.45%, 20.59%] | sufficient |
| **imd_x_gender** | 10-20%_M | 811 | 18.5% | 24.0% | 46.7% | 53.3% | 18.9% | 35.9% | 0.062 | -8.62% | [-20.42%, 3.16%] | sufficient |
| **imd_x_gender** | 20-30%_F | 803 | 19.8% | 20.2% | 38.4% | 61.6% | 15.7% | 37.6% | 0.080 | -0.31% | [-11.58%, 11.57%] | sufficient |
| **imd_x_gender** | 20-30%_M | 833 | 19.7% | 23.5% | 44.5% | 55.5% | 18.4% | 37.2% | 0.075 | -6.46% | [-18.64%, 5.56%] | sufficient |
| **imd_x_gender** | 30-40%_F | 700 | 15.7% | 19.1% | 30.9% | 69.1% | 17.0% | 25.4% | 0.043 | 7.14% | [-4.52%, 20.18%] | sufficient |
| **imd_x_gender** | 30-40%_M | 921 | 16.9% | 22.4% | 32.7% | 67.3% | 20.3% | 24.8% | 0.049 | 5.36% | [-6.26%, 17.22%] | sufficient |
| **imd_x_gender** | 40-50%_F | 687 | 15.6% | 16.9% | 34.6% | 65.4% | 13.6% | 31.9% | 0.043 | 3.47% | [-9.58%, 15.89%] | sufficient |
| **imd_x_gender** | 40-50%_M | 788 | 17.4% | 22.3% | 38.7% | 61.3% | 18.9% | 30.1% | 0.053 | -0.64% | [-12.67%, 11.95%] | sufficient |
| **imd_x_gender** | 50-60%_F | 653 | 11.8% | 14.7% | 33.8% | 66.2% | 12.2% | 27.1% | 0.013 | 4.28% | [-9.55%, 18.35%] | sufficient |
| **imd_x_gender** | 50-60%_M | 808 | 18.7% | 23.9% | 45.7% | 54.3% | 18.9% | 35.8% | 0.066 | -7.65% | [-18.68%, 4.78%] | sufficient |
| **imd_x_gender** | 60-70%_F | 579 | 14.2% | 17.3% | 37.8% | 62.2% | 13.9% | 31.0% | 0.029 | 0.25% | [-13.52%, 14.43%] | sufficient |
| **imd_x_gender** | 60-70%_M | 829 | 18.6% | 19.3% | 35.7% | 64.3% | 15.6% | 34.4% | 0.073 | 2.34% | [-9.76%, 14.69%] | sufficient |
| **imd_x_gender** | 70-80%_F | 504 | 12.1% | 17.7% | 29.5% | 70.5% | 16.0% | 20.2% | 0.024 | 8.54% | [-6.83%, 22.19%] | sufficient |
| **imd_x_gender** | 70-80%_M | 868 | 15.8% | 20.2% | 38.0% | 62.0% | 16.8% | 29.7% | 0.047 | 0.09% | [-12.76%, 11.71%] | sufficient |
| **imd_x_gender** | 80-90%_F | 511 | 12.9% | 14.5% | 31.8% | 68.2% | 11.9% | 28.4% | 0.022 | 6.23% | [-8.04%, 20.74%] | sufficient |
| **imd_x_gender** | 80-90%_M | 838 | 15.5% | 18.7% | 33.1% | 66.9% | 16.1% | 27.4% | 0.043 | 4.97% | [-6.64%, 17.01%] | sufficient |
| **imd_x_gender** | 90-100%_F | 437 | 11.4% | 16.5% | 34.0% | 66.0% | 14.2% | 23.6% | 0.011 | 4.05% | [-11.37%, 19.09%] | sufficient |
| **imd_x_gender** | 90-100%_M *(Ref)* | 792 | 14.3% | 17.2% | 38.0% | 62.0% | 13.7% | 31.6% | 0.051 | 0.00% | — | sufficient |
| **imd_x_gender** | Missing_F | 149 | 12.8% | 20.8% | 47.4% | 52.6% | 16.9% | 29.0% | 0.031 | -9.32% | [-32.31%, 15.89%] | sufficient |
| **imd_x_gender** | Missing_M | 447 | 14.5% | 13.2% | 30.8% | 69.2% | 10.2% | 33.9% | 0.047 | 7.28% | [-7.10%, 20.71%] | sufficient |

---

## 5. OULAD Presentation Shift Check: 2013 In-Sample vs. 2014 Temporal Holdout
Evaluates temporal stability of subgroup error rates and disparity gaps across academic years (2013 calendar presentations vs. 2014 calendar presentations at snapshot $t = 56$).

| Attribute | Group | $N$ (2013) | $N$ (2014) | Base Rate 2013 | Base Rate 2014 | FNR 2013 | FNR 2014 | FNR Shift (2014 - 2013) | FNR Gap 2013 | FNR Gap 2014 | Gap Shift (2014 - 2013) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **gender** | F | 5497 | 6458 | 13.8% | 15.0% | 54.9% | 65.4% | 10.47% | -11.20% | 5.17% | 16.37% |
| **gender** | M *(Ref)* | 5933 | 8634 | 14.7% | 17.3% | 66.1% | 60.2% | -5.90% | 0.00% | 0.00% | 0.00% |
| **age_band** | 0-35 *(Ref)* | 7964 | 10489 | 14.5% | 16.6% | 61.9% | 60.6% | -1.28% | 0.00% | 0.00% | 0.00% |
| **age_band** | 35-55 | 3393 | 4491 | 13.6% | 15.8% | 58.7% | 66.1% | 7.49% | -3.19% | 5.58% | 8.77% |
| **age_band** | 55<= | 73 | 112 | 11.0% | 13.4% | 50.0% | 66.7% | 16.67% | -11.85% | 6.10% | 17.95% |
| **disability** | N *(Ref)* | 10356 | 13681 | 13.4% | 15.6% | 63.3% | 63.7% | 0.34% | 0.00% | 0.00% | 0.00% |
| **disability** | Y | 1074 | 1411 | 22.2% | 23.2% | 46.6% | 52.7% | 6.10% | -16.69% | -10.93% | 5.76% |
| **imd_band** | 0-10% | 1128 | 1420 | 18.8% | 18.1% | 52.4% | 56.0% | 3.67% | -9.94% | -7.16% | 2.78% |
| **imd_band** | 10-20% | 1201 | 1525 | 15.4% | 17.8% | 60.0% | 61.0% | 1.03% | -2.30% | -2.16% | 0.14% |
| **imd_band** | 20-30% | 1229 | 1636 | 17.2% | 19.7% | 58.5% | 58.5% | 0.02% | -3.81% | -4.68% | -0.87% |
| **imd_band** | 30-40% | 1261 | 1621 | 13.6% | 16.4% | 59.3% | 68.0% | 8.75% | -3.00% | 4.86% | 7.86% |
| **imd_band** | 40-50% | 1136 | 1475 | 13.5% | 16.5% | 60.8% | 63.1% | 2.33% | -1.52% | -0.08% | 1.44% |
| **imd_band** | 50-60% | 1129 | 1461 | 12.3% | 15.6% | 67.6% | 58.3% | -9.30% | 5.33% | -4.86% | -10.19% |
| **imd_band** | 60-70% | 1011 | 1408 | 13.6% | 16.8% | 54.0% | 63.6% | 9.55% | -8.29% | 0.37% | 8.66% |
| **imd_band** | 70-80% | 1047 | 1372 | 13.4% | 14.4% | 66.4% | 64.6% | -1.78% | 4.13% | 1.46% | -2.67% |
| **imd_band** | 80-90% | 950 | 1349 | 12.1% | 14.5% | 70.4% | 67.3% | -3.08% | 8.13% | 4.16% | -3.97% |
| **imd_band** | 90-100% *(Ref)* | 936 | 1229 | 13.0% | 13.3% | 62.3% | 63.2% | 0.89% | 0.00% | 0.00% | 0.00% |
| **imd_band** | Missing | 402 | 596 | 9.7% | 14.1% | 79.5% | 65.5% | -14.01% | 17.19% | 2.29% | -14.90% |
| **highest_education** | A Level or Equivalent *(Ref)* | 4984 | 6682 | 12.8% | 15.2% | 62.7% | 64.4% | 1.64% | 0.00% | 0.00% | 0.00% |
| **highest_education** | HE Qualification | 1607 | 2424 | 12.8% | 15.7% | 72.7% | 73.5% | 0.81% | 9.94% | 9.11% | -0.83% |
| **highest_education** | Lower Than A Level | 4648 | 5655 | 16.2% | 18.0% | 56.3% | 55.4% | -0.91% | -6.42% | -8.97% | -2.55% |
| **highest_education** | No Formal quals | 109 | 136 | 22.0% | 16.9% | 37.5% | 56.5% | 19.02% | -25.24% | -7.86% | 17.38% |
| **highest_education** | Post Graduate Qualification | 82 | 195 | 12.2% | 14.4% | 100.0% | 82.1% | -17.86% | 37.26% | 17.76% | -19.50% |
| **region** | East Anglian Region | 1244 | 1466 | 13.2% | 14.5% | 63.4% | 60.9% | -2.56% | -1.90% | -4.73% | -2.83% |
| **region** | East Midlands Region | 778 | 1073 | 14.9% | 17.9% | 62.1% | 60.4% | -1.65% | -3.24% | -5.16% | -1.92% |
| **region** | Ireland | 480 | 590 | 13.8% | 15.8% | 62.1% | 69.9% | 7.77% | -3.19% | 4.31% | 7.50% |
| **region** | London Region | 1085 | 1409 | 14.6% | 16.4% | 46.5% | 66.7% | 20.13% | -18.77% | 1.09% | 19.86% |
| **region** | North Region | 611 | 869 | 13.8% | 17.2% | 73.8% | 64.4% | -9.38% | 8.50% | -1.15% | -9.65% |
| **region** | North Western Region | 966 | 1257 | 14.8% | 16.6% | 55.9% | 59.1% | 3.19% | -9.37% | -6.45% | 2.92% |
| **region** | Scotland | 1152 | 1908 | 14.8% | 18.4% | 64.7% | 63.0% | -1.75% | -0.60% | -2.62% | -2.02% |
| **region** | South East Region | 736 | 964 | 11.4% | 16.0% | 64.3% | 64.9% | 0.65% | -1.02% | -0.64% | 0.38% |
| **region** | South Region *(Ref)* | 1098 | 1427 | 13.4% | 15.1% | 65.3% | 65.6% | 0.27% | 0.00% | 0.00% | 0.00% |
| **region** | South West Region | 871 | 1121 | 14.1% | 17.1% | 55.3% | 58.3% | 3.05% | -10.03% | -7.25% | 2.78% |
| **region** | Wales | 803 | 1039 | 15.8% | 14.7% | 62.2% | 59.5% | -2.72% | -3.11% | -6.10% | -2.99% |
| **region** | West Midlands Region | 903 | 1098 | 16.9% | 16.4% | 65.4% | 60.0% | -5.36% | 0.05% | -5.58% | -5.63% |
| **region** | Yorkshire Region | 703 | 871 | 12.8% | 15.7% | 55.6% | 57.7% | 2.10% | -9.75% | -7.92% | 1.83% |

---

## 6. Income Feature Ablation Benchmark (Simulated Cohort)
Evaluates model performance and disaggregated False Negative Rates across household income brackets when `income_slab_idx` and `financial_stress_index` are included versus completely ablated from the 37-feature simulated pipeline.

| Feature Configuration | Predictor Features | ROC-AUC | PR-AUC | Recall | Precision | Brier Score | FNR (<2 LPA) | FNR (>8 LPA) | FNR Disparity Gap |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **With Income Features (37 features)** | 37 | 0.935 | 0.916 | 78.9% | 88.8% | 0.0867 | 46.3% | 45.1% | **1.25%** |
| **Without Income Features (35 features)** | 35 | 0.936 | 0.917 | 78.9% | 89.2% | 0.0861 | 46.9% | 44.0% | **2.95%** |

#### Disaggregated Performance Across Income Slabs (Simulated Cohort)

| Income Bracket | $N$ Students | Actual Dropouts | Base Rate | FNR (With Income Features) | FNR (Without Income Features) | FNR Difference (Without - With) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **<2 LPA** | 432 | 162 | 37.5% | 46.30% | 46.91% | 0.61% |
| **2-5 LPA** | 774 | 282 | 36.4% | 43.26% | 43.26% | 0.00% |
| **5-8 LPA** | 509 | 176 | 34.6% | 46.02% | 46.02% | 0.00% |
| **>8 LPA** | 285 | 91 | 31.9% | 45.05% | 43.96% | -1.09% |

---

## 7. Generator Sanity Check (Simulated Indian Cohort)
Verification of synthetic data generator properties across $N = 2,000$ simulated students.

> [!NOTE]
> **Generator Sanity Check Disclaimer**:
> The simulated Indian cohort is generated from known mathematical equations (`ml/simulation/estimate_parameters.py` and `ml/data_pipeline/generate_synthetic_indian.py`). The numbers below verify internal generator calibration and absence of unintentional statistical distortions; they are **NOT evidence of real-world predictive validity**.

| Demographic Slice | Single Held-Out Test Split ($N=300$) | 5-Fold Cross-Validation ($N=2,000$ Out-of-Fold) | Empirical Difference |
| :--- | :--- | :--- | :--- |
| **Gender Disparity Gap** (Female vs Male FNR) | **5.41%** | **0.05%** | **5.36%** |
| **Economic Proxy Gap** (<5 LPA vs $\ge$5 LPA) | **3.05%** | **4.07%** | **1.02%** |
| **First-Generation Gap** (First-Gen vs Non-First-Gen) | **0.35%** | **1.96%** | **1.61%** |

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
