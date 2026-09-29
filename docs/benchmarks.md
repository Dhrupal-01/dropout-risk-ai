# Real-Data Benchmark Evaluation Report

> Rigorous, leak-free empirical evaluation on official educational benchmarks: UCI ID 697 and OULAD (UCI ID 349).
> Conducted via the standardized evaluation harness across temporal holdouts, Leave-One-Course/Module-Out, and repeated cross-validation.
> Point estimates and 95% bootstrap confidence intervals (1,000 resamples of out-of-fold predictions).

---

## 1. Portuguese Higher Education Benchmark (UCI ID 697)

- **Dataset**: UCI ID 697 ($N = 4,424$, 17 Degree Programs / Courses)
- **Feature Sets**:
  - `ENROLMENT_TIME`: Baseline features available at matriculation (Demographics, admission credentials, socio-economic signals, macroeconomic indicators; 24 features).
  - `END_OF_SEM1`: `ENROLMENT_TIME` + 1st-semester curricular unit evaluations and approved units (30 features). Zero 2nd-semester features.
  - `FULL`: Everything, including 2nd-semester curricular units (36 features). Explicitly labeled as **not early warning**.
- **Label Variants**:
  - `Primary`: Binary outcome ($N = 3,630$). `Dropout = 1` vs `Graduate = 0`. Students with `Target == 'Enrolled'` are excluded.
  - `Sensitivity`: Binary outcome ($N = 4,424$). `Dropout = 1` vs `Graduate / Enrolled = 0`. Students with `Target == 'Enrolled'` are coded as $0$.

### Reliability & Probability Calibration

![UCI Benchmark Reliability Diagram](figures/uci_reliability_curves.png)

---

### UCI Primary Cohort Evaluation (Dropout vs Graduate, N = 3,630)

Excludes active Enrolled students (1,421 Dropouts, 2,209 Graduates; Prevalence = 39.15%).

#### Repeated Stratified 5-Fold CV (3 repeats)

| Feature Set | Model | ROC-AUC (95% CI) | PR-AUC (95% CI) | Top 10% Prec (95% CI) | Top 10% Recall (95% CI) | Top 20% Prec (95% CI) | Top 20% Recall (95% CI) | Brier Score (95% CI) | ECE (95% CI) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ENROLMENT_TIME (Admission / Baseline) | Majority Class (Prior) | 0.5012 [0.4844, 0.5191] | 0.3924 [0.3747, 0.4103] | 0.3912 [0.3499, 0.4490] | 0.0999 [0.0894, 0.1134] | 0.4022 [0.3595, 0.4284] | 0.2055 [0.1859, 0.2176] | 0.2382 [0.2347, 0.2419] | 0.0000 [0.0000, 0.0187] |
| ENROLMENT_TIME (Admission / Baseline) | Logistic Regression | 0.8216 [0.8072, 0.8345] | 0.7853 [0.7641, 0.8033] | 0.9752 [0.9587, 0.9945] | 0.2491 [0.2405, 0.2608] | 0.8430 [0.8113, 0.8788] | 0.4307 [0.4158, 0.4484] | 0.1613 [0.1546, 0.1676] | 0.0259 [0.0185, 0.0403] |
| ENROLMENT_TIME (Admission / Baseline) | XGBoost | 0.8571 [0.8446, 0.8692] | 0.8277 [0.8099, 0.8440] | 0.9890 [0.9751, 0.9972] | 0.2526 [0.2428, 0.2624] | 0.9063 [0.8760, 0.9298] | 0.4631 [0.4468, 0.4786] | 0.1439 [0.1371, 0.1507] | 0.0129 [0.0115, 0.0293] |
| END_OF_SEM1 (First Semester Completed) | Majority Class (Prior) | 0.5012 [0.4844, 0.5191] | 0.3924 [0.3747, 0.4103] | 0.3912 [0.3499, 0.4490] | 0.0999 [0.0894, 0.1134] | 0.4022 [0.3595, 0.4284] | 0.2055 [0.1859, 0.2176] | 0.2382 [0.2347, 0.2419] | 0.0000 [0.0000, 0.0187] |
| END_OF_SEM1 (First Semester Completed) | Logistic Regression | 0.9346 [0.9257, 0.9434] | 0.9292 [0.9197, 0.9388] | 0.9945 [0.9862, 1.0000] | 0.2540 [0.2438, 0.2654] | 0.9890 [0.9807, 0.9972] | 0.5053 [0.4864, 0.5277] | 0.0857 [0.0791, 0.0918] | 0.0154 [0.0131, 0.0279] |
| END_OF_SEM1 (First Semester Completed) | XGBoost | 0.9418 [0.9334, 0.9492] | 0.9362 [0.9273, 0.9445] | 1.0000 [1.0000, 1.0000] | 0.2555 [0.2449, 0.2665] | 0.9931 [0.9862, 0.9986] | 0.5074 [0.4881, 0.5290] | 0.0819 [0.0755, 0.0883] | 0.0110 [0.0097, 0.0231] |
| FULL (not early warning) | Majority Class (Prior) | 0.5012 [0.4844, 0.5191] | 0.3924 [0.3747, 0.4103] | 0.3912 [0.3499, 0.4490] | 0.0999 [0.0894, 0.1134] | 0.4022 [0.3595, 0.4284] | 0.2055 [0.1859, 0.2176] | 0.2382 [0.2347, 0.2419] | 0.0000 [0.0000, 0.0187] |
| FULL (not early warning) | Logistic Regression | 0.9529 [0.9452, 0.9603] | 0.9508 [0.9430, 0.9581] | 0.9972 [0.9890, 1.0000] | 0.2548 [0.2443, 0.2659] | 0.9972 [0.9931, 1.0000] | 0.5095 [0.4892, 0.5319] | 0.0682 [0.0626, 0.0739] | 0.0187 [0.0132, 0.0288] |
| FULL (not early warning) | XGBoost | 0.9587 [0.9520, 0.9651] | 0.9550 [0.9479, 0.9613] | 1.0000 [1.0000, 1.0000] | 0.2555 [0.2449, 0.2665] | 0.9986 [0.9959, 1.0000] | 0.5102 [0.4895, 0.5324] | 0.0673 [0.0618, 0.0735] | 0.0108 [0.0079, 0.0211] |

#### Leave-One-Course/Module-Out (LOGO)

| Feature Set | Model | ROC-AUC (95% CI) | PR-AUC (95% CI) | Top 10% Prec (95% CI) | Top 10% Recall (95% CI) | Top 20% Prec (95% CI) | Top 20% Recall (95% CI) | Brier Score (95% CI) | ECE (95% CI) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ENROLMENT_TIME (Admission / Baseline) | Majority Class (Prior) | 0.3056 [0.2890, 0.3223] | 0.2943 [0.2794, 0.3095] | 0.1901 [0.1377, 0.2149] | 0.0486 [0.0356, 0.0549] | 0.1736 [0.1501, 0.2080] | 0.0887 [0.0778, 0.1053] | 0.2446 [0.2412, 0.2480] | 0.1230 [0.1075, 0.1390] |
| ENROLMENT_TIME (Admission / Baseline) | Logistic Regression | 0.7932 [0.7775, 0.8080] | 0.7598 [0.7373, 0.7785] | 0.9752 [0.9559, 0.9890] | 0.2491 [0.2395, 0.2596] | 0.8264 [0.7961, 0.8567] | 0.4222 [0.4068, 0.4399] | 0.1720 [0.1650, 0.1786] | 0.0310 [0.0243, 0.0454] |
| ENROLMENT_TIME (Admission / Baseline) | XGBoost | 0.8071 [0.7924, 0.8235] | 0.7765 [0.7557, 0.7959] | 0.9807 [0.9641, 0.9945] | 0.2505 [0.2411, 0.2608] | 0.8581 [0.8223, 0.8912] | 0.4384 [0.4226, 0.4552] | 0.1683 [0.1604, 0.1754] | 0.0408 [0.0335, 0.0554] |
| END_OF_SEM1 (First Semester Completed) | Majority Class (Prior) | 0.3056 [0.2890, 0.3223] | 0.2943 [0.2794, 0.3095] | 0.1901 [0.1377, 0.2149] | 0.0486 [0.0356, 0.0549] | 0.1736 [0.1501, 0.2080] | 0.0887 [0.0778, 0.1053] | 0.2446 [0.2412, 0.2480] | 0.1230 [0.1075, 0.1390] |
| END_OF_SEM1 (First Semester Completed) | Logistic Regression | 0.9290 [0.9194, 0.9382] | 0.9221 [0.9117, 0.9325] | 0.9917 [0.9807, 1.0000] | 0.2533 [0.2432, 0.2648] | 0.9876 [0.9780, 0.9959] | 0.5046 [0.4856, 0.5261] | 0.0912 [0.0841, 0.0978] | 0.0229 [0.0175, 0.0350] |
| END_OF_SEM1 (First Semester Completed) | XGBoost | 0.9156 [0.9051, 0.9249] | 0.8974 [0.8839, 0.9093] | 0.9945 [0.9835, 1.0000] | 0.2540 [0.2446, 0.2652] | 0.9394 [0.9187, 0.9573] | 0.4799 [0.4640, 0.4986] | 0.1046 [0.0972, 0.1129] | 0.0378 [0.0313, 0.0493] |
| FULL (not early warning) | Majority Class (Prior) | 0.3056 [0.2890, 0.3223] | 0.2943 [0.2794, 0.3095] | 0.1901 [0.1377, 0.2149] | 0.0486 [0.0356, 0.0549] | 0.1736 [0.1501, 0.2080] | 0.0887 [0.0778, 0.1053] | 0.2446 [0.2412, 0.2480] | 0.1230 [0.1075, 0.1390] |
| FULL (not early warning) | Logistic Regression | 0.9454 [0.9371, 0.9540] | 0.9437 [0.9349, 0.9519] | 0.9972 [0.9890, 1.0000] | 0.2548 [0.2443, 0.2659] | 0.9972 [0.9931, 1.0000] | 0.5095 [0.4892, 0.5319] | 0.0737 [0.0673, 0.0803] | 0.0188 [0.0140, 0.0295] |
| FULL (not early warning) | XGBoost | 0.9299 [0.9205, 0.9388] | 0.8756 [0.8549, 0.8951] | 0.8733 [0.8402, 0.9091] | 0.2231 [0.2125, 0.2346] | 0.8967 [0.8733, 0.9187] | 0.4581 [0.4371, 0.4804] | 0.0856 [0.0786, 0.0936] | 0.0314 [0.0262, 0.0426] |

---

### UCI Sensitivity Cohort Evaluation (Enrolled as Negative, N = 4,424)

Treats Enrolled students as non-dropouts (1,421 Dropouts, 3,003 Non-Dropouts; Prevalence = 32.12%).

#### Repeated Stratified 5-Fold CV (3 repeats)

| Feature Set | Model | ROC-AUC (95% CI) | PR-AUC (95% CI) | Top 10% Prec (95% CI) | Top 10% Recall (95% CI) | Top 20% Prec (95% CI) | Top 20% Recall (95% CI) | Brier Score (95% CI) | ECE (95% CI) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ENROLMENT_TIME (Admission / Baseline) | Majority Class (Prior) | 0.5032 [0.4847, 0.5219] | 0.3242 [0.3074, 0.3433] | 0.3341 [0.2889, 0.3792] | 0.1042 [0.0902, 0.1164] | 0.3277 [0.2994, 0.3638] | 0.2041 [0.1886, 0.2238] | 0.2180 [0.2132, 0.2231] | 0.0000 [0.0000, 0.0165] |
| ENROLMENT_TIME (Admission / Baseline) | Logistic Regression | 0.7925 [0.7778, 0.8066] | 0.6977 [0.6743, 0.7209] | 0.9074 [0.8712, 0.9368] | 0.2829 [0.2695, 0.2948] | 0.7198 [0.6847, 0.7559] | 0.4483 [0.4297, 0.4659] | 0.1596 [0.1539, 0.1661] | 0.0130 [0.0109, 0.0275] |
| ENROLMENT_TIME (Admission / Baseline) | XGBoost | 0.8289 [0.8162, 0.8418] | 0.7483 [0.7280, 0.7703] | 0.9368 [0.9120, 0.9639] | 0.2920 [0.2797, 0.3048] | 0.7876 [0.7548, 0.8169] | 0.4905 [0.4710, 0.5074] | 0.1450 [0.1395, 0.1513] | 0.0140 [0.0106, 0.0276] |
| END_OF_SEM1 (First Semester Completed) | Majority Class (Prior) | 0.5032 [0.4847, 0.5219] | 0.3242 [0.3074, 0.3433] | 0.3341 [0.2889, 0.3792] | 0.1042 [0.0902, 0.1164] | 0.3277 [0.2994, 0.3638] | 0.2041 [0.1886, 0.2238] | 0.2180 [0.2132, 0.2231] | 0.0000 [0.0000, 0.0165] |
| END_OF_SEM1 (First Semester Completed) | Logistic Regression | 0.8974 [0.8867, 0.9079] | 0.8467 [0.8291, 0.8643] | 0.9661 [0.9481, 0.9842] | 0.3012 [0.2886, 0.3148] | 0.9006 [0.8768, 0.9266] | 0.5609 [0.5419, 0.5818] | 0.1079 [0.1019, 0.1144] | 0.0098 [0.0091, 0.0233] |
| END_OF_SEM1 (First Semester Completed) | XGBoost | 0.9095 [0.8988, 0.9189] | 0.8658 [0.8501, 0.8808] | 0.9887 [0.9729, 0.9977] | 0.3082 [0.2946, 0.3214] | 0.9186 [0.8983, 0.9412] | 0.5721 [0.5518, 0.5932] | 0.1022 [0.0962, 0.1088] | 0.0107 [0.0093, 0.0236] |
| FULL (not early warning) | Majority Class (Prior) | 0.5032 [0.4847, 0.5219] | 0.3242 [0.3074, 0.3433] | 0.3341 [0.2889, 0.3792] | 0.1042 [0.0902, 0.1164] | 0.3277 [0.2994, 0.3638] | 0.2041 [0.1886, 0.2238] | 0.2180 [0.2132, 0.2231] | 0.0000 [0.0000, 0.0165] |
| FULL (not early warning) | Logistic Regression | 0.9182 [0.9077, 0.9278] | 0.8767 [0.8610, 0.8917] | 0.9752 [0.9594, 0.9887] | 0.3040 [0.2907, 0.3176] | 0.9322 [0.9141, 0.9480] | 0.5806 [0.5567, 0.6031] | 0.0941 [0.0881, 0.1005] | 0.0107 [0.0090, 0.0218] |
| FULL (not early warning) | XGBoost | 0.9293 [0.9206, 0.9377] | 0.8913 [0.8773, 0.9055] | 0.9865 [0.9752, 0.9955] | 0.3075 [0.2939, 0.3212] | 0.9435 [0.9277, 0.9627] | 0.5876 [0.5665, 0.6115] | 0.0897 [0.0837, 0.0959] | 0.0153 [0.0109, 0.0257] |

#### Leave-One-Course/Module-Out (LOGO)

| Feature Set | Model | ROC-AUC (95% CI) | PR-AUC (95% CI) | Top 10% Prec (95% CI) | Top 10% Recall (95% CI) | Top 20% Prec (95% CI) | Top 20% Recall (95% CI) | Brier Score (95% CI) | ECE (95% CI) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ENROLMENT_TIME (Admission / Baseline) | Majority Class (Prior) | 0.3456 [0.3281, 0.3628] | 0.2474 [0.2355, 0.2624] | 0.1377 [0.1196, 0.1874] | 0.0429 [0.0378, 0.0577] | 0.1548 [0.1345, 0.1831] | 0.0964 [0.0849, 0.1121] | 0.2214 [0.2166, 0.2264] | 0.0052 [0.0004, 0.0192] |
| ENROLMENT_TIME (Admission / Baseline) | Logistic Regression | 0.7687 [0.7520, 0.7838] | 0.6770 [0.6536, 0.7015] | 0.8939 [0.8600, 0.9255] | 0.2787 [0.2658, 0.2917] | 0.7119 [0.6768, 0.7480] | 0.4433 [0.4241, 0.4612] | 0.1658 [0.1599, 0.1728] | 0.0199 [0.0160, 0.0344] |
| ENROLMENT_TIME (Admission / Baseline) | XGBoost | 0.7905 [0.7743, 0.8050] | 0.7075 [0.6849, 0.7313] | 0.9165 [0.8826, 0.9458] | 0.2857 [0.2725, 0.2976] | 0.7424 [0.7062, 0.7808] | 0.4624 [0.4431, 0.4799] | 0.1596 [0.1535, 0.1665] | 0.0286 [0.0233, 0.0425] |
| END_OF_SEM1 (First Semester Completed) | Majority Class (Prior) | 0.3456 [0.3281, 0.3628] | 0.2474 [0.2355, 0.2624] | 0.1377 [0.1196, 0.1874] | 0.0429 [0.0378, 0.0577] | 0.1548 [0.1345, 0.1831] | 0.0964 [0.0849, 0.1121] | 0.2214 [0.2166, 0.2264] | 0.0052 [0.0004, 0.0192] |
| END_OF_SEM1 (First Semester Completed) | Logistic Regression | 0.8928 [0.8814, 0.9036] | 0.8407 [0.8227, 0.8588] | 0.9661 [0.9458, 0.9842] | 0.3012 [0.2880, 0.3142] | 0.8938 [0.8734, 0.9186] | 0.5567 [0.5368, 0.5775] | 0.1100 [0.1039, 0.1167] | 0.0142 [0.0101, 0.0256] |
| END_OF_SEM1 (First Semester Completed) | XGBoost | 0.8800 [0.8685, 0.8901] | 0.7957 [0.7749, 0.8174] | 0.8736 [0.8397, 0.9074] | 0.2723 [0.2592, 0.2866] | 0.8328 [0.8090, 0.8610] | 0.5186 [0.4983, 0.5424] | 0.1239 [0.1171, 0.1313] | 0.0375 [0.0326, 0.0497] |
| FULL (not early warning) | Majority Class (Prior) | 0.3456 [0.3281, 0.3628] | 0.2474 [0.2355, 0.2624] | 0.1377 [0.1196, 0.1874] | 0.0429 [0.0378, 0.0577] | 0.1548 [0.1345, 0.1831] | 0.0964 [0.0849, 0.1121] | 0.2214 [0.2166, 0.2264] | 0.0052 [0.0004, 0.0192] |
| FULL (not early warning) | Logistic Regression | 0.9029 [0.8916, 0.9134] | 0.8601 [0.8437, 0.8774] | 0.9707 [0.9526, 0.9865] | 0.3026 [0.2893, 0.3163] | 0.9232 [0.9062, 0.9435] | 0.5749 [0.5538, 0.5987] | 0.1009 [0.0940, 0.1078] | 0.0219 [0.0168, 0.0330] |
| FULL (not early warning) | XGBoost | 0.9023 [0.8923, 0.9120] | 0.8145 [0.7936, 0.8350] | 0.8713 [0.8330, 0.9007] | 0.2716 [0.2581, 0.2844] | 0.8508 [0.8237, 0.8723] | 0.5299 [0.5075, 0.5508] | 0.1103 [0.1032, 0.1174] | 0.0349 [0.0305, 0.0470] |

---

## 2. Open University Learning Analytics Benchmark (OULAD / UCI ID 349)

- **Dataset**: OULAD ($N = 32,593$ student registrations across 7 degree modules).
- **Evaluation Horizons**: Time-bounded snapshots $t \in \{14, 28, 56, 84\}$ days from course start.
- **Population Filtering**: Only registrations where `date_unregistration` is null or $> t$. Students withdrawing on or before $t$ are excluded.
- **Temporal Cutoff**: Clickstream activity, weekly interaction sequences, and assessment submissions restricted strictly to `date <= t`.
- **Demographic Isolation**: `gender`, `age_band`, `imd_band`, `disability`, and `region` are strictly excluded from $X$ and held in an isolated audit frame.
- **Evaluation Protocol**:
  - Primary: Temporal split — train on `2013B + 2013J`, test on `2014B + 2014J`.
  - Secondary: Leave-One-Module-Out (LOGO across 7 modules).

### Early-Warning Earliness Curves

![OULAD Earliness Curves](figures/earliness_curve.png)

### Temporal Holdout Evaluation (Train 2013B/J, Test 2014B/J)

| Snapshot Horizon | Model | PR-AUC (95% CI) | ROC-AUC (95% CI) | Top 10% Prec (95% CI) | Top 10% Recall (95% CI) | Brier Score (95% CI) | ECE (95% CI) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Snapshot t=14 (2 weeks) | Majority Class (Prior) | 0.3669 [0.3149, 0.4189] | 0.5000 [0.5000, 0.5000] | 0.2903 [0.1935, 0.5484] | 0.0796 [0.0588, 0.1504] | 0.2352 [0.2158, 0.2547] | 0.0544 [0.0057, 0.1064] |
| Snapshot t=14 (2 weeks) | Logistic Regression | 0.3262 [0.2705, 0.3947] | 0.4474 [0.3861, 0.5146] | 0.3226 [0.1290, 0.4516] | 0.0885 [0.0396, 0.1207] | 0.2562 [0.2328, 0.2783] | 0.1000 [0.0670, 0.1602] |
| Snapshot t=14 (2 weeks) | XGBoost | 0.3686 [0.3081, 0.4493] | 0.5049 [0.4385, 0.5698] | 0.2581 [0.1290, 0.4516] | 0.0708 [0.0370, 0.1204] | 0.2626 [0.2357, 0.2884] | 0.1560 [0.1099, 0.2084] |
| Snapshot t=14 (2 weeks) | PyTorch GRU | 0.3364 [0.2792, 0.4055] | 0.4561 [0.3961, 0.5210] | 0.1935 [0.0645, 0.3548] | 0.0531 [0.0182, 0.0982] | 0.2412 [0.2223, 0.2599] | 0.0812 [0.0376, 0.1348] |
| Snapshot t=28 (4 weeks) | Majority Class (Prior) | 0.3596 [0.3048, 0.4144] | 0.5000 [0.5000, 0.5000] | 0.2667 [0.2000, 0.5333] | 0.0762 [0.0556, 0.1525] | 0.2344 [0.2120, 0.2568] | 0.0641 [0.0112, 0.1189] |
| Snapshot t=28 (4 weeks) | Logistic Regression | 0.3357 [0.2765, 0.4155] | 0.4649 [0.3935, 0.5310] | 0.3333 [0.1667, 0.5008] | 0.0952 [0.0500, 0.1429] | 0.2566 [0.2300, 0.2843] | 0.1250 [0.0868, 0.1858] |
| Snapshot t=28 (4 weeks) | XGBoost | 0.3657 [0.2961, 0.4482] | 0.5048 [0.4319, 0.5673] | 0.3000 [0.1667, 0.5000] | 0.0857 [0.0500, 0.1443] | 0.2641 [0.2333, 0.2972] | 0.1762 [0.1390, 0.2372] |
| Snapshot t=28 (4 weeks) | PyTorch GRU | 0.3418 [0.2770, 0.4246] | 0.4814 [0.4134, 0.5493] | 0.2667 [0.1333, 0.4667] | 0.0762 [0.0364, 0.1237] | 0.2420 [0.2170, 0.2672] | 0.0926 [0.0534, 0.1521] |
| Snapshot t=56 (8 weeks) | Majority Class (Prior) | 0.3596 [0.3034, 0.4157] | 0.5000 [0.5000, 0.5000] | 0.3704 [0.1852, 0.5556] | 0.1042 [0.0532, 0.1546] | 0.2341 [0.2114, 0.2569] | 0.0620 [0.0091, 0.1182] |
| Snapshot t=56 (8 weeks) | Logistic Regression | 0.3464 [0.2841, 0.4361] | 0.4826 [0.4097, 0.5546] | 0.2593 [0.1111, 0.4444] | 0.0729 [0.0312, 0.1188] | 0.2563 [0.2290, 0.2859] | 0.1573 [0.1098, 0.2172] |
| Snapshot t=56 (8 weeks) | XGBoost | 0.3684 [0.2984, 0.4604] | 0.5189 [0.4489, 0.5886] | 0.4074 [0.1852, 0.5926] | 0.1146 [0.0532, 0.1596] | 0.2672 [0.2339, 0.3018] | 0.1627 [0.1211, 0.2274] |
| Snapshot t=56 (8 weeks) | PyTorch GRU | 0.3743 [0.3066, 0.4743] | 0.5127 [0.4465, 0.5818] | 0.3704 [0.2593, 0.6296] | 0.1042 [0.0714, 0.1724] | 0.2337 [0.2113, 0.2555] | 0.0577 [0.0316, 0.1206] |
| Snapshot t=84 (12 weeks) | Majority Class (Prior) | 0.3663 [0.3086, 0.4280] | 0.5000 [0.5000, 0.5000] | 0.4000 [0.1600, 0.5600] | 0.1124 [0.0526, 0.1562] | 0.2378 [0.2137, 0.2636] | 0.0752 [0.0176, 0.1369] |
| Snapshot t=84 (12 weeks) | Logistic Regression | 0.3696 [0.2920, 0.4705] | 0.4663 [0.3911, 0.5438] | 0.4400 [0.2400, 0.6000] | 0.1236 [0.0649, 0.1711] | 0.2847 [0.2478, 0.3226] | 0.1907 [0.1507, 0.2631] |
| Snapshot t=84 (12 weeks) | XGBoost | 0.3854 [0.3057, 0.4883] | 0.4967 [0.4219, 0.5690] | 0.4000 [0.2000, 0.6000] | 0.1124 [0.0541, 0.1607] | 0.2720 [0.2406, 0.3066] | 0.1822 [0.1412, 0.2503] |
| Snapshot t=84 (12 weeks) | PyTorch GRU | 0.3724 [0.2971, 0.4741] | 0.4815 [0.4021, 0.5564] | 0.4000 [0.2000, 0.6000] | 0.1124 [0.0617, 0.1630] | 0.2419 [0.2178, 0.2652] | 0.0689 [0.0408, 0.1408] |

---

## 3. Sim-to-Real Cross-Domain Transfer Benchmark

- **Evaluation Scope**: Cross-domain generalization between empirical benchmark (UCI ID 697) and simulated Indian cohort.
- **Shared Proxies (6)**: `current_cgpa`, `backlog_count`, `has_scholarship`, `fee_payment_delay_days`, `is_first_generation`, `is_hosteler`.
- **Normalization**: Standardized within domain to reflect relative cohort position.
- **Confidence Intervals**: 95% bootstrap confidence intervals (1,000 resamples of holdout predictions).

| Evaluation Mode | Training Domain | Test Domain | N (Train / Test) | ROC-AUC (95% CI) | PR-AUC (95% CI) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline: Trained on UCI proxies, tested on UCI holdout (Real-on-Real) | UCI (ID 697) | UCI (ID 697) | 2904 / 726 | 0.9399 [0.9203, 0.9587] | 0.9329 [0.9119, 0.9525] |
| Baseline: Trained on Simulated proxies, tested on Simulated holdout (Sim-on-Sim) | Simulated Indian Cohort | Simulated Indian Cohort | 1600 / 400 | 0.9017 [0.8672, 0.9328] | 0.8627 [0.8116, 0.9045] |
| Transfer: Trained on Simulated proxies, tested on UCI holdout (Sim-to-Real) | Simulated Indian Cohort | UCI (ID 697) | 1600 / 726 | 0.9365 [0.9171, 0.9551] | 0.9276 [0.9050, 0.9476] |
| Transfer: Trained on UCI proxies, tested on Simulated holdout (Real-to-Sim) | UCI (ID 697) | Simulated Indian Cohort | 2904 / 400 | 0.8993 [0.8664, 0.9303] | 0.8480 [0.7932, 0.8963] |

---
