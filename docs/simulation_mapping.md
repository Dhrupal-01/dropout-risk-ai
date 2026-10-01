# Simulation Parameter Estimation & Proxy Mapping

> Grounding simulated Indian collegiate risk mechanisms in empirical benchmark data (UCI ID 697 and OULAD UCI ID 349).
> Coefficients estimated via standardized logistic regression and transferred to Indian cohort natural units via standard deviation scaling: $\beta_{\text{ind}} = \beta_{\text{std}} / \sigma(X_{\text{ind}})$.

---

## Empirical Feature Mapping & Transferred Effects

| Indian Feature | Source Dataset | Exact Source Column(s) | Transformation / Proxy Mapping | Standardised Effect [95% CI] | Transferred Indian-Unit Effect [95% CI] | Proxy Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `current_cgpa` | UCI (ID 697) | `cu_1st_sem_grade` | Linear rescaling from Portuguese 0–20 scale to Indian 0–10 CGPA scale (`grade / 2.0`). | -0.8914 [-1.0433, -0.7395] | -0.6157 [-0.7206, -0.5108] | |
| `backlog_count` | UCI (ID 697) | `cu_1st_sem_enrolled`, `cu_1st_sem_approved` | Difference between enrolled and approved curricular units (`enrolled - approved`). | +2.1943 [+1.9680, +2.4206] | +1.9535 [+1.7520, +2.1549] | |
| `has_scholarship` | UCI (ID 697) | `scholarship_holder` | Direct binary indicator (1 = active scholarship, 0 = no scholarship). | -0.4406 [-0.5704, -0.3109] | -0.9048 [-1.1714, -0.6385] | |
| `fee_payment_delay_days` | UCI (ID 697) | `tuition_fees_up_to_date`, `debtor` | **Binary** financial default indicator: 1 if `tuition_fees_up_to_date == 0` or `debtor == 1`, else 0. UCI records no delay duration (see note below). | +0.7982 [+0.6746, +0.9217] | +0.0507 [+0.0429, +0.0586] | |
| `is_first_generation` | UCI (ID 697) | `mothers_qualification`, `fathers_qualification` | Neither parent holding higher education degree (codes 2–6, 40–44). | +0.0120 [-0.1028, +0.1268] | +0.0247 [-0.2117, +0.2611] | |
| `is_hosteler` | UCI (ID 697) | `displaced` | Displaced from home region as institutional proxy for residential hosteler status. | +0.1148 [-0.0037, +0.2334] | +0.2306 [-0.0074, +0.4689] | |
| `lms_logins_per_week` | OULAD (ID 349) | `studentVle` (weeks 1–8) | Mean weekly click volume across first 8 weeks at snapshot t=56. | -0.1544 [-0.2001, -0.1087] | -0.0685 [-0.0888, -0.0482] | |
| `days_since_last_lms_activity` | OULAD (ID 349) | `studentVle.date` (<= t=56) | Recency of last digital interaction: `56 - max(date)` at snapshot t=56. | +0.1220 [+0.0912, +0.1528] | +0.0290 [+0.0217, +0.0364] | |
| `assignment_submission_lag_days` | OULAD (ID 349) | `studentAssessment.date_submitted`, `assessments.date` | Mean difference between submission date and assessment due date (`date_submitted - due_date`). | +0.3110 [+0.2760, +0.3460] | +0.1321 [+0.1173, +0.1470] | |

---

## Methodological Note on Effect Transfer

Because empirical source features (such as Portuguese curricular grades on a 0–20 scale or raw OULAD click volumes) have different measurement units and variances than the target Indian collegiate cohort features, direct unstandardized coefficient transfer would introduce arbitrary scale distortion.

To preserve empirical effect magnitude, each source regression is fit on z-score standardized features ($Z_j = (X_j - \mu_j) / \sigma_j$), yielding log-odds effects per one standard deviation change ($\beta_{\text{std}}$). The transferred effect in Indian collegiate units is computed as:

$$\beta_{\text{ind}} = \frac{\beta_{\text{std}}}{\sigma(X_{\text{ind}})}$$

where $\sigma(X_{\text{ind}})$ is the sample standard deviation of that indicator in a simulated cohort generated with a fixed seed (n = 2,000, seed = 42).

## UCI Estimation Split

UCI effects are estimated on 2,904 rows only. The remaining 726 rows (20% stratified holdout, seed 42) are reserved for the sim-to-real check (`ml/simulation/uci_proxies.py`), so that check never evaluates on data that informed these parameters.

## Fee-Delay Proxy Is Binary

The UCI source for `fee_payment_delay_days` is a 0/1 default indicator (`tuition_fees_up_to_date == 0` or `debtor == 1`); UCI has no delay duration. Its standardized effect is divided by the simulated SD of delay **days**, which treats one SD of being in default as one SD of delay days. This is an assumption, not an estimate of a per-day effect.
