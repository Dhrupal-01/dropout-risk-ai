# Simulation Parameter Estimation & Proxy Mapping

> Grounding simulated Indian collegiate risk mechanisms in empirical benchmark data (UCI ID 697 and OULAD UCI ID 349).
> Coefficients estimated via standardized logistic regression and transferred to Indian cohort natural units via standard deviation scaling: $\beta_{\text{ind}} = \beta_{\text{std}} / \sigma(X_{\text{ind}})$.

---

## Empirical Feature Mapping & Transferred Effects

| Indian Feature | Source Dataset | Exact Source Column(s) | Transformation / Proxy Mapping | Standardised Effect [95% CI] | Transferred Indian-Unit Effect [95% CI] | Proxy Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `current_cgpa` | UCI (ID 697) | `cu_1st_sem_grade` | Linear rescaling from Portuguese 0–20 scale to Indian 0–10 CGPA scale (`grade / 2.0`). | -0.8917 [-1.0288, -0.7546] | -0.6150 [-0.7095, -0.5204] | |
| `backlog_count` | UCI (ID 697) | `cu_1st_sem_enrolled`, `cu_1st_sem_approved` | Difference between enrolled and approved curricular units (`enrolled - approved`). | +2.2412 [+2.0370, +2.4454] | +1.7930 [+1.6296, +1.9563] | |
| `has_scholarship` | UCI (ID 697) | `scholarship_holder` | Direct binary indicator (1 = active scholarship, 0 = no scholarship). | -0.4674 [-0.5868, -0.3479] | -1.0387 [-1.3040, -0.7731] | |
| `fee_payment_delay_days` | UCI (ID 697) | `tuition_fees_up_to_date`, `debtor` | Financial default indicator (`tuition_fees == 0 | debtor == 1`) mapped to delay days. | +0.8355 [+0.7253, +0.9457] | +0.0341 [+0.0296, +0.0386] | |
| `is_first_generation` | UCI (ID 697) | `mothers_qualification`, `fathers_qualification` | Neither parent holding higher education degree (codes 2–6, 40–44). | -0.0072 [-0.1100, +0.0955] | -0.0150 [-0.2292, +0.1990] | |
| `is_hosteler` | UCI (ID 697) | `displaced` | Displaced from home region as institutional proxy for residential hosteler status. | +0.0667 [-0.0403, +0.1738] | +0.1334 [-0.0806, +0.3476] | |
| `lms_logins_per_week` | OULAD (ID 349) | `studentVle` (weeks 1–8) | Mean weekly click volume across first 8 weeks at snapshot t=56. | -0.1544 [-0.2001, -0.1087] | -0.0702 [-0.0910, -0.0494] | |
| `days_since_last_lms_activity` | OULAD (ID 349) | `studentVle.date` (<= t=56) | Recency of last digital interaction: `56 - max(date)` at snapshot t=56. | +0.1220 [+0.0912, +0.1528] | +0.0102 [+0.0076, +0.0127] | |
| `assignment_submission_lag_days` | OULAD (ID 349) | `studentAssessment.date_submitted`, `assessments.date` | Mean difference between submission date and assessment due date (`date_submitted - due_date`). | +0.3110 [+0.2760, +0.3460] | +0.1244 [+0.1104, +0.1384] | |

---

## Methodological Note on Effect Transfer

Because empirical source features (such as Portuguese curricular grades on a 0–20 scale or raw OULAD click volumes) have different measurement units and variances than the target Indian collegiate cohort features, direct unstandardized coefficient transfer would introduce arbitrary scale distortion.

To preserve empirical effect magnitude, each source regression is fit on z-score standardized features ($Z_j = (X_j - \mu_j) / \sigma_j$), yielding log-odds effects per one standard deviation change ($\beta_{\text{std}}$). The transferred effect in Indian collegiate units is computed as:

$$\beta_{\text{ind}} = \frac{\beta_{\text{std}}}{\sigma(X_{\text{ind}})}$$

where $\sigma(X_{\text{ind}})$ represents the expected standard deviation of that indicator in the simulated cohort.
