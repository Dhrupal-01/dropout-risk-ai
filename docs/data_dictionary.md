# DropoutGuard — Feature Data Dictionary

Every column of the simulated cohort (`data/processed/features.csv`, built by
`python -m ml.data_pipeline.feature_engineering`) and what the model does with it. Definitions
are taken from the code; each row names the file and line it comes from.

- **Model features: 36 = 27 raw + 9 engineered.** The lists are `RAW_FEATURE_COLUMNS` and
  `ENGINEERED_FEATURE_COLUMNS` in `backend/app/services/ml_service.py`; the model's column order is
  `ml/artifacts/feature_names.json`.
- **Protected attributes are never model features** (`age`, `gender`, `category`). They stay in the
  cohort for the fairness audit only, and `POST /api/v1/predict` rejects them with 422.
- The data is a **simulated Indian college cohort**. It has not been validated on Indian college records.

**Roles**

| Role | Meaning |
| :--- | :--- |
| `raw` | Supplied by the caller and fed to the model |
| `engineered` | Computed by `build_engineered_features` (`ml/data_pipeline/feature_engineering.py`); a caller may not send it (422), except `is_hosteler`, see below |
| `audit-only` | Protected attribute: never a model feature, used only to audit fairness |
| `excluded` | In the cohort, not a model feature (`ml.config.EXCLUDED_FEATURES`) |
| `input-only` | Accepted by the API to derive an engineered feature, not itself a model feature |
| `label` | Training target; never accepted as inference input |
| `identifier` | Row key |

Accepted ranges are the API bounds in `backend/app/schemas/student.py` (`StudentFeatureInput`);
that file is the source of truth. Percentages are on a 0–100 scale.

---

## 1. Raw model inputs (27)

| Column | Role | Type | Accepted range | Definition | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `commute_distance_km` | raw | float | 0–100 | One-way daily commute in km | `schemas/student.py:66` |
| `income_slab_idx` | raw | int | 0–3 | Family income bracket: 0 = `<2 LPA`, 1 = `2-5 LPA`, 2 = `5-8 LPA`, 3 = `>8 LPA`. A model input on purpose (need signal for financial support) and an audit group (`ml/fairness/attributes.py`) | `schemas/student.py:67` |
| `is_first_generation` | raw | int | 0 or 1 | First in the family to attend college | `schemas/student.py:68` |
| `has_scholarship` | raw | int | 0 or 1 | Holds a scholarship | `schemas/student.py:69` |
| `fee_payment_delay_days` | raw | int | 0–365 | Days the tuition fee payment is overdue | `schemas/student.py:70-72` |
| `att_core1` | raw | float | 0–100 | Attendance in the first core course | `schemas/student.py:79` |
| `att_core2` | raw | float | 0–100 | Attendance in the second core course | `schemas/student.py:80` |
| `att_lab` | raw | float | 0–100 | Laboratory attendance | `schemas/student.py:81` |
| `att_elective` | raw | float | 0–100 | Elective attendance | `schemas/student.py:82` |
| `attendance_month_1` | raw | float | 0–100 | Attendance in month 1 of the tracking window | `schemas/student.py:83` |
| `attendance_month_2` | raw | float | 0–100 | Attendance in month 2 | `schemas/student.py:84` |
| `attendance_month_3` | raw | float | 0–100 | Attendance in month 3 (most recent) | `schemas/student.py:85` |
| `attendance_percentage` | raw | float | 0–100 | Overall attendance over the window | `schemas/student.py:86` |
| `attendance_3m_trend` | raw | float | −50 to 50 | Optional. If omitted: `(attendance_month_3 - attendance_month_1) / 2`, rounded to 2 dp (API: `schemas/student.py:152-155`; pipeline: `feature_engineering.py:109-113`) | `schemas/student.py:91` |
| `consecutive_absences` | raw | int | 0–180 | Longest run of consecutive absent days | `schemas/student.py:87` |
| `attendance_risk_flag` | raw | int | 0 or 1 | Optional. If omitted: `1` when attendance is below the statutory threshold. Pipeline: `attendance_percentage < ATTENDANCE_THRESHOLD` (`feature_engineering.py:116-117`), from `regulations_and_thresholds.mandatory_attendance_threshold` in `ml/simulation/assumptions.yaml:98-101`. API: the same comparison with the threshold typed in (`schemas/student.py:156-157`) | `schemas/student.py:92` |
| `prev_sem_cgpa` | raw | float | 0–10 | Previous semester CGPA (10-point scale) | `schemas/student.py:95` |
| `current_cgpa` | raw | float | 0–10 | Current CGPA | `schemas/student.py:96` |
| `cgpa_delta` | raw | float | −10 to 10 | Optional. If omitted: `current_cgpa - prev_sem_cgpa`, rounded to 2 dp (API: `schemas/student.py:150-151`; pipeline: `feature_engineering.py:122-123`) | `schemas/student.py:98` |
| `backlog_count` | raw | int | 0–20 | Uncleared failed courses | `schemas/student.py:99` |
| `internal_exam_score_pct` | raw | float | 0–100 | Internal assessment score | `schemas/student.py:100` |
| `stem_core_fail_flag` | raw | int | 0 or 1 | Failed a core STEM course | `schemas/student.py:101` |
| `lms_logins_per_week` | raw | float | 0–50 | LMS logins per week | `schemas/student.py:104` |
| `assignment_submission_lag_days` | raw | float | −30 to 90 | Days relative to the deadline; negative = early | `schemas/student.py:105-110` |
| `resource_access_count` | raw | int | 0–1000 | LMS resources accessed | `schemas/student.py:111` |
| `days_since_last_lms_activity` | raw | int | 0–365 | Days since the last LMS activity | `schemas/student.py:112` |
| `forum_participation_count` | raw | int | 0–200 | Forum posts and replies | `schemas/student.py:113` |

## 2. Engineered model features (9)

All computed in `build_engineered_features`, `ml/data_pipeline/feature_engineering.py`. Training
and serving call the same function. `clip(x, 0, 1)` means `np.clip(x, 0.0, 1.0)`.

| Column | Role | Formula (as in the code) | Source |
| :--- | :--- | :--- | :--- |
| `subject_attendance_std` | engineered | Standard deviation (pandas, `ddof=1`) of `att_core1`, `att_core2`, `att_lab`, `att_elective`, rounded to 2 dp | `feature_engineering.py:102-104` |
| `academic_crisis_flag` | engineered | `1` if `current_cgpa < 5.0` **or** `backlog_count >= 2`, else `0` | `feature_engineering.py:126` |
| `behavioral_disengagement_index` | engineered | `0.35 * (1 - clip(lms_logins_per_week / 10)) + 0.35 * clip(assignment_submission_lag_days / 7) + 0.30 * clip(days_since_last_lms_activity / 30)`, rounded to 3 dp | `feature_engineering.py:133-136` |
| `is_hosteler` | engineered | `1` if `hostel_status == "Hosteler"`, else `0`. The API accepts either `hostel_status` or `is_hosteler`; a supplied `is_hosteler` is used as is | `feature_engineering.py:142-143`, `schemas/student.py:158-159` |
| `financial_stress_index` | engineered | `0.40 * (3 - income_slab_idx) / 3 + 0.40 * clip(fee_payment_delay_days / 60) + 0.20 * (1 - has_scholarship)`, rounded to 3 dp | `feature_engineering.py:152-155` |
| `interaction_att_x_fee` | engineered | `max(0, 100 - attendance_percentage) * fee_payment_delay_days / 30`, rounded to 2 dp | `feature_engineering.py:162-165` |
| `interaction_cgpa_x_backlog` | engineered | `max(0, 10 - current_cgpa) * backlog_count`, rounded to 2 dp | `feature_engineering.py:168-171` |
| `interaction_firstgen_x_inactivity` | engineered | `is_first_generation * days_since_last_lms_activity / 10`, rounded to 2 dp | `feature_engineering.py:175-178` |
| `interaction_att_x_cgpa_drop` | engineered | `attendance_risk_flag * max(0, -cgpa_delta)`, rounded to 2 dp | `feature_engineering.py:181-185` |

## 3. Columns that are not model inputs

| Column | Role | Definition | Source |
| :--- | :--- | :--- | :--- |
| `age` | audit-only | Protected attribute (`PROTECTED["simulated"]`, `ml/fairness/attributes.py`). Simulated from `demographics_distribution.age_*` in `assumptions.yaml`; not used in the label formula. The simulated fairness audit groups by age at or below vs above the cohort median (`compute_age_band_audit`, `ml/models/fairness_audit.py`). Rejected by `/predict` | `generate_synthetic_indian.py:151-155` |
| `gender` | audit-only | Protected attribute. Fairness audit group | `generate_synthetic_indian.py:146` |
| `category` | audit-only | Protected attribute (caste category). Fairness audit group | `generate_synthetic_indian.py:149` |
| `family_income_slab` | excluded | Text label of the income bracket; the model uses `income_slab_idx` instead | `ml/config.py` `EXCLUDED_FEATURES` |
| `hostel_status` | input-only | `"Hosteler"` or `"Day Scholar"`; source of `is_hosteler` | `schemas/student.py:75` |
| `is_dropout` | label | Simulated outcome, `is_dropout ~ Bernoulli(p)` with `p` the generator probability (`ground_truth_risk_prob` before rounding) | `generate_synthetic_indian.py:417,421,462` |
| `ground_truth_risk_prob` | label | The generator's latent probability (rounded to 4 dp); never shown to the model | `generate_synthetic_indian.py:461` |
| `student_id` | identifier | Row key | `generate_synthetic_indian.py:143` |
