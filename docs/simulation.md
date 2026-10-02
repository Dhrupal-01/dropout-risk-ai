# Simulated Indian Cohort Specification and Grounding

> Complete inventory of all structural coefficients, distribution parameters, and policy thresholds
> governing the simulated Indian collegiate dataset generator (`ml/data_pipeline/generate_synthetic_indian.py`).
> Generated automatically from [`ml/simulation/assumptions.yaml`](../ml/simulation/assumptions.yaml).

---

## Source Taxonomy

- `estimated_from_uci`: Empirically estimated via standardized logistic regression on UCI Student Dropout (ID 697).
- `estimated_from_oulad`: Empirically estimated via standardized logistic regression on OULAD Learning Analytics (UCI ID 349).
- `indian_regulation`: Grounded in Indian higher education regulatory mandates (e.g. AICTE/UGC 75% attendance rule, statutory reservation percentages).
- `assumption`: Explicit engineering or behavioral assumption documented for transparent audit.
- `TODO(citation)`: Temporary parameter placeholder pending published empirical Indian empirical survey citation.

---
## Cohort Metadata

| Parameter Key | Value | Source | Notes / Description |
| :--- | :--- | :--- | :--- |
| `cohort_metadata.target_base_rate` | `0.355` | `TODO(citation)` | Placeholder baseline cohort dropout rate for Indian collegiate technical education. |

## Risk Coefficients

| Parameter Key | Value | Source | Notes / Description |
| :--- | :--- | :--- | :--- |
| `risk_coefficients.current_cgpa` | `-0.61572` (from `estimated_parameters.json#indian_unit_coefficient.current_cgpa`) | `estimated_from_uci` | Indian-unit coefficient estimated from UCI ID 697 cu_1st_sem_grade rescaled to a 0-10 CGPA; fitted on the UCI train split (shared estimation/holdout split, holdout excluded). The value lives only in estimated_parameters.json (python -m ml.simulation.estimate_parameters). |
| `risk_coefficients.backlog_count` | `1.95349` (from `estimated_parameters.json#indian_unit_coefficient.backlog_count`) | `estimated_from_uci` | Indian-unit coefficient estimated from UCI ID 697 unapproved curricular units (enrolled - approved); fitted on the UCI train split (shared estimation/holdout split, holdout excluded). The value lives only in estimated_parameters.json (python -m ml.simulation.estimate_parameters). |
| `risk_coefficients.has_scholarship` | `-0.90481` (from `estimated_parameters.json#indian_unit_coefficient.has_scholarship`) | `estimated_from_uci` | Indian-unit coefficient estimated from UCI ID 697 scholarship_holder; fitted on the UCI train split (shared estimation/holdout split, holdout excluded). The value lives only in estimated_parameters.json (python -m ml.simulation.estimate_parameters). |
| `risk_coefficients.fee_payment_delay_days` | `0.05074` (from `estimated_parameters.json#indian_unit_coefficient.fee_payment_delay_days`) | `estimated_from_uci` | Indian-unit coefficient estimated from UCI ID 697 binary fee default / debtor indicator; fitted on the UCI train split (shared estimation/holdout split, holdout excluded). The value lives only in estimated_parameters.json (python -m ml.simulation.estimate_parameters). |
| `risk_coefficients.is_first_generation` | `0.02471` (from `estimated_parameters.json#indian_unit_coefficient.is_first_generation`) | `estimated_from_uci` | Indian-unit coefficient estimated from UCI ID 697 parents' qualification (no higher-education degree); fitted on the UCI train split (shared estimation/holdout split, holdout excluded). The value lives only in estimated_parameters.json (python -m ml.simulation.estimate_parameters). |
| `risk_coefficients.is_hosteler` | `0.23065` (from `estimated_parameters.json#indian_unit_coefficient.is_hosteler`) | `estimated_from_uci` | Indian-unit coefficient estimated from UCI ID 697 displaced status; fitted on the UCI train split (shared estimation/holdout split, holdout excluded). The value lives only in estimated_parameters.json (python -m ml.simulation.estimate_parameters). |
| `risk_coefficients.lms_logins_per_week` | `-0.06853` (from `estimated_parameters.json#indian_unit_coefficient.lms_logins_per_week`) | `estimated_from_oulad` | Indian-unit coefficient estimated from OULAD ID 349 mean weekly clicks at snapshot t=56; fitted on all registrations active at t=56 (no holdout; OULAD is not used by the sim-to-real check). The value lives only in estimated_parameters.json (python -m ml.simulation.estimate_parameters). |
| `risk_coefficients.days_since_last_lms_activity` | `0.02903` (from `estimated_parameters.json#indian_unit_coefficient.days_since_last_lms_activity`) | `estimated_from_oulad` | Indian-unit coefficient estimated from OULAD ID 349 inactivity recency at snapshot t=56; fitted on all registrations active at t=56 (no holdout; OULAD is not used by the sim-to-real check). The value lives only in estimated_parameters.json (python -m ml.simulation.estimate_parameters). |
| `risk_coefficients.assignment_submission_lag_days` | `0.13214` (from `estimated_parameters.json#indian_unit_coefficient.assignment_submission_lag_days`) | `estimated_from_oulad` | Indian-unit coefficient estimated from OULAD ID 349 assignment submission lag at snapshot t=56; fitted on all registrations active at t=56 (no holdout; OULAD is not used by the sim-to-real check). The value lives only in estimated_parameters.json (python -m ml.simulation.estimate_parameters). |
| `risk_coefficients.attendance_percentage` | `-0.04` | `indian_regulation` | Penalty for attendance deficit below statutory 75% rule (log-odds increase of 0.040 per percentage deficit). |
| `risk_coefficients.attendance_3m_trend` | `-0.08` | `assumption` | Deteriorating attendance trajectory slope over 3-month observation window. |
| `risk_coefficients.consecutive_absences` | `0.075` | `assumption` | Log-odds penalty per consecutive absent day indicating prolonged absence spell. |
| `risk_coefficients.cgpa_delta` | `-0.4` | `assumption` | Acute academic decline log-odds penalty per unit semester-over-semester CGPA drop. |
| `risk_coefficients.stem_core_fail_flag` | `0.35` | `assumption` | Prerequisite gatekeeping failure penalty in core foundational course. |
| `risk_coefficients.gender_male` | `0` | `assumption` | Protected attribute coefficient strictly zero to prevent algorithmic gender bias. |
| `risk_coefficients.category_obc` | `0` | `assumption` | Protected demographic category coefficient strictly zero. |
| `risk_coefficients.category_sc` | `0` | `assumption` | Protected demographic category coefficient strictly zero. |
| `risk_coefficients.category_st` | `0` | `assumption` | Protected demographic category coefficient strictly zero. |
| `risk_coefficients.category_ews` | `0` | `assumption` | Protected demographic category coefficient strictly zero. |
| `risk_coefficients.interaction_low_att_high_fee` | `0.35` | `assumption` | Compound interaction for concurrent attendance crisis (<70%) and chronic fee delay (>20d). |
| `risk_coefficients.interaction_low_cgpa_high_backlogs` | `0.35` | `assumption` | Compound interaction for severe academic distress (CGPA < 5.0 and backlogs >= 2). |
| `risk_coefficients.idiosyncratic_noise_std` | `0.85` | `assumption` | Standard deviation of unobserved idiosyncratic Gaussian noise in latent risk score. |

## Regulations And Thresholds

| Parameter Key | Value | Source | Notes / Description |
| :--- | :--- | :--- | :--- |
| `regulations_and_thresholds.mandatory_attendance_threshold` | `75` | `indian_regulation` | Statutory 75% attendance threshold for semester examination eligibility. |
| `regulations_and_thresholds.stem_core_pass_mark` | `42` | `indian_regulation` | Passing threshold out of 100 for academic credit in foundational STEM courses. |
| `regulations_and_thresholds.dual_crisis_att_threshold` | `70` | `assumption` | Threshold below which attendance crisis compounds with fee delay. |
| `regulations_and_thresholds.dual_crisis_fee_threshold` | `20` | `assumption` | Threshold above which fee payment delay compounds with attendance crisis. |
| `regulations_and_thresholds.academic_crisis_cgpa_threshold` | `5` | `assumption` | Threshold below which low CGPA triggers year-down / detaining risk. |
| `regulations_and_thresholds.academic_crisis_backlog_threshold` | `2` | `assumption` | Threshold for minimum backlogs compounding low CGPA. |

## Demographics Distribution

| Parameter Key | Value | Source | Notes / Description |
| :--- | :--- | :--- | :--- |
| `demographics_distribution.student_id_prefix` | `IND_2026_` | `assumption` | Identifier prefix formatted with year and sequential 4-digit student index. |
| `demographics_distribution.gender_proportions` | `{Female: 0.42, Male: 0.58}` | `TODO(citation)` | Estimated gender distribution in Indian STEM technical collegiate degree programs. |
| `demographics_distribution.category_proportions` | `{EWS: 0.06, General: 0.35, OBC: 0.32, SC: 0.18, ST: 0.09}` | `indian_regulation` | Central educational institution statutory reservation allocation proportions. |
| `demographics_distribution.age_mean` | `20.1` | `assumption` | Mean student age in 2nd/3rd collegiate year. |
| `demographics_distribution.age_std` | `1.3` | `assumption` | Standard deviation of collegiate student age. |
| `demographics_distribution.age_min` | `17.5` | `indian_regulation` | Statutory minimum age for admission to degree programs. |
| `demographics_distribution.age_max` | `27` | `assumption` | Upper bound of typical undergraduate student age distribution. |
| `demographics_distribution.hostel_status_proportions` | `{Day Scholar: 0.55, Hosteler: 0.45}` | `assumption` | Collegiate residential distribution between on-campus hostel and local commute. |
| `demographics_distribution.commute_hosteler_km` | `0.5` | `assumption` | Fixed nominal commute distance for residential hostel students. |
| `demographics_distribution.commute_day_scholar_shape` | `2.5` | `assumption` | Gamma distribution shape parameter for day scholar commute distance. |
| `demographics_distribution.commute_day_scholar_scale` | `4` | `assumption` | Gamma distribution scale parameter for day scholar commute distance. |
| `demographics_distribution.commute_day_scholar_min_km` | `1` | `assumption` | Minimum commute distance for day scholar students. |
| `demographics_distribution.commute_day_scholar_max_km` | `45` | `assumption` | Maximum commute distance for day scholar students. |
| `demographics_distribution.commute_day_scholar_shift_km` | `1.5` | `assumption` | Constant baseline shift added to Gamma-drawn day scholar commute distance. |

## Socioeconomic Distribution

| Parameter Key | Value | Source | Notes / Description |
| :--- | :--- | :--- | :--- |
| `socioeconomic_distribution.income_slab_labels` | `[<2 LPA, 2-5 LPA, 5-8 LPA, '>8 LPA']` | `assumption` | Standard family income categorization brackets in Indian collegiate admissions. |
| `socioeconomic_distribution.income_slab_proportions` | `[0.22, 0.38, 0.25, 0.15]` | `TODO(citation)` | Income bracket distribution in state-affiliated technical colleges. |
| `socioeconomic_distribution.first_gen_probs_by_slab` | `[0.65, 0.45, 0.2, 0.08]` | `assumption` | Probability of first-generation learner status conditioned on income bracket index. |
| `socioeconomic_distribution.scholarship_quota_prob` | `0.65` | `indian_regulation` | Post-matric and merit-cum-means scholarship coverage for SC/ST/EWS or <2 LPA income. |
| `socioeconomic_distribution.scholarship_mid_prob` | `0.25` | `assumption` | Scholarship coverage probability for 2-5 LPA income slab. |
| `socioeconomic_distribution.scholarship_high_prob` | `0.05` | `assumption` | Merit scholarship coverage probability for higher income brackets. |
| `socioeconomic_distribution.fee_delay_with_scholarship_choices` | `[0, 5, 15, 30]` | `assumption` | Discrete fee delay day choices for scholarship recipients. |
| `socioeconomic_distribution.fee_delay_with_scholarship_probs` | `[0.75, 0.15, 0.07, 0.03]` | `assumption` | Probabilities of fee delay days for scholarship recipients. |
| `socioeconomic_distribution.fee_delay_slab0_choices` | `[0, 15, 30, 45, 60, 90, 120]` | `assumption` | Discrete fee delay day choices for low income <2 LPA students without scholarship. |
| `socioeconomic_distribution.fee_delay_slab0_probs` | `[0.15, 0.2, 0.25, 0.2, 0.1, 0.07, 0.03]` | `assumption` | Probabilities of fee delay days for low income <2 LPA students. |
| `socioeconomic_distribution.fee_delay_slab1_choices` | `[0, 10, 20, 30, 45, 60]` | `assumption` | Discrete fee delay day choices for 2-5 LPA students without scholarship. |
| `socioeconomic_distribution.fee_delay_slab1_probs` | `[0.45, 0.25, 0.15, 0.08, 0.05, 0.02]` | `assumption` | Probabilities of fee delay days for 2-5 LPA students. |
| `socioeconomic_distribution.fee_delay_high_slab_choices` | `[0, 5, 15]` | `assumption` | Discrete fee delay day choices for >5 LPA students without scholarship. |
| `socioeconomic_distribution.fee_delay_high_slab_probs` | `[0.85, 0.12, 0.03]` | `assumption` | Probabilities of fee delay days for >5 LPA students. |
| `socioeconomic_distribution.fee_delay_jitter_max` | `5` | `assumption` | Maximum uniform integer jitter added to base fee delay days. |

## Attendance Distribution

| Parameter Key | Value | Source | Notes / Description |
| :--- | :--- | :--- | :--- |
| `attendance_distribution.latent_alpha` | `5` | `assumption` | Beta distribution alpha parameter for latent student attendance discipline. |
| `attendance_distribution.latent_beta` | `1.8` | `assumption` | Beta distribution beta parameter for latent student attendance discipline. |
| `attendance_distribution.core1_noise_std` | `6` | `assumption` | Core subject 1 attendance random variation standard deviation. |
| `attendance_distribution.core2_noise_std` | `5.5` | `assumption` | Core subject 2 attendance random variation standard deviation. |
| `attendance_distribution.lab_boost` | `4` | `assumption` | Attendance percentage shift for mandatory practical lab sessions. |
| `attendance_distribution.lab_noise_std` | `4` | `assumption` | Practical lab attendance random variation standard deviation. |
| `attendance_distribution.elective_penalty` | `2` | `assumption` | Attendance percentage shift for elective courses. |
| `attendance_distribution.elective_noise_std` | `7` | `assumption` | Elective course attendance random variation standard deviation. |
| `attendance_distribution.month_weights` | `[0.25, 0.35, 0.4]` | `assumption` | Recency weights applied to 3-month attendance records (Month 1, Month 2, Month 3). |
| `attendance_distribution.commute_long_threshold_km` | `25` | `assumption` | Distance threshold triggering severe commute-related attendance penalty. |
| `attendance_distribution.commute_long_penalty` | `0.08` | `assumption` | Attendance penalty applied for commute exceeding long distance threshold. |
| `attendance_distribution.commute_medium_threshold_km` | `15` | `assumption` | Distance threshold triggering moderate commute-related attendance penalty. |
| `attendance_distribution.commute_medium_penalty` | `0.04` | `assumption` | Attendance penalty applied for commute exceeding medium distance threshold. |
| `attendance_distribution.attendance_min` | `10` | `assumption` | Floor percentage for attendance metrics. |
| `attendance_distribution.attendance_max` | `100` | `assumption` | Ceiling percentage for attendance metrics. |
| `attendance_distribution.latent_attendance_min_clip` | `0.15` | `assumption` | Lower clipping bound for commute-adjusted latent attendance factor. |
| `attendance_distribution.latent_attendance_max_clip` | `0.98` | `assumption` | Upper clipping bound for commute-adjusted latent attendance factor. |
| `attendance_distribution.attendance_slope_noise_std` | `4.5` | `assumption` | Standard deviation of latent monthly attendance trend slope. |
| `attendance_distribution.consecutive_absence_thresholds` | `[50.0, 65.0, 75.0]` | `assumption` | Attendance tier cutoffs determining consecutive absence days window. |
| `attendance_distribution.consecutive_absence_ranges` | `[[8, 25], [4, 12], [2, 7], [0, 3]]` | `assumption` | Min and max integer bounds for consecutive absence day draws by attendance tier. |

## Academic Distribution

| Parameter Key | Value | Source | Notes / Description |
| :--- | :--- | :--- | :--- |
| `academic_distribution.prior_cgpa_mean` | `7.2` | `assumption` | Mean prior academic ability CGPA score on 10-point scale. |
| `academic_distribution.prior_cgpa_std` | `1.4` | `assumption` | Standard deviation of prior academic ability CGPA score. |
| `academic_distribution.cgpa_noise_std` | `0.45` | `assumption` | Semester-over-semester random academic shock standard deviation. |
| `academic_distribution.attendance_cgpa_slope` | `0.025` | `assumption` | CGPA impact per percentage point deviation from 75% attendance benchmark. |
| `academic_distribution.backlog_base_lambda` | `4.5` | `assumption` | Base multiplier for Poisson backlog rate function. |
| `academic_distribution.backlog_cgpa_decay` | `0.8` | `assumption` | Exponential decay coefficient of backlog rate with respect to current CGPA. |
| `academic_distribution.backlog_attendance_boost` | `1.8` | `assumption` | Additive backlog Poisson rate increase when attendance drops below 65%. |
| `academic_distribution.backlog_max` | `7` | `indian_regulation` | Statutory academic limit of active uncleared backlogs before student is detained. |
| `academic_distribution.internal_exam_cgpa_multiplier` | `9.5` | `assumption` | Linear multiplier converting 10-point CGPA to 100-point internal examination score. |
| `academic_distribution.internal_exam_noise_std` | `5` | `assumption` | Standard deviation of internal examination evaluation noise. |
| `academic_distribution.stem_exam_att_weight` | `0.45` | `assumption` | Weight of attendance in STEM core exam score simulation. |
| `academic_distribution.stem_exam_cgpa_weight` | `4.8` | `assumption` | Weight of CGPA in STEM core exam score simulation. |
| `academic_distribution.stem_exam_noise_std` | `11` | `assumption` | Noise standard deviation in STEM core exam score simulation. |
| `academic_distribution.cgpa_min` | `2.5` | `assumption` | Floor bounding for current semester CGPA. |
| `academic_distribution.cgpa_max` | `10` | `indian_regulation` | Statutory 10-point maximum ceiling for Indian university CGPA. |
| `academic_distribution.prev_cgpa_min` | `3.5` | `assumption` | Floor bounding for previous semester CGPA. |
| `academic_distribution.prev_cgpa_max` | `9.95` | `assumption` | Ceiling bounding for previous semester CGPA. |
| `academic_distribution.internal_exam_min` | `15` | `assumption` | Floor percentage for internal examination score. |
| `academic_distribution.internal_exam_max` | `99` | `assumption` | Ceiling percentage for internal examination score. |
| `academic_distribution.core1_exam_score_min` | `5` | `assumption` | Floor percentage bounding for core 1 exam score simulation. |
| `academic_distribution.core1_exam_score_max` | `98` | `assumption` | Ceiling percentage bounding for core 1 exam score simulation. |

## Learning Behavior Distribution

| Parameter Key | Value | Source | Notes / Description |
| :--- | :--- | :--- | :--- |
| `learning_behavior_distribution.lms_mean_multiplier` | `9` | `assumption` | Multiplier converting engagement score to weekly LMS login frequency. |
| `learning_behavior_distribution.lms_noise_std` | `2` | `assumption` | Standard deviation of weekly LMS logins. |
| `learning_behavior_distribution.lms_max_weekly` | `14` | `assumption` | Upper bound cap on weekly LMS logins. |
| `learning_behavior_distribution.submission_lag_base_scale` | `6` | `assumption` | Sensitivity of assignment submission lag in days to student disengagement. |
| `learning_behavior_distribution.submission_lag_offset` | `2.5` | `assumption` | Baseline submission timeliness offset in days. |
| `learning_behavior_distribution.submission_lag_noise_std` | `2.2` | `assumption` | Standard deviation of assignment submission lag. |
| `learning_behavior_distribution.resource_lognormal_mean` | `2.8` | `assumption` | Base parameter mu for lognormal digital resource access distribution. |
| `learning_behavior_distribution.resource_lognormal_slope` | `1.2` | `assumption` | Linear effect of engagement on lognormal resource access mean. |
| `learning_behavior_distribution.resource_lognormal_sigma` | `0.45` | `assumption` | Dispersion parameter sigma for lognormal resource access distribution. |
| `learning_behavior_distribution.resource_access_min` | `2` | `assumption` | Minimum resource access count. |
| `learning_behavior_distribution.resource_access_max` | `220` | `assumption` | Maximum resource access count. |
| `learning_behavior_distribution.inactivity_scale` | `25` | `assumption` | Maximum baseline LMS inactivity days due to disengagement. |
| `learning_behavior_distribution.inactivity_exp_scale` | `3` | `assumption` | Exponential distribution scale for inactivity recency noise in days. |
| `learning_behavior_distribution.inactivity_max_days` | `60` | `assumption` | Maximum inactivity recency in days. |
| `learning_behavior_distribution.forum_base_lambda` | `4` | `assumption` | Multiplier for Poisson discussion forum participation frequency. |
| `learning_behavior_distribution.forum_participation_max` | `25` | `assumption` | Maximum discussion forum participation count. |
