"""
Synthetic Indian College Dataset Generator for DropoutGuard
Context: Indian Higher Education Engineering/Degree College (SDG 4 / SIH 2026 PSID 7-L)

Generates ~2,000 realistic student records across the 4 core pillars:
1. Attendance: Monthly % per subject, 3-month trend slope, consecutive absent days, mandatory 75% rule violation flag.
2. Academic Performance: CGPA trajectory, semester-over-semester delta, backlog count, internal exam scores, core subject failure.
3. Learning Behavior: LMS login frequency, assignment submission lag (days), digital resource views, forum activity, inactivity recency.
4. Socio-Economic Indicators: Family income slab, first-generation learner status, hostel vs day-scholar status, fee payment delay (days), scholarship status, commute distance.

All structural coefficients, distribution parameters, and policy thresholds are loaded dynamically
from ml/simulation/assumptions.yaml without hardcoded constants in Python.
The baseline intercept beta_0 is solved numerically at generation time to calibrate the expected cohort
dropout rate exactly to the target base rate, and binary labels are drawn via Bernoulli trials.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
SYNTHETIC_DATA_DIR = BASE_DIR / "data" / "synthetic"
DEFAULT_OUTPUT_PATH = SYNTHETIC_DATA_DIR / "indian_college_students.csv"
ASSUMPTIONS_PATH = BASE_DIR / "ml" / "simulation" / "assumptions.yaml"


ESTIMATED_SOURCES = ("estimated_from_uci", "estimated_from_oulad")


def _resolve_value_from(reference: str, yaml_dir: Path, entry_name: str) -> float:
    """
    Resolves `<json file>#<field>.<entry key>` (JSON path relative to the YAML's directory) to
    json[<entry key>][<field>]. Raises ValueError if the file, key or field is missing.
    """
    try:
        file_part, field_path = reference.split("#", 1)
        field, key = field_path.split(".", 1)
    except ValueError as exc:
        raise ValueError(f"{entry_name}: malformed value_from '{reference}' (expected '<file>#<field>.<key>')") from exc

    json_path = yaml_dir / file_part
    if not json_path.exists():
        raise ValueError(f"{entry_name}: value_from file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if key not in data or field not in data[key]:
        raise ValueError(f"{entry_name}: value_from '{reference}' does not resolve in {json_path.name}")
    value = data[key][field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{entry_name}: value_from '{reference}' resolved to non-numeric {value!r}")
    return float(value)


def load_simulation_assumptions(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Loads simulation parameters and assumptions from YAML specification.
    Entries with `value_from` are resolved from the referenced JSON (estimated coefficients live only
    in ml/simulation/estimated_parameters.json); `estimated_from_*` entries must use `value_from`.
    """
    yaml_path = Path(path or ASSUMPTIONS_PATH)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Simulation assumptions file not found at: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    for section, items in data.items():
        if not isinstance(items, dict):
            continue
        for key, entry in items.items():
            if not isinstance(entry, dict):
                continue
            name = f"{section}.{key}"
            has_value, has_ref = "value" in entry, "value_from" in entry
            if has_value and has_ref:
                raise ValueError(f"{name}: has both 'value' and 'value_from'; an estimate must live in one place")
            if entry.get("source") in ESTIMATED_SOURCES and not has_ref:
                raise ValueError(f"{name}: source '{entry['source']}' must use 'value_from', not a literal value")
            if has_ref:
                entry["value"] = _resolve_value_from(entry["value_from"], yaml_path.parent, name)
    return data


def get_param(assumptions: Dict[str, Any], section: str, key: str) -> Any:
    """Helper to safely retrieve parameter values from structured assumptions."""
    try:
        return assumptions[section][key]["value"]
    except KeyError as exc:
        raise KeyError(f"Parameter '{section}.{key}' missing from assumptions.yaml") from exc


def solve_intercept_for_base_rate(z_uncentered: np.ndarray, target_rate: float) -> float:
    """
    Solves for beta_0 such that mean(sigmoid(beta_0 + z_uncentered)) == target_rate.
    Since mean(sigmoid(beta_0 + z)) is strictly monotonically increasing in beta_0,
    Brent's root finding algorithm guarantees rapid and exact numerical convergence.
    """
    def obj(b: float) -> float:
        p = 1.0 / (1.0 + np.exp(-np.clip(b + z_uncentered, -25.0, 25.0)))
        return float(np.mean(p) - target_rate)

    b_min, b_max = -20.0, 20.0
    if obj(b_min) > 0:
        b_min = -50.0
    if obj(b_max) < 0:
        b_max = 50.0

    beta_0 = float(brentq(obj, b_min, b_max, xtol=1e-6))
    return beta_0


def generate_indian_student_cohort(
    n_students: int = 2000,
    seed: int = 42,
    output_path: Optional[Path] = DEFAULT_OUTPUT_PATH,
    assumptions_path: Optional[Path] = None,
    target_base_rate: Optional[float] = None,
) -> pd.DataFrame:
    """
    Generates a realistic cohort of Indian collegiate students using empirical parameters
    grounded in UCI ID 697, OULAD UCI ID 349, and Indian regulatory mandates.
    `target_base_rate` overrides cohort_metadata.target_base_rate (sensitivity analysis only).
    """
    logger.info("Generating Indian college student cohort (N=%d, seed=%d)...", n_students, seed)
    rng = np.random.default_rng(seed)

    assumptions = load_simulation_assumptions(assumptions_path)

    # Base rate: from assumptions.yaml (a missing key raises), unless explicitly overridden
    if target_base_rate is None:
        target_base_rate = float(get_param(assumptions, "cohort_metadata", "target_base_rate"))

    # 1. Demographics & Socioeconomic Parameters
    id_prefix = str(get_param(assumptions, "demographics_distribution", "student_id_prefix"))
    student_ids = [f"{id_prefix}{i:04d}" for i in range(1, n_students + 1)]

    gender_props = get_param(assumptions, "demographics_distribution", "gender_proportions")
    gender = rng.choice(list(gender_props.keys()), size=n_students, p=list(gender_props.values()))

    category_props = get_param(assumptions, "demographics_distribution", "category_proportions")
    category = rng.choice(list(category_props.keys()), size=n_students, p=list(category_props.values()))

    age_mean = float(get_param(assumptions, "demographics_distribution", "age_mean"))
    age_std = float(get_param(assumptions, "demographics_distribution", "age_std"))
    age_min = float(get_param(assumptions, "demographics_distribution", "age_min"))
    age_max = float(get_param(assumptions, "demographics_distribution", "age_max"))
    age = np.clip(np.round(rng.normal(age_mean, age_std, size=n_students), 1), age_min, age_max)

    hostel_props = get_param(assumptions, "demographics_distribution", "hostel_status_proportions")
    hostel_status = rng.choice(list(hostel_props.keys()), size=n_students, p=list(hostel_props.values()))
    is_hosteler = (hostel_status == "Hosteler").astype(int)

    commute_hosteler_km = float(get_param(assumptions, "demographics_distribution", "commute_hosteler_km"))
    commute_shape = float(get_param(assumptions, "demographics_distribution", "commute_day_scholar_shape"))
    commute_scale = float(get_param(assumptions, "demographics_distribution", "commute_day_scholar_scale"))
    commute_min = float(get_param(assumptions, "demographics_distribution", "commute_day_scholar_min_km"))
    commute_max = float(get_param(assumptions, "demographics_distribution", "commute_day_scholar_max_km"))
    commute_shift = float(get_param(assumptions, "demographics_distribution", "commute_day_scholar_shift_km"))
    day_scholar_commute = np.round(np.clip(rng.gamma(shape=commute_shape, scale=commute_scale, size=n_students) + commute_shift, commute_min, commute_max), 1)
    commute_distance_km = np.where(is_hosteler == 1, commute_hosteler_km, day_scholar_commute)

    income_labels = list(get_param(assumptions, "socioeconomic_distribution", "income_slab_labels"))
    income_props = list(get_param(assumptions, "socioeconomic_distribution", "income_slab_proportions"))
    income_slab_idx = rng.choice(len(income_labels), size=n_students, p=income_props)
    family_income_slab = np.array([income_labels[i] for i in income_slab_idx])

    first_gen_probs = list(get_param(assumptions, "socioeconomic_distribution", "first_gen_probs_by_slab"))
    p_first_gen = np.array([first_gen_probs[i] for i in income_slab_idx])
    is_first_generation = rng.binomial(1, p_first_gen, size=n_students)

    scholarship_quota_prob = float(get_param(assumptions, "socioeconomic_distribution", "scholarship_quota_prob"))
    scholarship_mid_prob = float(get_param(assumptions, "socioeconomic_distribution", "scholarship_mid_prob"))
    scholarship_high_prob = float(get_param(assumptions, "socioeconomic_distribution", "scholarship_high_prob"))
    is_quota_eligible = np.isin(category, ["SC", "ST", "EWS"]) | (income_slab_idx == 0)
    scholarship_prob = np.where(
        is_quota_eligible,
        scholarship_quota_prob,
        np.where(income_slab_idx == 1, scholarship_mid_prob, scholarship_high_prob),
    )
    has_scholarship = rng.binomial(1, scholarship_prob, size=n_students)

    fee_schol_choices = list(get_param(assumptions, "socioeconomic_distribution", "fee_delay_with_scholarship_choices"))
    fee_schol_probs = list(get_param(assumptions, "socioeconomic_distribution", "fee_delay_with_scholarship_probs"))
    fee_s0_choices = list(get_param(assumptions, "socioeconomic_distribution", "fee_delay_slab0_choices"))
    fee_s0_probs = list(get_param(assumptions, "socioeconomic_distribution", "fee_delay_slab0_probs"))
    fee_s1_choices = list(get_param(assumptions, "socioeconomic_distribution", "fee_delay_slab1_choices"))
    fee_s1_probs = list(get_param(assumptions, "socioeconomic_distribution", "fee_delay_slab1_probs"))
    fee_high_choices = list(get_param(assumptions, "socioeconomic_distribution", "fee_delay_high_slab_choices"))
    fee_high_probs = list(get_param(assumptions, "socioeconomic_distribution", "fee_delay_high_slab_probs"))
    fee_jitter_max = int(get_param(assumptions, "socioeconomic_distribution", "fee_delay_jitter_max"))

    draw_schol = rng.choice(fee_schol_choices, size=n_students, p=fee_schol_probs)
    draw_s0 = rng.choice(fee_s0_choices, size=n_students, p=fee_s0_probs)
    draw_s1 = rng.choice(fee_s1_choices, size=n_students, p=fee_s1_probs)
    draw_high = rng.choice(fee_high_choices, size=n_students, p=fee_high_probs)

    fee_delay_base = np.where(
        has_scholarship == 1,
        draw_schol,
        np.where(income_slab_idx == 0, draw_s0, np.where(income_slab_idx == 1, draw_s1, draw_high)),
    )
    fee_payment_delay_days = fee_delay_base + rng.integers(0, fee_jitter_max, size=n_students)

    # 2. Attendance Dynamics
    latent_alpha = float(get_param(assumptions, "attendance_distribution", "latent_alpha"))
    latent_beta = float(get_param(assumptions, "attendance_distribution", "latent_beta"))
    latent_att_factor = rng.beta(latent_alpha, latent_beta, size=n_students)

    commute_long_thresh = float(get_param(assumptions, "attendance_distribution", "commute_long_threshold_km"))
    commute_long_pen = float(get_param(assumptions, "attendance_distribution", "commute_long_penalty"))
    commute_med_thresh = float(get_param(assumptions, "attendance_distribution", "commute_medium_threshold_km"))
    commute_med_pen = float(get_param(assumptions, "attendance_distribution", "commute_medium_penalty"))

    commute_penalty = np.where(commute_distance_km > commute_long_thresh, commute_long_pen, np.where(commute_distance_km > commute_med_thresh, commute_med_pen, 0.0))
    latent_att_min_clip = float(get_param(assumptions, "attendance_distribution", "latent_attendance_min_clip"))
    latent_att_max_clip = float(get_param(assumptions, "attendance_distribution", "latent_attendance_max_clip"))
    adj_att_factor = np.clip(latent_att_factor - commute_penalty, latent_att_min_clip, latent_att_max_clip)

    att_min = float(get_param(assumptions, "attendance_distribution", "attendance_min"))
    att_max = float(get_param(assumptions, "attendance_distribution", "attendance_max"))
    core1_std = float(get_param(assumptions, "attendance_distribution", "core1_noise_std"))
    core2_std = float(get_param(assumptions, "attendance_distribution", "core2_noise_std"))
    lab_boost = float(get_param(assumptions, "attendance_distribution", "lab_boost"))
    lab_std = float(get_param(assumptions, "attendance_distribution", "lab_noise_std"))
    elec_pen = float(get_param(assumptions, "attendance_distribution", "elective_penalty"))
    elec_std = float(get_param(assumptions, "attendance_distribution", "elective_noise_std"))

    att_core1 = np.clip(rng.normal(adj_att_factor * 100.0, core1_std), att_min, att_max)
    att_core2 = np.clip(rng.normal(adj_att_factor * 100.0, core2_std), att_min, att_max)
    lab_att_min = float(get_param(assumptions, "attendance_distribution", "lab_attendance_min"))
    att_lab = np.clip(rng.normal(adj_att_factor * 100.0 + lab_boost, lab_std), lab_att_min, att_max)
    att_elective = np.clip(rng.normal(adj_att_factor * 100.0 - elec_pen, elec_std), att_min, att_max)

    slope_noise_std = float(get_param(assumptions, "attendance_distribution", "attendance_slope_noise_std"))
    att_slope_latent = rng.normal(0.0, slope_noise_std, size=n_students)
    monthly_std = float(get_param(assumptions, "attendance_distribution", "monthly_noise_std"))
    attendance_month_1 = np.clip(rng.normal(adj_att_factor * 100.0 - att_slope_latent, monthly_std), att_min, att_max)
    attendance_month_2 = np.clip(rng.normal(adj_att_factor * 100.0, monthly_std), att_min, att_max)
    attendance_month_3 = np.clip(rng.normal(adj_att_factor * 100.0 + att_slope_latent, monthly_std), att_min, att_max)

    weights = list(get_param(assumptions, "attendance_distribution", "month_weights"))
    attendance_percentage = np.round(weights[0] * attendance_month_1 + weights[1] * attendance_month_2 + weights[2] * attendance_month_3, 1)
    attendance_3m_trend = np.round((attendance_month_3 - attendance_month_1) / 2.0, 2)

    abs_ranges = list(get_param(assumptions, "attendance_distribution", "consecutive_absence_ranges"))
    abs_thresh = list(get_param(assumptions, "attendance_distribution", "consecutive_absence_thresholds"))
    draw_abs_0 = rng.integers(abs_ranges[0][0], abs_ranges[0][1], size=n_students)
    draw_abs_1 = rng.integers(abs_ranges[1][0], abs_ranges[1][1], size=n_students)
    draw_abs_2 = rng.integers(abs_ranges[2][0], abs_ranges[2][1], size=n_students)
    draw_abs_3 = rng.integers(abs_ranges[3][0], abs_ranges[3][1], size=n_students)

    consecutive_absences = np.where(
        attendance_percentage < abs_thresh[0],
        draw_abs_0,
        np.where(
            attendance_percentage < abs_thresh[1],
            draw_abs_1,
            np.where(attendance_percentage < abs_thresh[2], draw_abs_2, draw_abs_3),
        ),
    )

    mandatory_att_thresh = float(get_param(assumptions, "regulations_and_thresholds", "mandatory_attendance_threshold"))
    attendance_risk_flag = (attendance_percentage < mandatory_att_thresh).astype(int)

    # 3. Academic Performance
    prior_cgpa_mean = float(get_param(assumptions, "academic_distribution", "prior_cgpa_mean"))
    prior_cgpa_std = float(get_param(assumptions, "academic_distribution", "prior_cgpa_std"))
    prev_cgpa_min = float(get_param(assumptions, "academic_distribution", "prev_cgpa_min"))
    prev_cgpa_max = float(get_param(assumptions, "academic_distribution", "prev_cgpa_max"))
    prev_sem_cgpa = np.clip(np.round(rng.normal(prior_cgpa_mean, prior_cgpa_std, size=n_students), 2), prev_cgpa_min, prev_cgpa_max)

    cgpa_noise_std = float(get_param(assumptions, "academic_distribution", "cgpa_noise_std"))
    cgpa_noise = rng.normal(0.0, cgpa_noise_std, size=n_students)
    att_cgpa_slope = float(get_param(assumptions, "academic_distribution", "attendance_cgpa_slope"))
    att_effect_on_cgpa = (attendance_percentage - mandatory_att_thresh) * att_cgpa_slope

    cgpa_min = float(get_param(assumptions, "academic_distribution", "cgpa_min"))
    cgpa_max = float(get_param(assumptions, "academic_distribution", "cgpa_max"))
    current_cgpa = np.clip(np.round(prev_sem_cgpa + att_effect_on_cgpa + cgpa_noise, 2), cgpa_min, cgpa_max)
    cgpa_delta = np.round(current_cgpa - prev_sem_cgpa, 2)

    backlog_base_lam = float(get_param(assumptions, "academic_distribution", "backlog_base_lambda"))
    backlog_cgpa_decay = float(get_param(assumptions, "academic_distribution", "backlog_cgpa_decay"))
    backlog_att_boost = float(get_param(assumptions, "academic_distribution", "backlog_attendance_boost"))
    backlog_max = int(get_param(assumptions, "academic_distribution", "backlog_max"))
    backlog_lambda_floor = float(get_param(assumptions, "academic_distribution", "backlog_lambda_floor"))
    backlog_att_thresh = float(get_param(assumptions, "academic_distribution", "backlog_attendance_threshold"))
    backlog_lambda = np.maximum(
        backlog_lambda_floor,
        backlog_base_lam * np.exp(-backlog_cgpa_decay * current_cgpa) + backlog_att_boost * (attendance_percentage < backlog_att_thresh),
    )
    backlog_count = np.clip(rng.poisson(lam=backlog_lambda, size=n_students), 0, backlog_max)

    internal_mult = float(get_param(assumptions, "academic_distribution", "internal_exam_cgpa_multiplier"))
    internal_std = float(get_param(assumptions, "academic_distribution", "internal_exam_noise_std"))
    internal_min = float(get_param(assumptions, "academic_distribution", "internal_exam_min"))
    internal_max = float(get_param(assumptions, "academic_distribution", "internal_exam_max"))
    internal_exam_score_pct = np.clip(np.round(current_cgpa * internal_mult + rng.normal(0, internal_std, size=n_students), 1), internal_min, internal_max)

    stem_exam_att_w = float(get_param(assumptions, "academic_distribution", "stem_exam_att_weight"))
    stem_exam_cgpa_w = float(get_param(assumptions, "academic_distribution", "stem_exam_cgpa_weight"))
    stem_exam_noise = float(get_param(assumptions, "academic_distribution", "stem_exam_noise_std"))
    stem_pass_mark = float(get_param(assumptions, "regulations_and_thresholds", "stem_core_pass_mark"))
    core1_exam_min = float(get_param(assumptions, "academic_distribution", "core1_exam_score_min"))
    core1_exam_max = float(get_param(assumptions, "academic_distribution", "core1_exam_score_max"))
    core1_exam_score = np.clip(att_core1 * stem_exam_att_w + current_cgpa * stem_exam_cgpa_w + rng.normal(0, stem_exam_noise, size=n_students), core1_exam_min, core1_exam_max)
    stem_core_fail_flag = (core1_exam_score < stem_pass_mark).astype(int)

    # 4. Learning Behavior Dynamics
    w_eng_cgpa = float(get_param(assumptions, "learning_behavior_distribution", "engagement_cgpa_weight"))
    w_eng_att = float(get_param(assumptions, "learning_behavior_distribution", "engagement_attendance_weight"))
    base_engagement = (current_cgpa / cgpa_max) * w_eng_cgpa + (attendance_percentage / att_max) * w_eng_att

    lms_mult = float(get_param(assumptions, "learning_behavior_distribution", "lms_mean_multiplier"))
    lms_std = float(get_param(assumptions, "learning_behavior_distribution", "lms_noise_std"))
    lms_cap = float(get_param(assumptions, "learning_behavior_distribution", "lms_max_weekly"))
    lms_logins_per_week = np.clip(np.round(rng.normal(base_engagement * lms_mult, lms_std, size=n_students), 1), 0.0, lms_cap)

    sub_scale = float(get_param(assumptions, "learning_behavior_distribution", "submission_lag_base_scale"))
    sub_offset = float(get_param(assumptions, "learning_behavior_distribution", "submission_lag_offset"))
    sub_std = float(get_param(assumptions, "learning_behavior_distribution", "submission_lag_noise_std"))
    assignment_submission_lag_days = np.round(rng.normal((1.0 - base_engagement) * sub_scale - sub_offset, sub_std, size=n_students), 1)

    res_mu = float(get_param(assumptions, "learning_behavior_distribution", "resource_lognormal_mean"))
    res_slope = float(get_param(assumptions, "learning_behavior_distribution", "resource_lognormal_slope"))
    res_sigma = float(get_param(assumptions, "learning_behavior_distribution", "resource_lognormal_sigma"))
    res_min = int(get_param(assumptions, "learning_behavior_distribution", "resource_access_min"))
    res_max = int(get_param(assumptions, "learning_behavior_distribution", "resource_access_max"))
    resource_access_count = np.clip(np.round(rng.lognormal(mean=res_mu + res_slope * base_engagement, sigma=res_sigma, size=n_students)), res_min, res_max).astype(int)

    inact_scale = float(get_param(assumptions, "learning_behavior_distribution", "inactivity_scale"))
    inact_exp = float(get_param(assumptions, "learning_behavior_distribution", "inactivity_exp_scale"))
    inact_max = int(get_param(assumptions, "learning_behavior_distribution", "inactivity_max_days"))
    days_since_last_lms_activity = np.clip(
        np.round((1.0 - base_engagement) * inact_scale + rng.exponential(scale=inact_exp, size=n_students)),
        0, inact_max
    ).astype(int)

    forum_lam = float(get_param(assumptions, "learning_behavior_distribution", "forum_base_lambda"))
    forum_max = int(get_param(assumptions, "learning_behavior_distribution", "forum_participation_max"))
    forum_floor = float(get_param(assumptions, "learning_behavior_distribution", "forum_lambda_floor"))
    forum_participation_count = np.clip(rng.poisson(lam=np.maximum(forum_floor, base_engagement * forum_lam), size=n_students), 0, forum_max)

    # 5. Risk Formula and Calibrated Intercept Beta_0
    # Retrieve risk coefficients from assumptions.yaml:
    b_cgpa = float(get_param(assumptions, "risk_coefficients", "current_cgpa"))
    b_backlog = float(get_param(assumptions, "risk_coefficients", "backlog_count"))
    b_scholarship = float(get_param(assumptions, "risk_coefficients", "has_scholarship"))
    b_fee_delay = float(get_param(assumptions, "risk_coefficients", "fee_payment_delay_days"))
    b_first_gen = float(get_param(assumptions, "risk_coefficients", "is_first_generation"))
    b_hosteler = float(get_param(assumptions, "risk_coefficients", "is_hosteler"))
    b_lms_login = float(get_param(assumptions, "risk_coefficients", "lms_logins_per_week"))
    b_lms_inactive = float(get_param(assumptions, "risk_coefficients", "days_since_last_lms_activity"))
    b_sub_lag = float(get_param(assumptions, "risk_coefficients", "assignment_submission_lag_days"))
    b_att_deficit = float(get_param(assumptions, "risk_coefficients", "attendance_deficit_slope"))
    b_att_trend = float(get_param(assumptions, "risk_coefficients", "attendance_3m_trend"))
    b_absences = float(get_param(assumptions, "risk_coefficients", "consecutive_absences"))
    b_cgpa_delta = float(get_param(assumptions, "risk_coefficients", "cgpa_delta"))
    b_stem_fail = float(get_param(assumptions, "risk_coefficients", "stem_core_fail_flag"))

    # Protected attributes strictly from assumptions.yaml (defaults 0.0)
    b_gender_male = float(get_param(assumptions, "risk_coefficients", "gender_male"))
    b_cat_obc = float(get_param(assumptions, "risk_coefficients", "category_obc"))
    b_cat_sc = float(get_param(assumptions, "risk_coefficients", "category_sc"))
    b_cat_st = float(get_param(assumptions, "risk_coefficients", "category_st"))
    b_cat_ews = float(get_param(assumptions, "risk_coefficients", "category_ews"))

    # Compound non-linear interaction terms
    dual_crisis_att_th = float(get_param(assumptions, "regulations_and_thresholds", "dual_crisis_att_threshold"))
    dual_crisis_fee_th = float(get_param(assumptions, "regulations_and_thresholds", "dual_crisis_fee_threshold"))
    acad_crisis_cgpa_th = float(get_param(assumptions, "regulations_and_thresholds", "academic_crisis_cgpa_threshold"))
    acad_crisis_backlog_th = float(get_param(assumptions, "regulations_and_thresholds", "academic_crisis_backlog_threshold"))

    b_inter_att_fee = float(get_param(assumptions, "risk_coefficients", "interaction_low_att_high_fee"))
    b_inter_cgpa_backlog = float(get_param(assumptions, "risk_coefficients", "interaction_low_cgpa_high_backlogs"))
    noise_std = float(get_param(assumptions, "risk_coefficients", "idiosyncratic_noise_std"))

    # Compute uncentered log-odds z_uncentered
    z_uncentered = (
        b_cgpa * current_cgpa
        + b_backlog * backlog_count
        + b_scholarship * has_scholarship
        + b_fee_delay * fee_payment_delay_days
        + b_first_gen * is_first_generation
        + b_hosteler * is_hosteler
        + b_lms_login * lms_logins_per_week
        + b_lms_inactive * days_since_last_lms_activity
        + b_sub_lag * assignment_submission_lag_days
        + b_att_deficit * np.maximum(0.0, mandatory_att_thresh - attendance_percentage)
        + b_att_trend * attendance_3m_trend
        + b_absences * consecutive_absences
        + b_cgpa_delta * cgpa_delta
        + b_stem_fail * stem_core_fail_flag
        + b_gender_male * (gender == "Male").astype(float)
        + b_cat_obc * (category == "OBC").astype(float)
        + b_cat_sc * (category == "SC").astype(float)
        + b_cat_st * (category == "ST").astype(float)
        + b_cat_ews * (category == "EWS").astype(float)
        + b_inter_att_fee * ((attendance_percentage < dual_crisis_att_th) & (fee_payment_delay_days > dual_crisis_fee_th)).astype(float)
        + b_inter_cgpa_backlog * ((current_cgpa < acad_crisis_cgpa_th) & (backlog_count >= acad_crisis_backlog_th)).astype(float)
        + rng.normal(0, noise_std, size=n_students)
    )

    # Solve baseline intercept beta_0 numerically so E[sigmoid(beta_0 + z)] == target_base_rate
    beta_0 = solve_intercept_for_base_rate(z_uncentered, target_rate=target_base_rate)
    z = beta_0 + z_uncentered

    # Calibrated Ground-Truth Probability via Logistic Sigmoid
    dropout_probability = 1.0 / (1.0 + np.exp(-np.clip(z, -25.0, 25.0)))

    # Bernoulli Trial Sampling for realistic individual variation:
    # is_dropout ~ Bernoulli(p_i)
    is_dropout = (rng.uniform(0.0, 1.0, size=n_students) < dropout_probability).astype(int)

    df = pd.DataFrame({
        "student_id": student_ids,
        "gender": gender,
        "category": category,
        "age": age,
        "hostel_status": hostel_status,
        "commute_distance_km": commute_distance_km,
        "family_income_slab": family_income_slab,
        "income_slab_idx": income_slab_idx,
        "is_first_generation": is_first_generation,
        "has_scholarship": has_scholarship,
        "fee_payment_delay_days": fee_payment_delay_days,
        # Pillar 1: Attendance
        "att_core1": np.round(att_core1, 1),
        "att_core2": np.round(att_core2, 1),
        "att_lab": np.round(att_lab, 1),
        "att_elective": np.round(att_elective, 1),
        "attendance_month_1": np.round(attendance_month_1, 1),
        "attendance_month_2": np.round(attendance_month_2, 1),
        "attendance_month_3": np.round(attendance_month_3, 1),
        "attendance_percentage": attendance_percentage,
        "attendance_3m_trend": attendance_3m_trend,
        "consecutive_absences": consecutive_absences,
        "attendance_risk_flag": attendance_risk_flag,
        # Pillar 2: Academic
        "prev_sem_cgpa": prev_sem_cgpa,
        "current_cgpa": current_cgpa,
        "cgpa_delta": cgpa_delta,
        "backlog_count": backlog_count,
        "internal_exam_score_pct": internal_exam_score_pct,
        "stem_core_fail_flag": stem_core_fail_flag,
        # Pillar 3: Learning Behavior
        "lms_logins_per_week": lms_logins_per_week,
        "assignment_submission_lag_days": assignment_submission_lag_days,
        "resource_access_count": resource_access_count,
        "days_since_last_lms_activity": days_since_last_lms_activity,
        "forum_participation_count": forum_participation_count,
        # Targets
        "ground_truth_risk_prob": np.round(dropout_probability, 4),
        "is_dropout": is_dropout,
    })

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(
            "Saved synthetic Indian cohort dataset to %s (Rows=%d, Cols=%d, Mean Dropout Rate=%.2f%%, Calibrated beta_0=%.4f)",
            output_path, len(df), len(df.columns), df["is_dropout"].mean() * 100, beta_0
        )

    return df


if __name__ == "__main__":
    df = generate_indian_student_cohort(n_students=2000, seed=42)
    print("Generated Indian Dataset Summary:")
    print("Shape:", df.shape)
    print("Dropout Class Breakdown:\n", df["is_dropout"].value_counts(normalize=True))
    print("Correlation with is_dropout:")
    corr = df.select_dtypes(include=[np.number]).corr()["is_dropout"].sort_values()
    print(corr)
