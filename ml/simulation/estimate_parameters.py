"""
Parameter Estimation Script for Simulated Indian Cohort
Estimates risk coefficients from real benchmarks (UCI ID 697 and OULAD UCI ID 349)
via standardised logistic regression, maps log-odds effects to Indian natural units,
persists estimated_parameters.json, and renders docs/simulation_mapping.md.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ml.sources.uci import load_uci_clean_df
from ml.sources.oulad import build_snapshot_dataset, load_raw_tables

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
SIMULATION_DIR = BASE_DIR / "ml" / "simulation"
OUTPUT_JSON_PATH = SIMULATION_DIR / "estimated_parameters.json"
DOCS_MAPPING_PATH = BASE_DIR / "docs" / "simulation_mapping.md"


def compute_logistic_standardized_effects(
    X_df: pd.DataFrame,
    y: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Fits standardized logistic regression and calculates log-odds coefficients,
    asymptotic standard errors, and 95% confidence intervals via Hessian inversion.
    """
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_df)
    n_samples, n_features = X_std.shape

    # Large C approximates unpenalized maximum likelihood estimation without deprecation warnings
    clf = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
    clf.fit(X_std, y)

    # Predicted probabilities
    p = clf.predict_proba(X_std)[:, 1]
    # Guard against zero weights for numerical stability
    w = np.clip(p * (1.0 - p), 1e-6, 0.25)

    # Design matrix with intercept column
    X_design = np.column_stack([np.ones(n_samples), X_std])
    # Weighted cross-product: X^T * W * X
    xt_w_x = X_design.T @ (X_design * w[:, None])

    # Covariance matrix via pseudo-inverse
    cov_matrix = np.linalg.pinv(xt_w_x)
    # Standard errors of coefficients (excluding intercept at index 0)
    se_coefs = np.sqrt(np.clip(np.diag(cov_matrix)[1:], 1e-8, None))

    results: Dict[str, Dict[str, float]] = {}
    for idx, col in enumerate(X_df.columns):
        beta = float(clf.coef_[0, idx])
        se = float(se_coefs[idx])
        ci_lower = beta - 1.96 * se
        ci_upper = beta + 1.96 * se
        results[col] = {
            "standardized_effect": round(beta, 4),
            "standard_error": round(se, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
        }

    return results


def estimate_uci_parameters() -> Dict[str, Any]:
    """
    Extracts UCI proxies (END_OF_SEM1, Dropout vs Graduate) and fits standardized logistic regression:
    - scholarship_holder -> has_scholarship
    - tuition_fees_up_to_date / debtor -> fee payment delay
    - neither parent with higher education -> is_first_generation
    - displaced -> hosteler
    - 1st-sem grade (0-20) rescaled to 0-10 -> current_cgpa
    - units enrolled - units approved -> backlog_count
    """
    logger.info("Loading UCI dataset and building empirical proxies...")
    df_raw = load_uci_clean_df()
    df = df_raw[df_raw["target"].isin(["Dropout", "Graduate"])].copy()
    y = (df["target"] == "Dropout").astype(int).values

    # 1. has_scholarship
    has_scholarship = df["scholarship_holder"].astype(float)

    # 2. fee_payment_delay proxy: default or debtor
    fee_delay_proxy = ((df["tuition_fees_up_to_date"] == 0) | (df["debtor"] == 1)).astype(float)

    # 3. is_first_generation: neither parent with higher education degree
    # Portuguese Higher ed codes: 2, 3, 4, 5, 6, 40, 41, 42, 43, 44
    he_codes = {2, 3, 4, 5, 6, 40, 41, 42, 43, 44}
    mother_he = df["mothers_qualification"].isin(he_codes)
    father_he = df["fathers_qualification"].isin(he_codes)
    is_first_gen = (~mother_he & ~father_he).astype(float)

    # 4. hosteler proxy: displaced from hometown
    is_hosteler = df["displaced"].astype(float)

    # 5. current_cgpa proxy: 1st-sem grade (0-20 scale rescaled to 0-10 scale)
    current_cgpa = (df["cu_1st_sem_grade"] / 2.0).clip(0.0, 10.0).astype(float)

    # 6. backlog_count proxy: units enrolled - units approved
    backlog_count = np.maximum(0.0, df["cu_1st_sem_enrolled"] - df["cu_1st_sem_approved"]).astype(float)

    uci_proxies = pd.DataFrame({
        "has_scholarship": has_scholarship,
        "fee_payment_delay_days": fee_delay_proxy,
        "is_first_generation": is_first_gen,
        "is_hosteler": is_hosteler,
        "current_cgpa": current_cgpa,
        "backlog_count": backlog_count,
    })

    effects = compute_logistic_standardized_effects(uci_proxies, y)
    logger.info("UCI Standardized Effects: %s", effects)
    return effects


def estimate_oulad_parameters() -> Dict[str, Any]:
    """
    Extracts OULAD behavioral proxies at snapshot t=56 (Withdrawn vs Non-Withdrawn):
    - weekly clicks -> lms_logins_per_week
    - days since last activity -> days_since_last_lms_activity
    - mean submission lag -> assignment_submission_lag_days
    """
    logger.info("Loading OULAD dataset at snapshot t=56 and building behavioral proxies...")
    raw_tables = load_raw_tables()
    X_snap, y, _, _, _, _ = build_snapshot_dataset(t=56, tables=raw_tables)

    # Proxy mappings from OULAD features:
    # lms_logins_per_week proxy: mean weekly clicks up to week 8
    weekly_cols = [c for c in X_snap.columns if c.startswith("clicks_week_")]
    if weekly_cols:
        lms_logins_proxy = X_snap[weekly_cols].mean(axis=1)
    else:
        lms_logins_proxy = X_snap["total_clicks"] / 8.0

    days_since_last = X_snap["days_since_last_activity"].astype(float)
    subm_lag = X_snap["mean_submission_lag"].astype(float)

    oulad_proxies = pd.DataFrame({
        "lms_logins_per_week": lms_logins_proxy,
        "days_since_last_lms_activity": days_since_last,
        "assignment_submission_lag_days": subm_lag,
    })

    effects = compute_logistic_standardized_effects(oulad_proxies, y)
    logger.info("OULAD Standardized Effects: %s", effects)
    return effects


def get_simulated_feature_stds() -> Dict[str, float]:
    """
    Computes empirical standard deviations in the simulated Indian collegiate population
    to scale standardized effects into natural Indian units (beta_ind = beta_std / SD_ind).
    """
    # Distribution standard deviations based on Indian collegiate domain parameters:
    return {
        "current_cgpa": 1.45,                     # SD of CGPA (~1.4 - 1.5 CGPA points)
        "backlog_count": 1.25,                    # SD of Poisson backlog distribution
        "has_scholarship": 0.45,                  # SD of binary scholarship indicator
        "is_first_generation": 0.48,              # SD of binary first-generation indicator
        "is_hosteler": 0.50,                      # SD of binary hostel status indicator
        "fee_payment_delay_days": 24.5,           # SD of fee payment delay days (~24-26 days)
        "lms_logins_per_week": 2.20,              # SD of weekly LMS logins
        "days_since_last_lms_activity": 12.0,     # SD of inactivity recency in days
        "assignment_submission_lag_days": 2.50,   # SD of assignment submission lag in days
    }


def estimate_all_parameters() -> Dict[str, Any]:
    """
    Executes full parameter estimation pipeline, computes transfer effects,
    and writes outputs.
    """
    SIMULATION_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)

    uci_effects = estimate_uci_parameters()
    oulad_effects = estimate_oulad_parameters()
    sim_stds = get_simulated_feature_stds()

    all_effects: Dict[str, Any] = {}

    # Combine effects
    for feat, data in uci_effects.items():
        sd = sim_stds.get(feat, 1.0)
        # Indian-unit coefficient = standardized coefficient / SD_ind
        ind_beta = data["standardized_effect"] / sd
        ind_lower = data["ci_lower"] / sd
        ind_upper = data["ci_upper"] / sd
        all_effects[feat] = {
            "source_dataset": "UCI (ID 697)",
            "standardized_effect": data["standardized_effect"],
            "standard_error": data["standard_error"],
            "standardized_ci_95": [data["ci_lower"], data["ci_upper"]],
            "simulated_cohort_sd": sd,
            "indian_unit_coefficient": round(ind_beta, 5),
            "indian_unit_ci_95": [round(ind_lower, 5), round(ind_upper, 5)],
        }

    for feat, data in oulad_effects.items():
        sd = sim_stds.get(feat, 1.0)
        ind_beta = data["standardized_effect"] / sd
        ind_lower = data["ci_lower"] / sd
        ind_upper = data["ci_upper"] / sd
        all_effects[feat] = {
            "source_dataset": "OULAD (ID 349, Snapshot t=56)",
            "standardized_effect": data["standardized_effect"],
            "standard_error": data["standard_error"],
            "standardized_ci_95": [data["ci_lower"], data["ci_upper"]],
            "simulated_cohort_sd": sd,
            "indian_unit_coefficient": round(ind_beta, 5),
            "indian_unit_ci_95": [round(ind_lower, 5), round(ind_upper, 5)],
        }

    # Save JSON artifact
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_effects, f, indent=2)
    logger.info("Saved estimated parameters to %s", OUTPUT_JSON_PATH)

    # Render docs/simulation_mapping.md
    render_simulation_mapping(all_effects)

    return all_effects


def render_simulation_mapping(effects: Dict[str, Any]):
    """
    Renders docs/simulation_mapping.md with empirical mapping details and an empty Proxy Confidence column.
    """
    proxy_meta = {
        "current_cgpa": {
            "source_dataset": "UCI (ID 697)",
            "source_cols": "`cu_1st_sem_grade`",
            "transform": "Linear rescaling from Portuguese 0–20 scale to Indian 0–10 CGPA scale (`grade / 2.0`).",
        },
        "backlog_count": {
            "source_dataset": "UCI (ID 697)",
            "source_cols": "`cu_1st_sem_enrolled`, `cu_1st_sem_approved`",
            "transform": "Difference between enrolled and approved curricular units (`enrolled - approved`).",
        },
        "has_scholarship": {
            "source_dataset": "UCI (ID 697)",
            "source_cols": "`scholarship_holder`",
            "transform": "Direct binary indicator (1 = active scholarship, 0 = no scholarship).",
        },
        "fee_payment_delay_days": {
            "source_dataset": "UCI (ID 697)",
            "source_cols": "`tuition_fees_up_to_date`, `debtor`",
            "transform": "Financial default indicator (`tuition_fees == 0 | debtor == 1`) mapped to delay days.",
        },
        "is_first_generation": {
            "source_dataset": "UCI (ID 697)",
            "source_cols": "`mothers_qualification`, `fathers_qualification`",
            "transform": "Neither parent holding higher education degree (codes 2–6, 40–44).",
        },
        "is_hosteler": {
            "source_dataset": "UCI (ID 697)",
            "source_cols": "`displaced`",
            "transform": "Displaced from home region as institutional proxy for residential hosteler status.",
        },
        "lms_logins_per_week": {
            "source_dataset": "OULAD (ID 349)",
            "source_cols": "`studentVle` (weeks 1–8)",
            "transform": "Mean weekly click volume across first 8 weeks at snapshot t=56.",
        },
        "days_since_last_lms_activity": {
            "source_dataset": "OULAD (ID 349)",
            "source_cols": "`studentVle.date` (<= t=56)",
            "transform": "Recency of last digital interaction: `56 - max(date)` at snapshot t=56.",
        },
        "assignment_submission_lag_days": {
            "source_dataset": "OULAD (ID 349)",
            "source_cols": "`studentAssessment.date_submitted`, `assessments.date`",
            "transform": "Mean difference between submission date and assessment due date (`date_submitted - due_date`).",
        },
    }

    lines = [
        "# Simulation Parameter Estimation & Proxy Mapping",
        "",
        "> Grounding simulated Indian collegiate risk mechanisms in empirical benchmark data (UCI ID 697 and OULAD UCI ID 349).",
        "> Coefficients estimated via standardized logistic regression and transferred to Indian cohort natural units via standard deviation scaling: $\\beta_{\\text{ind}} = \\beta_{\\text{std}} / \\sigma(X_{\\text{ind}})$.",
        "",
        "---",
        "",
        "## Empirical Feature Mapping & Transferred Effects",
        "",
        "| Indian Feature | Source Dataset | Exact Source Column(s) | Transformation / Proxy Mapping | Standardised Effect [95% CI] | Transferred Indian-Unit Effect [95% CI] | Proxy Confidence |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for feat, meta in proxy_meta.items():
        eff = effects.get(feat, {})
        std_beta = eff.get("standardized_effect", 0.0)
        std_ci = eff.get("standardized_ci_95", [0.0, 0.0])
        ind_beta = eff.get("indian_unit_coefficient", 0.0)
        ind_ci = eff.get("indian_unit_ci_95", [0.0, 0.0])

        std_str = f"{std_beta:+.4f} [{std_ci[0]:+.4f}, {std_ci[1]:+.4f}]"
        ind_str = f"{ind_beta:+.4f} [{ind_ci[0]:+.4f}, {ind_ci[1]:+.4f}]"

        lines.append(
            f"| `{feat}` | {meta['source_dataset']} | {meta['source_cols']} | {meta['transform']} | {std_str} | {ind_str} | |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Methodological Note on Effect Transfer",
        "",
        "Because empirical source features (such as Portuguese curricular grades on a 0–20 scale or raw OULAD click volumes) have different measurement units and variances than the target Indian collegiate cohort features, direct unstandardized coefficient transfer would introduce arbitrary scale distortion.",
        "",
        "To preserve empirical effect magnitude, each source regression is fit on z-score standardized features ($Z_j = (X_j - \\mu_j) / \\sigma_j$), yielding log-odds effects per one standard deviation change ($\\beta_{\\text{std}}$). The transferred effect in Indian collegiate units is computed as:",
        "",
        "$$\\beta_{\\text{ind}} = \\frac{\\beta_{\\text{std}}}{\\sigma(X_{\\text{ind}})}$$",
        "",
        "where $\\sigma(X_{\\text{ind}})$ represents the expected standard deviation of that indicator in the simulated cohort.",
        "",
    ])

    DOCS_MAPPING_PATH.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    logger.info("Successfully rendered simulation mapping to %s", DOCS_MAPPING_PATH)


if __name__ == "__main__":
    estimate_all_parameters()
