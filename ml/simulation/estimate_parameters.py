"""
Parameter Estimation Script for Simulated Indian Cohort
Estimates risk coefficients from real benchmarks (UCI ID 697 and OULAD UCI ID 349)
via standardised logistic regression, maps log-odds effects to Indian natural units,
persists estimated_parameters.json. docs/simulation_mapping.md is rendered from that JSON by
`--render-mapping` (run by `python -m ml.pipeline run-all` after every step has succeeded).
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ml.config import RANDOM_SEED
from ml.data_pipeline.generate_synthetic_indian import generate_indian_student_cohort
from ml.simulation.uci_proxies import HOLDOUT_FRACTION, SPLIT_SEED, build_uci_proxies, split_uci_estimation_holdout
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
    Fits standardized logistic regression on the UCI proxies (ml/simulation/uci_proxies.py) using the
    ESTIMATION rows of the shared split only. The holdout rows are reserved for sim_to_real.py, so the
    sim-to-real check never evaluates on data that informed the simulator's parameters.
    """
    logger.info("Loading UCI dataset and building empirical proxies...")
    X, y = build_uci_proxies()
    train_index, holdout_index = split_uci_estimation_holdout(X, y)
    logger.info("UCI estimation rows: %d (holdout rows %d excluded)", len(train_index), len(holdout_index))

    effects = compute_logistic_standardized_effects(X.loc[train_index], y.loc[train_index].values)
    for data in effects.values():
        data["n_estimation_rows"] = int(len(train_index))
        data["n_holdout_rows_excluded"] = int(len(holdout_index))
    logger.info("UCI Standardized Effects: %s", effects)
    return effects


def estimate_oulad_parameters() -> Dict[str, Any]:
    """
    Extracts OULAD behavioral proxies at snapshot t=56 (Withdrawn vs Non-Withdrawn) from the
    shared OULAD builder (ml.sources.oulad.build_snapshot_dataset, same X as the benchmark):
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


SIMULATED_SD_FEATURES: List[str] = [
    "current_cgpa",
    "backlog_count",
    "has_scholarship",
    "is_first_generation",
    "is_hosteler",
    "fee_payment_delay_days",
    "lms_logins_per_week",
    "days_since_last_lms_activity",
    "assignment_submission_lag_days",
]
SIMULATED_SD_N_STUDENTS = 2000


def get_simulated_feature_stds(
    n_students: int = SIMULATED_SD_N_STUDENTS,
    seed: int = RANDOM_SEED,
) -> Dict[str, float]:
    """
    Sample standard deviations (ddof=1) of the transferred features in a simulated Indian cohort
    generated in memory with a fixed seed. Used to scale standardized effects into natural Indian
    units (beta_ind = beta_std / SD_ind). Feature draws precede the risk coefficients in the
    generator, so these SDs do not depend on the estimated effects.
    """
    df = generate_indian_student_cohort(n_students=n_students, seed=seed, output_path=None)
    df["is_hosteler"] = (df["hostel_status"] == "Hosteler").astype(int)
    return {feat: float(df[feat].std(ddof=1)) for feat in SIMULATED_SD_FEATURES}


def estimate_all_parameters(allow_dirty: bool = False) -> Dict[str, Any]:
    """
    Executes full parameter estimation pipeline, computes transfer effects,
    and writes outputs. Refuses to run on a dirty tree unless allow_dirty
    (recorded in the top-level `provenance` of estimated_parameters.json).
    """
    from ml.provenance import require_clean_tree
    require_clean_tree(allow_dirty)

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
            "simulated_cohort_sd": round(sd, 4),
            "simulated_cohort_sd_source": {"n_students": SIMULATED_SD_N_STUDENTS, "seed": RANDOM_SEED},
            "indian_unit_coefficient": round(ind_beta, 5),
            "indian_unit_ci_95": [round(ind_lower, 5), round(ind_upper, 5)],
            "n_estimation_rows": data["n_estimation_rows"],
            "n_holdout_rows_excluded": data["n_holdout_rows_excluded"],
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
            "simulated_cohort_sd": round(sd, 4),
            "simulated_cohort_sd_source": {"n_students": SIMULATED_SD_N_STUDENTS, "seed": RANDOM_SEED},
            "indian_unit_coefficient": round(ind_beta, 5),
            "indian_unit_ci_95": [round(ind_lower, 5), round(ind_upper, 5)],
        }

    # Provenance: real-data inputs of the estimation (no coefficient key is named "provenance")
    from ml.provenance import build_provenance, merge_input_files, oulad_inputs, uci_inputs
    all_effects["provenance"] = build_provenance(merge_input_files(uci_inputs(), oulad_inputs()), allow_dirty=allow_dirty)

    # Save JSON artifact
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_effects, f, indent=2)
    logger.info("Saved estimated parameters to %s", OUTPUT_JSON_PATH)

    return all_effects


def render_simulation_mapping_from_json(json_path: Path = OUTPUT_JSON_PATH) -> None:
    """Renders docs/simulation_mapping.md from estimated_parameters.json; refuses without provenance."""
    from ml.provenance import assert_consistent_provenance, load_labelled_json

    artifacts = load_labelled_json([json_path])
    assert_consistent_provenance(artifacts)
    render_simulation_mapping(artifacts[Path(json_path).name])


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
            "transform": "**Binary** financial default indicator: 1 if `tuition_fees_up_to_date == 0` or `debtor == 1`, else 0. UCI records no delay duration (see note below).",
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

    uci_effects = [e for e in effects.values() if str(e.get("source_dataset", "")).startswith("UCI")]
    n_est = uci_effects[0]["n_estimation_rows"]
    n_hold = uci_effects[0]["n_holdout_rows_excluded"]
    sd_source = next(iter(effects.values()))["simulated_cohort_sd_source"]

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
        "where $\\sigma(X_{\\text{ind}})$ is the sample standard deviation of that indicator in a simulated cohort "
        f"generated with a fixed seed (n = {sd_source['n_students']:,}, seed = {sd_source['seed']}).",
        "",
        "## UCI Estimation Split",
        "",
        f"UCI effects are estimated on {n_est:,} rows only. The remaining {n_hold:,} rows "
        f"({HOLDOUT_FRACTION:.0%} stratified holdout, seed {SPLIT_SEED}) are reserved for the sim-to-real check "
        "(`ml/simulation/uci_proxies.py`), so that check never evaluates on data that informed these parameters.",
        "",
        "## Fee-Delay Proxy Is Binary",
        "",
        "The UCI source for `fee_payment_delay_days` is a 0/1 default indicator (`tuition_fees_up_to_date == 0` or "
        "`debtor == 1`); UCI has no delay duration. Its standardized effect is divided by the simulated SD of delay "
        "**days**, which treats one SD of being in default as one SD of delay days. This is an assumption, not an "
        "estimate of a per-day effect.",
        "",
    ])

    DOCS_MAPPING_PATH.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    logger.info("Successfully rendered simulation mapping to %s", DOCS_MAPPING_PATH)


def build_arg_parser() -> argparse.ArgumentParser:
    from ml.provenance import add_allow_dirty_argument
    parser = argparse.ArgumentParser(description="Estimate simulation parameters from UCI and OULAD")
    parser.add_argument(
        "--render-mapping",
        action="store_true",
        help="Only render docs/simulation_mapping.md from the existing estimated_parameters.json",
    )
    return add_allow_dirty_argument(parser)


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if args.render_mapping:
        render_simulation_mapping_from_json()
    else:
        estimate_all_parameters(allow_dirty=args.allow_dirty)
