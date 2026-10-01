"""
Sim-to-Real Cross-Domain Transfer Evaluation
Evaluates generalization between empirical benchmarks (UCI ID 697) and the
empirically calibrated simulated Indian collegiate cohort across 6 shared proxy features:
- current_cgpa
- backlog_count
- has_scholarship
- fee_payment_delay_days
- is_first_generation
- is_hosteler

Evaluations:
1. Real-on-Real: Train on UCI proxies, test on UCI holdout.
2. Sim-on-Sim: Train on simulated proxies, test on simulated holdout.
3. Sim-to-Real: Train on simulated proxies, test on UCI holdout.
4. Real-to-Sim: Train on UCI proxies, test on simulated holdout.

Computes ROC-AUC and PR-AUC with 95% bootstrap confidence intervals (1,000 resamples),
saves ml/artifacts/benchmarks/sim_to_real.json, and updates docs/benchmarks.md.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml.data_pipeline.generate_synthetic_indian import generate_indian_student_cohort
from ml.evaluation.harness import compute_bootstrap_cis
from ml.provenance import build_provenance, uci_inputs
from ml.sources.uci import load_uci_clean_df

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = BASE_DIR / "ml" / "artifacts" / "benchmarks"
OUTPUT_JSON_PATH = BENCHMARK_DIR / "sim_to_real.json"
ASSUMPTIONS_PATH = BASE_DIR / "ml" / "simulation" / "assumptions.yaml"
SHARED_PROXIES = [
    "current_cgpa",
    "backlog_count",
    "has_scholarship",
    "fee_payment_delay_days",
    "is_first_generation",
    "is_hosteler",
]


def load_uci_proxies() -> Tuple[pd.DataFrame, np.ndarray]:
    """Extracts the 6 shared proxies and binary labels from UCI dataset (Dropout vs Graduate)."""
    df_raw = load_uci_clean_df()
    df = df_raw[df_raw["target"].isin(["Dropout", "Graduate"])].copy()
    y = (df["target"] == "Dropout").astype(int).values

    has_scholarship = df["scholarship_holder"].astype(float)
    fee_delay_proxy = ((df["tuition_fees_up_to_date"] == 0) | (df["debtor"] == 1)).astype(float)
    he_codes = {2, 3, 4, 5, 6, 40, 41, 42, 43, 44}
    mother_he = df["mothers_qualification"].isin(he_codes)
    father_he = df["fathers_qualification"].isin(he_codes)
    is_first_gen = (~mother_he & ~father_he).astype(float)
    is_hosteler = df["displaced"].astype(float)
    current_cgpa = (df["cu_1st_sem_grade"] / 2.0).clip(0.0, 10.0).astype(float)
    backlog_count = np.maximum(0.0, df["cu_1st_sem_enrolled"] - df["cu_1st_sem_approved"]).astype(float)

    X = pd.DataFrame({
        "current_cgpa": current_cgpa,
        "backlog_count": backlog_count,
        "has_scholarship": has_scholarship,
        "fee_payment_delay_days": fee_delay_proxy,
        "is_first_generation": is_first_gen,
        "is_hosteler": is_hosteler,
    })
    return X, y


def load_simulated_proxies(n_students: int = 2000, seed: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
    """Generates simulated Indian cohort and extracts the 6 shared proxy features."""
    df_sim = generate_indian_student_cohort(n_students=n_students, seed=seed, output_path=None)
    y = df_sim["is_dropout"].values

    X = pd.DataFrame({
        "current_cgpa": df_sim["current_cgpa"].astype(float),
        "backlog_count": df_sim["backlog_count"].astype(float),
        "has_scholarship": df_sim["has_scholarship"].astype(float),
        "fee_payment_delay_days": df_sim["fee_payment_delay_days"].astype(float),
        "is_first_generation": df_sim["is_first_generation"].astype(float),
        "is_hosteler": (df_sim["hostel_status"] == "Hosteler").astype(float),
    })
    return X, y


def run_sim_to_real_benchmark(n_bootstraps: int = 1000) -> Dict[str, Any]:
    """
    Executes cross-domain and within-domain transfer evaluations.
    """
    logger.info("Loading UCI benchmark proxies...")
    X_uci, y_uci = load_uci_proxies()
    logger.info("Generating calibrated simulated cohort proxies...")
    X_sim, y_sim = load_simulated_proxies()

    # Stratified 80/20 train/test splits within each domain
    X_uci_tr, X_uci_te, y_uci_tr, y_uci_te = train_test_split(
        X_uci, y_uci, test_size=0.2, random_state=42, stratify=y_uci
    )
    X_sim_tr, X_sim_te, y_sim_tr, y_sim_te = train_test_split(
        X_sim, y_sim, test_size=0.2, random_state=42, stratify=y_sim
    )

    # Standardize features within each domain so models operate on standardized relative standing
    scaler_uci = StandardScaler()
    X_uci_tr_std = pd.DataFrame(scaler_uci.fit_transform(X_uci_tr), columns=SHARED_PROXIES)
    X_uci_te_std = pd.DataFrame(scaler_uci.transform(X_uci_te), columns=SHARED_PROXIES)

    scaler_sim = StandardScaler()
    X_sim_tr_std = pd.DataFrame(scaler_sim.fit_transform(X_sim_tr), columns=SHARED_PROXIES)
    X_sim_te_std = pd.DataFrame(scaler_sim.transform(X_sim_te), columns=SHARED_PROXIES)

    # Train within-domain models
    m_uci = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
    m_uci.fit(X_uci_tr_std, y_uci_tr)

    m_sim = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
    m_sim.fit(X_sim_tr_std, y_sim_tr)

    # Predictions
    # 1. Real-on-Real (UCI test)
    p_real_real = m_uci.predict_proba(X_uci_te_std)[:, 1]
    # 2. Sim-on-Sim (Sim test)
    p_sim_sim = m_sim.predict_proba(X_sim_te_std)[:, 1]
    # 3. Sim-to-Real: trained on Sim, tested on UCI
    p_sim_to_real = m_sim.predict_proba(X_uci_te_std)[:, 1]
    # 4. Real-to-Sim: trained on UCI, tested on Sim
    p_real_to_sim = m_uci.predict_proba(X_sim_te_std)[:, 1]

    logger.info("Computing 95%% bootstrap confidence intervals (B=%d)...", n_bootstraps)
    ci_real_real = compute_bootstrap_cis(y_uci_te, p_real_real, n_bootstraps=n_bootstraps, seed=42)
    ci_sim_sim = compute_bootstrap_cis(y_sim_te, p_sim_sim, n_bootstraps=n_bootstraps, seed=42)
    ci_sim_to_real = compute_bootstrap_cis(y_uci_te, p_sim_to_real, n_bootstraps=n_bootstraps, seed=42)
    ci_real_to_sim = compute_bootstrap_cis(y_sim_te, p_real_to_sim, n_bootstraps=n_bootstraps, seed=42)

    results: Dict[str, Any] = {
        "benchmark": "sim_to_real_transfer",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_bootstraps": n_bootstraps,
        "shared_proxies": SHARED_PROXIES,
        "transfer_evaluations": {
            "real_on_real": {
                "description": "Baseline: Trained on UCI proxies, tested on UCI holdout (Real-on-Real)",
                "train_domain": "UCI (ID 697)",
                "test_domain": "UCI (ID 697)",
                "n_train": len(X_uci_tr),
                "n_test": len(X_uci_te),
                "metrics": {
                    "roc_auc": ci_real_real["roc_auc"],
                    "pr_auc": ci_real_real["pr_auc"],
                },
            },
            "sim_on_sim": {
                "description": "Baseline: Trained on Simulated proxies, tested on Simulated holdout (Sim-on-Sim)",
                "train_domain": "Simulated Indian Cohort",
                "test_domain": "Simulated Indian Cohort",
                "n_train": len(X_sim_tr),
                "n_test": len(X_sim_te),
                "metrics": {
                    "roc_auc": ci_sim_sim["roc_auc"],
                    "pr_auc": ci_sim_sim["pr_auc"],
                },
            },
            "sim_to_real": {
                "description": "Transfer: Trained on Simulated proxies, tested on UCI holdout (Sim-to-Real)",
                "train_domain": "Simulated Indian Cohort",
                "test_domain": "UCI (ID 697)",
                "n_train": len(X_sim_tr),
                "n_test": len(X_uci_te),
                "metrics": {
                    "roc_auc": ci_sim_to_real["roc_auc"],
                    "pr_auc": ci_sim_to_real["pr_auc"],
                },
            },
            "real_to_sim": {
                "description": "Transfer: Trained on UCI proxies, tested on Simulated holdout (Real-to-Sim)",
                "train_domain": "UCI (ID 697)",
                "test_domain": "Simulated Indian Cohort",
                "n_train": len(X_uci_tr),
                "n_test": len(X_sim_te),
                "metrics": {
                    "roc_auc": ci_real_to_sim["roc_auc"],
                    "pr_auc": ci_real_to_sim["pr_auc"],
                },
            },
        },
    }

    # Inputs: the real UCI file and the generator assumptions the simulated cohort is drawn from
    results["provenance"] = build_provenance({**uci_inputs(), ASSUMPTIONS_PATH.name: ASSUMPTIONS_PATH})

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved Sim-to-Real benchmark results to %s", OUTPUT_JSON_PATH)

    return results


if __name__ == "__main__":
    run_sim_to_real_benchmark()
