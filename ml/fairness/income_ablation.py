"""
Simulated Cohort Income Feature Ablation
Context: Evaluates the empirical effect of removing explicit income features on overall predictive performance
and disaggregated False Negative Rates (FNR) across family income brackets:
- Model A (With Income Features): Includes income_slab_idx and financial_stress_index (37 features)
- Model B (Without Income Features): Strictly ablates income_slab_idx and financial_stress_index (35 features)

Measures overall metrics (ROC-AUC, PR-AUC, F1, Recall, Precision, Brier) and disaggregated FNR by income slab.
"""

import json
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ml.config import FEATURE_NAMES_PATH, PROCESSED_DATA_PATH
from ml.fairness.audit import compute_group_metrics

logger = logging.getLogger(__name__)

INCOME_SLAB_LABELS = {
    0: "<2 LPA",
    1: "2-5 LPA",
    2: "5-8 LPA",
    3: ">8 LPA",
}


def run_income_ablation_experiment(
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes 5-fold cross-validation on the simulated cohort with vs without income features.
    """
    logger.info("Loading processed simulated cohort from %s...", PROCESSED_DATA_PATH)
    df = pd.read_csv(PROCESSED_DATA_PATH)
    with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
        all_features = json.load(f)

    # Define feature sets
    income_features_to_drop = ["income_slab_idx", "financial_stress_index"]
    ablated_features = [f for f in all_features if f not in income_features_to_drop]

    y = df["is_dropout"].values
    income_slabs = df["income_slab_idx"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_probs_with = np.zeros(len(df))
    oof_probs_without = np.zeros(len(df))

    # Cross-validation loop
    for _fold, (train_idx, val_idx) in enumerate(skf.split(df, y), 1):
        # Model With Income Features
        X_with = df[all_features]
        scaler_with = StandardScaler()
        X_tr_with = scaler_with.fit_transform(X_with.iloc[train_idx])
        X_val_with = scaler_with.transform(X_with.iloc[val_idx])

        clf_with = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        clf_with.fit(X_tr_with, y[train_idx])
        oof_probs_with[val_idx] = clf_with.predict_proba(X_val_with)[:, 1]

        # Model Without Income Features
        X_without = df[ablated_features]
        scaler_without = StandardScaler()
        X_tr_without = scaler_without.fit_transform(X_without.iloc[train_idx])
        X_val_without = scaler_without.transform(X_without.iloc[val_idx])

        clf_without = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        clf_without.fit(X_tr_without, y[train_idx])
        oof_probs_without[val_idx] = clf_without.predict_proba(X_val_without)[:, 1]

    # Evaluate Overall Performance
    def eval_overall(p: np.ndarray) -> Dict[str, float]:
        pred = (p >= 0.50).astype(int)
        return {
            "roc_auc": round(float(roc_auc_score(y, p)), 4),
            "pr_auc": round(float(average_precision_score(y, p)), 4),
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "brier_score": round(float(brier_score_loss(y, p)), 4),
        }

    overall_with = eval_overall(oof_probs_with)
    overall_without = eval_overall(oof_probs_without)

    # Disaggregated Metrics by Income Slab (at 0.50 threshold and top-20% threshold)
    thresh_with = float(np.quantile(oof_probs_with, 0.80))
    thresh_without = float(np.quantile(oof_probs_without, 0.80))

    pred_with_20 = (oof_probs_with >= thresh_with).astype(int)
    pred_without_20 = (oof_probs_without >= thresh_without).astype(int)

    slabs_summary: Dict[str, Any] = {}
    for idx_val, label in INCOME_SLAB_LABELS.items():
        mask = income_slabs == idx_val
        y_slice = y[mask]

        m_with = compute_group_metrics(y_slice, oof_probs_with[mask], pred_with_20[mask])
        m_without = compute_group_metrics(y_slice, oof_probs_without[mask], pred_without_20[mask])

        slabs_summary[label] = {
            "income_slab_idx": idx_val,
            "n_samples": int(np.sum(mask)),
            "n_dropouts": int(np.sum(y_slice == 1)),
            "base_rate": m_with["base_rate"],
            "with_income_features": {
                "tpr": m_with["tpr_recall"],
                "fnr": m_with["fnr_miss_rate"],
                "selection_rate": m_with["selection_rate"],
                "precision": m_with["precision"],
                "within_group_ece": m_with["within_group_ece"],
            },
            "without_income_features": {
                "tpr": m_without["tpr_recall"],
                "fnr": m_without["fnr_miss_rate"],
                "selection_rate": m_without["selection_rate"],
                "precision": m_without["precision"],
                "within_group_ece": m_without["within_group_ece"],
            },
            "fnr_delta_without_minus_with": round(m_without["fnr_miss_rate"] - m_with["fnr_miss_rate"], 4),
        }

    # Summary table rows
    comparison_table = [
        {
            "configuration": "With Income Features (37 features)",
            "n_features": len(all_features),
            "roc_auc": overall_with["roc_auc"],
            "pr_auc": overall_with["pr_auc"],
            "f1": overall_with["f1"],
            "recall": overall_with["recall"],
            "precision": overall_with["precision"],
            "brier_score": overall_with["brier_score"],
            "fnr_lowest_slab": slabs_summary["<2 LPA"]["with_income_features"]["fnr"],
            "fnr_highest_slab": slabs_summary[">8 LPA"]["with_income_features"]["fnr"],
            "fnr_gap_low_vs_high": round(slabs_summary["<2 LPA"]["with_income_features"]["fnr"] - slabs_summary[">8 LPA"]["with_income_features"]["fnr"], 4),
        },
        {
            "configuration": "Without Income Features (35 features)",
            "n_features": len(ablated_features),
            "roc_auc": overall_without["roc_auc"],
            "pr_auc": overall_without["pr_auc"],
            "f1": overall_without["f1"],
            "recall": overall_without["recall"],
            "precision": overall_without["precision"],
            "brier_score": overall_without["brier_score"],
            "fnr_lowest_slab": slabs_summary["<2 LPA"]["without_income_features"]["fnr"],
            "fnr_highest_slab": slabs_summary[">8 LPA"]["without_income_features"]["fnr"],
            "fnr_gap_low_vs_high": round(slabs_summary["<2 LPA"]["without_income_features"]["fnr"] - slabs_summary[">8 LPA"]["without_income_features"]["fnr"], 4),
        },
    ]

    return {
        "benchmark": "simulated_income_ablation",
        "n_samples": len(df),
        "all_features_count": len(all_features),
        "ablated_features_count": len(ablated_features),
        "ablated_features_list": income_features_to_drop,
        "overall_performance": {
            "with_income": overall_with,
            "without_income": overall_without,
        },
        "slabs_summary": slabs_summary,
        "comparison_table": comparison_table,
    }
