"""
OULAD Presentation Shift Check
Evaluates fairness metric stability and subgroup disparity gaps across calendar presentations:
- 2013 Presentations (2013B + 2013J: Training Cohort)
- 2014 Presentations (2014B + 2014J: Temporal Holdout Cohort)

Audited attributes (see ml/fairness/attributes.py):
- PROTECTED, never model features: gender, age_band, imd_band, disability, region (+ imd_x_gender)
- AUDIT_GROUPS, also a documented model feature: highest_education
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ml.fairness.audit import audit_model_fairness
from ml.sources.oulad import build_snapshot_dataset, load_raw_tables

logger = logging.getLogger(__name__)


def run_oulad_presentation_shift_check(
    t: int = 56,
    tables: Optional[Dict[str, pd.DataFrame]] = None,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes temporal shift evaluation on OULAD snapshot t.
    Compares 2013 (in-sample / cross-validated) vs 2014 (temporal holdout) fairness metrics.
    """
    logger.info("Building OULAD snapshot t=%d for presentation shift check...", t)
    raw_tables = tables or load_raw_tables()
    X, y, splits, groups, audit_df, feature_names = build_snapshot_dataset(t=t, tables=raw_tables)

    # X is used exactly as built by the shared OULAD builder (highest_education is an AUDIT_GROUP
    # feature); the audit frame already carries highest_education, normalised imd_band and imd_x_gender.
    train_idx, test_idx = splits[0]
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]

    audit_train = audit_df.iloc[train_idx].reset_index(drop=True)
    audit_test = audit_df.iloc[test_idx].reset_index(drop=True)

    # Scale features
    scaler = StandardScaler()
    X_tr_std = scaler.fit_transform(X_train)
    X_te_std = scaler.transform(X_test)

    # Fit model on 2013
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    clf.fit(X_tr_std, y_train)

    p_train = clf.predict_proba(X_tr_std)[:, 1]
    p_test = clf.predict_proba(X_te_std)[:, 1]

    # Audit attributes configuration with canonical reference groups
    attr_configs = {
        "gender": "M",
        "age_band": "0-35",
        "disability": "N",
        "imd_band": "90-100%",  # least deprived as reference
        "highest_education": "A Level or Equivalent",
        "region": "South Region",
    }

    logger.info("Auditing 2013 presentations (N=%d)...", len(y_train))
    audit_2013 = audit_model_fairness(
        y_true=y_train,
        y_prob=p_train,
        audit_df=audit_train,
        attribute_configs=attr_configs,
        top_k_pct=0.20,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )

    logger.info("Auditing 2014 presentations (N=%d)...", len(y_test))
    audit_2014 = audit_model_fairness(
        y_true=y_test,
        y_prob=p_test,
        audit_df=audit_test,
        attribute_configs=attr_configs,
        top_k_pct=0.20,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )

    # Compute Shift Differences (2014 - 2013)
    shift_comparison: Dict[str, Any] = {}
    for attr, ref in attr_configs.items():
        if attr not in audit_2013["attributes"] or attr not in audit_2014["attributes"]:
            continue

        res_13 = audit_2013["attributes"][attr]
        res_14 = audit_2014["attributes"][attr]

        groups_13 = res_13.get("groups", {})
        groups_14 = res_14.get("groups", {})
        gaps_13 = res_13.get("gaps_vs_reference", {})
        gaps_14 = res_14.get("gaps_vs_reference", {})

        attr_shift = []
        all_groups = sorted(list(set(groups_13.keys()) | set(groups_14.keys())))

        for g in all_groups:
            g_13 = groups_13.get(g, {})
            g_14 = groups_14.get(g, {})
            gap_13 = gaps_13.get(g, {})
            gap_14 = gaps_14.get(g, {})

            fnr_13 = g_13.get("fnr_miss_rate")
            fnr_14 = g_14.get("fnr_miss_rate")
            fnr_shift = round(fnr_14 - fnr_13, 4) if (fnr_13 is not None and fnr_14 is not None) else None

            gap_val_13 = gap_13.get("fnr_gap")
            gap_val_14 = gap_14.get("fnr_gap")
            gap_shift = round(gap_val_14 - gap_val_13, 4) if (gap_val_13 is not None and gap_val_14 is not None) else None

            attr_shift.append({
                "group": g,
                "is_reference": (g == ref),
                "n_2013": g_13.get("n_samples", 0),
                "n_2014": g_14.get("n_samples", 0),
                "base_rate_2013": g_13.get("base_rate"),
                "base_rate_2014": g_14.get("base_rate"),
                "fnr_2013": fnr_13,
                "fnr_2014": fnr_14,
                "fnr_shift_2014_minus_2013": fnr_shift,
                "fnr_gap_2013": gap_val_13,
                "fnr_gap_2014": gap_val_14,
                "gap_shift_2014_minus_2013": gap_shift,
            })

        shift_comparison[attr] = {
            "reference_group": ref,
            "groups": attr_shift,
        }

    return {
        "benchmark": "oulad_presentation_shift_check",
        "snapshot_horizon_t": t,
        "n_train_2013": len(y_train),
        "n_test_2014": len(y_test),
        "audit_2013": audit_2013,
        "audit_2014": audit_2014,
        "shift_comparison": shift_comparison,
    }
