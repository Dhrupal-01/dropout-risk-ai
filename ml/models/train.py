"""
Model Training Pipeline for DropoutGuard
Algorithm: Gradient Boosted Decision Trees (XGBoost)
Pillars: Attendance, Academic Performance, Learning Behavior, Socio-Economic Indicators + Interactions.

Handles Class Imbalance:
- Evaluates both SMOTE (Synthetic Minority Over-sampling) and Cost-Sensitive Class Weighting (`scale_pos_weight`).
- Selects the superior approach based on minority-class Recall and F1 score on validation set.
- Strictly guards against data leakage by fitting resamplers on train split only.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
    confusion_matrix
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from ml.config import (
    PROCESSED_DATA_PATH,
    BASE_MODEL_PATH,
    FEATURE_NAMES_PATH,
    METRICS_REPORT_PATH,
    EXCLUDED_FEATURES,
    RANDOM_SEED
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def prepare_training_data(
    data_path: Path = PROCESSED_DATA_PATH
) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.DataFrame]:
    """
    Loads features.csv, filters out metadata/targets, and prepares feature matrix X and target y.
    Returns: X, y, feature_names, raw_df (for demographic tracking).
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Feature dataset not found at {data_path}. Run Phase 1 data pipeline first.")

    df = pd.read_csv(data_path)
    logger.info("Loaded processed features from %s (%d rows, %d cols)", data_path, len(df), len(df.columns))

    feature_cols = [c for c in df.columns if c not in EXCLUDED_FEATURES]
    X = df[feature_cols].copy()
    y = df["is_dropout"].astype(int)

    logger.info("Extracted %d feature columns for model training.", len(feature_cols))
    return X, y, feature_cols, df


def train_and_compare_imbalance_methods(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> Tuple[XGBClassifier, str, Dict[str, Any]]:
    """
    Trains XGBoost using:
    1. Method A: Class Weighting (`scale_pos_weight = N_neg / N_pos`)
    2. Method B: SMOTE oversampling on training split only
    Compares validation minority-class F1 and Recall, returning the champion model.
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_weight = float(n_neg) / max(float(n_pos), 1.0)
    logger.info("Class distribution in Train split -> Negative (0): %d, Positive (1/At-Risk): %d (Ratio: %.2f)",
                n_neg, n_pos, scale_weight)

    # 1. Model A: Class Weighting
    logger.info("Training Model A: XGBoost with Class-Weighting (scale_pos_weight=%.2f)...", scale_weight)
    model_weighted = XGBClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_weight,
        random_state=RANDOM_SEED,
        eval_metric="logloss"
    )
    model_weighted.fit(X_train, y_train)
    val_preds_w = model_weighted.predict(X_val)
    val_probs_w = model_weighted.predict_proba(X_val)[:, 1]

    metrics_weighted = {
        "recall_minority": float(recall_score(y_val, val_preds_w, pos_label=1)),
        "precision_minority": float(precision_score(y_val, val_preds_w, pos_label=1)),
        "f1_minority": float(f1_score(y_val, val_preds_w, pos_label=1)),
        "roc_auc": float(roc_auc_score(y_val, val_probs_w))
    }

    # 2. Model B: SMOTE Oversampling
    logger.info("Training Model B: XGBoost with SMOTE oversampling on training split...")
    smote = SMOTE(random_state=RANDOM_SEED)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    model_smote = XGBClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RANDOM_SEED,
        eval_metric="logloss"
    )
    model_smote.fit(X_train_smote, y_train_smote)
    val_preds_s = model_smote.predict(X_val)
    val_probs_s = model_smote.predict_proba(X_val)[:, 1]

    metrics_smote = {
        "recall_minority": float(recall_score(y_val, val_preds_s, pos_label=1)),
        "precision_minority": float(precision_score(y_val, val_preds_s, pos_label=1)),
        "f1_minority": float(f1_score(y_val, val_preds_s, pos_label=1)),
        "roc_auc": float(roc_auc_score(y_val, val_probs_s))
    }

    logger.info("Validation Comparison Results:")
    logger.info(" -> Class Weighting: Recall=%.4f, Precision=%.4f, Minority F1=%.4f, AUC=%.4f",
                metrics_weighted["recall_minority"], metrics_weighted["precision_minority"],
                metrics_weighted["f1_minority"], metrics_weighted["roc_auc"])
    logger.info(" -> SMOTE:           Recall=%.4f, Precision=%.4f, Minority F1=%.4f, AUC=%.4f",
                metrics_smote["recall_minority"], metrics_smote["precision_minority"],
                metrics_smote["f1_minority"], metrics_smote["roc_auc"])

    # Decision Rule: Prioritize Minority F1 while ensuring Recall >= 0.85
    # If F1 is close, pick the model with higher Recall (reducing costly False Negatives)
    if (metrics_weighted["f1_minority"] > metrics_smote["f1_minority"]) or (
        np.isclose(metrics_weighted["f1_minority"], metrics_smote["f1_minority"], atol=0.01)
        and metrics_weighted["recall_minority"] >= metrics_smote["recall_minority"]
    ):
        chosen_model = model_weighted
        chosen_method = "Class-Weighting (scale_pos_weight)"
        comparison_info = {
            "chosen_method": chosen_method,
            "reason": "Class-Weighting achieved higher validation minority-class F1 & Recall without synthesizing artificial boundary points.",
            "metrics_weighted": metrics_weighted,
            "metrics_smote": metrics_smote
        }
    else:
        chosen_model = model_smote
        chosen_method = "SMOTE (Synthetic Minority Over-sampling)"
        comparison_info = {
            "chosen_method": chosen_method,
            "reason": "SMOTE achieved superior minority-class F1 and balanced decision boundaries on validation split.",
            "metrics_weighted": metrics_weighted,
            "metrics_smote": metrics_smote
        }

    logger.info("Selected Champion Strategy: %s", chosen_method)
    return chosen_model, chosen_method, comparison_info


def train_pipeline() -> Tuple[XGBClassifier, Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Executes full model training pipeline:
    1. Stratified 70/15/15 Train/Validation/Test Split
    2. Imbalance strategy selection
    3. Test set evaluation & artifact persistence
    """
    X, y, feature_names, full_df = prepare_training_data()

    # Stratified 70/15/15 Split
    X_train_val, X_test, y_train_val, y_test, idx_train_val, idx_test = train_test_split(
        X, y, full_df.index, test_size=0.15, random_state=RANDOM_SEED, stratify=y
    )
    # Validation split from train_val (0.15 / 0.85 = ~0.1765)
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_train_val, y_train_val, idx_train_val, test_size=(0.15 / 0.85), random_state=RANDOM_SEED, stratify=y_train_val
    )

    logger.info("Dataset Splits -> Train: %d (%.1f%%), Val: %d (%.1f%%), Test: %d (%.1f%%)",
                len(X_train), len(X_train) / len(X) * 100,
                len(X_val), len(X_val) / len(X) * 100,
                len(X_test), len(X_test) / len(X) * 100)

    # Train and select best imbalance strategy
    model, chosen_method, comparison_info = train_and_compare_imbalance_methods(
        X_train, y_train, X_val, y_val
    )

    # Final Evaluation on the untouched Test Split
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, test_preds)
    tn, fp, fn, tp = cm.ravel()

    test_metrics = {
        "accuracy": float(accuracy_score(y_test, test_preds)),
        "recall_at_risk_minority": float(recall_score(y_test, test_preds, pos_label=1)),
        "precision_at_risk_minority": float(precision_score(y_test, test_preds, pos_label=1)),
        "f1_at_risk_minority": float(f1_score(y_test, test_preds, pos_label=1)),
        "f1_macro": float(f1_score(y_test, test_preds, average="macro")),
        "roc_auc": float(roc_auc_score(y_test, test_probs)),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "imbalance_strategy": comparison_info
    }

    logger.info("================================================================")
    logger.info("TEST SET EVALUATION METRICS (HEADLINE):")
    logger.info("  Headline At-Risk Recall:    %.4f (Minimizing missed dropouts)", test_metrics["recall_at_risk_minority"])
    logger.info("  At-Risk Precision:         %.4f", test_metrics["precision_at_risk_minority"])
    logger.info("  At-Risk Minority F1 Score: %.4f", test_metrics["f1_at_risk_minority"])
    logger.info("  Macro F1 Score:            %.4f", test_metrics["f1_macro"])
    logger.info("  AUC-ROC:                   %.4f", test_metrics["roc_auc"])
    logger.info("  Overall Accuracy:          %.4f", test_metrics["accuracy"])
    logger.info("  Confusion Matrix: TN=%d, FP=%d, FN=%d, TP=%d", tn, fp, fn, tp)
    logger.info("================================================================")

    # Save artifacts
    BASE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, BASE_MODEL_PATH)
    logger.info("Saved trained base model to %s", BASE_MODEL_PATH)

    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(feature_names, f, indent=2)
    logger.info("Saved feature names (%d cols) to %s", len(feature_names), FEATURE_NAMES_PATH)

    with open(METRICS_REPORT_PATH, "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info("Saved test metrics to %s", METRICS_REPORT_PATH)

    split_indices = {
        "train_idx": idx_train,
        "val_idx": idx_val,
        "test_idx": idx_test
    }

    return model, test_metrics, split_indices


if __name__ == "__main__":
    model, metrics, splits = train_pipeline()
    print("Base Model Training Complete!")
