"""
Probability Calibration Pipeline for DropoutGuard
Calibrates raw classifier output into mathematically sound posterior probabilities.
Maps calibrated probabilities to Low/Medium/High risk tiers using central configurable thresholds.
"""

import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, List
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

from ml.config import (
    PROCESSED_DATA_PATH,
    BASE_MODEL_PATH,
    MODEL_ARTIFACT_PATH,
    FEATURE_NAMES_PATH,
    EXCLUDED_FEATURES,
    RISK_THRESHOLD_LOW,
    RISK_THRESHOLD_HIGH,
    RANDOM_SEED,
    get_risk_tier
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_calibration_diagnostics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Computes reliability curve points and Brier score loss.
    Brier Score: MSE of predicted probabilities vs actual outcomes (0 = perfect calibration).
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    brier = float(brier_score_loss(y_true, y_prob))
    loss = float(log_loss(y_true, y_prob))

    # Calculate Expected Calibration Error (ECE)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_assignments = np.digitize(y_prob, bin_edges) - 1
    bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)
    
    ece = 0.0
    for b in range(n_bins):
        mask = (bin_assignments == b)
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (np.sum(mask) / len(y_prob)) * np.abs(bin_acc - bin_conf)

    return {
        "brier_score": round(brier, 4),
        "log_loss": round(loss, 4),
        "expected_calibration_error": round(float(ece), 4),
        "prob_true_curve": [round(float(x), 4) for x in prob_true],
        "prob_pred_curve": [round(float(x), 4) for x in prob_pred]
    }


def fit_best_calibrator(
    base_model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[CalibratedClassifierCV, str, Dict[str, Any]]:
    """
    Fits and compares Isotonic Regression vs Sigmoid (Platt Scaling) calibration via cross-validation.
    Selects method minimizing Brier Score Loss on validation data.
    """
    from sklearn.base import clone

    # Combine train + val for 5-fold cross-validated calibration
    X_cv = pd.concat([X_train, X_val], axis=0)
    y_cv = pd.concat([y_train, y_val], axis=0)

    # 1. Evaluate Uncalibrated Base Model on Validation
    raw_val_probs = base_model.predict_proba(X_val)[:, 1]
    raw_diagnostics = compute_calibration_diagnostics(y_val.values, raw_val_probs)
    logger.info("Uncalibrated Base Model Diagnostics (Val): Brier Score=%.4f, ECE=%.4f",
                raw_diagnostics["brier_score"], raw_diagnostics["expected_calibration_error"])

    # 2. Method 1: Isotonic Calibration (Non-parametric step function, 5-fold CV)
    calibrator_isotonic = CalibratedClassifierCV(estimator=clone(base_model), method="isotonic", cv=5)
    calibrator_isotonic.fit(X_cv, y_cv)
    iso_val_probs = calibrator_isotonic.predict_proba(X_val)[:, 1]
    iso_diagnostics = compute_calibration_diagnostics(y_val.values, iso_val_probs)

    # 3. Method 2: Sigmoid Calibration (Platt Scaling parametric logistic, 5-fold CV)
    calibrator_sigmoid = CalibratedClassifierCV(estimator=clone(base_model), method="sigmoid", cv=5)
    calibrator_sigmoid.fit(X_cv, y_cv)
    sig_val_probs = calibrator_sigmoid.predict_proba(X_val)[:, 1]
    sig_diagnostics = compute_calibration_diagnostics(y_val.values, sig_val_probs)

    logger.info("Calibration Comparison (5-Fold CV):")
    logger.info(" -> Isotonic: Brier Score=%.4f, LogLoss=%.4f, ECE=%.4f",
                iso_diagnostics["brier_score"], iso_diagnostics["log_loss"], iso_diagnostics["expected_calibration_error"])
    logger.info(" -> Sigmoid:  Brier Score=%.4f, LogLoss=%.4f, ECE=%.4f",
                sig_diagnostics["brier_score"], sig_diagnostics["log_loss"], sig_diagnostics["expected_calibration_error"])

    if iso_diagnostics["brier_score"] <= sig_diagnostics["brier_score"]:
        champion_calibrator = calibrator_isotonic
        champion_method = "isotonic"
        chosen_diagnostics = iso_diagnostics
    else:
        champion_calibrator = calibrator_sigmoid
        champion_method = "sigmoid"
        chosen_diagnostics = sig_diagnostics

    logger.info("Selected Champion Calibration Method: '%s' (Brier Score on Val: %.4f)",
                champion_method, chosen_diagnostics["brier_score"])

    # Evaluate on untouched test split
    test_probs = champion_calibrator.predict_proba(X_test)[:, 1]
    test_diagnostics = compute_calibration_diagnostics(y_test.values, test_probs)

    comparison_report = {
        "selected_method": champion_method,
        "validation_brier_score": chosen_diagnostics["brier_score"],
        "validation_ece": chosen_diagnostics["expected_calibration_error"],
        "test_diagnostics": test_diagnostics,
        "raw_diagnostics": raw_diagnostics,
        "isotonic_diagnostics": iso_diagnostics,
        "sigmoid_diagnostics": sig_diagnostics,
        "risk_thresholds": {
            "low_threshold": RISK_THRESHOLD_LOW,
            "high_threshold": RISK_THRESHOLD_HIGH
        }
    }

    return champion_calibrator, champion_method, comparison_report


def predict_student_risk(
    model: Any,
    student_features: pd.DataFrame
) -> Tuple[np.ndarray, List[str]]:
    """
    Given a dataframe of student features, returns:
    - calibrated_probs: array of calibrated risk probabilities in [0.0, 1.0]
    - risk_tiers: list of 'Low', 'Medium', 'High' strings
    """
    calibrated_probs = model.predict_proba(student_features)[:, 1]
    calibrated_probs = np.clip(calibrated_probs, 0.0, 1.0)
    risk_tiers = [get_risk_tier(p) for p in calibrated_probs]
    return calibrated_probs, risk_tiers


def run_calibration_pipeline() -> Tuple[CalibratedClassifierCV, Dict[str, Any]]:
    """
    Loads trained base model, fits probability calibrator, evaluates on test split, and persists artifact.
    """
    # 1. Load base model and data
    if not BASE_MODEL_PATH.exists():
        logger.info("Base model not found. Running training pipeline first...")
        from ml.models.train import train_pipeline
        train_pipeline()

    base_model = joblib.load(BASE_MODEL_PATH)
    df = pd.read_csv(PROCESSED_DATA_PATH)
    with open(FEATURE_NAMES_PATH, "r") as f:
        feature_names = json.load(f)

    X = df[feature_names]
    y = df["is_dropout"].astype(int)

    # Recreate the exact stratified splits
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=(0.15 / 0.85), random_state=RANDOM_SEED, stratify=y_train_val
    )

    # Fit calibration
    calibrated_model, method, report = fit_best_calibrator(base_model, X_train, y_train, X_val, y_val, X_test, y_test)

    # Evaluate calibration on test split
    test_probs, test_tiers = predict_student_risk(calibrated_model, X_test)
    test_calibration_diag = compute_calibration_diagnostics(y_test.values, test_probs)

    tier_counts = pd.Series(test_tiers).value_counts().to_dict()
    logger.info("Test Split Calibration Diagnostics: Brier Score=%.4f, ECE=%.4f",
                test_calibration_diag["brier_score"], test_calibration_diag["expected_calibration_error"])
    logger.info("Test Split Risk Tier Breakdown: %s", tier_counts)

    report["test_diagnostics"] = test_calibration_diag
    report["test_tier_distribution"] = tier_counts

    # Save calibrated model artifact
    MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated_model, MODEL_ARTIFACT_PATH)
    logger.info("Saved calibrated model artifact to %s", MODEL_ARTIFACT_PATH)

    return calibrated_model, report


if __name__ == "__main__":
    cal_model, report = run_calibration_pipeline()
    print("Probability Calibration Complete!")
    print(json.dumps(report, indent=2))
