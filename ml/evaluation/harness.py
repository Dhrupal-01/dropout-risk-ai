"""
Source-Agnostic Benchmark Evaluation Harness
Supports:
- Repeated Stratified 5-Fold CV (3 repeats)
- Leave-One-Group-Out (LOGO)
- Predefined split mode
- Majority class, Logistic Regression, XGBoost
- Sklearn Pipeline wrapping so preprocessors fit exclusively on train folds
- Metrics: ROC-AUC, PR-AUC, Precision@10%, Recall@10%, Precision@20%, Recall@20%, Brier Score, ECE (10 bins)
- 95% Bootstrap Confidence Intervals (1,000 resamples of out-of-fold predictions)
- Benchmark JSON artifact output
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_ARTIFACTS_DIR = BASE_DIR / "ml" / "artifacts" / "benchmarks"


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Computes all standard benchmark metrics on ground truth labels and predicted probabilities.

    Metrics:
    - ROC-AUC
    - PR-AUC (Average Precision)
    - Precision at top 10% predicted risk
    - Recall at top 10% predicted risk
    - Precision at top 20% predicted risk
    - Recall at top 20% predicted risk
    - Brier score
    - Expected Calibration Error (ECE, 10 bins)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_samples = len(y_true)
    total_positives = int(np.sum(y_true))

    # ROC-AUC & PR-AUC
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        pr_auc = float(np.mean(y_true))

    # Precision & Recall at Top K%
    # Sort samples by predicted probability descending
    sort_idx = np.argsort(-y_prob)
    sorted_labels = y_true[sort_idx]

    # Top 10%
    k_10 = max(1, int(np.ceil(0.10 * n_samples)))
    top_10_labels = sorted_labels[:k_10]
    prec_10 = float(np.mean(top_10_labels))
    rec_10 = float(np.sum(top_10_labels) / max(1, total_positives))

    # Top 20%
    k_20 = max(1, int(np.ceil(0.20 * n_samples)))
    top_20_labels = sorted_labels[:k_20]
    prec_20 = float(np.mean(top_20_labels))
    rec_20 = float(np.sum(top_20_labels) / max(1, total_positives))

    # Brier score
    brier = float(brier_score_loss(y_true, y_prob))

    # Expected Calibration Error (ECE) over 10 equal bins in [0, 1]
    bin_edges = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for i in range(10):
        low_b, high_b = bin_edges[i], bin_edges[i + 1]
        if i == 9:
            mask = (y_prob >= low_b) & (y_prob <= high_b)
        else:
            mask = (y_prob >= low_b) & (y_prob < high_b)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (bin_count / n_samples) * abs(bin_acc - bin_conf)

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision_top_10": round(prec_10, 4),
        "recall_top_10": round(rec_10, 4),
        "precision_top_20": round(prec_20, 4),
        "recall_top_20": round(rec_20, 4),
        "brier_score": round(brier, 4),
        "expected_calibration_error": round(float(ece), 4),
    }


def compute_calibration_curve_data(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> List[Dict[str, Any]]:
    """
    Computes binned reliability diagnostic data across n_bins equal buckets.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins_data = []

    for i in range(n_bins):
        low_b, high_b = float(bin_edges[i]), float(bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= low_b) & (y_prob <= high_b)
        else:
            mask = (y_prob >= low_b) & (y_prob < high_b)

        count = int(np.sum(mask))
        if count > 0:
            mean_pred = float(np.mean(y_prob[mask]))
            true_rate = float(np.mean(y_true[mask]))
        else:
            mean_pred = (low_b + high_b) / 2.0
            true_rate = 0.0

        bins_data.append({
            "bin_index": i,
            "bin_low": round(low_b, 2),
            "bin_high": round(high_b, 2),
            "count": count,
            "mean_pred": round(mean_pred, 4),
            "true_rate": round(true_rate, 4),
        })

    return bins_data


def compute_bootstrap_cis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Computes 95% bootstrap confidence intervals across 1,000 resamples of out-of-fold predictions.
    Guarantees that the point estimate is bounded within [ci_lower, ci_upper].
    """
    point_estimates = compute_metrics(y_true, y_prob)
    n_samples = len(y_true)
    rng = np.random.default_rng(seed)

    boot_metrics: Dict[str, List[float]] = {m: [] for m in point_estimates}

    attempts = 0
    max_attempts = n_bootstraps * 3
    valid_bootstraps = 0

    while valid_bootstraps < n_bootstraps and attempts < max_attempts:
        attempts += 1
        boot_idx = rng.integers(0, n_samples, size=n_samples)
        y_b = y_true[boot_idx]
        p_b = y_prob[boot_idx]

        # Resample must contain at least one positive and one negative sample for AUC calculations
        if len(np.unique(y_b)) < 2:
            continue

        res = compute_metrics(y_b, p_b)
        for m in point_estimates:
            boot_metrics[m].append(res[m])
        valid_bootstraps += 1

    ci_summary: Dict[str, Dict[str, float]] = {}
    for m, point_val in point_estimates.items():
        vals = boot_metrics[m]
        if vals:
            ci_low = float(np.percentile(vals, 2.5))
            ci_high = float(np.percentile(vals, 97.5))
        else:
            ci_low = point_val
            ci_high = point_val

        # Ensure point estimate is strictly bounded within interval
        ci_low = round(min(ci_low, point_val), 4)
        ci_high = round(max(ci_high, point_val), 4)

        ci_summary[m] = {
            "point": point_val,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
        }

    return ci_summary


def build_models(seed: int = 42, include_gru: bool = False) -> Dict[str, Any]:
    """
    Constructs model evaluation pipelines.
    Preprocessing (StandardScaler) is placed inside Pipeline so it is fit exclusively on train folds.
    """
    models: Dict[str, Any] = {
        "majority_class": Pipeline([
            ("clf", DummyClassifier(strategy="prior"))
        ]),
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed))
        ]),
        "xgboost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=180,
                max_depth=4,
                learning_rate=0.06,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=seed,
                eval_metric="logloss"
            ))
        ])
    }
    if include_gru:
        try:
            from ml.models.gru import HAS_TORCH, PyTorchGRUEstimator
            if HAS_TORCH:
                models["pytorch_gru"] = PyTorchGRUEstimator(random_state=seed)
        except Exception as exc:
            logger.warning("Could not instantiate PyTorchGRUEstimator: %s", exc)

    return models


def evaluate_split_strategy(
    X: pd.DataFrame,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    models: Dict[str, Any],
    n_repeats: int = 1,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Runs cross-validation across provided train/test index splits, generating out-of-fold probability
    predictions and computing point estimates and 95% bootstrap CIs.

    If n_repeats > 1, out-of-fold predictions from each repeat are averaged.
    """
    n_samples = len(y)
    results = {}

    for model_name, pipeline in models.items():
        logger.info("Evaluating model '%s' across %d splits...", model_name, len(splits))

        # Array to accumulate out-of-fold probability predictions
        accumulated_probs = np.zeros(n_samples, dtype=float)
        prediction_counts = np.zeros(n_samples, dtype=int)

        for train_idx, test_idx in splits:
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_te = X.iloc[test_idx]

            # Fit pipeline on training fold only
            pipeline.fit(X_tr, y_tr)

            # Predict probabilities on test fold
            probs = pipeline.predict_proba(X_te)[:, 1]
            accumulated_probs[test_idx] += probs
            prediction_counts[test_idx] += 1

        # Identify all indices evaluated across test splits
        all_test_indices = []
        for _, test_idx in splits:
            all_test_indices.extend(test_idx)
        evaluated_indices = np.unique(all_test_indices)

        assert len(evaluated_indices) > 0, "No test indices provided in splits!"

        # Extract evaluated subset
        y_eval = y[evaluated_indices]
        oof_probs = accumulated_probs[evaluated_indices] / prediction_counts[evaluated_indices]

        # Compute point estimates and 95% bootstrap CIs
        cis = compute_bootstrap_cis(y_eval, oof_probs, n_bootstraps=1000, seed=seed)
        cal_bins = compute_calibration_curve_data(y_eval, oof_probs, n_bins=10)

        results[model_name] = {
            "metrics": cis,
            "calibration_bins": cal_bins,
            "n_evaluated": len(evaluated_indices),
            "oof_probabilities_sample": [round(float(p), 4) for p in oof_probs[:10]],
        }

    return results


def run_benchmark_for_dataset(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    dataset_name: str = "uci_697",
    feature_set: str = "enrolment_time",
    label_variant: str = "primary",
    predefined_splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    include_repeated_cv: bool = True,
    include_gru: bool = False,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Executes full benchmark evaluation across split strategies:
    1. Predefined Splits (if provided)
    2. Repeated Stratified 5-Fold CV (if include_repeated_cv=True)
    3. Leave-One-Group-Out (if groups provided)

    Writes JSON benchmark artifact to ml/artifacts/benchmarks/
    """
    out_dir = output_dir or BENCHMARK_ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    models = build_models(seed=seed, include_gru=include_gru)
    eval_results = {}

    # 1. Predefined splits (if supplied, e.g. for OULAD temporal validation)
    if predefined_splits is not None:
        logger.info("Running evaluation on predefined splits...")
        eval_results["predefined_split"] = evaluate_split_strategy(
            X, y, predefined_splits, models, n_repeats=1, seed=seed
        )

    # 2. Repeated Stratified 5-Fold CV (3 repeats)
    if include_repeated_cv:
        logger.info("Generating Repeated Stratified 5-Fold CV splits (3 repeats, seed=%d)...", seed)
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
        rskf_splits = list(rskf.split(X, y))
        eval_results["repeated_stratified_cv"] = evaluate_split_strategy(
            X, y, rskf_splits, models, n_repeats=3, seed=seed
        )

    # 3. Leave-One-Group-Out (using course / program / module groups)
    if groups is not None:
        logger.info("Generating Leave-One-Group-Out splits across %d groups...", len(np.unique(groups)))
        logo = LeaveOneGroupOut()
        logo_splits = list(logo.split(X, y, groups=groups))
        eval_results["leave_one_group_out"] = evaluate_split_strategy(
            X, y, logo_splits, models, n_repeats=1, seed=seed
        )

    benchmark_artifact = {
        "dataset": dataset_name,
        "feature_set": feature_set,
        "label_variant": label_variant,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "feature_names": list(X.columns),
        "total_positives": int(np.sum(y)),
        "prevalence": round(float(np.mean(y)), 4),
        "split_strategies": list(eval_results.keys()),
        "results": eval_results,
    }

    artifact_filename = f"{dataset_name.split('_')[0]}_{feature_set}_{label_variant}.json"
    artifact_path = out_dir / artifact_filename
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_artifact, f, indent=2)

    logger.info("Successfully saved benchmark artifact to %s", artifact_path)
    return benchmark_artifact
