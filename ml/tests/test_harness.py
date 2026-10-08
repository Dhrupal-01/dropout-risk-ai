"""
Unit Tests for Benchmark Evaluation Harness
Context: Phase 1 — Real-data benchmark & reusable evaluation harness

Tests:
1. Integrity check: no 2nd-semester columns in END_OF_SEM1 feature set.
2. Leakage check: preprocessing (StandardScaler) is fit strictly on train folds only.
3. Statistical check: 95% bootstrap confidence interval strictly contains the point estimate.
4. Reproducibility check: identical evaluation results and CIs when evaluated with the same seed.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

from ml.evaluation.harness import (
    compute_per_fold_metrics,
    compute_bootstrap_cis,
    evaluate_split_strategy,
)
from ml.sources.uci import (
    END_OF_SEM1,
    ENROLMENT_TIME,
    FULL,
)


class TestFeatureSetIntegrity:
    """Verifies temporal separation of features across horizons."""

    def test_no_second_sem_column_in_end_of_sem1(self):
        """Verifies strictly zero 2nd-semester features exist in END_OF_SEM1."""
        second_sem_markers = ["2nd_sem", "second_sem", "2nd", "sem2"]
        for col in END_OF_SEM1:
            for marker in second_sem_markers:
                assert marker not in col.lower(), (
                    f"Temporal leakage detected: Feature '{col}' in END_OF_SEM1 matches 2nd-semester marker '{marker}'!"
                )

    def test_no_sem_columns_in_enrolment_time(self):
        """Verifies strictly zero semester-progression features exist in ENROLMENT_TIME."""
        sem_markers = ["1st_sem", "2nd_sem", "sem1", "sem2"]
        for col in ENROLMENT_TIME:
            for marker in sem_markers:
                assert marker not in col.lower(), (
                    f"Temporal leakage detected: Feature '{col}' in ENROLMENT_TIME matches semester marker '{marker}'!"
                )

    def test_full_feature_set_contains_both_semesters(self):
        """Verifies FULL feature set includes both 1st and 2nd semester features."""
        has_1st = any("1st_sem" in c for c in FULL)
        has_2nd = any("2nd_sem" in c for c in FULL)
        assert has_1st and has_2nd, "FULL feature set must encompass both 1st and 2nd semester units."
        assert len(FULL) > len(END_OF_SEM1) > len(ENROLMENT_TIME)


class TestPreprocessingFoldIsolation:
    """Verifies that preprocessing fit operations occur exclusively on train folds."""

    def test_standard_scaler_fits_only_on_train_fold(self):
        """
        Constructs a dataset where test samples have an extreme distribution shift (e.g. mean=1000).
        Verifies that the fitted StandardScaler in the pipeline matches the train fold mean,
        proving no test data leaked into the scaler.
        """
        n_train = 80
        n_test = 20
        n_total = n_train + n_test

        rng = np.random.default_rng(42)
        # Train features ~ N(0, 1)
        X_train = rng.normal(loc=0.0, scale=1.0, size=(n_train, 3))
        # Test features ~ N(1000, 1)
        X_test = rng.normal(loc=1000.0, scale=1.0, size=(n_test, 3))

        X = pd.DataFrame(np.vstack([X_train, X_test]), columns=["a", "b", "c"])
        y = np.array([0, 1] * (n_total // 2))

        train_idx = np.arange(0, n_train)
        test_idx = np.arange(n_train, n_total)
        splits = [(train_idx, test_idx)]

        # Custom spy transformer to record fitted input shape
        fit_shapes = []

        class SpyScaler(BaseEstimator, TransformerMixin):
            def __init__(self):
                self.scaler = StandardScaler()

            def fit(self, X_in, y_in=None):
                fit_shapes.append(len(X_in))
                self.scaler.fit(X_in, y_in)
                return self

            def transform(self, X_in):
                return self.scaler.transform(X_in)

        models = {
            "spy_model": Pipeline([
                ("spy_scaler", SpyScaler()),
                ("clf", LogisticRegression(max_iter=100))
            ])
        }

        evaluate_split_strategy(X, y, splits, models, n_repeats=1, seed=42)

        # The transformer should have been fit only on the training fold (80 rows, never 100)
        assert len(fit_shapes) == 1
        assert fit_shapes[0] == n_train, (
            f"Data leakage detected! Preprocessor was fit on {fit_shapes[0]} rows instead of train fold ({n_train})."
        )


class TestBootstrapConfidenceIntervals:
    """Verifies bootstrap CI properties."""

    def test_ci_contains_point_estimate(self):
        """Verifies that point estimate is strictly bounded within [ci_lower, ci_upper] for all metrics."""
        rng = np.random.default_rng(42)
        n = 500
        y_true = rng.binomial(1, 0.35, size=n)
        y_prob = np.clip(y_true * 0.5 + rng.uniform(0.1, 0.6, size=n), 0.01, 0.99)

        ci_results = compute_bootstrap_cis(y_true, y_prob, n_bootstraps=500, seed=42)

        expected_metrics = [
            "roc_auc",
            "pr_auc",
            "precision_top_10",
            "recall_top_10",
            "precision_top_20",
            "recall_top_20",
            "brier_score",
            "expected_calibration_error",
        ]

        for metric in expected_metrics:
            assert metric in ci_results, f"Missing metric {metric} in CI results."
            stat = ci_results[metric]
            point = stat["point"]
            lo = stat["ci_lower"]
            hi = stat["ci_upper"]

            assert lo <= point, f"Metric '{metric}': ci_lower ({lo}) > point ({point})"
            assert point <= hi, f"Metric '{metric}': point ({point}) > ci_upper ({hi})"
            assert lo <= hi, f"Metric '{metric}': ci_lower ({lo}) > ci_upper ({hi})"


class TestHarnessReproducibility:
    """Verifies deterministic evaluation with fixed random seeds."""

    def test_identical_results_with_same_seed(self):
        """Running evaluation harness twice with identical seed produces identical metrics and CIs."""
        rng = np.random.default_rng(123)
        n = 120
        X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["f1", "f2", "f3", "f4"])
        y = rng.binomial(1, 0.4, size=n)

        # 2 splits
        splits = [
            (np.arange(0, 80), np.arange(80, 120)),
            (np.arange(40, 120), np.arange(0, 40)),
        ]

        models_run1 = {
            "lr": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=42))])
        }
        res1 = evaluate_split_strategy(X, y, splits, models_run1, n_repeats=1, seed=42)

        models_run2 = {
            "lr": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=42))])
        }
        res2 = evaluate_split_strategy(X, y, splits, models_run2, n_repeats=1, seed=42)

        metrics1 = res1["lr"]["metrics"]
        metrics2 = res2["lr"]["metrics"]

        for metric_name in metrics1:
            for key in ["point", "ci_lower", "ci_upper"]:
                assert metrics1[metric_name][key] == metrics2[metric_name][key], (
                    f"Reproducibility mismatch in {metric_name}['{key}']: {metrics1[metric_name][key]} != {metrics2[metric_name][key]}"
                )


class _ScoreColumnModel(BaseEstimator):
    """Stub estimator: fit is a no-op, predict_proba returns the `score` column as P(y=1)."""

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        p = X["score"].to_numpy(dtype=float)
        return np.column_stack([1.0 - p, p])


class TestLeaveOneGroupOutPerFold:
    """LOGO reports per-fold mean ± SD as primary and pooled out-of-fold metrics as secondary."""

    def test_logo_per_fold_differs_from_pooled(self):
        # Within each group the scores rank perfectly (fold ROC-AUC = 1.0), but group B's scores are
        # shifted low: pooled, B's positives (0.3, 0.4) rank below A's negatives (0.6, 0.7).
        # Pooled ROC-AUC = 12 correctly ordered pairs / 16 = 0.75.
        X = pd.DataFrame({"score": [0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4]})
        y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
        splits = list(LeaveOneGroupOut().split(X, y, groups=groups))
        labels = [groups[test_idx][0] for _, test_idx in splits]

        res = evaluate_split_strategy(
            X, y, splits, {"stub": _ScoreColumnModel()}, seed=0, fold_labels=labels
        )["stub"]

        per_fold_auc = res["per_fold"]["summary"]["roc_auc"]
        assert per_fold_auc == {"mean": 1.0, "sd": 0.0, "n_folds_defined": 2, "n_folds_total": 2}
        assert [f["group"] for f in res["per_fold"]["folds"]] == ["A", "B"]

        assert res["metrics_pooling"] == "pooled_out_of_fold"
        assert res["metrics"]["roc_auc"]["point"] == 0.75
        assert res["metrics"]["roc_auc"]["point"] == round(roc_auc_score(y, X["score"]), 4)
        assert per_fold_auc["mean"] != res["metrics"]["roc_auc"]["point"]

    def test_per_fold_metrics_exclude_single_class_folds(self):
        # Fold "A" has no positives: ROC-AUC, PR-AUC and recall are undefined there and must not be
        # averaged in (compute_metrics alone would report ROC-AUC 0.5 for it).
        y = np.array([0, 0, 0, 0, 0, 1, 0, 1])
        probs_a = np.array([0.1, 0.2, 0.3, 0.4])
        probs_b = np.array([0.2, 0.9, 0.3, 0.6])
        fold_probs = [(np.arange(0, 4), probs_a), (np.arange(4, 8), probs_b)]

        result = compute_per_fold_metrics(y, fold_probs, ["A", "B"])
        fold_a, fold_b = result["folds"]

        for m in ("roc_auc", "pr_auc", "recall_top_10", "recall_top_20"):
            assert fold_a["metrics"][m] is None, f"{m} must be undefined on a fold with no positives"
        assert fold_a["metrics"]["brier_score"] is not None

        auc_summary = result["summary"]["roc_auc"]
        assert auc_summary["n_folds_defined"] == 1 and auc_summary["n_folds_total"] == 2
        assert auc_summary["mean"] == fold_b["metrics"]["roc_auc"] == 1.0
        assert auc_summary["sd"] is None
        assert result["summary"]["brier_score"]["n_folds_defined"] == 2

    def test_format_fold_summary(self):
        from scripts.render_benchmark_report import format_fold_summary

        assert format_fold_summary({"mean": 0.71234, "sd": 0.0456, "n_folds_defined": 17, "n_folds_total": 17}) == "0.7123 ± 0.0456 (k=17/17)"
        assert format_fold_summary({"mean": 1.0, "sd": None, "n_folds_defined": 1, "n_folds_total": 2}) == "1.0000 ± n/a (k=1/2)"
        assert format_fold_summary({"mean": None, "sd": None, "n_folds_defined": 0, "n_folds_total": 2}) == "N/A"


class TestRequestedModelsAreNeverDropped:
    """Audit M5: a benchmark never silently drops a model it was asked to run."""

    def test_gru_without_pytorch_raises(self, monkeypatch):
        import pytest
        import ml.models.gru as gru
        from ml.evaluation.harness import build_models

        monkeypatch.setattr(gru, "HAS_TORCH", False)
        with pytest.raises(ImportError, match="PyTorch is not installed"):
            build_models(seed=0, include_gru=True)

    def test_gru_is_built_when_requested(self):
        import ml.models.gru as gru
        from ml.evaluation.harness import build_models

        assert gru.HAS_TORCH, "PyTorch is pinned in requirements.lock and must be installed"
        assert "pytorch_gru" in build_models(seed=0, include_gru=True)
        assert "pytorch_gru" not in build_models(seed=0, include_gru=False)
