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
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.evaluation.harness import (
    build_models,
    compute_bootstrap_cis,
    compute_metrics,
    evaluate_split_strategy,
)
from ml.sources.uci import (
    END_OF_SEM1,
    ENROLMENT_TIME,
    FEATURE_SETS,
    FULL,
    get_uci_benchmark_dataset,
    load_uci_clean_df,
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
