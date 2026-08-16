"""
Unit & Integration Tests for DropoutGuard Model Pipeline
Tests:
- XGBoost training and artifact persistence
- CalibratedClassifierCV probability bounds [0, 1] and risk tier thresholds
- SHAP explanation schema and plain-language sentence generation
- Fairness audit computation and non-empty markdown output
"""

import pytest
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from ml.config import (
    MODEL_ARTIFACT_PATH,
    BASE_MODEL_PATH,
    FEATURE_NAMES_PATH,
    FAIRNESS_REPORT_PATH,
    RISK_THRESHOLD_LOW,
    RISK_THRESHOLD_HIGH,
    get_risk_tier
)
from ml.models.train import train_pipeline, prepare_training_data
from ml.models.calibrate import run_calibration_pipeline, predict_student_risk
from ml.models.explain_shap import SHAPExplainerService, build_plain_language_sentence
from ml.models.fairness_audit import run_comprehensive_fairness_audit


class TestModelTrainingAndCalibration:
    """Tests for model training and probability calibration."""

    @pytest.fixture(scope="class", autouse=True)
    def setup_model_artifacts(self):
        """Ensures model training and calibration artifacts exist."""
        train_pipeline()
        run_calibration_pipeline()

    def test_artifacts_exist(self):
        """Verify saved model and metadata artifacts."""
        assert BASE_MODEL_PATH.exists()
        assert MODEL_ARTIFACT_PATH.exists()
        assert FEATURE_NAMES_PATH.exists()

    def test_calibrated_prediction_bounds_and_tiers(self):
        """Verify calibrated probabilities are strictly in [0, 1] and map correctly to tiers."""
        model = joblib.load(MODEL_ARTIFACT_PATH)
        X, y, feature_names, _ = prepare_training_data()
        
        sample_X = X.head(50)
        probs, tiers = predict_student_risk(model, sample_X)

        assert len(probs) == 50
        assert len(tiers) == 50
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

        for p, t in zip(probs, tiers):
            if p < RISK_THRESHOLD_LOW:
                assert t == "Low"
            elif p <= RISK_THRESHOLD_HIGH:
                assert t == "Medium"
            else:
                assert t == "High"

    def test_risk_tier_boundary_logic(self):
        """Test edge cases for get_risk_tier mapping function."""
        assert get_risk_tier(0.0) == "Low"
        assert get_risk_tier(0.329) == "Low"
        assert get_risk_tier(0.33) == "Medium"
        assert get_risk_tier(0.50) == "Medium"
        assert get_risk_tier(0.66) == "Medium"
        assert get_risk_tier(0.661) == "High"
        assert get_risk_tier(1.0) == "High"


class TestSHAPExplainability:
    """Tests for SHAP TreeExplainer and plain language generation."""

    @pytest.fixture(scope="class")
    def shap_service(self):
        return SHAPExplainerService()

    def test_global_importance_structure(self, shap_service):
        """Verify global SHAP feature ranking structure."""
        X, _, _, _ = prepare_training_data()
        global_summary = shap_service.explain_global(X.head(100), top_k=5)

        assert len(global_summary) == 5
        for item in global_summary:
            assert "feature_name" in item
            assert "display_name" in item
            assert "mean_abs_shap" in item
            assert "importance_percentage" in item
            assert item["mean_abs_shap"] >= 0.0

    def test_local_student_explanation_schema(self, shap_service):
        """Verify per-student local explanation returns top 3-5 drivers with sentences."""
        X, _, _, _ = prepare_training_data()
        sample_student = X.iloc[0]
        explanations = shap_service.explain_local_student(sample_student, top_k=4)

        assert len(explanations) == 4
        for exp in explanations:
            assert "feature_name" in exp
            assert "display_name" in exp
            assert "feature_value" in exp
            assert "shap_value" in exp
            assert exp["impact_direction"] in ["RISK_INCREASING", "RISK_DECREASING"]
            assert "risk_delta_percentage_points" in exp
            assert "plain_language_explanation" in exp
            assert len(exp["plain_language_explanation"]) > 10

    def test_sentence_generator_plain_language(self):
        """Test domain specific sentence templates."""
        s1 = build_plain_language_sentence("attendance_percentage", 62.0, 0.185, 18.5)
        assert "increasing risk by 18.5 percentage points" in s1
        assert "75%" in s1

        s2 = build_plain_language_sentence("backlog_count", 3.0, 0.142, 14.2)
        assert "3 uncleared exam backlog" in s2
        assert "increasing risk by 14.2 percentage points" in s2

        s3 = build_plain_language_sentence("fee_payment_delay_days", 45.0, 0.098, 9.8)
        assert "45 days" in s3
        assert "financial distress" in s3


class TestFairnessAudit:
    """Tests for the fairness and bias audit reporting."""

    def test_fairness_audit_execution_and_report_file(self):
        """Verify fairness audit runs and populates docs/ethics_and_fairness.md."""
        audit_results = run_comprehensive_fairness_audit()

        assert "gender" in audit_results
        assert "economic_proxy" in audit_results
        assert "first_generation" in audit_results

        assert FAIRNESS_REPORT_PATH.exists()
        content = FAIRNESS_REPORT_PATH.read_text()
        assert len(content) > 500
        assert "Gender Disparity Audit" in content
        assert "Socio-Economic Proxy Audit" in content
        assert "False Negative Rate" in content
