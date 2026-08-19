"""
ML service: artifact loading and the 37-feature contract.

These tests treat ml/artifacts/feature_names.json and the trained model as the source of
truth — never the Markdown docs.
"""

import json

import pytest

from backend.app.services.ml_service import (
    ENGINEERED_FEATURE_COLUMNS,
    LABEL_COLUMNS,
    RAW_FEATURE_COLUMNS,
    MLArtifactsNotLoaded,
    MLService,
)
from ml.config import FEATURE_NAMES_PATH


class TestArtifactLoading:
    def test_artifacts_load(self, ml):
        assert ml.is_loaded, f"artifacts failed to load: {ml.load_error}"
        assert ml.load_error is None

    def test_model_version_is_stable_fingerprint(self, ml):
        assert ml.model_version and ml.model_version.startswith("calibrated-")
        assert ml.model_version == MLService()._compute_model_version()

    def test_loading_is_idempotent(self, ml):
        """A second load must not re-read joblib artifacts."""
        before = id(ml._model)
        assert ml.load() is True
        assert id(ml._model) == before

    def test_inference_before_load_raises_clearly(self, sample_raw_features):
        fresh = MLService()
        with pytest.raises(MLArtifactsNotLoaded):
            fresh.predict(sample_raw_features)


class TestFeatureContract:
    def test_exactly_37_features(self, ml):
        assert len(ml.feature_names) == 37

    def test_matches_artifact_order_exactly(self, ml):
        on_disk = json.loads(FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
        assert ml.feature_names == on_disk

    def test_split_is_28_raw_plus_9_engineered(self, ml):
        assert len(RAW_FEATURE_COLUMNS) == 28
        assert len(ENGINEERED_FEATURE_COLUMNS) == 9
        assert sorted(list(RAW_FEATURE_COLUMNS) + list(ENGINEERED_FEATURE_COLUMNS)) == sorted(
            ml.feature_names
        )

    def test_age_is_a_model_feature(self, ml):
        """Documented as 'demographics_protected' but genuinely fed to the model."""
        assert "age" in ml.feature_names

    def test_family_income_slab_excluded_but_encoding_included(self, ml):
        assert "family_income_slab" not in ml.feature_names
        assert "income_slab_idx" in ml.feature_names

    def test_protected_string_attributes_never_reach_the_model(self, ml):
        for column in ("gender", "category", "hostel_status", "student_id"):
            assert column not in ml.feature_names

    def test_frame_has_exact_columns_in_exact_order(self, ml, sample_raw_features):
        frame = ml.build_feature_frame(sample_raw_features)
        assert list(frame.columns) == ml.feature_names
        assert frame.shape == (1, 37)

    def test_engineered_columns_are_derived_not_supplied(self, ml, sample_raw_features):
        """The caller sends no engineered values, yet all 9 appear in the frame."""
        for column in ENGINEERED_FEATURE_COLUMNS:
            assert column not in sample_raw_features
        frame = ml.build_feature_frame(sample_raw_features)
        for column in ENGINEERED_FEATURE_COLUMNS:
            assert column in frame.columns

    def test_engineering_matches_the_training_pipeline(self, ml, sample_raw_features):
        """Values must equal build_engineered_features, the function used for training."""
        import pandas as pd

        from ml.data_pipeline.feature_engineering import build_engineered_features

        expected = build_engineered_features(pd.DataFrame([sample_raw_features]))
        actual = ml.build_feature_frame(sample_raw_features)
        for column in ENGINEERED_FEATURE_COLUMNS:
            assert actual[column].iloc[0] == pytest.approx(expected[column].iloc[0])

    def test_is_hosteler_derived_from_hostel_status(self, ml, sample_raw_features):
        payload = dict(sample_raw_features, hostel_status="Hosteler")
        assert ml.build_feature_frame(payload)["is_hosteler"].iloc[0] == 1
        payload["hostel_status"] = "Day Scholar"
        assert ml.build_feature_frame(payload)["is_hosteler"].iloc[0] == 0

    def test_missing_raw_feature_is_rejected(self, ml, sample_raw_features):
        payload = dict(sample_raw_features)
        del payload["current_cgpa"]
        with pytest.raises(ValueError, match="current_cgpa"):
            ml.build_feature_frame(payload)

    def test_missing_residency_is_rejected(self, ml, sample_raw_features):
        payload = dict(sample_raw_features)
        del payload["hostel_status"]
        with pytest.raises(ValueError, match="hostel_status"):
            ml.build_feature_frame(payload)


class TestLabelLeakage:
    @pytest.mark.parametrize("label", sorted(LABEL_COLUMNS))
    def test_labels_are_refused_as_input(self, ml, sample_raw_features, label):
        payload = dict(sample_raw_features, **{label: 1})
        with pytest.raises(ValueError, match="Label columns"):
            ml.build_feature_frame(payload)

    def test_labels_are_not_model_features(self, ml):
        assert not LABEL_COLUMNS.intersection(ml.feature_names)


class TestInference:
    def test_predict_returns_calibrated_result(self, ml, sample_raw_features):
        result = ml.predict(sample_raw_features)
        assert 0.0 <= result["calibrated_risk_probability"] <= 1.0
        assert result["risk_tier"] in ("Low", "Medium", "High")
        assert result["risk_score_percentage"] == pytest.approx(
            result["calibrated_risk_probability"] * 100.0, abs=0.01
        )
        assert result["model_version"] == ml.model_version
        assert len(result["input_features"]) == 37

    def test_tier_agrees_with_ml_core(self, ml, sample_raw_features):
        from ml.config import get_risk_tier

        result = ml.predict(sample_raw_features)
        assert result["risk_tier"] == get_risk_tier(result["calibrated_risk_probability"])

    def test_high_risk_profile_scores_high(self, ml, sample_raw_features):
        """44% attendance, 75-day fee default, 2 backlogs must not read as Low risk."""
        result = ml.predict(sample_raw_features)
        assert result["risk_tier"] == "High", result

    def test_low_risk_profile_scores_low(self, ml, sample_raw_features):
        healthy = dict(
            sample_raw_features,
            att_core1=95.0, att_core2=94.0, att_lab=98.0, att_elective=92.0,
            attendance_month_1=92.0, attendance_month_2=95.0, attendance_month_3=96.0,
            attendance_percentage=94.5, attendance_3m_trend=2.0,
            consecutive_absences=0, attendance_risk_flag=0,
            prev_sem_cgpa=8.5, current_cgpa=8.9, cgpa_delta=0.4,
            backlog_count=0, internal_exam_score_pct=92.0, stem_core_fail_flag=0,
            lms_logins_per_week=11.5, assignment_submission_lag_days=-2.5,
            resource_access_count=95, days_since_last_lms_activity=1,
            forum_participation_count=12,
            fee_payment_delay_days=0, has_scholarship=1, is_first_generation=0,
            income_slab_idx=3, hostel_status="Hosteler", commute_distance_km=0.5,
        )
        assert ml.predict(healthy)["risk_tier"] == "Low"

    def test_prediction_is_deterministic(self, ml, sample_raw_features):
        a = ml.predict(sample_raw_features)
        b = ml.predict(sample_raw_features)
        assert a["calibrated_risk_probability"] == b["calibrated_risk_probability"]

    def test_snapshot_is_json_serialisable_for_jsonb(self, ml, sample_raw_features):
        """
        input_features goes straight into a JSONB column. pandas returns numpy scalars,
        which json.dumps cannot encode — they must be converted at the service boundary.
        """
        import json

        snapshot = ml.predict(sample_raw_features)["input_features"]
        json.dumps(snapshot)
        for name, value in snapshot.items():
            assert type(value) in (int, float, bool, str), f"{name} is {type(value)}"


class TestExplainAndRecourse:
    def test_shap_drivers_have_the_documented_schema(self, ml, sample_raw_features):
        drivers = ml.explain(sample_raw_features, top_k=4)
        assert len(drivers) == 4
        for driver in drivers:
            assert driver["feature_name"] in ml.feature_names
            assert driver["impact_direction"] in ("RISK_INCREASING", "RISK_DECREASING")
            assert len(driver["plain_language_explanation"]) > 10

    def test_interventions_map_from_drivers(self, ml, sample_raw_features):
        drivers = ml.explain(sample_raw_features, top_k=4)
        recommendations = ml.recommend_interventions(drivers, max_recommendations=3)
        assert 1 <= len(recommendations) <= 3
        for item in recommendations:
            assert item["intervention_id"].startswith("INT_")
            assert item["urgency"] in ("HIGH", "MEDIUM", "LOW")

    def test_counterfactual_reduces_risk(self, ml, sample_raw_features):
        drivers = ml.explain(sample_raw_features, top_k=4)
        recourse = ml.generate_counterfactual(sample_raw_features, top_shap_drivers=drivers)
        assert recourse["projected_risk_prob"] < recourse["current_risk_prob"]
        assert recourse["required_actions"]
