"""
API tests for GET /api/v1/students/{student_id}/explanation.

The central guarantee: the explanation is derived from the persisted prediction snapshot,
never from the live (mutable) student feature record.
"""

import pytest

from backend.app.services.ml_service import ml_service
from backend.tests.conftest import requires_db

PREDICT_URL = "/api/v1/predict"


def explanation_url(student_id: str) -> str:
    return f"/api/v1/students/{student_id}/explanation"


def body(student_id, features, **extra):
    return {"student_id": student_id, "features": features, **extra}


@pytest.fixture
def high_risk_features(sample_raw_features):
    return dict(sample_raw_features)


@pytest.fixture
def low_risk_features(sample_raw_features):
    return dict(
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


@requires_db
class TestExplanationEndpoint:
    def test_returns_drivers_for_scored_student(self, client, db_engine, high_risk_features):
        client.post(PREDICT_URL, json=body("EXP_OK", high_risk_features))
        response = client.get(explanation_url("EXP_OK"))
        assert response.status_code == 200

        payload = response.json()
        assert payload["student_id"] == "EXP_OK"
        assert 0.0 <= payload["risk_probability"] <= 1.0
        assert payload["risk_tier"] in ("Low", "Medium", "High")
        assert len(payload["top_drivers"]) == 5

    def test_driver_schema_preserves_ml_semantics(self, client, db_engine, high_risk_features):
        """Field names and sign conventions must pass through from explain_shap untouched."""
        client.post(PREDICT_URL, json=body("EXP_SCHEMA", high_risk_features))
        drivers = client.get(explanation_url("EXP_SCHEMA")).json()["top_drivers"]

        for driver in drivers:
            for field in (
                "feature_name",
                "display_name",
                "feature_value",
                "shap_value",
                "impact_direction",
                "risk_delta_percentage_points",
                "plain_language_explanation",
            ):
                assert field in driver
            assert driver["feature_name"] in ml_service.feature_names
            assert driver["impact_direction"] in ("RISK_INCREASING", "RISK_DECREASING")
            assert len(driver["plain_language_explanation"]) > 10

    def test_shap_sign_semantics_unchanged(self, client, db_engine, high_risk_features):
        """Positive SHAP must mean RISK_INCREASING, exactly as the ML layer defines it."""
        client.post(PREDICT_URL, json=body("EXP_SIGN", high_risk_features))
        for driver in client.get(explanation_url("EXP_SIGN")).json()["top_drivers"]:
            if driver["shap_value"] > 0:
                assert driver["impact_direction"] == "RISK_INCREASING"
            else:
                assert driver["impact_direction"] == "RISK_DECREASING"

    def test_probability_matches_the_persisted_prediction(self, client, db_engine, high_risk_features):
        predicted = client.post(PREDICT_URL, json=body("EXP_MATCH", high_risk_features)).json()
        explained = client.get(explanation_url("EXP_MATCH")).json()
        assert explained["risk_probability"] == predicted["calibrated_risk_probability"]
        assert explained["risk_tier"] == predicted["risk_tier"]

    def test_explains_latest_prediction_not_current_features(
        self, client, db_engine, high_risk_features, low_risk_features
    ):
        """
        After re-scoring with different features, the explanation must describe the LATEST
        prediction — and its probability must reconcile with that prediction's score.
        """
        client.post(PREDICT_URL, json=body("EXP_LATEST", high_risk_features))
        second = client.post(PREDICT_URL, json=body("EXP_LATEST", low_risk_features)).json()

        explained = client.get(explanation_url("EXP_LATEST")).json()
        assert explained["risk_probability"] == second["calibrated_risk_probability"]
        assert explained["risk_tier"] == second["risk_tier"] == "Low"

    def test_explanation_is_stable_across_calls(self, client, db_engine, high_risk_features):
        """The snapshot is immutable, so repeated calls must agree."""
        client.post(PREDICT_URL, json=body("EXP_STABLE", high_risk_features))
        first = client.get(explanation_url("EXP_STABLE")).json()["top_drivers"]
        second = client.get(explanation_url("EXP_STABLE")).json()["top_drivers"]
        assert [d["feature_name"] for d in first] == [d["feature_name"] for d in second]
        assert [d["shap_value"] for d in first] == [d["shap_value"] for d in second]

    def test_top_k_honoured(self, client, db_engine, high_risk_features):
        client.post(PREDICT_URL, json=body("EXP_TOPK", high_risk_features))
        assert len(client.get(explanation_url("EXP_TOPK") + "?top_k=3").json()["top_drivers"]) == 3
        # Above the 5 persisted drivers, recomputed from the snapshot.
        assert len(client.get(explanation_url("EXP_TOPK") + "?top_k=8").json()["top_drivers"]) == 8

    def test_invalid_top_k_is_422(self, client, db_engine, high_risk_features):
        client.post(PREDICT_URL, json=body("EXP_BADK", high_risk_features))
        assert client.get(explanation_url("EXP_BADK") + "?top_k=0").status_code == 422
        assert client.get(explanation_url("EXP_BADK") + "?top_k=99").status_code == 422

    def test_unknown_student_is_404(self, client, db_engine):
        response = client.get(explanation_url("DOES_NOT_EXIST"))
        assert response.status_code == 404
        assert response.json()["error"] == "student_not_found"

    def test_student_without_prediction_is_404(self, client, db_engine, high_risk_features):
        """A student can exist via seeding yet never have been scored."""
        from backend.app.services import student_service
        from sqlalchemy.orm import Session

        with Session(db_engine) as session:
            student_service.upsert_student(
                session, student_id="EXP_NOPRED", features=high_risk_features
            )
            session.commit()

        response = client.get(explanation_url("EXP_NOPRED"))
        assert response.status_code == 404
        assert response.json()["error"] == "no_prediction_history"

    def test_no_traceback_leaked_on_404(self, client, db_engine):
        raw = client.get(explanation_url("NOPE")).text.lower()
        for marker in ("traceback", "site-packages", ".py\", line"):
            assert marker not in raw
