"""
API tests for POST /api/v1/students/{student_id}/rescore.

The endpoint takes no body and scores the student's stored raw record (`students.features`), so
these tests change that record directly in the database and check the new prediction follows it.
"""

import json
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from backend.tests.conftest import TEST_ADMIN_TOKEN, requires_db

PREDICT_URL = "/api/v1/predict"


def rescore_url(student_id):
    return f"/api/v1/students/{student_id}/rescore"


@pytest.fixture(autouse=True)
def clean_tables(db_engine):
    with db_engine.begin() as conn:
        conn.execute(text("TRUNCATE latest_predictions, intervention_logs, predictions, students CASCADE;"))


@pytest.fixture(scope="module")
def anon(client):
    """No Authorization header; reuses the app the shared `client` started (see test_auth.py)."""
    from backend.app.main import app

    return TestClient(app)


@pytest.fixture
def low_risk_features(sample_raw_features):
    """A different, valid raw record (low risk), in the canonical stored form."""
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


def set_stored_features(db_engine, student_id, features):
    with db_engine.begin() as conn:
        conn.execute(
            text("UPDATE students SET features = CAST(:features AS jsonb) WHERE student_id = :sid"),
            {"features": json.dumps(features), "sid": student_id},
        )


def insert_unscored_student(db_engine, student_id, features):
    with db_engine.begin() as conn:
        conn.execute(
            text("INSERT INTO students (id, student_id, features) VALUES (:id, :sid, CAST(:features AS jsonb))"),
            {"id": uuid.uuid4(), "sid": student_id, "features": json.dumps(features)},
        )


def prediction_rows(db_engine, student_id):
    with db_engine.connect() as conn:
        return conn.execute(
            text(
                "SELECT p.id, p.calibrated_risk_probability, p.input_features FROM predictions p "
                "JOIN students s ON s.id = p.student_id WHERE s.student_id = :sid ORDER BY p.evaluated_at"
            ),
            {"sid": student_id},
        ).all()


@requires_db
class TestRescoreUsesStoredFeatures:
    def test_scores_the_stored_record_not_the_last_snapshot(self, client, db_engine, sample_raw_features, low_risk_features):
        assert client.post(PREDICT_URL, json={"student_id": "RS_1", "features": sample_raw_features}).status_code == 201
        set_stored_features(db_engine, "RS_1", low_risk_features)

        response = client.post(rescore_url("RS_1"))
        assert response.status_code == 201, response.text
        body = response.json()

        rows = prediction_rows(db_engine, "RS_1")
        assert len(rows) == 2  # appended, never overwritten
        first, second = rows
        assert str(second.id) == body["prediction_id"]
        for column in ("attendance_percentage", "current_cgpa", "fee_payment_delay_days", "backlog_count"):
            assert second.input_features[column] == low_risk_features[column]
            assert first.input_features[column] == sample_raw_features[column]

        # The same record scored through /predict gives the same probability.
        reference = client.post(PREDICT_URL, json={"student_id": "RS_REF", "features": low_risk_features})
        assert reference.status_code == 201
        assert body["calibrated_risk_probability"] == reference.json()["calibrated_risk_probability"]
        assert body["risk_tier"] == reference.json()["risk_tier"]

    def test_latest_prediction_points_at_the_rescore(self, client, db_engine, sample_raw_features):
        client.post(PREDICT_URL, json={"student_id": "RS_2", "features": sample_raw_features, "department": "Civil Engineering"})
        body = client.post(rescore_url("RS_2")).json()
        with db_engine.connect() as conn:
            latest = conn.execute(
                text(
                    "SELECT lp.prediction_id, lp.department FROM latest_predictions lp "
                    "JOIN students s ON s.id = lp.student_id WHERE s.student_id = 'RS_2'"
                )
            ).one()
        assert str(latest.prediction_id) == body["prediction_id"]
        assert latest.department == "Civil Engineering"  # metadata kept, not cleared

    def test_scores_a_student_who_was_never_scored(self, client, db_engine, sample_raw_features):
        insert_unscored_student(db_engine, "RS_NEW", sample_raw_features)
        assert client.get("/api/v1/students/RS_NEW/explanation").status_code == 404
        response = client.post(rescore_url("RS_NEW"))
        assert response.status_code == 201, response.text
        assert len(prediction_rows(db_engine, "RS_NEW")) == 1
        assert client.get("/api/v1/students/RS_NEW/explanation").status_code == 200

    def test_a_legacy_age_key_is_ignored_not_scored(self, client, db_engine, sample_raw_features):
        insert_unscored_student(db_engine, "RS_AGE", dict(sample_raw_features, age=20))
        response = client.post(rescore_url("RS_AGE"))
        assert response.status_code == 201, response.text
        (row,) = prediction_rows(db_engine, "RS_AGE")
        assert "age" not in row.input_features


@requires_db
class TestRescoreErrors:
    def test_unknown_student_is_404(self, client):
        response = client.post(rescore_url("RS_MISSING"))
        assert response.status_code == 404
        assert response.json()["error"] == "student_not_found"

    def test_empty_stored_record_is_422_and_writes_nothing(self, client, db_engine):
        insert_unscored_student(db_engine, "RS_EMPTY", {})
        response = client.post(rescore_url("RS_EMPTY"))
        assert response.status_code == 422
        assert response.json()["error"] == "invalid_feature_payload"
        assert prediction_rows(db_engine, "RS_EMPTY") == []

    def test_record_that_fails_the_contract_is_422_and_writes_nothing(self, client, db_engine, sample_raw_features):
        incomplete = {k: v for k, v in sample_raw_features.items() if k != "attendance_percentage"}
        insert_unscored_student(db_engine, "RS_BAD", incomplete)
        response = client.post(rescore_url("RS_BAD"))
        assert response.status_code == 422
        assert response.json()["error"] == "invalid_feature_payload"
        assert prediction_rows(db_engine, "RS_BAD") == []


@requires_db
class TestRescoreNeedsTheAdminToken:
    def test_missing_token_is_401_and_writes_nothing(self, anon, db_engine, sample_raw_features):
        insert_unscored_student(db_engine, "RS_AUTH", sample_raw_features)
        response = anon.post(rescore_url("RS_AUTH"))
        assert response.status_code == 401
        assert response.json()["error"] == "unauthorized"
        assert prediction_rows(db_engine, "RS_AUTH") == []

    def test_wrong_token_is_401(self, anon, db_engine, sample_raw_features):
        insert_unscored_student(db_engine, "RS_AUTH", sample_raw_features)
        response = anon.post(rescore_url("RS_AUTH"), headers={"Authorization": f"Bearer {TEST_ADMIN_TOKEN}x"})
        assert response.status_code == 401
        assert prediction_rows(db_engine, "RS_AUTH") == []

    def test_right_token_scores(self, anon, db_engine, sample_raw_features):
        insert_unscored_student(db_engine, "RS_AUTH", sample_raw_features)
        response = anon.post(rescore_url("RS_AUTH"), headers={"Authorization": f"Bearer {TEST_ADMIN_TOKEN}"})
        assert response.status_code == 201, response.text


@requires_db
class TestExplanation404SaysWhetherRescoreCanRun:
    def test_stored_features_present(self, client, db_engine, sample_raw_features):
        insert_unscored_student(db_engine, "RS_HAS", sample_raw_features)
        body = client.get("/api/v1/students/RS_HAS/explanation").json()
        assert body["error"] == "no_prediction_history"
        assert body["has_stored_features"] is True

    def test_stored_features_absent(self, client, db_engine):
        insert_unscored_student(db_engine, "RS_NONE", {})
        body = client.get("/api/v1/students/RS_NONE/explanation").json()
        assert body["error"] == "no_prediction_history"
        assert body["has_stored_features"] is False
