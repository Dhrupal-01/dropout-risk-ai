"""
API tests for GET /api/v1/stats/summary.
"""

import pytest
from sqlalchemy import text
from backend.tests.conftest import requires_db

PREDICT_URL = "/api/v1/predict"
STATS_URL = "/api/v1/stats/summary"


@pytest.fixture(autouse=True)
def clean_tables(db_engine):
    with db_engine.begin() as conn:
        conn.execute(text("TRUNCATE latest_predictions, intervention_logs, predictions, students CASCADE;"))


def body(student_id, features, **extra):
    return {"student_id": student_id, "features": features, **extra}


@pytest.fixture
def seeded_cohort(client, db_engine, sample_raw_features):
    high_features = dict(sample_raw_features)
    low_features = dict(
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

    students = [
        ("STAT_CSE_H1", high_features, "Computer Science & Engineering"),
        ("STAT_CSE_H2", high_features, "Computer Science & Engineering"),
        ("STAT_CSE_L1", low_features, "Computer Science & Engineering"),
        ("STAT_MECH_H1", high_features, "Mechanical Engineering"),
        ("STAT_MECH_L1", low_features, "Mechanical Engineering"),
    ]
    for sid, feat, dept in students:
        client.post(
            PREDICT_URL,
            json=body(sid, feat, name=f"Student {sid}", department=dept),
        )
    return students


@requires_db
class TestStatsSummary:
    def test_empty_database_returns_zeros(self, client, db_engine):
        res = client.get(STATS_URL)
        assert res.status_code == 200
        data = res.json()
        assert data["total"] == 0
        assert data["by_tier"] == {"High": 0, "Medium": 0, "Low": 0}
        assert data["by_department"] == {}

    def test_summary_aggregates_tiers_and_departments(self, client, seeded_cohort):
        res = client.get(STATS_URL)
        assert res.status_code == 200
        data = res.json()

        assert data["total"] == 5
        assert data["by_tier"]["High"] == 3
        assert data["by_tier"]["Low"] == 2
        assert data["by_tier"]["Medium"] == 0

        assert data["by_department"]["Computer Science & Engineering"] == 3
        assert data["by_department"]["Mechanical Engineering"] == 2

    def test_rescoring_updates_without_increasing_total(self, client, seeded_cohort, sample_raw_features):
        low_features = dict(
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
        # Rescore high risk student STAT_CSE_H1 as low risk
        client.post(
            PREDICT_URL,
            json=body("STAT_CSE_H1", low_features, department="Computer Science & Engineering"),
        )

        res = client.get(STATS_URL)
        assert res.status_code == 200
        data = res.json()

        assert data["total"] == 5  # Still 5 unique students
        assert data["by_tier"]["High"] == 2  # 3 decreased to 2
        assert data["by_tier"]["Low"] == 3  # 2 increased to 3
