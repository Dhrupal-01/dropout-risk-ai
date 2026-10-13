"""
API tests for the Overview aggregates:

    GET /api/v1/stats/distribution
    GET /api/v1/stats/drivers
    GET /api/v1/stats/interventions
    GET /api/v1/stats/alerts

Each aggregate is checked against an independent oracle: a direct SQL read of the rows the
endpoint summarises, or (for alerts) ml.intervention.engine.rule_based_alerts itself, which is
what the student detail endpoint reports. The seeded cohort mixes high- and low-risk students so
no expected count is trivially zero or trivially everyone.
"""

import json
from collections import Counter

import pytest
from sqlalchemy import text

from backend.tests.conftest import requires_db
from ml import config as ml_config
from ml.intervention.engine import rule_based_alerts

PREDICT_URL = "/api/v1/predict"
LOG_URL = "/api/v1/interventions/log"
DISTRIBUTION_URL = "/api/v1/stats/distribution"
DRIVERS_URL = "/api/v1/stats/drivers"
INTERVENTIONS_URL = "/api/v1/stats/interventions"
ALERTS_URL = "/api/v1/stats/alerts"


@pytest.fixture(autouse=True)
def clean_tables(db_engine):
    with db_engine.begin() as conn:
        conn.execute(text("TRUNCATE latest_predictions, intervention_logs, predictions, students CASCADE;"))


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


@pytest.fixture
def seeded_cohort(client, sample_raw_features, low_risk_features):
    students = [
        ("OV_H1", sample_raw_features, "Computer Science & Engineering"),
        ("OV_H2", sample_raw_features, "Mechanical Engineering"),
        ("OV_H3", sample_raw_features, "Computer Science & Engineering"),
        ("OV_L1", low_risk_features, "Computer Science & Engineering"),
        ("OV_L2", low_risk_features, "Mechanical Engineering"),
    ]
    for sid, features, dept in students:
        res = client.post(PREDICT_URL, json={"student_id": sid, "features": features, "department": dept})
        assert res.status_code == 201, res.text
    return [sid for sid, _, _ in students]


def _latest_rows(db_engine):
    """Latest prediction per student, read directly: (student_id, tier, probability, snapshot, drivers)."""
    with db_engine.connect() as conn:
        return conn.execute(
            text(
                """
                SELECT s.student_id, lp.risk_tier, lp.calibrated_risk_probability, p.input_features, p.top_drivers
                FROM latest_predictions lp
                JOIN predictions p ON p.id = lp.prediction_id
                JOIN students s ON s.id = lp.student_id
                """
            )
        ).all()


def _set_latest_snapshot(db_engine, student_id, **changes):
    """Edit fields of a student's latest prediction snapshot (None removes the key)."""
    with db_engine.begin() as conn:
        snapshot = conn.execute(
            text(
                "SELECT p.input_features FROM predictions p JOIN latest_predictions lp ON lp.prediction_id = p.id "
                "JOIN students s ON s.id = lp.student_id WHERE s.student_id = :sid"
            ),
            {"sid": student_id},
        ).scalar_one()
        for key, value in changes.items():
            if value is None:
                snapshot.pop(key, None)
            else:
                snapshot[key] = value
        conn.execute(
            text(
                "UPDATE predictions p SET input_features = CAST(:features AS JSONB) FROM latest_predictions lp, students s "
                "WHERE lp.prediction_id = p.id AND s.id = lp.student_id AND s.student_id = :sid"
            ),
            {"features": json.dumps(snapshot), "sid": student_id},
        )


def _assert_no_student_ids(response, student_ids):
    body = response.text
    leaked = [sid for sid in student_ids if sid in body]
    assert not leaked, f"aggregate response contains student identifiers: {leaked}"


@requires_db
class TestDistribution:
    def test_empty_database_returns_zero_bins(self, client):
        res = client.get(DISTRIBUTION_URL)
        assert res.status_code == 200
        data = res.json()
        assert data["total"] == 0
        assert data["bin_count"] == 10
        assert len(data["bins"]) == 10
        assert all(b["count"] == 0 for b in data["bins"])
        assert data["bins"][0]["lower"] == 0.0 and data["bins"][-1]["upper"] == 1.0

    def test_bins_match_latest_probabilities(self, client, db_engine, seeded_cohort):
        bins = 5
        res = client.get(DISTRIBUTION_URL, params={"bins": bins})
        assert res.status_code == 200
        data = res.json()

        probabilities = [row.calibrated_risk_probability for row in _latest_rows(db_engine)]
        expected = Counter(min(int(p * bins), bins - 1) for p in probabilities)
        assert len(expected) >= 2, "seeded cohort should span more than one bin"

        assert data["total"] == len(seeded_cohort)
        assert [b["count"] for b in data["bins"]] == [expected.get(i, 0) for i in range(bins)]
        assert [(b["lower"], b["upper"]) for b in data["bins"]] == [
            (round(i / bins, 10), round((i + 1) / bins, 10)) for i in range(bins)
        ]
        _assert_no_student_ids(res, seeded_cohort)

    @pytest.mark.parametrize("bins", [1, 51])
    def test_bin_count_out_of_range_is_rejected(self, client, bins):
        res = client.get(DISTRIBUTION_URL, params={"bins": bins})
        assert res.status_code == 422
        assert res.json()["error"] == "validation_error"


@requires_db
class TestDrivers:
    @staticmethod
    def _expected(rows, tier=None):
        counts, names = Counter(), {}
        for row in rows:
            if tier and row.risk_tier != tier:
                continue
            features = {d["feature_name"] for d in (row.top_drivers or []) if d["impact_direction"] == "RISK_INCREASING"}
            counts.update(features)
            for d in row.top_drivers or []:
                names.setdefault(d["feature_name"], d["display_name"])
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(f, names[f], c) for f, c in ordered]

    def test_counts_match_stored_drivers(self, client, db_engine, seeded_cohort):
        rows = _latest_rows(db_engine)
        expected = self._expected(rows)
        assert expected, "seeded predictions should store risk-increasing drivers"

        res = client.get(DRIVERS_URL, params={"limit": 50})
        assert res.status_code == 200
        data = res.json()
        assert data["risk_tier"] is None
        assert data["students_considered"] == len(seeded_cohort)
        assert data["students_with_drivers"] == sum(1 for r in rows if r.top_drivers)
        assert [(d["feature_name"], d["display_name"], d["student_count"]) for d in data["drivers"]] == expected
        _assert_no_student_ids(res, seeded_cohort)

    def test_limit_returns_the_most_common(self, client, db_engine, seeded_cohort):
        expected = self._expected(_latest_rows(db_engine))
        res = client.get(DRIVERS_URL, params={"limit": 2})
        assert [d["feature_name"] for d in res.json()["drivers"]] == [f for f, _, _ in expected[:2]]

    def test_tier_filter(self, client, db_engine, seeded_cohort):
        rows = _latest_rows(db_engine)
        tier = "High"
        considered = sum(1 for r in rows if r.risk_tier == tier)
        assert 0 < considered < len(rows), "seeded cohort should have High and non-High students"

        res = client.get(DRIVERS_URL, params={"risk_tier": tier, "limit": 50})
        assert res.status_code == 200
        data = res.json()
        assert data["risk_tier"] == tier
        assert data["students_considered"] == considered
        assert [(d["feature_name"], d["display_name"], d["student_count"]) for d in data["drivers"]] == self._expected(rows, tier)

    def test_predictions_without_stored_drivers_are_counted_as_such(self, client, db_engine, seeded_cohort):
        with db_engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE predictions p SET top_drivers = NULL FROM latest_predictions lp, students s "
                    "WHERE lp.prediction_id = p.id AND s.id = lp.student_id AND s.student_id = :sid"
                ),
                {"sid": seeded_cohort[0]},
            )
        data = client.get(DRIVERS_URL).json()
        assert data["students_considered"] == len(seeded_cohort)
        assert data["students_with_drivers"] == len(seeded_cohort) - 1

    def test_unknown_tier_is_rejected(self, client):
        res = client.get(DRIVERS_URL, params={"risk_tier": "Critical"})
        assert res.status_code == 422


@requires_db
class TestInterventions:
    def test_empty_database_has_every_status_key(self, client):
        data = client.get(INTERVENTIONS_URL).json()
        assert data["total"] == 0 and data["open"] == 0 and data["students_with_open_interventions"] == 0
        assert data["by_status"] == {"ASSIGNED": 0, "IN_PROGRESS": 0, "APPLIED": 0, "COMPLETED": 0}
        assert data["by_outcome_status"] == {"PENDING_EVALUATION": 0, "IMPROVED": 0, "NO_CHANGE": 0, "DETERIORATED": 0}

    def test_counts_match_intervention_logs(self, client, db_engine, seeded_cohort):
        h1, h2, h3 = seeded_cohort[:3]
        log = lambda **body: client.post(LOG_URL, json=body)  # noqa: E731
        assert log(student_id=h1, intervention_id="INT_ATT_01", status="ASSIGNED").status_code == 201
        assert log(student_id=h1, intervention_id="INT_ACAD_01", status="ASSIGNED").status_code == 201
        assert log(
            student_id=h1, intervention_id="INT_ACAD_01", status="COMPLETED",
            baseline_risk_probability=0.9, post_intervention_risk_probability=0.6,
        ).status_code == 201
        assert log(student_id=h2, intervention_id="INT_ATT_01", status="IN_PROGRESS").status_code == 201
        assert log(student_id=h3, intervention_id="INT_ATT_02", status="APPLIED").status_code == 201

        with db_engine.connect() as conn:
            rows = conn.execute(text("SELECT student_id, status, outcome_status FROM intervention_logs")).all()
        by_status = Counter(r.status for r in rows)
        by_outcome = Counter(r.outcome_status for r in rows)
        open_rows = [r for r in rows if r.status != "COMPLETED"]
        assert by_status["COMPLETED"] == 1 and len(open_rows) == 3

        res = client.get(INTERVENTIONS_URL)
        assert res.status_code == 200
        data = res.json()
        assert data["total"] == len(rows)
        assert data["open"] == len(open_rows)
        assert data["students_with_open_interventions"] == len({r.student_id for r in open_rows})
        assert data["by_status"] == {s: by_status.get(s, 0) for s in ("ASSIGNED", "IN_PROGRESS", "APPLIED", "COMPLETED")}
        assert data["by_outcome_status"] == {
            s: by_outcome.get(s, 0) for s in ("PENDING_EVALUATION", "IMPROVED", "NO_CHANGE", "DETERIORATED")
        }
        _assert_no_student_ids(res, seeded_cohort)


@requires_db
class TestAlerts:
    @staticmethod
    def _expected(db_engine):
        per_student = [{a["code"] for a in rule_based_alerts(row.input_features)} for row in _latest_rows(db_engine)]
        counts = Counter(code for codes in per_student for code in codes)
        return counts, sum(1 for codes in per_student if codes), len(per_student)

    def _assert_matches_engine(self, client, db_engine):
        counts, any_alert, considered = self._expected(db_engine)
        res = client.get(ALERTS_URL)
        assert res.status_code == 200
        data = res.json()
        assert data["students_considered"] == considered
        assert data["students_with_any_alert"] == any_alert
        assert data["attendance_threshold"] == float(ml_config.ATTENDANCE_THRESHOLD)
        assert {a["code"]: a["student_count"] for a in data["alerts"]} == {
            "ATTENDANCE_BELOW_REQUIREMENT": counts.get("ATTENDANCE_BELOW_REQUIREMENT", 0),
            "ACADEMIC_CRISIS_FLAG": counts.get("ACADEMIC_CRISIS_FLAG", 0),
        }
        return res, counts

    def test_empty_database(self, client):
        data = client.get(ALERTS_URL).json()
        assert data["students_considered"] == 0 and data["students_with_any_alert"] == 0
        assert all(a["student_count"] == 0 for a in data["alerts"])

    def test_counts_match_rule_based_alerts(self, client, db_engine, seeded_cohort):
        res, counts = self._assert_matches_engine(client, db_engine)
        assert 0 < counts["ATTENDANCE_BELOW_REQUIREMENT"] < len(seeded_cohort), "cohort should mix alert and no-alert students"
        _assert_no_student_ids(res, seeded_cohort)

    def test_edge_cases_follow_the_engine(self, client, db_engine, seeded_cohort):
        threshold = float(ml_config.ATTENDANCE_THRESHOLD)
        low_1, low_2 = seeded_cohort[3], seeded_cohort[4]
        # Exactly at the threshold is not below it; the flag alone still alerts when attendance is missing.
        _set_latest_snapshot(db_engine, low_1, attendance_percentage=threshold, attendance_risk_flag=0)
        _set_latest_snapshot(db_engine, low_2, attendance_percentage=None, attendance_risk_flag=1)
        self._assert_matches_engine(client, db_engine)
