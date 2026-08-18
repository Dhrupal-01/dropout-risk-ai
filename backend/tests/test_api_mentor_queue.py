"""
API tests for GET /api/v1/mentors/queue.

Central guarantees: ranked by latest prediction only, descending calibrated risk, with
filters and pagination that keep a continuous cohort-wide rank.
"""

import pytest

from backend.tests.conftest import requires_db

PREDICT_URL = "/api/v1/predict"
QUEUE_URL = "/api/v1/mentors/queue"
LOG_URL = "/api/v1/interventions/log"


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


@pytest.fixture
def seeded_queue(client, db_engine, high_risk_features, low_risk_features):
    """A small cohort spread across two departments and both risk extremes."""
    students = [
        ("Q_CSE_H1", high_risk_features, "Computer Science & Engineering", "FAC_Q1"),
        ("Q_CSE_H2", high_risk_features, "Computer Science & Engineering", "FAC_Q1"),
        ("Q_CSE_L1", low_risk_features, "Computer Science & Engineering", "FAC_Q2"),
        ("Q_MECH_H1", high_risk_features, "Mechanical Engineering", "FAC_Q2"),
        ("Q_MECH_L1", low_risk_features, "Mechanical Engineering", "FAC_Q2"),
    ]
    for student_id, features, department, mentor in students:
        client.post(
            PREDICT_URL,
            json=body(
                student_id,
                features,
                name=f"Student {student_id}",
                department=department,
                assigned_mentor_id=mentor,
            ),
        )
    return [s[0] for s in students]


@requires_db
class TestQueueOrdering:
    def test_sorted_descending_by_risk(self, client, seeded_queue):
        items = client.get(f"{QUEUE_URL}?limit=200").json()["items"]
        probabilities = [i["risk_probability"] for i in items]
        assert probabilities == sorted(probabilities, reverse=True)

    def test_priority_rank_is_contiguous_from_one(self, client, seeded_queue):
        items = client.get(f"{QUEUE_URL}?limit=200").json()["items"]
        assert [i["priority_rank"] for i in items] == list(range(1, len(items) + 1))

    def test_high_risk_students_rank_above_low_risk(self, client, seeded_queue):
        items = client.get(f"{QUEUE_URL}?limit=200").json()["items"]
        ranks = {i["student_id"]: i["priority_rank"] for i in items}
        assert ranks["Q_CSE_H1"] < ranks["Q_CSE_L1"]
        assert ranks["Q_MECH_H1"] < ranks["Q_MECH_L1"]


@requires_db
class TestLatestPredictionOnly:
    def test_rescoring_does_not_duplicate_a_student(
        self, client, db_engine, high_risk_features, low_risk_features
    ):
        """predictions is append-only; the queue must still list each student once."""
        client.post(PREDICT_URL, json=body("Q_DUP", high_risk_features, department="Civil Engineering"))
        client.post(PREDICT_URL, json=body("Q_DUP", high_risk_features, department="Civil Engineering"))
        client.post(PREDICT_URL, json=body("Q_DUP", low_risk_features, department="Civil Engineering"))

        items = client.get(f"{QUEUE_URL}?department=Civil Engineering&limit=200").json()["items"]
        matching = [i for i in items if i["student_id"] == "Q_DUP"]
        assert len(matching) == 1

    def test_uses_the_most_recent_score(self, client, db_engine, high_risk_features, low_risk_features):
        client.post(PREDICT_URL, json=body("Q_LATEST", high_risk_features, department="Civil Engineering"))
        newest = client.post(
            PREDICT_URL, json=body("Q_LATEST", low_risk_features, department="Civil Engineering")
        ).json()

        items = client.get(f"{QUEUE_URL}?department=Civil Engineering&limit=200").json()["items"]
        row = next(i for i in items if i["student_id"] == "Q_LATEST")
        assert row["risk_probability"] == newest["calibrated_risk_probability"]
        assert row["risk_tier"] == newest["risk_tier"] == "Low"


@requires_db
class TestFilters:
    def test_department_filter(self, client, seeded_queue):
        items = client.get(f"{QUEUE_URL}?department=Mechanical Engineering&limit=200").json()["items"]
        assert items
        assert all(i["department"] == "Mechanical Engineering" for i in items)

    def test_department_filter_is_case_insensitive(self, client, seeded_queue):
        lower = client.get(f"{QUEUE_URL}?department=mechanical engineering&limit=200").json()
        exact = client.get(f"{QUEUE_URL}?department=Mechanical Engineering&limit=200").json()
        assert lower["total"] == exact["total"] > 0

    def test_risk_tier_filter(self, client, seeded_queue):
        payload = client.get(f"{QUEUE_URL}?risk_tier=High&limit=200").json()
        assert payload["items"]
        assert all(i["risk_tier"] == "High" for i in payload["items"])

    def test_invalid_risk_tier_is_422(self, client, db_engine):
        assert client.get(f"{QUEUE_URL}?risk_tier=Critical").status_code == 422

    def test_mentor_filter(self, client, seeded_queue):
        items = client.get(f"{QUEUE_URL}?assigned_mentor_id=FAC_Q1&limit=200").json()["items"]
        assert items
        assert all(i["assigned_mentor_id"] == "FAC_Q1" for i in items)

    def test_combined_filters(self, client, seeded_queue):
        items = client.get(
            f"{QUEUE_URL}?department=Computer Science %26 Engineering&risk_tier=High&limit=200"
        ).json()["items"]
        for item in items:
            assert item["department"] == "Computer Science & Engineering"
            assert item["risk_tier"] == "High"

    def test_unmatched_filter_returns_empty(self, client, seeded_queue):
        payload = client.get(f"{QUEUE_URL}?department=Nonexistent Dept").json()
        assert payload["items"] == []
        assert payload["total"] == 0


@requires_db
class TestPagination:
    def test_limit_and_offset(self, client, seeded_queue):
        first = client.get(f"{QUEUE_URL}?limit=2&offset=0").json()
        second = client.get(f"{QUEUE_URL}?limit=2&offset=2").json()

        assert len(first["items"]) == 2
        assert first["total"] == second["total"]
        assert {i["student_id"] for i in first["items"]}.isdisjoint(
            {i["student_id"] for i in second["items"]}
        )

    def test_rank_continues_across_pages(self, client, seeded_queue):
        """Ranks are cohort-wide, so page 2 must not restart at 1."""
        first = client.get(f"{QUEUE_URL}?limit=2&offset=0").json()["items"]
        second = client.get(f"{QUEUE_URL}?limit=2&offset=2").json()["items"]
        assert [i["priority_rank"] for i in first] == [1, 2]
        assert [i["priority_rank"] for i in second] == [3, 4]

    def test_total_is_independent_of_page_size(self, client, seeded_queue):
        assert (
            client.get(f"{QUEUE_URL}?limit=1").json()["total"]
            == client.get(f"{QUEUE_URL}?limit=200").json()["total"]
        )

    def test_offset_past_end_is_empty(self, client, seeded_queue):
        payload = client.get(f"{QUEUE_URL}?limit=10&offset=100000").json()
        assert payload["items"] == []
        assert payload["total"] > 0

    def test_invalid_pagination_is_422(self, client, db_engine):
        assert client.get(f"{QUEUE_URL}?limit=0").status_code == 422
        assert client.get(f"{QUEUE_URL}?limit=500").status_code == 422
        assert client.get(f"{QUEUE_URL}?offset=-1").status_code == 422


@requires_db
class TestQueueFields:
    def test_frontend_ready_fields_present(self, client, seeded_queue):
        item = client.get(f"{QUEUE_URL}?limit=1").json()["items"][0]
        for field in (
            "priority_rank",
            "student_id",
            "name",
            "department",
            "risk_probability",
            "risk_tier",
            "attendance",
            "cgpa",
            "backlogs",
            "fee_delay_days",
            "primary_intervention",
            "intervention_status",
        ):
            assert field in item, f"missing field: {field}"

    def test_metadata_comes_from_persisted_student_record(self, client, seeded_queue):
        """name/department do not exist in the ML dataset; they come from the DB."""
        item = next(
            i
            for i in client.get(f"{QUEUE_URL}?limit=200").json()["items"]
            if i["student_id"] == "Q_CSE_H1"
        )
        assert item["name"] == "Student Q_CSE_H1"
        assert item["department"] == "Computer Science & Engineering"

    def test_academic_fields_match_the_prediction_snapshot(self, client, seeded_queue, high_risk_features):
        item = next(
            i
            for i in client.get(f"{QUEUE_URL}?limit=200").json()["items"]
            if i["student_id"] == "Q_CSE_H1"
        )
        assert item["attendance"] == pytest.approx(high_risk_features["attendance_percentage"])
        assert item["cgpa"] == pytest.approx(high_risk_features["current_cgpa"])
        assert item["backlogs"] == high_risk_features["backlog_count"]
        assert item["fee_delay_days"] == high_risk_features["fee_payment_delay_days"]

    def test_intervention_status_surfaced(self, client, seeded_queue):
        client.post(
            LOG_URL,
            json={
                "student_id": "Q_CSE_H1",
                "intervention_id": "INT_ATT_01",
                "assigned_faculty_id": "FAC_Q1",
            },
        )
        item = next(
            i
            for i in client.get(f"{QUEUE_URL}?limit=200").json()["items"]
            if i["student_id"] == "Q_CSE_H1"
        )
        assert item["primary_intervention"] == "INT_ATT_01"
        assert item["intervention_status"] == "ASSIGNED"

    def test_students_without_predictions_are_absent(self, client, db_engine, high_risk_features):
        from sqlalchemy.orm import Session

        from backend.app.services import student_service

        with Session(db_engine) as session:
            student_service.upsert_student(
                session,
                student_id="Q_UNSCORED",
                features=high_risk_features,
                department="Civil Engineering",
            )
            session.commit()

        items = client.get(f"{QUEUE_URL}?limit=200").json()["items"]
        assert all(i["student_id"] != "Q_UNSCORED" for i in items)


@requires_db
class TestRankingSafety:
    def test_ranking_never_falls_back_to_ground_truth(self, client, db_engine, seeded_queue):
        """
        build_prioritized_mentor_queue sorts by ground_truth_risk_prob when
        calibrated_prob is absent. The service must always supply calibrated_prob.
        """
        from sqlalchemy.orm import Session

        from backend.app.services import mentor_queue_service

        with Session(db_engine) as session:
            rows, total = mentor_queue_service.build_queue(session, limit=200)

        assert total > 0
        raw = str(rows)
        assert "ground_truth_risk_prob" not in raw
        assert "is_dropout" not in raw

    def test_queue_carries_support_framing(self, client, seeded_queue):
        disclaimer = client.get(f"{QUEUE_URL}?limit=1").json()["disclaimer"].lower()
        assert "not" in disclaimer
        assert "punitive" in disclaimer or "support" in disclaimer
