"""
API tests for the intervention endpoints.

    GET  /api/v1/students/{id}/interventions
    POST /api/v1/interventions/log
    GET  /api/v1/interventions/catalog
"""

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.models.intervention_log import InterventionLog
from backend.app.models.student import Student
from backend.app.services import intervention_service
from backend.tests.conftest import requires_db

PREDICT_URL = "/api/v1/predict"
LOG_URL = "/api/v1/interventions/log"
CATALOG_URL = "/api/v1/interventions/catalog"


def plan_url(student_id: str) -> str:
    return f"/api/v1/students/{student_id}/interventions"


def body(student_id, features, **extra):
    return {"student_id": student_id, "features": features, **extra}


@pytest.fixture
def high_risk_features(sample_raw_features):
    return dict(sample_raw_features)


# ------------------------------------------------------------------------- catalog

class TestCatalog:
    def test_catalog_served_from_ml_artifact(self, client):
        items = client.get(CATALOG_URL).json()
        assert len(items) == 12
        assert {i["intervention_id"] for i in items} == set(
            intervention_service.load_intervention_catalog()
        )

    def test_no_competing_catalog_table_exists(self):
        """The ML artifact is the only intervention definition source."""
        import backend.app.models  # noqa: F401
        from backend.app.db.base import Base

        assert set(Base.metadata.tables) == {"students", "predictions", "intervention_logs"}
        assert "interventions" not in Base.metadata.tables


# -------------------------------------------------------------- recommendations

@requires_db
class TestStudentInterventions:
    def test_returns_recommendations_and_recourse(self, client, db_engine, high_risk_features):
        client.post(PREDICT_URL, json=body("INT_OK", high_risk_features))
        response = client.get(plan_url("INT_OK"))
        assert response.status_code == 200

        payload = response.json()
        assert payload["student_id"] == "INT_OK"
        assert payload["current_risk_tier"] == "High"
        assert 0.0 <= payload["current_risk_probability"] <= 1.0
        assert 1 <= len(payload["recommended_interventions"]) <= 3
        assert payload["counterfactual_recourse"] is not None

    def test_intervention_ids_come_from_the_artifact(self, client, db_engine, high_risk_features):
        client.post(PREDICT_URL, json=body("INT_IDS", high_risk_features))
        known = set(intervention_service.load_intervention_catalog())

        for item in client.get(plan_url("INT_IDS")).json()["recommended_interventions"]:
            assert item["intervention_id"] in known
            assert item["urgency"] in ("HIGH", "MEDIUM", "LOW")
            assert item["title"]

    def test_recommendations_trace_back_to_shap_drivers(self, client, db_engine, high_risk_features):
        """map_shap_drivers_to_interventions annotates each pick with its trigger."""
        client.post(PREDICT_URL, json=body("INT_TRACE", high_risk_features))
        payload = client.get(plan_url("INT_TRACE")).json()

        from backend.app.services.ml_service import ml_service

        for item in payload["recommended_interventions"]:
            assert "rationale" in item
            if item.get("matched_driver_feature"):
                assert item["matched_driver_feature"] in ml_service.feature_names

    def test_matches_the_persisted_prediction(self, client, db_engine, high_risk_features):
        predicted = client.post(PREDICT_URL, json=body("INT_MATCH", high_risk_features)).json()
        plan = client.get(plan_url("INT_MATCH")).json()
        assert plan["current_risk_probability"] == predicted["calibrated_risk_probability"]
        assert plan["current_risk_tier"] == predicted["risk_tier"]

    def test_counterfactual_reduces_risk_and_lists_actions(self, client, db_engine, high_risk_features):
        client.post(PREDICT_URL, json=body("INT_CF", high_risk_features))
        recourse = client.get(plan_url("INT_CF")).json()["counterfactual_recourse"]

        assert recourse["projected_risk_prob"] < recourse["current_risk_prob"]
        assert recourse["risk_reduction_pct"] > 0
        assert recourse["required_actions"]
        for action in recourse["required_actions"]:
            assert action["plain_language_action"]

    def test_counterfactual_uses_the_real_student_id(self, client, db_engine, high_risk_features):
        """The engine defaults to the literal 'STUDENT' when no id column is present."""
        client.post(PREDICT_URL, json=body("INT_CFID", high_risk_features))
        recourse = client.get(plan_url("INT_CFID")).json()["counterfactual_recourse"]
        assert recourse["student_id"] == "INT_CFID"

    def test_unknown_student_is_404(self, client, db_engine):
        response = client.get(plan_url("NO_SUCH_STUDENT"))
        assert response.status_code == 404
        assert response.json()["error"] == "student_not_found"

    def test_student_without_prediction_is_404(self, client, db_engine, high_risk_features):
        from backend.app.services import student_service

        with Session(db_engine) as session:
            student_service.upsert_student(
                session, student_id="INT_NOPRED", features=high_risk_features
            )
            session.commit()

        response = client.get(plan_url("INT_NOPRED"))
        assert response.status_code == 404
        assert response.json()["error"] == "no_prediction_history"


# ------------------------------------------------------------- responsible AI

@requires_db
class TestResponsibleAiFraming:
    def test_plan_carries_a_support_framing_disclaimer(self, client, db_engine, high_risk_features):
        client.post(PREDICT_URL, json=body("RAI_PLAN", high_risk_features))
        payload = client.get(plan_url("RAI_PLAN")).json()

        disclaimer = payload["disclaimer"].lower()
        assert "not" in disclaimer
        assert "risk estimate" in disclaimer or "decision-support" in disclaimer

    def test_recourse_is_labelled_a_projection(self, client, db_engine, high_risk_features):
        client.post(PREDICT_URL, json=body("RAI_CF", high_risk_features))
        recourse = client.get(plan_url("RAI_CF")).json()["counterfactual_recourse"]

        assert recourse["is_projection"] is True
        assert "not a causal guarantee" in recourse["disclaimer"].lower()

    def test_no_punitive_or_deterministic_language(self, client, db_engine, high_risk_features):
        """
        The API must never present outputs as verdicts or authorise punishment.
        'debarment' is permitted only as the pre-existing statutory attendance rule the
        ML layer describes -- never as an action this system takes.
        """
        client.post(PREDICT_URL, json=body("RAI_LANG", high_risk_features))
        raw = client.get(plan_url("RAI_LANG")).text.lower()

        for forbidden in (
            "will drop out",
            "will fail",
            "expel",
            "expulsion",
            "terminate enrolment",
            "punish",
            "penalis",
            "penaliz",
            "automatically debar",
        ):
            assert forbidden not in raw, f"punitive/deterministic phrasing found: {forbidden}"

    def test_queue_and_plan_never_expose_ground_truth(self, client, db_engine, high_risk_features):
        client.post(PREDICT_URL, json=body("RAI_GT", high_risk_features))
        for url in (plan_url("RAI_GT"), "/api/v1/mentors/queue"):
            raw = client.get(url).text
            assert "ground_truth_risk_prob" not in raw
            assert "is_dropout" not in raw


# ------------------------------------------------------------------ logging

@requires_db
class TestInterventionLogging:
    def test_creates_a_log(self, client, db_engine, high_risk_features):
        client.post(PREDICT_URL, json=body("LOG_NEW", high_risk_features))
        response = client.post(
            LOG_URL,
            json={
                "student_id": "LOG_NEW",
                "intervention_id": "INT_ATT_01",
                "assigned_faculty_id": "FAC_001",
                "notes": "Initial attendance counselling scheduled",
                "scheduled_followup_date": "2026-09-01",
                "baseline_risk_probability": 0.92,
            },
        )
        assert response.status_code == 201

        payload = response.json()
        assert payload["created"] is True
        assert payload["log"]["status"] == "ASSIGNED"
        assert payload["log"]["outcome_status"] == "PENDING_EVALUATION"
        # Enriched from the catalog artifact.
        assert payload["log"]["title"]
        assert payload["log"]["pillar"] == "attendance"

    def test_persisted_to_postgres(self, client, db_engine, high_risk_features):
        client.post(PREDICT_URL, json=body("LOG_PERSIST", high_risk_features))
        client.post(
            LOG_URL, json={"student_id": "LOG_PERSIST", "intervention_id": "INT_FIN_01"}
        )

        with Session(db_engine) as session:
            student = session.execute(
                select(Student).where(Student.student_id == "LOG_PERSIST")
            ).scalar_one()
            count = session.execute(
                select(func.count())
                .select_from(InterventionLog)
                .where(InterventionLog.student_id == student.id)
            ).scalar()
        assert count == 1

    def test_unknown_student_is_404(self, client, db_engine):
        response = client.post(
            LOG_URL, json={"student_id": "GHOST", "intervention_id": "INT_ATT_01"}
        )
        assert response.status_code == 404
        assert response.json()["error"] == "student_not_found"

    def test_unknown_intervention_id_is_422(self, client, db_engine, high_risk_features):
        client.post(PREDICT_URL, json=body("LOG_BADINT", high_risk_features))
        response = client.post(
            LOG_URL, json={"student_id": "LOG_BADINT", "intervention_id": "INT_NOT_REAL"}
        )
        assert response.status_code == 422
        assert "Unknown intervention_id" in response.json()["message"]

    def test_invalid_status_value_is_422(self, client, db_engine, high_risk_features):
        client.post(PREDICT_URL, json=body("LOG_BADSTATUS", high_risk_features))
        response = client.post(
            LOG_URL,
            json={
                "student_id": "LOG_BADSTATUS",
                "intervention_id": "INT_ATT_01",
                "status": "CANCELLED",  # not in the tracker's lifecycle
            },
        )
        assert response.status_code == 422


@requires_db
class TestLifecycleTransitions:
    def _assign(self, client, student_id, features, **extra):
        client.post(PREDICT_URL, json=body(student_id, features))
        return client.post(
            LOG_URL,
            json={
                "student_id": student_id,
                "intervention_id": "INT_ATT_01",
                "baseline_risk_probability": 0.90,
                **extra,
            },
        )

    def test_valid_forward_transitions(self, client, db_engine, high_risk_features):
        """ASSIGNED -> IN_PROGRESS -> APPLIED -> COMPLETED, updating one log in place."""
        self._assign(client, "LC_FWD", high_risk_features)

        for status_value in ("IN_PROGRESS", "APPLIED", "COMPLETED"):
            response = client.post(
                LOG_URL,
                json={
                    "student_id": "LC_FWD",
                    "intervention_id": "INT_ATT_01",
                    "status": status_value,
                },
            )
            assert response.status_code == 201, response.text
            payload = response.json()
            assert payload["created"] is False, "must advance the existing log, not create one"
            assert payload["log"]["status"] == status_value

    def test_backward_transition_is_409(self, client, db_engine, high_risk_features):
        self._assign(client, "LC_BACK", high_risk_features)
        client.post(
            LOG_URL,
            json={"student_id": "LC_BACK", "intervention_id": "INT_ATT_01", "status": "APPLIED"},
        )
        response = client.post(
            LOG_URL,
            json={
                "student_id": "LC_BACK",
                "intervention_id": "INT_ATT_01",
                "status": "IN_PROGRESS",
            },
        )
        assert response.status_code == 409
        payload = response.json()
        assert payload["error"] == "invalid_lifecycle_transition"
        assert payload["current_status"] == "APPLIED"
        assert payload["requested_status"] == "IN_PROGRESS"

    def test_completed_is_terminal(self, client, db_engine, high_risk_features):
        self._assign(client, "LC_TERM", high_risk_features)
        client.post(
            LOG_URL,
            json={"student_id": "LC_TERM", "intervention_id": "INT_ATT_01", "status": "COMPLETED"},
        )
        # A new assignment after closure starts a fresh cycle rather than reopening.
        response = client.post(
            LOG_URL,
            json={"student_id": "LC_TERM", "intervention_id": "INT_ATT_01", "status": "ASSIGNED"},
        )
        assert response.status_code == 201
        assert response.json()["created"] is True

    def test_invalid_transition_does_not_mutate_the_record(self, client, db_engine, high_risk_features):
        """A rejected transition must leave the stored status untouched."""
        self._assign(client, "LC_ATOMIC", high_risk_features)
        client.post(
            LOG_URL,
            json={"student_id": "LC_ATOMIC", "intervention_id": "INT_ATT_01", "status": "APPLIED"},
        )
        client.post(
            LOG_URL,
            json={
                "student_id": "LC_ATOMIC",
                "intervention_id": "INT_ATT_01",
                "status": "ASSIGNED",
                "notes": "this note must not be saved either",
            },
        )

        history = client.get("/api/v1/students/LC_ATOMIC/interventions/history").json()
        assert history[0]["status"] == "APPLIED"
        assert "must not be saved" not in (history[0]["notes"] or "")

    def test_outcome_classified_from_risk_delta(self, client, db_engine, high_risk_features):
        """Mirrors InterventionStatusTracker's +/-0.05 thresholds."""
        self._assign(client, "LC_OUTCOME", high_risk_features)
        response = client.post(
            LOG_URL,
            json={
                "student_id": "LC_OUTCOME",
                "intervention_id": "INT_ATT_01",
                "status": "APPLIED",
                "post_intervention_risk_probability": 0.30,
            },
        )
        log = response.json()["log"]
        assert log["outcome_status"] == "IMPROVED"
        assert log["risk_delta"] == pytest.approx(0.60, abs=1e-6)

    def test_created_at_preserved_across_updates(self, client, db_engine, high_risk_features):
        """Architecture (A): update in place, so the original assignment time survives."""
        first = self._assign(client, "LC_TS", high_risk_features).json()["log"]
        second = client.post(
            LOG_URL,
            json={"student_id": "LC_TS", "intervention_id": "INT_ATT_01", "status": "IN_PROGRESS"},
        ).json()["log"]

        assert second["id"] == first["id"]
        assert second["created_at"] == first["created_at"]

    def test_notes_are_appended_not_overwritten(self, client, db_engine, high_risk_features):
        self._assign(client, "LC_NOTES", high_risk_features, notes="first note")
        log = client.post(
            LOG_URL,
            json={
                "student_id": "LC_NOTES",
                "intervention_id": "INT_ATT_01",
                "status": "IN_PROGRESS",
                "notes": "second note",
            },
        ).json()["log"]
        assert "first note" in log["notes"]
        assert "second note" in log["notes"]


@requires_db
class TestHistoryProtection:
    def test_student_delete_is_blocked_while_history_exists(self, client, db_engine, high_risk_features):
        """
        ON DELETE RESTRICT: prediction and intervention history must not disappear as a
        side effect of removing a student record.
        """
        from sqlalchemy.exc import IntegrityError

        client.post(PREDICT_URL, json=body("DEL_GUARD", high_risk_features))

        with Session(db_engine) as session:
            student = session.execute(
                select(Student).where(Student.student_id == "DEL_GUARD")
            ).scalar_one()
            session.delete(student)
            with pytest.raises(IntegrityError):
                session.commit()
            session.rollback()

        with Session(db_engine) as session:
            still_there = session.execute(
                select(Student).where(Student.student_id == "DEL_GUARD")
            ).scalar_one_or_none()
        assert still_there is not None
