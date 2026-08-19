"""
ORM model definitions, constraints and indexes.

Schema-shape assertions run without a database. Anything needing a live connection is
guarded by `requires_db` (set TEST_DATABASE_URL to enable).
"""

import uuid
from datetime import date

import pytest
from sqlalchemy import inspect

import backend.app.models  # noqa: F401 — populate metadata
from backend.app.db.base import Base
from backend.app.models.intervention_log import (
    INTERVENTION_OUTCOME_STATUSES,
    INTERVENTION_STATUSES,
    InterventionLog,
)
from backend.app.models.prediction import Prediction
from backend.app.models.student import Student
from backend.app.services.ml_service import LABEL_COLUMNS
from backend.tests.conftest import requires_db


class TestSchemaShape:
    def test_all_three_tables_registered(self):
        assert set(Base.metadata.tables) == {"students", "predictions", "intervention_logs"}

    def test_students_columns(self):
        cols = {c.name: c for c in Student.__table__.columns}
        assert cols["student_id"].unique is True
        assert cols["student_id"].nullable is False
        assert cols["features"].nullable is False
        for optional in ("name", "department", "assigned_mentor_id"):
            assert cols[optional].nullable is True

    def test_students_features_never_holds_labels(self):
        """Labels are columns of the CSV but must not be modelled on the students table."""
        assert not LABEL_COLUMNS.intersection({c.name for c in Student.__table__.columns})

    def test_predictions_columns(self):
        cols = {c.name: c for c in Prediction.__table__.columns}
        assert cols["calibrated_risk_probability"].nullable is False
        assert cols["risk_tier"].nullable is False
        assert cols["input_features"].nullable is False
        assert cols["top_drivers"].nullable is True
        assert cols["model_version"].nullable is True

    def test_predictions_are_protected_from_student_delete(self):
        """
        RESTRICT, not CASCADE: prediction history is an audit record and must not vanish
        as a side effect of deleting a student. Removing one now requires deliberate
        archival of their history first.
        """
        fk = list(Prediction.__table__.foreign_keys)[0]
        assert fk.column.table.name == "students"
        assert fk.ondelete == "RESTRICT"

    def test_intervention_logs_are_protected_from_student_delete(self):
        fk = list(InterventionLog.__table__.foreign_keys)[0]
        assert fk.column.table.name == "students"
        assert fk.ondelete == "RESTRICT"

    def test_student_relationships_do_not_cascade_deletes(self):
        """An ORM delete must not quietly remove history either."""
        for relationship_name in ("predictions", "intervention_logs"):
            cascade = Student.__mapper__.relationships[relationship_name].cascade
            assert "delete" not in cascade
            assert "delete-orphan" not in cascade

    def test_intervention_statuses_match_the_tracker_implementation(self):
        """
        Sourced from ml.intervention.engine.InterventionStatusTracker, not the docs.
        The tracker validates nothing, so these are the values it documents and emits.
        """
        assert INTERVENTION_STATUSES == ("ASSIGNED", "IN_PROGRESS", "APPLIED", "COMPLETED")
        assert INTERVENTION_OUTCOME_STATUSES == (
            "PENDING_EVALUATION",
            "IMPROVED",
            "NO_CHANGE",
            "DETERIORATED",
        )

    def test_deteriorated_is_included_despite_being_undocumented(self):
        """Discrepancy #1: the code produces DETERIORATED; the docstring omits it."""
        assert "DETERIORATED" in INTERVENTION_OUTCOME_STATUSES

    def test_resolved_is_excluded_because_no_code_path_emits_it(self):
        """Discrepancy #2: the docstring advertises RESOLVED; it is unreachable."""
        assert "RESOLVED" not in INTERVENTION_OUTCOME_STATUSES


class TestIndexes:
    def _index_names(self, model) -> set:
        return {ix.name for ix in model.__table__.indexes}

    def test_required_student_indexes(self):
        names = self._index_names(Student)
        assert "ix_students_student_id" in names
        assert "ix_students_department" in names

    def test_required_prediction_indexes(self):
        names = self._index_names(Prediction)
        assert "ix_predictions_student_id_evaluated_at" in names
        assert "ix_predictions_risk_tier_evaluated_at" in names

    def test_prediction_indexes_are_descending_on_time(self):
        """History queries are always 'most recent first'."""
        ix = next(
            i for i in Prediction.__table__.indexes
            if i.name == "ix_predictions_student_id_evaluated_at"
        )
        assert any("DESC" in str(expr).upper() for expr in ix.expressions)

    def test_required_intervention_indexes(self):
        names = self._index_names(InterventionLog)
        assert "ix_intervention_logs_student_id_updated_at" in names
        assert "ix_intervention_logs_faculty_status" in names


class TestOutcomeClassification:
    """Must match InterventionStatusTracker.update_intervention_status exactly."""

    @pytest.mark.parametrize(
        "baseline,post,expected",
        [
            (0.88, 0.35, "IMPROVED"),      # delta +0.53
            (0.50, 0.44, "IMPROVED"),      # delta +0.06 (just over threshold)
            (0.50, 0.46, "NO_CHANGE"),     # delta +0.04
            (0.50, 0.50, "NO_CHANGE"),
            (0.50, 0.54, "NO_CHANGE"),     # delta -0.04
            (0.35, 0.88, "DETERIORATED"),  # delta -0.53
            (None, 0.4, "PENDING_EVALUATION"),
            (0.4, None, "PENDING_EVALUATION"),
        ],
    )
    def test_thresholds(self, baseline, post, expected):
        from backend.app.services.intervention_service import classify_outcome

        assert classify_outcome(baseline, post) == expected

    def test_agrees_with_the_ml_tracker(self):
        """Cross-check the backend against the real tracker on the ML side."""
        from ml.intervention.engine import InterventionStatusTracker

        from backend.app.services.intervention_service import classify_outcome

        tracker = InterventionStatusTracker()
        record = tracker.log_intervention_assignment(
            student_id="IND_2026_9999",
            intervention_id="INT_ATT_01",
            mentor_name="Dr. Sharma",
            baseline_risk_prob=0.88,
        )
        updated = tracker.update_intervention_status(
            record_id=record["record_id"], new_status="APPLIED", post_intervention_risk_prob=0.35
        )
        assert updated["outcome_status"] == classify_outcome(0.88, 0.35) == "IMPROVED"


@requires_db
class TestLiveDatabase:
    def test_tables_exist(self, db_engine):
        tables = set(inspect(db_engine).get_table_names())
        assert {"students", "predictions", "intervention_logs"}.issubset(tables)

    def test_student_round_trip(self, db_session, sample_raw_features):
        from backend.app.services import student_service

        student = student_service.upsert_student(
            db_session,
            student_id="TEST_0001",
            features=sample_raw_features,
            name="Test Student",
            department="Computer Science & Engineering",
        )
        db_session.commit()

        assert isinstance(student.id, uuid.UUID)
        assert student.created_at is not None

        fetched = student_service.get_by_student_id(db_session, "TEST_0001")
        assert fetched.features["current_cgpa"] == sample_raw_features["current_cgpa"]

    def test_upsert_is_idempotent(self, db_session, sample_raw_features):
        from backend.app.services import student_service

        first = student_service.upsert_student(
            db_session, student_id="TEST_0002", features=sample_raw_features
        )
        db_session.commit()
        second = student_service.upsert_student(
            db_session, student_id="TEST_0002", features=dict(sample_raw_features, backlog_count=5)
        )
        db_session.commit()

        assert first.id == second.id
        assert second.features["backlog_count"] == 5

    def test_storing_a_label_is_refused(self, db_session, sample_raw_features):
        from backend.app.services import student_service

        with pytest.raises(ValueError, match="label column"):
            student_service.upsert_student(
                db_session,
                student_id="TEST_0003",
                features=dict(sample_raw_features, is_dropout=1),
            )

    def test_prediction_history_is_append_only(self, db_session, sample_raw_features):
        from backend.app.services import student_service

        student = student_service.upsert_student(
            db_session, student_id="TEST_0004", features=sample_raw_features
        )
        for probability, tier in ((0.91, "High"), (0.42, "Medium")):
            student_service.record_prediction(
                db_session,
                student=student,
                calibrated_risk_probability=probability,
                risk_tier=tier,
                risk_score_percentage=probability * 100,
                input_features=sample_raw_features,
                model_version="test-version",
            )
        db_session.commit()

        assert len(student_service.list_students(db_session)) >= 1
        latest = student_service.latest_prediction(db_session, student)
        assert latest.calibrated_risk_probability == 0.42
        assert db_session.query(Prediction).filter_by(student_id=student.id).count() == 2

    def test_invalid_risk_tier_violates_check_constraint(self, db_session, sample_raw_features):
        from sqlalchemy.exc import IntegrityError

        from backend.app.services import student_service

        student = student_service.upsert_student(
            db_session, student_id="TEST_0005", features=sample_raw_features
        )
        db_session.add(
            Prediction(
                student_id=student.id,
                calibrated_risk_probability=0.5,
                risk_tier="Critical",  # not a valid tier
                risk_score_percentage=50.0,
                input_features={},
            )
        )
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_intervention_lifecycle(self, db_session, sample_raw_features):
        from backend.app.services import intervention_service, student_service

        student = student_service.upsert_student(
            db_session, student_id="TEST_0006", features=sample_raw_features
        )
        log = intervention_service.create_log(
            db_session,
            student=student,
            intervention_id="INT_ATT_01",
            assigned_faculty_id="FAC_001",
            baseline_risk_probability=0.88,
            scheduled_followup_date=date(2026, 9, 1),
        )
        db_session.commit()
        assert log.status == "ASSIGNED"
        assert log.outcome_status == "PENDING_EVALUATION"

        intervention_service.update_log(
            db_session, log=log, status="APPLIED", post_intervention_risk_probability=0.35
        )
        db_session.commit()
        assert log.status == "APPLIED"
        assert log.outcome_status == "IMPROVED"
        assert log.risk_delta == 0.53

    def test_unknown_intervention_id_is_rejected(self, db_session, sample_raw_features):
        from backend.app.services import intervention_service, student_service

        student = student_service.upsert_student(
            db_session, student_id="TEST_0007", features=sample_raw_features
        )
        with pytest.raises(ValueError, match="Unknown intervention_id"):
            intervention_service.create_log(
                db_session, student=student, intervention_id="INT_DOES_NOT_EXIST"
            )
