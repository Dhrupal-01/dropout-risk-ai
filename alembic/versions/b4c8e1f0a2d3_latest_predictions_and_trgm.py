"""create latest_predictions table and pg_trgm indexes

Revision ID: b4c8e1f0a2d3
Revises: 9a1c7f4b2e10
Create Date: 2026-09-28 20:40:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b4c8e1f0a2d3"
down_revision: Union[str, None] = "9a1c7f4b2e10"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == "sqlite"

    # 1. Create latest_predictions table
    op.create_table(
        "latest_predictions",
        sa.Column("student_id", sa.Uuid(), sa.ForeignKey("students.id", ondelete="CASCADE"), primary_key=True, nullable=False),
        sa.Column("prediction_id", sa.Uuid(), sa.ForeignKey("predictions.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("calibrated_risk_probability", sa.Float(), nullable=False),
        sa.Column("risk_tier", sa.String(length=16), nullable=False),
        sa.Column("priority_score", sa.Float(), nullable=False),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("department", sa.String(length=64), nullable=True),
        sa.Column("assigned_mentor_id", sa.String(length=64), nullable=True),
    )

    # 2. Indexes on latest_predictions
    op.create_index(
        "ix_latest_predictions_priority_score_student_id",
        "latest_predictions",
        [sa.text("priority_score DESC"), "student_id"],
    )
    op.create_index(
        "ix_latest_predictions_risk_tier_priority_score",
        "latest_predictions",
        ["risk_tier", sa.text("priority_score DESC")],
    )
    op.create_index(
        "ix_latest_predictions_department",
        "latest_predictions",
        ["department"],
    )
    op.create_index(
        "ix_latest_predictions_assigned_mentor_id",
        "latest_predictions",
        ["assigned_mentor_id"],
    )

    # 3. Backfill from newest prediction per student
    if is_sqlite:
        op.execute("""
            INSERT OR REPLACE INTO latest_predictions (
                student_id,
                prediction_id,
                calibrated_risk_probability,
                risk_tier,
                priority_score,
                evaluated_at,
                department,
                assigned_mentor_id
            )
            SELECT
                p.student_id,
                p.id AS prediction_id,
                p.calibrated_risk_probability,
                p.risk_tier,
                (
                    p.calibrated_risk_probability
                    + COALESCE(json_extract(p.input_features, '$.backlog_count'), 0.0) * 1e-6
                    + (100.0 - COALESCE(json_extract(p.input_features, '$.attendance_percentage'), 100.0)) * 1e-9
                ) AS priority_score,
                p.evaluated_at,
                s.department,
                s.assigned_mentor_id
            FROM predictions p
            JOIN students s ON s.id = p.student_id
            WHERE p.id IN (
                SELECT p2.id FROM predictions p2
                WHERE p2.student_id = p.student_id
                ORDER BY p2.evaluated_at DESC, p2.id DESC
                LIMIT 1
            );
        """)
    else:
        op.execute("""
            INSERT INTO latest_predictions (
                student_id,
                prediction_id,
                calibrated_risk_probability,
                risk_tier,
                priority_score,
                evaluated_at,
                department,
                assigned_mentor_id
            )
            SELECT DISTINCT ON (p.student_id)
                p.student_id,
                p.id AS prediction_id,
                p.calibrated_risk_probability,
                p.risk_tier,
                (
                    round(p.calibrated_risk_probability::numeric, 4)::float8
                    + COALESCE((p.input_features->>'backlog_count')::float8, 0.0) * 1e-6
                    + (100.0 - COALESCE((p.input_features->>'attendance_percentage')::float8, 100.0)) * 1e-9
                ) AS priority_score,
                p.evaluated_at,
                s.department,
                s.assigned_mentor_id
            FROM predictions p
            JOIN students s ON s.id = p.student_id
            ORDER BY p.student_id, p.evaluated_at DESC, p.id DESC
            ON CONFLICT (student_id) DO UPDATE SET
                prediction_id = EXCLUDED.prediction_id,
                calibrated_risk_probability = EXCLUDED.calibrated_risk_probability,
                risk_tier = EXCLUDED.risk_tier,
                priority_score = EXCLUDED.priority_score,
                evaluated_at = EXCLUDED.evaluated_at,
                department = EXCLUDED.department,
                assigned_mentor_id = EXCLUDED.assigned_mentor_id;
        """)

    # 5. Enable pg_trgm and add GIN trigram indexes on students (Postgres only)
    if not is_sqlite:
        op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        op.execute("CREATE INDEX IF NOT EXISTS ix_students_name_trgm ON students USING gin (name gin_trgm_ops);")
        op.execute("CREATE INDEX IF NOT EXISTS ix_students_student_id_trgm ON students USING gin (student_id gin_trgm_ops);")


def downgrade() -> None:
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == "sqlite"

    # Drop trigram indexes (Postgres only)
    if not is_sqlite:
        op.execute("DROP INDEX IF EXISTS ix_students_student_id_trgm;")
        op.execute("DROP INDEX IF EXISTS ix_students_name_trgm;")


    # Drop indexes on latest_predictions
    op.drop_index("ix_latest_predictions_assigned_mentor_id", table_name="latest_predictions")
    op.drop_index("ix_latest_predictions_department", table_name="latest_predictions")
    op.drop_index("ix_latest_predictions_risk_tier_priority_score", table_name="latest_predictions")
    op.drop_index("ix_latest_predictions_priority_score_student_id", table_name="latest_predictions")

    # Drop table
    op.drop_table("latest_predictions")
