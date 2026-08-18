"""protect history: student FK cascade -> restrict

Prediction and intervention rows are an audit record. Under ON DELETE CASCADE, removing a
student silently erased every score they were ever given and every support action offered
to them. RESTRICT forces that to be a deliberate, explicit archival decision instead of a
side effect.

Autogenerate emitted `op.drop_constraint(None, ...)` in downgrade(), which cannot compile.
The constraint names are therefore spelled out explicitly in both directions; they match
PostgreSQL's default `<table>_<column>_fkey` naming, which is what the initial migration
produced.

Revision ID: 82d34e5dd4e7
Revises: 0f349a75250d
Create Date: 2026-08-18 18:52:39.285552
"""

from typing import Sequence, Union

from alembic import op

revision: str = "82d34e5dd4e7"
down_revision: Union[str, None] = "0f349a75250d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

PREDICTIONS_FK = "predictions_student_id_fkey"
INTERVENTION_LOGS_FK = "intervention_logs_student_id_fkey"


def upgrade() -> None:
    op.drop_constraint(INTERVENTION_LOGS_FK, "intervention_logs", type_="foreignkey")
    op.create_foreign_key(
        INTERVENTION_LOGS_FK,
        "intervention_logs",
        "students",
        ["student_id"],
        ["id"],
        ondelete="RESTRICT",
    )

    op.drop_constraint(PREDICTIONS_FK, "predictions", type_="foreignkey")
    op.create_foreign_key(
        PREDICTIONS_FK,
        "predictions",
        "students",
        ["student_id"],
        ["id"],
        ondelete="RESTRICT",
    )


def downgrade() -> None:
    op.drop_constraint(PREDICTIONS_FK, "predictions", type_="foreignkey")
    op.create_foreign_key(
        PREDICTIONS_FK,
        "predictions",
        "students",
        ["student_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.drop_constraint(INTERVENTION_LOGS_FK, "intervention_logs", type_="foreignkey")
    op.create_foreign_key(
        INTERVENTION_LOGS_FK,
        "intervention_logs",
        "students",
        ["student_id"],
        ["id"],
        ondelete="CASCADE",
    )
