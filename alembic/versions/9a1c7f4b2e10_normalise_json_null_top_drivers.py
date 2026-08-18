"""normalise JSON 'null' top_drivers to SQL NULL

SQLAlchemy maps Python None on a JSON/JSONB column to the JSON scalar `null` unless
`none_as_null=True` is set. Every prediction written without SHAP drivers therefore stored
`'null'::jsonb` instead of SQL NULL, which meant:

  * `WHERE top_drivers IS NULL` matched zero rows, and
  * `jsonb_array_length(top_drivers)` raised "cannot get array length of a scalar".

The model now sets none_as_null=True; this backfills the rows already written.

Revision ID: 9a1c7f4b2e10
Revises: 82d34e5dd4e7
Create Date: 2026-08-18 20:30:00.000000
"""

from typing import Sequence, Union

from alembic import op

revision: str = "9a1c7f4b2e10"
down_revision: Union[str, None] = "82d34e5dd4e7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("UPDATE predictions SET top_drivers = NULL WHERE top_drivers = 'null'::jsonb")


def downgrade() -> None:
    # Restore the previous (incorrect but original) representation so the migration is
    # reversible without data loss.
    op.execute("UPDATE predictions SET top_drivers = 'null'::jsonb WHERE top_drivers IS NULL")
