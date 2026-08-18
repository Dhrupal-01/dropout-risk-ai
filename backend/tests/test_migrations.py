"""
Alembic migration validity.

Structural checks run offline. The live upgrade/downgrade round-trip and the
"no pending model changes" autogenerate check require TEST_DATABASE_URL.
"""

from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory

import backend.app.models  # noqa: F401 — populate metadata
from backend.app.db.base import Base
from backend.tests.conftest import TEST_DATABASE_URL, requires_db

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"


@pytest.fixture(scope="module")
def alembic_config():
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    return config


class TestMigrationStructure:
    def test_alembic_ini_exists(self):
        assert ALEMBIC_INI.exists()

    def test_ini_contains_no_credentials(self):
        """The URL must come from the environment, never from a tracked file."""
        text = ALEMBIC_INI.read_text(encoding="utf-8")
        for marker in ("password", "@ep-", "neon.tech", "postgresql://"):
            assert marker not in text.lower()

    def test_exactly_one_head(self, alembic_config):
        """Multiple heads mean a branched history that cannot be upgraded cleanly."""
        assert len(ScriptDirectory.from_config(alembic_config).get_heads()) == 1

    def test_revisions_are_linear_and_importable(self, alembic_config):
        revisions = list(ScriptDirectory.from_config(alembic_config).walk_revisions())
        assert revisions, "no migration revisions found"
        for revision in revisions:
            assert revision.module is not None

    def test_initial_migration_creates_all_three_tables(self, alembic_config):
        script = ScriptDirectory.from_config(alembic_config)
        source = "\n".join(
            Path(rev.path).read_text(encoding="utf-8") for rev in script.walk_revisions()
        )
        for table in ("students", "predictions", "intervention_logs"):
            assert f'"{table}"' in source or f"'{table}'" in source

    def test_migration_covers_every_model_table(self, alembic_config):
        script = ScriptDirectory.from_config(alembic_config)
        source = "\n".join(
            Path(rev.path).read_text(encoding="utf-8") for rev in script.walk_revisions()
        )
        for table in Base.metadata.tables:
            assert table in source, f"table '{table}' has no migration"

    def test_required_indexes_are_in_the_migration(self, alembic_config):
        script = ScriptDirectory.from_config(alembic_config)
        source = "\n".join(
            Path(rev.path).read_text(encoding="utf-8") for rev in script.walk_revisions()
        )
        for index in (
            "ix_students_student_id",
            "ix_students_department",
            "ix_predictions_student_id_evaluated_at",
            "ix_predictions_risk_tier_evaluated_at",
            "ix_intervention_logs_student_id_updated_at",
            "ix_intervention_logs_faculty_status",
        ):
            assert index in source, f"missing index in migration: {index}"

    def test_downgrade_is_implemented(self, alembic_config):
        script = ScriptDirectory.from_config(alembic_config)
        for revision in script.walk_revisions():
            source = Path(revision.path).read_text(encoding="utf-8")
            downgrade = source.split("def downgrade()", 1)[1]
            assert "pass" not in downgrade.split("\n")[1], "downgrade() is a no-op"


@requires_db
class TestMigrationAgainstLiveDatabase:
    def test_upgrade_downgrade_round_trip(self, alembic_config):
        from alembic import command
        from sqlalchemy import create_engine, inspect

        alembic_config.set_main_option("sqlalchemy.url", TEST_DATABASE_URL)
        engine = create_engine(TEST_DATABASE_URL)

        command.upgrade(alembic_config, "head")
        tables = set(inspect(engine).get_table_names())
        assert {"students", "predictions", "intervention_logs"}.issubset(tables)

        command.downgrade(alembic_config, "base")
        remaining = set(inspect(engine).get_table_names())
        assert not {"students", "predictions", "intervention_logs"}.intersection(remaining)

        command.upgrade(alembic_config, "head")
        engine.dispose()

    def test_no_pending_schema_drift(self, alembic_config):
        """Autogenerate must produce an empty diff: models and migration agree."""
        from alembic import command
        from alembic.autogenerate import compare_metadata
        from alembic.migration import MigrationContext
        from sqlalchemy import create_engine

        alembic_config.set_main_option("sqlalchemy.url", TEST_DATABASE_URL)
        command.upgrade(alembic_config, "head")

        engine = create_engine(TEST_DATABASE_URL)
        with engine.connect() as connection:
            context = MigrationContext.configure(connection)
            diff = compare_metadata(context, Base.metadata)
        engine.dispose()

        assert not diff, f"models drifted from migration: {diff}"
