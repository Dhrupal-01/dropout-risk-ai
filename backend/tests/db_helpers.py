"""
Database helpers shared by backend/tests and verification.

Importing this module has no side effects (unlike backend/tests/conftest.py, which rewrites
DATABASE_URL and redirects ml artifact paths to a temp dir at import time).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def alembic_config_for(url: str):
    """Alembic config pointed at an explicit database."""
    from alembic.config import Config

    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", url)
    return config


def _reset_schema(engine) -> None:
    """Drop every table plus Alembic's bookkeeping, leaving a clean slate."""
    from sqlalchemy import text

    import backend.app.models  # noqa: F401 — populate metadata
    from backend.app.db.base import Base

    Base.metadata.drop_all(engine)
    with engine.begin() as connection:
        connection.execute(text("DROP TABLE IF EXISTS alembic_version"))
