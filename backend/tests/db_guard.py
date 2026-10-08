"""
Guard against running a test session that drops tables on the application database.

The DB-backed suites (backend/tests, verification) point the app at TEST_DATABASE_URL and drop
tables. If TEST_DATABASE_URL names the same database as the app's DATABASE_URL (from .env, or the
process environment), the session is aborted before any engine is created.

Two URLs name the same database when, after dropping query parameters and normalising the driver
prefix (postgres://, postgresql://, postgresql+psycopg:// ...), they share host and database name.
Neon pooled hosts ("<endpoint>-pooler.<region>...") are treated as the same host as the direct
endpoint. SQLite URLs compare by resolved file path. Messages never include credentials.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import unquote, urlsplit

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILES_OVERRIDE = "DROPOUTGUARD_DB_GUARD_ENV_FILES"
# The app's DATABASE_URL as seen by the first guard call. backend/tests/conftest.py later overwrites
# DATABASE_URL with TEST_DATABASE_URL, so later guard calls (verification/conftest.py in the same
# session, or subprocesses that inherit the environment) must compare against this recorded value.
ORIGINAL_DATABASE_URL_ENV = "DROPOUTGUARD_APP_DATABASE_URL"


def database_identity(url: str) -> Tuple[str, ...]:
    """(family, host, database) for server databases; (family, path) for SQLite."""
    parts = urlsplit(url.strip())
    family = parts.scheme.split("+", 1)[0].lower()
    if family == "postgres":
        family = "postgresql"

    if family == "sqlite":
        raw = unquote(parts.path)
        raw = raw[1:] if raw.startswith("/") else raw  # sqlite:///rel.db -> rel.db; sqlite:////abs -> /abs
        if raw in ("", ":memory:"):
            return ("sqlite", ":memory:")
        path = Path(raw)
        return ("sqlite", str((path if path.is_absolute() else PROJECT_ROOT / path).resolve()))

    host = (parts.hostname or "").lower()
    first, _, rest = host.partition(".")
    if first.endswith("-pooler"):
        host = first[: -len("-pooler")] + ("." + rest if rest else "")
    database = unquote(parts.path.lstrip("/"))
    return (family, host, database)


def _describe(identity: Tuple[str, ...]) -> str:
    if identity[0] == "sqlite":
        return f"sqlite file {identity[1]}"
    return f"{identity[0]} host {identity[1]!r}, database {identity[2]!r}"


def env_file_paths() -> List[Path]:
    override = os.environ.get(ENV_FILES_OVERRIDE)
    if override:
        return [Path(p) for p in override.split(os.pathsep) if p]
    return [PROJECT_ROOT / ".env", PROJECT_ROOT / "backend" / ".env"]


def find_app_database_conflict(test_url: Optional[str], env_files: List[Path]) -> Optional[str]:
    """Error message if test_url names the app database, else None. Never echoes credentials."""
    if not test_url:
        return None
    test_identity = database_identity(test_url)
    if test_identity == ("sqlite", ":memory:"):
        return None

    candidates = []
    for env_file in env_files:
        if env_file.exists():
            value = dotenv_values(env_file).get("DATABASE_URL")
            if value:
                candidates.append((f"{env_file} DATABASE_URL", value))
    app_env_url = os.environ.get(ORIGINAL_DATABASE_URL_ENV, os.environ.get("DATABASE_URL"))
    if app_env_url:
        candidates.append(("environment DATABASE_URL", app_env_url))

    for source, app_url in candidates:
        if database_identity(app_url) == test_identity:
            return (
                f"Refusing to run tests: TEST_DATABASE_URL names the application database "
                f"({_describe(test_identity)}, same as {source}). The DB-backed tests drop tables. "
                "Point TEST_DATABASE_URL at a throwaway database."
            )
    return None


def abort_if_test_db_is_app_db() -> None:
    """Called from conftest import, before any engine is created."""
    if ORIGINAL_DATABASE_URL_ENV not in os.environ:
        os.environ[ORIGINAL_DATABASE_URL_ENV] = os.environ.get("DATABASE_URL", "")
    message = find_app_database_conflict(os.environ.get("TEST_DATABASE_URL"), env_file_paths())
    if message:
        import pytest

        pytest.exit(message, returncode=pytest.ExitCode.USAGE_ERROR)
