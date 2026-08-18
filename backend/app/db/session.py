"""
SQLAlchemy engine and session management.

Tuned for Neon (hosted Postgres): `pool_pre_ping` discards connections the serverless
endpoint closed underneath us, and `pool_recycle` retires them before it can.
"""

import uuid
import datetime
import sqlite3
from typing import Iterator
from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.postgresql import JSONB

from backend.app.core.config import get_settings

_settings = get_settings()

# Custom compiler mapping JSONB to JSON on SQLite
@compiles(JSONB, 'sqlite')
def compile_jsonb_sqlite(element, compiler, **kw):
    return "JSON"

# Global SQLite default function mappings on engine connections
@event.listens_for(Engine, "connect")
def register_sqlite_functions(dbapi_connection, connection_record):
    if isinstance(dbapi_connection, sqlite3.Connection):
        # Enable foreign keys constraint enforcement in SQLite
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
        # Register PostgreSQL function mocks
        dbapi_connection.create_function("gen_random_uuid", 0, lambda: str(uuid.uuid4()))
        dbapi_connection.create_function("clock_timestamp", 0, lambda: datetime.datetime.now().isoformat())
        dbapi_connection.create_function("now", 0, lambda: datetime.datetime.now().isoformat())

is_sqlite = _settings.DATABASE_URL.startswith("sqlite://")

engine_kwargs = {
    "echo": _settings.DB_ECHO,
    "future": True,
}

if not is_sqlite:
    engine_kwargs.update({
        "pool_pre_ping": True,
        "pool_size": _settings.DB_POOL_SIZE,
        "max_overflow": _settings.DB_MAX_OVERFLOW,
        "pool_recycle": _settings.DB_POOL_RECYCLE_SECONDS,
    })

engine: Engine = create_engine(
    _settings.DATABASE_URL,
    **engine_kwargs
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_database_connection() -> bool:
    """Cheap liveness probe for /health. Never raises; returns False on any failure."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
