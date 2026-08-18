"""
Alembic environment.

The database URL comes from backend settings (environment / .env), never from alembic.ini,
so credentials are never committed.
"""

import uuid
import datetime
import sqlite3
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool, event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.postgresql import JSONB

from backend.app.core.config import get_settings
from backend.app.db.base import Base

# Custom compiler mapping JSONB to JSON on SQLite during migrations
@compiles(JSONB, 'sqlite')
def compile_jsonb_sqlite(element, compiler, **kw):
    return "JSON"

# Register SQLite connection functions for PostgreSQL mocks during DDL executions
@event.listens_for(Engine, "connect")
def register_sqlite_functions(dbapi_connection, connection_record):
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
        dbapi_connection.create_function("gen_random_uuid", 0, lambda: str(uuid.uuid4()))
        dbapi_connection.create_function("clock_timestamp", 0, lambda: datetime.datetime.now().isoformat())
        dbapi_connection.create_function("now", 0, lambda: datetime.datetime.now().isoformat())

# Populates Base.metadata with every table. Required for autogenerate.
import backend.app.models  # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Prefer the direct (non-pooled) Neon endpoint for DDL. Tests override this explicitly.
if not config.get_main_option("sqlalchemy.url", None):
    config.set_main_option("sqlalchemy.url", get_settings().ddl_database_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
