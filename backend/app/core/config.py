"""
Backend configuration for DropoutGuard (Phase 4 FastAPI service).

Risk thresholds and ML artifact paths are NOT redefined here — they are re-exported
from `ml.config`, which stays the single source of truth for the ML core.
"""

from functools import lru_cache
from typing import Annotated, List, Literal, Optional

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Re-exported from the ML core so the backend can never drift from the model contract.
from ml.config import (  # noqa: F401
    ARTIFACTS_DIR,
    BASE_MODEL_PATH,
    FEATURE_NAMES_PATH,
    MODEL_ARTIFACT_PATH,
    PROCESSED_DATA_PATH,
    RANDOM_SEED,
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_LOW,
    SHAP_EXPLAINER_PATH,
    get_risk_tier,
    validate_risk_thresholds,
)

BASE_DIR = MODEL_ARTIFACT_PATH.parents[2]
INTERVENTION_CATALOG_PATH = ARTIFACTS_DIR / "interventions.json"


class Settings(BaseSettings):
    """Environment-driven settings. Secrets come from the environment, never from code."""

    model_config = SettingsConfigDict(
        env_file=(BASE_DIR / ".env", BASE_DIR / "backend" / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    PROJECT_NAME: str = "DropoutGuard API"
    API_V1_PREFIX: str = "/api/v1"
    ENVIRONMENT: Literal["development", "staging", "production", "test"] = "development"

    # PostgreSQL (Neon). Must be supplied via environment — no default credentials.
    DATABASE_URL: str = Field(
        ...,
        description="SQLAlchemy PostgreSQL URL, e.g. postgresql+psycopg://user:pw@host/db?sslmode=require",
    )

    # Optional direct (non-pooled) endpoint used for DDL. Neon recommends running
    # migrations against the direct endpoint and serving traffic through the pooler,
    # since PgBouncer transaction pooling is a poor fit for multi-statement DDL.
    MIGRATION_DATABASE_URL: Optional[str] = None
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_RECYCLE_SECONDS: int = 280  # Neon drops idle connections; recycle before it does.
    DB_ECHO: bool = False

    # CORS: explicit origins only. React dev server + deployed frontend via env.
    # NoDecode stops pydantic-settings from JSON-decoding the env value, so the
    # comma-separated form below reaches the validator intact.
    CORS_ORIGINS: Annotated[List[str], NoDecode] = [
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5176",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
        "http://127.0.0.1:5176",
    ]
    CORS_ALLOW_CREDENTIALS: bool = True

    # Access control, first stage (owner decision): one shared admin bearer token for every write;
    # reads stay public while PUBLIC_READ_ONLY is true. Per-mentor login comes later. Do not load
    # real student data until per-mentor authentication exists.
    API_ADMIN_TOKEN: Optional[SecretStr] = None
    PUBLIC_READ_ONLY: bool = True

    # Risk tiering and seed. Same source and precedence as ml.config (env > backend/.env > .env >
    # default), so the server, the tests and the pipeline always use one value; validated below.
    RISK_THRESHOLD_LOW: float = RISK_THRESHOLD_LOW
    RISK_THRESHOLD_HIGH: float = RISK_THRESHOLD_HIGH
    RANDOM_SEED: int = RANDOM_SEED

    # CSV upload limit (owner decision: 2 MB of file bytes). The envelope allowance covers multipart
    # boundaries and part headers when the request Content-Length is checked before parsing.
    MAX_UPLOAD_BYTES: int = Field(default=2_000_000, gt=0)
    MAX_UPLOAD_ENVELOPE_BYTES: int = Field(default=65_536, ge=0)

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _split_origins(cls, v):
        """Accept either a comma-separated string (env var) or a real list (code/tests)."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("DATABASE_URL", "MIGRATION_DATABASE_URL")
    @classmethod
    def _normalise_driver(cls, v: Optional[str]) -> Optional[str]:
        """
        Pin the psycopg3 driver. Neon hands out `postgresql://...` URLs, which SQLAlchemy
        would otherwise resolve to psycopg2 (not installed).
        """
        if v is None:
            return v
        if v.startswith("postgres://"):
            v = v.replace("postgres://", "postgresql://", 1)
        if v.startswith("postgresql://"):
            v = v.replace("postgresql://", "postgresql+psycopg://", 1)
        return v

    @property
    def ddl_database_url(self) -> str:
        """URL used for migrations: the direct endpoint when configured, else the default."""
        return self.MIGRATION_DATABASE_URL or self.DATABASE_URL

    @model_validator(mode="after")
    def _risk_thresholds_valid_and_shared(self) -> "Settings":
        """0 < low < high < 1, and identical to ml.config, which does the actual tiering."""
        validate_risk_thresholds(self.RISK_THRESHOLD_LOW, self.RISK_THRESHOLD_HIGH)
        shared = {
            "RISK_THRESHOLD_LOW": RISK_THRESHOLD_LOW,
            "RISK_THRESHOLD_HIGH": RISK_THRESHOLD_HIGH,
            "RANDOM_SEED": RANDOM_SEED,
        }
        mismatched = {k: (getattr(self, k), v) for k, v in shared.items() if getattr(self, k) != v}
        if mismatched:
            raise ValueError(
                f"Settings disagree with ml.config (settings, ml.config): {mismatched}. "
                "Set these only via the environment or .env so both read the same value."
            )
        return self

    @field_validator("API_ADMIN_TOKEN", mode="before")
    @classmethod
    def _blank_token_is_unset(cls, v):
        return None if v is None or (isinstance(v, str) and not v.strip()) else v

    @model_validator(mode="after")
    def _production_requires_admin_token(self) -> "Settings":
        """A deployed API must not accept writes without a configured admin token."""
        if self.ENVIRONMENT == "production" and self.API_ADMIN_TOKEN is None:
            raise ValueError("ENVIRONMENT=production requires API_ADMIN_TOKEN (writes need a bearer token).")
        return self

    @model_validator(mode="after")
    def _production_requires_postgres(self) -> "Settings":
        """
        Deployed environments must use the managed PostgreSQL instance. A local SQLite
        file on an ephemeral container filesystem silently loses every prediction and
        intervention log on restart, so it is rejected outright rather than allowed to
        look like it works.
        """
        if self.ENVIRONMENT == "production" and not self.DATABASE_URL.startswith("postgresql"):
            raise ValueError(
                "ENVIRONMENT=production requires a PostgreSQL DATABASE_URL "
                f"(got '{self.DATABASE_URL.split(':', 1)[0]}://...'). "
                "File-backed databases do not survive a container restart."
            )
        return self

    @model_validator(mode="after")
    def _no_wildcard_with_credentials(self) -> "Settings":
        """
        A wildcard origin combined with credentialed requests is rejected by browsers and
        is an auth-leak footgun. If someone sets '*', credentials are force-disabled.
        """
        if "*" in self.CORS_ORIGINS and self.CORS_ALLOW_CREDENTIALS:
            self.CORS_ALLOW_CREDENTIALS = False
        return self

    @property
    def is_neon(self) -> bool:
        return "neon.tech" in self.DATABASE_URL

    def safe_database_summary(self) -> str:
        """Host/db only — never the password. Used in logs and diagnostics."""
        try:
            from sqlalchemy.engine import make_url

            url = make_url(self.DATABASE_URL)
            return f"{url.drivername}://{url.host}/{url.database}"
        except Exception:
            return "unparsable-database-url"


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so the .env file is parsed once per process."""
    return Settings()  # type: ignore[call-arg]
