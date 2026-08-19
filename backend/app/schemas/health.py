"""Health/readiness schema. Deliberately carries no credentials or connection strings."""

from typing import Literal, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    database: Literal["connected", "disconnected"]
    model_loaded: bool

    # Diagnostics that are safe to expose publicly.
    environment: Optional[str] = None
    model_version: Optional[str] = None
    feature_count: Optional[int] = None
    detail: Optional[str] = None
