"""
DropoutGuard FastAPI application.

Startup loads the expensive ML artifacts exactly once via the lifespan handler; request
handlers reuse that single in-memory instance and never touch joblib.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.v1.router import api_router
from backend.app.core.config import get_settings
from backend.app.core.errors import register_exception_handlers
from backend.app.db.session import check_database_connection
from backend.app.schemas.health import HealthResponse
from backend.app.services.ml_service import ml_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artifacts once at startup; release them at shutdown."""
    logger.info("Starting %s (%s)", settings.PROJECT_NAME, settings.ENVIRONMENT)
    logger.info("Database target: %s", settings.safe_database_summary())

    if ml_service.load():
        logger.info(
            "ML artifacts ready (version=%s, features=%d)",
            ml_service.model_version,
            len(ml_service.feature_names),
        )
    else:
        # Deliberately non-fatal: the service still starts so /health can report the
        # failure, rather than crash-looping with no diagnostics.
        logger.error("Starting WITHOUT ML artifacts: %s", ml_service.load_error)

    yield

    ml_service.unload()
    logger.info("Shutdown complete")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI-powered academic dropout prediction and intervention system (SIH 2026, PSID 7-L).",
    version="0.1.0",
    lifespan=lifespan,
)

# Explicit origins from the environment — never a wildcard alongside credentials.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# 404 / 422 / 503 / 500 handling. Tracebacks are logged, never returned.
register_exception_handlers(app)

app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """
    Liveness and readiness probe.

    Reports database reachability and ML artifact state. Exposes no credentials — only the
    driver/host/database summary is ever logged, and never returned here.
    """
    db_ok = check_database_connection()
    model_ok = ml_service.is_loaded

    detail = None
    if not model_ok:
        detail = ml_service.load_error
    elif not db_ok:
        detail = "Database connection failed"

    return HealthResponse(
        status="healthy" if (db_ok and model_ok) else "degraded",
        database="connected" if db_ok else "disconnected",
        model_loaded=model_ok,
        environment=settings.ENVIRONMENT,
        model_version=ml_service.model_version,
        feature_count=len(ml_service.feature_names) or None,
        detail=detail,
    )


@app.get("/", tags=["system"])
def root() -> dict:
    return {
        "service": settings.PROJECT_NAME,
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    # Deployment entrypoint: `python -m backend.app.main`.
    # Hosting platforms (Render, Railway, Fly, Heroku) inject the port to bind via $PORT
    # and route external traffic to it; binding a hardcoded port fails health checks.
    # 0.0.0.0 is required inside a container - 127.0.0.1 is unreachable from outside it.
    import os

    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),  # noqa: S104 - containers must bind all interfaces
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
