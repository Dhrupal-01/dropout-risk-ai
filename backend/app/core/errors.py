"""
Centralised API error handling.

Contract:
  404  student not found
  422  invalid feature payload (Pydantic validation or contract violation)
  503  ML artifacts unavailable
  500  unexpected internal error

Tracebacks are NEVER returned to clients. Every 500 logs the full exception server-side
with a correlation id that is echoed to the caller, so a support request can be tied to
a specific log line without leaking internals.
"""

import logging
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.services.intervention_service import InvalidLifecycleTransition
from backend.app.services.ml_service import MLArtifactsNotLoaded

logger = logging.getLogger(__name__)


class StudentNotFoundError(Exception):
    """Raised when a student_id has no record."""

    def __init__(self, student_id: str) -> None:
        self.student_id = student_id
        super().__init__(f"Student '{student_id}' not found")


class NoPredictionError(Exception):
    """Raised when a student exists but has never been scored."""

    def __init__(self, student_id: str) -> None:
        self.student_id = student_id
        super().__init__(f"Student '{student_id}' has no prediction history")


class FeatureContractError(Exception):
    """Raised when a payload violates the 37-feature contract."""


def _body(error: str, message: str, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"error": error, "message": message}
    payload.update({k: v for k, v in extra.items() if v is not None})
    return payload


def register_exception_handlers(app: FastAPI) -> None:
    """Attach every handler. Called once from main."""

    @app.exception_handler(StudentNotFoundError)
    async def _student_not_found(request: Request, exc: StudentNotFoundError):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=_body("student_not_found", str(exc), student_id=exc.student_id),
        )

    @app.exception_handler(NoPredictionError)
    async def _no_prediction(request: Request, exc: NoPredictionError):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=_body(
                "no_prediction_history",
                f"{exc} — score the student via POST /api/v1/predict first.",
                student_id=exc.student_id,
            ),
        )

    @app.exception_handler(MLArtifactsNotLoaded)
    async def _model_unavailable(request: Request, exc: MLArtifactsNotLoaded):
        # The reason is operational (missing artifact path), not sensitive, and is what
        # an operator needs to fix the deployment.
        logger.error("Inference attempted without ML artifacts: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_body(
                "model_unavailable",
                "The prediction model is not loaded. Check /health.",
                detail=str(exc),
            ),
            headers={"Retry-After": "30"},
        )

    @app.exception_handler(InvalidLifecycleTransition)
    async def _invalid_transition(request: Request, exc: InvalidLifecycleTransition):
        """
        409, not 422: the payload is well formed but conflicts with the record's current
        state. The client needs to re-read the log, not fix its JSON.
        """
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content=_body(
                "invalid_lifecycle_transition",
                str(exc),
                current_status=exc.current,
                requested_status=exc.requested,
            ),
        )

    @app.exception_handler(FeatureContractError)
    async def _feature_contract(request: Request, exc: FeatureContractError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_body("invalid_feature_payload", str(exc)),
        )

    @app.exception_handler(ValueError)
    async def _value_error(request: Request, exc: ValueError):
        """
        Contract violations raised by MLService surface as 422, not 500 — they are caused
        by the request, not by a server fault.
        """
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_body("invalid_feature_payload", str(exc)),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_body(
                "validation_error",
                "Request payload failed validation.",
                details=_safe_validation_details(exc),
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=_body("http_error", str(exc.detail)),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        """Last resort. Full detail to the log, an opaque id to the client."""
        incident_id = uuid.uuid4().hex[:12]
        logger.exception(
            "Unhandled error [%s] on %s %s", incident_id, request.method, request.url.path
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_body(
                "internal_error",
                "An unexpected internal error occurred. Quote the incident id to support.",
                incident_id=incident_id,
            ),
        )


def _safe_validation_details(exc: RequestValidationError) -> list:
    """
    Field-level validation feedback without echoing submitted values.

    Pydantic includes the offending input in `ctx`/`input`; echoing it back would reflect
    student data into logs and error surfaces, so only location and reason are returned.
    """
    details = []
    for error in exc.errors():
        location = [str(part) for part in error.get("loc", []) if part != "body"]
        details.append(
            {
                "field": ".".join(location) or "body",
                "type": error.get("type", "value_error"),
                "message": error.get("msg", "invalid value"),
            }
        )
    return details


def optional_str(value: Optional[str]) -> Optional[str]:
    return value or None
