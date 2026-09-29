"""
Aggregate v1 router.

`interventions.student_router` is mounted separately so the student-scoped intervention
paths match the handover spec (/students/{id}/interventions) while the log and catalog
routes stay under /interventions.
"""

from fastapi import APIRouter

from backend.app.api.v1.endpoints import (
    explain,
    geo_analytics,
    interventions,
    mentors,
    predict,
    stats,
    universal,
)

api_router = APIRouter()

api_router.include_router(predict.router, tags=["predictions"])
api_router.include_router(universal.router, tags=["universal-prediction"])
api_router.include_router(geo_analytics.router, tags=["geo-analytics"])
api_router.include_router(explain.router, tags=["explanations"])
api_router.include_router(interventions.student_router, tags=["interventions"])
api_router.include_router(interventions.router, tags=["interventions"])
api_router.include_router(mentors.router, tags=["mentors"])
api_router.include_router(stats.router, tags=["stats"])
