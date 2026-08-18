"""
Aggregate v1 router.

`interventions.student_router` is mounted separately so the student-scoped intervention
paths match the handover spec (/students/{id}/interventions) while the log and catalog
routes stay under /interventions.
"""

from fastapi import APIRouter

from backend.app.api.v1.endpoints import explain, interventions, mentors, predict

api_router = APIRouter()

api_router.include_router(predict.router, tags=["predictions"])
api_router.include_router(explain.router, tags=["explanations"])
api_router.include_router(interventions.student_router, tags=["interventions"])
api_router.include_router(interventions.router, tags=["interventions"])
api_router.include_router(mentors.router, tags=["mentors"])
