"""
Mentor queue endpoint.

    GET /api/v1/mentors/queue

Ranked by `ml.intervention.engine.build_prioritized_mentor_queue` over each student's
LATEST prediction only — historical predictions never contribute a second row.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.schemas.mentor_queue import MentorFiltersResponse, MentorQueueResponse
from backend.app.services import mentor_queue_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mentors")


@router.get(
    "/queue",
    response_model=MentorQueueResponse,
    summary="Prioritised support worklist, highest estimated risk first",
)
def get_mentor_queue(
    department: Optional[str] = Query(default=None, description="Case-insensitive exact match."),
    risk_tier: Optional[str] = Query(default=None, pattern="^(Low|Medium|High)$"),
    assigned_mentor_id: Optional[str] = Query(default=None, description="Filter to one mentor."),
    search: Optional[str] = Query(default=None, max_length=64, description="Case-insensitive search across student ID and name."),
    cursor: Optional[str] = Query(default=None, description="Keyset pagination cursor."),
    limit: int = Query(default=25, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> MentorQueueResponse:
    """
    `priority_rank` is cohort-wide across the filtered set, so paging through the queue
    yields a continuous 1..N ordering rather than restarting each page.

    This is a support triage list. It orders outreach, not punishment.
    """
    items, total, next_cursor = mentor_queue_service.build_queue(
        db,
        department=department,
        risk_tier=risk_tier,
        assigned_mentor_id=assigned_mentor_id,
        search=search,
        cursor=cursor,
        limit=limit,
        offset=offset,
        return_cursor=True,
    )
    return MentorQueueResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        cursor=cursor,
        next_cursor=next_cursor,
        department=department,
        risk_tier=risk_tier,
        assigned_mentor_id=assigned_mentor_id,
        search=search,
    )


@router.get(
    "/filters",
    response_model=MentorFiltersResponse,
    summary="Distinct filter values for mentor triage worklist",
)
def get_mentor_filters(db: Session = Depends(get_db)) -> MentorFiltersResponse:
    """Return distinct, non-null, sorted departments and mentor IDs."""
    filters = mentor_queue_service.get_filters(db)
    return MentorFiltersResponse(
        departments=filters["departments"],
        mentor_ids=filters["mentor_ids"],
    )

