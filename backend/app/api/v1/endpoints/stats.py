"""
Cohort summary statistics endpoint.

    GET /api/v1/stats/summary

Computes cohort totals, risk tier distributions, and department breakdowns
directly in PostgreSQL using indexed GROUP BY queries on latest_predictions.
"""

import logging
from typing import Dict

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.models.latest_prediction import LatestPrediction
from backend.app.schemas.stats import StatsSummaryResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stats")


@router.get(
    "/summary",
    response_model=StatsSummaryResponse,
    summary="Cohort risk tier and department summary statistics",
)
def get_stats_summary(db: Session = Depends(get_db)) -> StatsSummaryResponse:
    """
    Returns aggregate counts for all active students with a prediction:
      - total: overall active student count
      - by_tier: count broken down by High / Medium / Low risk
      - by_department: count broken down by academic department
    """
    # 1. Tier counts via SQL GROUP BY
    tier_rows = db.execute(
        select(LatestPrediction.risk_tier, func.count(LatestPrediction.student_id)).group_by(
            LatestPrediction.risk_tier
        )
    ).all()

    by_tier: Dict[str, int] = {"High": 0, "Medium": 0, "Low": 0}
    total = 0
    for tier, count in tier_rows:
        if tier:
            by_tier[tier] = count
            total += count

    # 2. Department counts via SQL GROUP BY
    dept_rows = db.execute(
        select(LatestPrediction.department, func.count(LatestPrediction.student_id))
        .where(
            LatestPrediction.department.isnot(None),
            LatestPrediction.department != "",
        )
        .group_by(LatestPrediction.department)
        .order_by(LatestPrediction.department.asc())
    ).all()

    by_department: Dict[str, int] = {dept: count for dept, count in dept_rows}

    return StatsSummaryResponse(
        total=total,
        by_tier=by_tier,
        by_department=by_department,
    )
