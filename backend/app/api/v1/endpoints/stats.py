"""
Cohort statistics endpoints (read-only).

    GET /api/v1/stats/summary
    GET /api/v1/stats/distribution
    GET /api/v1/stats/drivers
    GET /api/v1/stats/interventions
    GET /api/v1/stats/alerts

Computes cohort totals, risk tier distributions, and department breakdowns
directly in PostgreSQL using indexed GROUP BY queries on latest_predictions.
"""

import logging
from typing import Dict, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.models.latest_prediction import LatestPrediction
from backend.app.schemas.prediction import RiskTier
from backend.app.schemas.stats import (
    StatsAlertsResponse,
    StatsDistributionResponse,
    StatsDriversResponse,
    StatsInterventionsResponse,
    StatsSummaryResponse,
)
from backend.app.services import stats_service

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


@router.get(
    "/distribution",
    response_model=StatsDistributionResponse,
    summary="Histogram of latest calibrated risk probabilities",
)
def get_risk_distribution(
    bins: int = Query(10, ge=2, le=50, description="Number of equal-width bins over [0, 1]"),
    db: Session = Depends(get_db),
) -> StatsDistributionResponse:
    return StatsDistributionResponse(**stats_service.risk_score_distribution(db, bins))


@router.get(
    "/drivers",
    response_model=StatsDriversResponse,
    summary="Most common risk-increasing drivers among students' latest predictions",
)
def get_top_drivers(
    limit: int = Query(10, ge=1, le=50, description="Maximum number of drivers to return"),
    risk_tier: Optional[RiskTier] = Query(None, description="Only count students in this tier"),
    db: Session = Depends(get_db),
) -> StatsDriversResponse:
    return StatsDriversResponse(**stats_service.top_risk_drivers(db, limit, risk_tier))


@router.get(
    "/interventions",
    response_model=StatsInterventionsResponse,
    summary="Intervention log entries by lifecycle and outcome status",
)
def get_intervention_counts(db: Session = Depends(get_db)) -> StatsInterventionsResponse:
    return StatsInterventionsResponse(**stats_service.intervention_counts(db))


@router.get(
    "/alerts",
    response_model=StatsAlertsResponse,
    summary="Students triggering each rule-based alert in their latest prediction",
)
def get_alert_counts(db: Session = Depends(get_db)) -> StatsAlertsResponse:
    return StatsAlertsResponse(**stats_service.rule_based_alert_counts(db))
