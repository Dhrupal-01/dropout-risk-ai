"""
Read-only cohort aggregates for the Overview dashboard.

Everything is computed in SQL over `latest_predictions` (one row per scored student), the
prediction it points to, and `intervention_logs`. No pandas, no per-student Python loops: the
only Python here fills empty bins and shapes the response. Responses carry counts and bin
edges only, never student identifiers or personal fields.
"""

from typing import Dict, List, Optional

from sqlalchemy import and_, case, func, or_, select, text
from sqlalchemy.orm import Session

# Module import (not `from ml.config import ATTENDANCE_THRESHOLD`) so the threshold is read from the
# same source as the rule-based alerts in ml.intervention.engine.
from ml import config as ml_config

from backend.app.models.intervention_log import INTERVENTION_OUTCOME_STATUSES, INTERVENTION_STATUSES, InterventionLog
from backend.app.models.latest_prediction import LatestPrediction
from backend.app.models.prediction import Prediction

# Codes match ml.intervention.engine.rule_based_alerts, which the student detail endpoint reports.
ATTENDANCE_ALERT_CODE = "ATTENDANCE_BELOW_REQUIREMENT"
ACADEMIC_CRISIS_ALERT_CODE = "ACADEMIC_CRISIS_FLAG"

RISK_INCREASING = "RISK_INCREASING"


def risk_score_distribution(db: Session, bin_count: int) -> Dict:
    """Equal-width histogram of the latest calibrated risk probability over [0, 1]."""
    probability = LatestPrediction.calibrated_risk_probability
    # width_bucket puts exactly 1.0 in bucket bin_count + 1; fold it into the last bin.
    bucket = func.greatest(1, func.least(func.width_bucket(probability, 0.0, 1.0, bin_count), bin_count)).label("bucket")
    counts = dict(db.execute(select(bucket, func.count()).group_by(bucket)).all())

    bins = [
        {"lower": round((i - 1) / bin_count, 10), "upper": round(i / bin_count, 10), "count": counts.get(i, 0)}
        for i in range(1, bin_count + 1)
    ]
    return {"total": sum(b["count"] for b in bins), "bin_count": bin_count, "bins": bins}


_DRIVER_COUNTS_SQL = text(
    """
    SELECT d.value ->> 'feature_name' AS feature_name,
           MAX(d.value ->> 'display_name') AS display_name,
           COUNT(DISTINCT lp.student_id) AS student_count
    FROM latest_predictions lp
    JOIN predictions p ON p.id = lp.prediction_id
    CROSS JOIN LATERAL jsonb_array_elements(p.top_drivers) AS d(value)
    WHERE jsonb_typeof(p.top_drivers) = 'array'
      AND d.value ->> 'impact_direction' = :direction
      AND (CAST(:risk_tier AS TEXT) IS NULL OR lp.risk_tier = :risk_tier)
    GROUP BY d.value ->> 'feature_name'
    ORDER BY student_count DESC, feature_name ASC
    LIMIT :limit
    """
)


def top_risk_drivers(db: Session, limit: int, risk_tier: Optional[str]) -> Dict:
    """Features that most often appear among students' stored risk-increasing SHAP drivers."""
    tier_filter = [LatestPrediction.risk_tier == risk_tier] if risk_tier else []
    has_drivers = func.jsonb_typeof(Prediction.top_drivers) == "array"

    considered, with_drivers = db.execute(
        select(func.count(), func.count().filter(has_drivers))
        .select_from(LatestPrediction)
        .join(Prediction, Prediction.id == LatestPrediction.prediction_id)
        .where(*tier_filter)
    ).one()

    rows = db.execute(
        _DRIVER_COUNTS_SQL, {"direction": RISK_INCREASING, "risk_tier": risk_tier, "limit": limit}
    ).all()
    drivers = [
        {"feature_name": r.feature_name, "display_name": r.display_name or r.feature_name, "student_count": r.student_count}
        for r in rows
    ]
    return {
        "risk_tier": risk_tier,
        "students_considered": considered,
        "students_with_drivers": with_drivers,
        "drivers": drivers,
    }


def intervention_counts(db: Session) -> Dict:
    """Intervention log entries by lifecycle status and by outcome status."""
    by_status = {status: 0 for status in INTERVENTION_STATUSES}
    by_status.update(
        dict(db.execute(select(InterventionLog.status, func.count()).group_by(InterventionLog.status)).all())
    )
    by_outcome = {status: 0 for status in INTERVENTION_OUTCOME_STATUSES}
    by_outcome.update(
        dict(
            db.execute(
                select(InterventionLog.outcome_status, func.count()).group_by(InterventionLog.outcome_status)
            ).all()
        )
    )

    is_open = InterventionLog.status != "COMPLETED"
    total, open_count, students_open = db.execute(
        select(
            func.count(),
            func.count().filter(is_open),
            func.count(func.distinct(case((is_open, InterventionLog.student_id)))),
        ).select_from(InterventionLog)
    ).one()

    return {
        "total": total,
        "open": open_count,
        "students_with_open_interventions": students_open,
        "by_status": by_status,
        "by_outcome_status": by_outcome,
    }


def rule_based_alert_counts(db: Session) -> Dict:
    """
    Students whose latest prediction snapshot triggers each rule-based alert. Mirrors
    ml.intervention.engine.rule_based_alerts:
      attendance: attendance_percentage < ATTENDANCE_THRESHOLD, or attendance_risk_flag == 1
      academic crisis: academic_crisis_flag == 1
    """
    threshold = float(ml_config.ATTENDANCE_THRESHOLD)
    features = Prediction.input_features
    attendance = features["attendance_percentage"].as_float()
    attendance_flag = features["attendance_risk_flag"].as_float()
    crisis_flag = features["academic_crisis_flag"].as_float()

    # NULL comparisons count as false, matching the engine's "missing value never alerts".
    attendance_alert = or_(and_(attendance.isnot(None), attendance < threshold), attendance_flag == 1)
    crisis_alert = crisis_flag == 1

    considered, attendance_count, crisis_count, any_count = db.execute(
        select(
            func.count(),
            func.count().filter(attendance_alert),
            func.count().filter(crisis_alert),
            func.count().filter(or_(attendance_alert, crisis_alert)),
        )
        .select_from(LatestPrediction)
        .join(Prediction, Prediction.id == LatestPrediction.prediction_id)
    ).one()

    alerts: List[Dict] = [
        {"code": ATTENDANCE_ALERT_CODE, "student_count": attendance_count},
        {"code": ACADEMIC_CRISIS_ALERT_CODE, "student_count": crisis_count},
    ]
    return {
        "students_considered": considered,
        "students_with_any_alert": any_count,
        "attendance_threshold": threshold,
        "alerts": alerts,
    }
