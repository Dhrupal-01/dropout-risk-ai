"""ORM models. Importing this package registers every table on Base.metadata."""

from backend.app.models.intervention_log import InterventionLog
from backend.app.models.latest_prediction import LatestPrediction
from backend.app.models.prediction import Prediction
from backend.app.models.student import Student

__all__ = ["Student", "Prediction", "LatestPrediction", "InterventionLog"]
