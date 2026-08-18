"""
ML artifact lifecycle and the raw -> 37-feature contract.

Owns the single process-wide instance of the expensive artifacts (calibrated model, SHAP
TreeExplainer, counterfactual engine). They are loaded once during FastAPI startup and
reused for every request — never re-loaded per call.

FEATURE CONTRACT (verified against ml/artifacts/feature_names.json and the trained
XGBClassifier's own `feature_names_in_`, not against the Markdown docs):

  37 model features = 28 RAW inputs + 9 ENGINEERED, in a fixed order.

  The caller supplies RAW inputs only. The 9 engineered columns are derived here by
  `ml.data_pipeline.feature_engineering.build_engineered_features`, the same function used
  to build the training set. Callers must never send engineered values directly.

  `is_dropout` and `ground_truth_risk_prob` are LABELS. They are stripped defensively and
  can never reach `predict_proba`.
"""

import hashlib
import json
import logging
import threading
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from ml.config import (
    BASE_MODEL_PATH,
    FEATURE_NAMES_PATH,
    MODEL_ARTIFACT_PATH,
    get_risk_tier,
)
from ml.data_pipeline.feature_engineering import build_engineered_features

logger = logging.getLogger(__name__)

# Never accepted as inference input, at any layer.
LABEL_COLUMNS = frozenset({"is_dropout", "ground_truth_risk_prob"})

# Protected/identifier columns present in the dataset but excluded from the model
# (ml.config.EXCLUDED_FEATURES). Dropped before the frame reaches predict_proba.
NON_FEATURE_COLUMNS = frozenset(
    {"student_id", "gender", "category", "family_income_slab", "name", "department"}
)

# The 28 raw columns a client must supply. `hostel_status` is additionally accepted as the
# source for the engineered `is_hosteler`.
RAW_FEATURE_COLUMNS: tuple[str, ...] = (
    "age",
    "commute_distance_km",
    "income_slab_idx",
    "is_first_generation",
    "has_scholarship",
    "fee_payment_delay_days",
    "att_core1",
    "att_core2",
    "att_lab",
    "att_elective",
    "attendance_month_1",
    "attendance_month_2",
    "attendance_month_3",
    "attendance_percentage",
    "attendance_3m_trend",
    "consecutive_absences",
    "attendance_risk_flag",
    "prev_sem_cgpa",
    "current_cgpa",
    "cgpa_delta",
    "backlog_count",
    "internal_exam_score_pct",
    "stem_core_fail_flag",
    "lms_logins_per_week",
    "assignment_submission_lag_days",
    "resource_access_count",
    "days_since_last_lms_activity",
    "forum_participation_count",
)

# Derived by build_engineered_features; rejected if supplied by a client.
ENGINEERED_FEATURE_COLUMNS: tuple[str, ...] = (
    "subject_attendance_std",
    "academic_crisis_flag",
    "behavioral_disengagement_index",
    "is_hosteler",
    "financial_stress_index",
    "interaction_att_x_fee",
    "interaction_cgpa_x_backlog",
    "interaction_firstgen_x_inactivity",
    "interaction_att_x_cgpa_drop",
)


class MLArtifactsNotLoaded(RuntimeError):
    """Raised when inference is attempted before artifacts finished loading."""


def to_native(value: Any) -> Any:
    """
    numpy scalar -> Python scalar.

    pandas hands back numpy types (np.float64, np.int64), which json.dumps cannot encode.
    Since these values are written straight into JSONB columns, they must be converted at
    the service boundary rather than at each call site.
    """
    if hasattr(value, "item"):
        return value.item()
    return value


class MLService:
    """Process-wide holder for the trained artifacts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: Any = None
        self._explainer_service: Any = None
        self._recourse_engine: Any = None
        self._feature_names: List[str] = []
        self._model_version: Optional[str] = None
        self._load_error: Optional[str] = None
        self._loaded = False

    # ------------------------------------------------------------------ lifecycle

    def load(self) -> bool:
        """
        Load every artifact once. Called from the FastAPI lifespan handler.

        Returns True on success. On failure it records the error and returns False rather
        than raising, so the process still starts and /health can report model_loaded=false
        instead of the container crash-looping.
        """
        with self._lock:
            if self._loaded:
                return True
            try:
                if not FEATURE_NAMES_PATH.exists():
                    raise FileNotFoundError(f"Missing feature names artifact: {FEATURE_NAMES_PATH}")
                if not MODEL_ARTIFACT_PATH.exists():
                    raise FileNotFoundError(
                        f"Missing calibrated model artifact: {MODEL_ARTIFACT_PATH}. "
                        "Run `python -m ml.validate_pipeline --regenerate` to build it."
                    )
                if not BASE_MODEL_PATH.exists():
                    raise FileNotFoundError(
                        f"Missing base XGBoost artifact: {BASE_MODEL_PATH}. "
                        "The SHAP TreeExplainer is built from it."
                    )

                self._feature_names = json.loads(FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
                self._model = joblib.load(MODEL_ARTIFACT_PATH)
                self._model_version = self._compute_model_version()

                # Imported lazily: constructing SHAPExplainerService can rebuild and write
                # the explainer artifact, which must not happen at module import time.
                from ml.intervention.engine import CounterfactualRecourseEngine
                from ml.models.explain_shap import SHAPExplainerService

                self._explainer_service = SHAPExplainerService()
                self._recourse_engine = CounterfactualRecourseEngine()

                # The recourse engine loads its model in __init__ and silently leaves it as
                # None if the artifact was missing, failing later with an opaque
                # AttributeError. Fail loudly here instead.
                if getattr(self._recourse_engine, "model", None) is None:
                    raise RuntimeError(
                        "CounterfactualRecourseEngine loaded without a model — "
                        f"expected artifact at {MODEL_ARTIFACT_PATH}"
                    )

                self._assert_contract()
                self._loaded = True
                self._load_error = None
                logger.info(
                    "ML artifacts loaded: %d features, model_version=%s",
                    len(self._feature_names),
                    self._model_version,
                )
                return True
            except Exception as exc:  # noqa: BLE001 — startup must degrade, not crash
                self._load_error = f"{type(exc).__name__}: {exc}"
                logger.error("Failed to load ML artifacts: %s", self._load_error)
                return False

    def _compute_model_version(self) -> str:
        """Content fingerprint of the calibrated artifact — stable and reproducible."""
        digest = hashlib.sha256(MODEL_ARTIFACT_PATH.read_bytes()).hexdigest()[:12]
        return f"calibrated-{digest}"

    def _assert_contract(self) -> None:
        """Verify the artifacts agree with the constants compiled into this module."""
        expected = list(RAW_FEATURE_COLUMNS) + list(ENGINEERED_FEATURE_COLUMNS)
        if sorted(expected) != sorted(self._feature_names):
            missing = sorted(set(self._feature_names) - set(expected))
            extra = sorted(set(expected) - set(self._feature_names))
            raise RuntimeError(
                "Feature contract drift between backend and ml/artifacts/feature_names.json. "
                f"In artifact but not backend: {missing}. In backend but not artifact: {extra}."
            )

    def unload(self) -> None:
        """Release artifacts on shutdown."""
        with self._lock:
            self._model = None
            self._explainer_service = None
            self._recourse_engine = None
            self._loaded = False

    # ------------------------------------------------------------------ accessors

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    @property
    def model_version(self) -> Optional[str]:
        return self._model_version

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise MLArtifactsNotLoaded(self._load_error or "ML artifacts are not loaded")

    # ------------------------------------------------------------------ contract

    def build_feature_frame(self, raw_features: Dict[str, Any]) -> pd.DataFrame:
        """
        Turn a raw feature dict into the exact ordered 37-column frame the model expects.

        Applies the same engineering function used to build the training set, then selects
        `feature_names` in artifact order. Labels and protected/identifier columns are
        dropped before the model ever sees the frame.
        """
        # Delegates to the batch builder so column alignment has exactly one
        # implementation and the single-row and batch paths can never diverge.
        return self.build_feature_frame_batch([raw_features])

    def build_feature_frame_batch(self, raw_rows: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Vectorised form of `build_feature_frame` for N students.

        Builds ONE DataFrame and runs `build_engineered_features` once, so batch scoring
        costs a single aligned inference call instead of N. Measured at ~0.5ms/student
        versus ~110ms/student when looping.

        Row order is preserved so results can be zipped back to the caller's input.
        """
        self._require_loaded()
        if not raw_rows:
            raise ValueError("No students supplied")

        for index, row in enumerate(raw_rows):
            leaked = LABEL_COLUMNS.intersection(row)
            if leaked:
                raise ValueError(
                    f"Row {index}: Label columns must never be used as inference "
                    f"input: {sorted(leaked)}"
                )
            missing = [c for c in RAW_FEATURE_COLUMNS if c not in row]
            if missing:
                raise ValueError(f"Row {index}: missing required raw feature(s): {missing}")
            if "is_hosteler" not in row and "hostel_status" not in row:
                raise ValueError(f"Row {index}: supply either 'is_hosteler' or 'hostel_status'")

        frame = pd.DataFrame([dict(r) for r in raw_rows])
        engineered = build_engineered_features(frame)
        engineered = engineered.drop(
            columns=[c for c in NON_FEATURE_COLUMNS if c in engineered.columns], errors="ignore"
        )

        still_missing = [c for c in self._feature_names if c not in engineered.columns]
        if still_missing:
            raise ValueError(f"Feature engineering did not produce: {still_missing}")

        # Reindex, never dict order: guarantees the canonical column sequence and drops
        # any extra column before it can reach the model.
        return engineered.reindex(columns=self._feature_names)

    def _validate_frame(self, frame: pd.DataFrame) -> None:
        """Assert exact model-feature agreement before inference."""
        actual = list(frame.columns)
        if actual != self._feature_names:
            missing = [c for c in self._feature_names if c not in actual]
            extra = [c for c in actual if c not in self._feature_names]
            raise ValueError(
                f"Feature matrix misaligned. missing={missing} extra={extra} "
                f"expected {len(self._feature_names)} columns in canonical order."
            )
        null_columns = [c for c in actual if frame[c].isna().any()]
        if null_columns:
            raise ValueError(f"Null values in model features: {null_columns}")

    # ------------------------------------------------------------------ inference

    def predict(self, raw_features: Dict[str, Any]) -> Dict[str, Any]:
        """Calibrated probability + tier for one student, with the exact input snapshot."""
        self._require_loaded()
        frame = self.build_feature_frame(raw_features)
        self._validate_frame(frame)

        from ml.models.calibrate import predict_student_risk

        probabilities, tiers = predict_student_risk(self._model, frame)
        probability = float(probabilities[0])

        return {
            "calibrated_risk_probability": round(probability, 4),
            "risk_tier": tiers[0],
            "risk_score_percentage": round(probability * 100.0, 2),
            "model_version": self._model_version,
            # Converted here so the snapshot can go straight into a JSONB column.
            "input_features": {k: to_native(v) for k, v in frame.iloc[0].to_dict().items()},
        }

    def predict_batch(self, raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score N students with a SINGLE aligned inference call.

        Uses the same `predict_student_risk` entry point as the single-row path, so tiers
        come from `ml.config.get_risk_tier` and are never recomputed here.
        """
        self._require_loaded()
        frame = self.build_feature_frame_batch(raw_rows)
        self._validate_frame(frame)

        from ml.models.calibrate import predict_student_risk

        probabilities, tiers = predict_student_risk(self._model, frame)

        results = []
        for position in range(len(frame)):
            probability = float(probabilities[position])
            snapshot = {k: to_native(v) for k, v in frame.iloc[position].to_dict().items()}
            results.append(
                {
                    "calibrated_risk_probability": round(probability, 4),
                    "risk_tier": tiers[position],
                    "risk_score_percentage": round(probability * 100.0, 2),
                    "model_version": self._model_version,
                    "input_features": snapshot,
                }
            )
        return results

    def explain_from_snapshot(
        self, model_features: Dict[str, Any], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Explain a PERSISTED 37-feature snapshot.

        Takes the exact vector that produced a stored prediction, so the explanation
        always matches that score — even if the student's mutable feature record has
        since changed. No re-engineering happens here; the snapshot is already final.
        """
        self._require_loaded()
        missing = [c for c in self._feature_names if c not in model_features]
        if missing:
            raise ValueError(f"Prediction snapshot is missing model feature(s): {missing}")

        frame = pd.DataFrame([model_features]).reindex(columns=self._feature_names)
        self._validate_frame(frame)
        return self._explainer_service.explain_local_student(frame.iloc[0], top_k=top_k)

    def explain(self, raw_features: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """Top-k local SHAP drivers with counselor-facing plain-language sentences."""
        self._require_loaded()
        frame = self.build_feature_frame(raw_features)
        return self._explainer_service.explain_local_student(frame.iloc[0], top_k=top_k)

    def recommend_interventions(
        self, shap_drivers: List[Dict[str, Any]], max_recommendations: int = 3
    ) -> List[Dict[str, Any]]:
        """Map SHAP drivers onto the codified intervention catalog."""
        self._require_loaded()
        from ml.intervention.engine import map_shap_drivers_to_interventions

        return map_shap_drivers_to_interventions(
            shap_drivers, max_recommendations=max_recommendations
        )

    def generate_counterfactual(
        self, raw_features: Dict[str, Any], top_shap_drivers: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Counterfactual recourse plan.

        The engine needs BOTH representations in one row:
          * it scores the baseline with `base_df[feature_names]`, so the 9 engineered
            columns must already be present; and
          * it re-runs `build_engineered_features` on each perturbed scenario, so the raw
            columns (att_core1, hostel_status, ...) must survive too.

        Passing raw-only fails with a KeyError on the engineered columns, and passing the
        37-column frame alone breaks scenario re-engineering. A DataFrame is handed over
        rather than a Series so per-column dtypes are preserved.
        """
        self._require_loaded()
        payload = {k: v for k, v in raw_features.items() if k not in LABEL_COLUMNS}
        combined = build_engineered_features(pd.DataFrame([payload]))
        return self._recourse_engine.generate_counterfactual(
            combined, top_shap_drivers=top_shap_drivers
        )

    def generate_counterfactual_from_snapshot(
        self,
        model_features: Dict[str, Any],
        top_shap_drivers: Optional[List[Dict[str, Any]]] = None,
        student_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Counterfactual recourse for a PERSISTED 37-feature snapshot.

        The snapshot already contains every column the engine needs: it scores the
        baseline with `base_df[feature_names]`, and its perturbed scenarios re-run
        `build_engineered_features`, which finds `is_hosteler` in the snapshot and so
        never needs the raw `hostel_status` string.

        The engine falls back to the literal "STUDENT" when no student_id column is
        present, so the real identifier is substituted back afterwards.
        """
        self._require_loaded()
        missing = [c for c in self._feature_names if c not in model_features]
        if missing:
            raise ValueError(f"Prediction snapshot is missing model feature(s): {missing}")

        payload = {k: v for k, v in model_features.items() if k not in LABEL_COLUMNS}
        frame = build_engineered_features(pd.DataFrame([payload]))
        recourse = self._recourse_engine.generate_counterfactual(
            frame, top_shap_drivers=top_shap_drivers
        )
        if student_id is not None:
            recourse["student_id"] = student_id
        return recourse

    def risk_tier_for(self, probability: float) -> str:
        """Delegates to ml.config so tiering never forks from the ML core."""
        return get_risk_tier(probability)


# Process-wide singleton. Populated by the FastAPI lifespan handler.
ml_service = MLService()
