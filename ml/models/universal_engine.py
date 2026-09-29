"""
Universal Multi-Tier Machine Learning & Explainability Engine for DropoutGuard.
Supports 5 educational stages with hierarchical macro-geographic context:
  1. PRE_10TH
  2. HIGHER_SECONDARY
  3. UNDERGRADUATE
  4. POSTGRADUATE
  5. DOCTORATE
"""

import json
import logging
import joblib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / "universal"
UNIVERSAL_ARTIFACTS_DIR = BASE_DIR / "ml" / "artifacts" / "universal"
UNIVERSAL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
CATALOG_PATH = BASE_DIR / "ml" / "artifacts" / "universal_interventions.json"

TIER_CONFIG = {
    "PRE_10TH": {
        "file": "tier_pre_10th.csv",
        "features": [
            "distance_to_school_km", "attendance_percentage", "attendance_risk_flag",
            "mid_day_meal_attendance_pct", "parental_literacy_years", "seasonal_migration_flag",
            "fln_score_pct", "sibling_count_under_5", "toilets_available_flag",
            "state_literacy_rate", "district_mpi_pct", "is_aspirational_district",
            "district_rurality_pct", "district_ptr_ratio"
        ]
    },
    "HIGHER_SECONDARY": {
        "file": "tier_higher_secondary.csv",
        "features": [
            "class_10_board_pct", "attendance_percentage", "attendance_risk_flag",
            "stream_preference_match", "coaching_financial_strain", "mid_term_board_sim_score",
            "vocational_practical_att_pct", "family_debt_pressure_flag", "consecutive_absences",
            "state_literacy_rate", "district_mpi_pct", "is_aspirational_district",
            "district_rurality_pct"
        ]
    },
    "UNDERGRADUATE": {
        "file": "tier_undergraduate.csv",
        "features": [
            "attendance_percentage", "attendance_month_1", "attendance_month_2", "attendance_month_3",
            "attendance_3m_trend", "consecutive_absences", "attendance_risk_flag", "subject_attendance_std",
            "att_core1", "att_core2", "att_lab", "att_elective",
            "current_cgpa", "prev_sem_cgpa", "cgpa_delta", "backlog_count",
            "internal_exam_score_pct", "stem_core_fail_flag", "academic_crisis_flag",
            "lms_logins_per_week", "assignment_submission_lag_days", "resource_access_count",
            "days_since_last_lms_activity", "forum_participation_count", "behavioral_disengagement_index",
            "income_slab_idx", "is_first_generation", "fee_payment_delay_days", "has_scholarship",
            "is_hosteler", "commute_distance_km", "financial_stress_index",
            "interaction_att_x_fee", "interaction_cgpa_x_backlog", "interaction_firstgen_x_inactivity",
            "interaction_att_x_cgpa_drop", "age",
            "state_literacy_rate", "district_mpi_pct", "is_aspirational_district"
        ]
    },
    "POSTGRADUATE": {
        "file": "tier_postgraduate.csv",
        "features": [
            "ug_graduation_cgpa", "current_pg_cgpa", "attendance_percentage",
            "dissertation_milestone_pct", "education_loan_burden_inr", "placement_readiness_score",
            "part_time_work_hours_per_week", "backlog_count", "fee_payment_delay_days",
            "state_literacy_rate", "district_mpi_pct", "is_aspirational_district"
        ]
    },
    "DOCTORATE": {
        "file": "tier_doctorate.csv",
        "features": [
            "phd_tenure_months", "fellowship_disbursement_delay_days", "advisor_meeting_frequency_per_month",
            "peer_reviewed_submissions", "comprehensive_exam_attempts", "research_stagnation_index",
            "stipend_adequacy_ratio", "lab_isolation_index",
            "state_literacy_rate", "district_mpi_pct", "is_aspirational_district"
        ]
    }
}

def get_tier_label(prob: float) -> str:
    if prob < 0.33:
        return "Low"
    elif prob <= 0.66:
        return "Medium"
    else:
        return "High"

class UniversalModelEngine:
    def __init__(self):
        self.models: Dict[str, CalibratedClassifierCV] = {}
        self.base_models: Dict[str, XGBClassifier] = {}
        self.explainers: Dict[str, shap.TreeExplainer] = {}
        self.feature_sets: Dict[str, List[str]] = {}
        self.catalogs: Dict[str, List[Dict[str, Any]]] = {}
        self._load_catalog()

    def _load_catalog(self):
        if CATALOG_PATH.exists():
            with open(CATALOG_PATH, "r") as f:
                self.catalogs = json.load(f)["interventions_by_tier"]

    def train_tier_model(self, tier: str) -> Dict[str, Any]:
        cfg = TIER_CONFIG[tier]
        data_path = DATA_DIR / cfg["file"]
        if not data_path.exists():
            raise FileNotFoundError(f"Missing data for tier {tier} at {data_path}")

        df = pd.read_csv(data_path)
        feature_cols = [c for c in cfg["features"] if c in df.columns]
        X = df[feature_cols].copy()
        y = df["is_dropout"].values

        # SMOTE oversampling
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)

        # Base XGBoost
        base_xgb = XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            random_state=42,
            eval_metric="logloss"
        )
        base_xgb.fit(X_res, y_res)

        # 5-fold Platt Sigmoid Calibration
        calibrated = CalibratedClassifierCV(estimator=base_xgb, method="sigmoid", cv=5)
        calibrated.fit(X, y)

        # TreeSHAP Explainer
        explainer = shap.TreeExplainer(base_xgb)

        # Save artifacts
        tier_lower = tier.lower()
        joblib.dump(base_xgb, UNIVERSAL_ARTIFACTS_DIR / f"base_{tier_lower}.joblib")
        joblib.dump(calibrated, UNIVERSAL_ARTIFACTS_DIR / f"calibrated_{tier_lower}.joblib")
        joblib.dump(explainer, UNIVERSAL_ARTIFACTS_DIR / f"shap_{tier_lower}.joblib")
        with open(UNIVERSAL_ARTIFACTS_DIR / f"features_{tier_lower}.json", "w") as f:
            json.dump(feature_cols, f, indent=2)

        self.base_models[tier] = base_xgb
        self.models[tier] = calibrated
        self.explainers[tier] = explainer
        self.feature_sets[tier] = feature_cols

        # Quick evaluation (Evaluates at intervention trigger threshold prob >= 0.33)
        probs = calibrated.predict_proba(X)[:, 1]
        preds = (probs >= 0.33).astype(int)
        recall = np.sum((preds == 1) & (y == 1)) / max(1, np.sum(y == 1))
        precision = np.sum((preds == 1) & (y == 1)) / max(1, np.sum(preds == 1))

        logger.info(f"Trained [{tier}]: Features={len(feature_cols)}, Recall (at Risk >= 33%)={recall:.2%}, Precision={precision:.2%}")
        return {"tier": tier, "features_count": len(feature_cols), "recall": float(recall), "precision": float(precision)}

    def train_all_tiers(self):
        logger.info("Training Universal Multi-Tier ML Engine (All 5 Stages)...")
        results = []
        for tier in TIER_CONFIG.keys():
            res = self.train_tier_model(tier)
            results.append(res)
        logger.info("All 5 Educational Tier Models Trained & Serialized Successfully!")
        return results

    def load_all_artifacts(self):
        for tier in TIER_CONFIG.keys():
            tier_lower = tier.lower()
            calib_p = UNIVERSAL_ARTIFACTS_DIR / f"calibrated_{tier_lower}.joblib"
            base_p = UNIVERSAL_ARTIFACTS_DIR / f"base_{tier_lower}.joblib"
            shap_p = UNIVERSAL_ARTIFACTS_DIR / f"shap_{tier_lower}.joblib"
            feat_p = UNIVERSAL_ARTIFACTS_DIR / f"features_{tier_lower}.json"

            if calib_p.exists() and base_p.exists() and shap_p.exists() and feat_p.exists():
                self.models[tier] = joblib.load(calib_p)
                self.base_models[tier] = joblib.load(base_p)
                self.explainers[tier] = joblib.load(shap_p)
                with open(feat_p, "r") as f:
                    self.feature_sets[tier] = json.load(f)

    def explain_tier_student(self, tier: str, student_dict: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        if tier not in self.explainers:
            self.load_all_artifacts()
        
        feature_cols = self.feature_sets[tier]
        explainer = self.explainers[tier]

        # Extract features vector
        row_vals = [float(student_dict.get(col, 0.0)) for col in feature_cols]
        X_row = pd.DataFrame([row_vals], columns=feature_cols)

        shap_values = explainer.shap_values(X_row)
        if isinstance(shap_values, list):
            vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif len(shap_values.shape) == 2:
            vals = shap_values[0]
        else:
            vals = shap_values

        sorted_indices = np.argsort(np.abs(vals))[::-1][:top_k]
        drivers = []
        for rank, idx in enumerate(sorted_indices, 1):
            feat_name = feature_cols[idx]
            val = row_vals[idx]
            s_val = float(vals[idx])
            direction = "RISK_INCREASING" if s_val > 0 else "PROTECTIVE"
            impact_pp = round(abs(s_val) * 25.0, 1)

            # Plain language generation
            plain_text = self._build_plain_language(tier, feat_name, val, s_val, impact_pp)

            drivers.append({
                "rank": rank,
                "feature_name": feat_name,
                "feature_value": val,
                "shap_value": round(s_val, 4),
                "impact_direction": direction,
                "risk_delta_percentage_points": impact_pp,
                "plain_language_explanation": plain_text
            })
        return drivers

    def _build_plain_language(self, tier: str, feat: str, val: float, s_val: float, impact_pp: float) -> str:
        d_str = "increasing" if s_val > 0 else "reducing"
        arrow = "[▲ RISK_UP]" if s_val > 0 else "[▼ RISK_DOWN]"

        # Stage-specific templates
        if feat == "attendance_percentage":
            return f"{arrow} Class attendance ({val:.1f}%) is {d_str} risk by {impact_pp} percentage points."
        elif feat == "distance_to_school_km":
            return f"{arrow} Long transit distance to school ({val:.1f} km) is {d_str} risk by {impact_pp} percentage points."
        elif feat == "fln_score_pct":
            return f"{arrow} Foundational Literacy & Numeracy score ({val:.1f}%) is {d_str} risk by {impact_pp} percentage points."
        elif feat == "seasonal_migration_flag":
            return f"{arrow} Seasonal agricultural labor migration is {d_str} risk by {impact_pp} percentage points."
        elif feat == "stream_preference_match":
            return f"{arrow} Forced academic stream mismatch is {d_str} risk by {impact_pp} percentage points."
        elif feat == "coaching_financial_strain":
            return f"{arrow} High out-of-pocket private coaching expense strain ({val:.2f}) is {d_str} risk by {impact_pp} percentage points."
        elif feat == "dissertation_milestone_pct":
            return f"{arrow} Master's dissertation progress milestone ({val:.1f}%) is {d_str} risk by {impact_pp} percentage points."
        elif feat == "education_loan_burden_inr":
            return f"{arrow} Outstanding education loan liability (₹{val:,.0f}) is {d_str} risk by {impact_pp} percentage points."
        elif feat == "fellowship_disbursement_delay_days":
            return f"{arrow} Arrears in fellowship stipend disbursement ({int(val)} days delay) is {d_str} risk by {impact_pp} percentage points."
        elif feat == "advisor_meeting_frequency_per_month":
            return f"{arrow} Monthly advisor consultation frequency ({int(val)} meetings/mo) is {d_str} risk by {impact_pp} percentage points."
        elif feat == "research_stagnation_index":
            return f"{arrow} Doctoral research stagnation index ({val:.2f}) is {d_str} risk by {impact_pp} percentage points."
        elif feat == "district_mpi_pct":
            return f"{arrow} High district poverty index (MPI headcount {val:.1f}%) is {d_str} baseline risk by {impact_pp} percentage points."
        elif feat == "is_aspirational_district" and val == 1:
            return f"{arrow} Residence in NITI Aayog Aspirational District is contributing {impact_pp} percentage points to contextual vulnerability."
        else:
            clean_name = feat.replace("_", " ").title()
            return f"{arrow} {clean_name} ({val}) is {d_str} risk by {impact_pp} percentage points."

    def predict_universal_student(self, student_dict: Dict[str, Any]) -> Dict[str, Any]:
        tier = student_dict.get("educational_tier", "UNDERGRADUATE").upper()
        if tier not in self.models:
            self.load_all_artifacts()

        if tier not in self.models:
            raise ValueError(f"Unknown or untrained educational tier: {tier}")

        feature_cols = self.feature_sets[tier]
        model = self.models[tier]

        row_vals = [float(student_dict.get(col, 0.0)) for col in feature_cols]
        X_row = pd.DataFrame([row_vals], columns=feature_cols)

        prob = float(model.predict_proba(X_row)[0, 1])
        tier_label = get_tier_label(prob)
        drivers = self.explain_tier_student(tier, student_dict, top_k=5)

        # Mapped interventions
        tier_catalog = self.catalogs.get(tier, [])
        rec_interventions = []
        top_feat_names = [d["feature_name"] for d in drivers if d["impact_direction"] == "RISK_INCREASING"]

        for item in tier_catalog:
            for feat in item.get("recommended_for", []):
                if feat in top_feat_names and item not in rec_interventions:
                    rec_interventions.append(item)
                    break
            if len(rec_interventions) >= 3:
                break

        return {
            "student_id": student_dict.get("student_id", "UNKNOWN"),
            "educational_tier": tier,
            "calibrated_risk_probability": round(prob, 4),
            "risk_tier": tier_label,
            "risk_score_percentage": round(prob * 100.0, 2),
            "top_drivers": drivers,
            "recommended_interventions": rec_interventions
        }

universal_engine = UniversalModelEngine()

if __name__ == "__main__":
    universal_engine.train_all_tiers()
