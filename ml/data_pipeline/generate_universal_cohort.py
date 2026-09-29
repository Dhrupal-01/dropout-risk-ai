"""
Universal Multi-Tier Synthetic Cohort Generator for DropoutGuard.
Generates 10,000 stratified student records across 5 educational tiers:
  1. PRE_10TH (School K-10)
  2. HIGHER_SECONDARY (11th-12th / ITI / Diploma)
  3. UNDERGRADUATE (B.Tech / B.Sc / B.Com / B.A.)
  4. POSTGRADUATE (M.Tech / M.Sc / MBA / M.A.)
  5. DOCTORATE (PhD)

Includes Macro-Geographic Socio-Economic Embeddings:
  - State Literacy & GER (UDISE+ / AISHE)
  - District Multidimensional Poverty Index (MPI) Headcount
  - NITI Aayog Aspirational District Flags
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
GEO_INDEX_PATH = BASE_DIR / "data" / "macro" / "state_district_indices.json"
UNIVERSAL_PROCESSED_DIR = BASE_DIR / "data" / "processed" / "universal"
UNIVERSAL_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_PER_TIER = 2000

def load_geo_index() -> Dict[str, Any]:
    with open(GEO_INDEX_PATH, "r") as f:
        return json.load(f)["states"]

def sample_geo_context(geo_data: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    state_names = list(geo_data.keys())
    state = rng.choice(state_names)
    s_info = geo_data[state]
    dist_names = list(s_info["districts"].keys())
    district = rng.choice(dist_names)
    d_info = s_info["districts"][district]
    
    return {
        "state": state,
        "state_code": s_info["state_code"],
        "state_literacy_rate": s_info["literacy_rate"],
        "state_female_literacy": s_info["female_literacy_rate"],
        "state_ger_higher_ed": s_info["ger_higher_ed"],
        "state_dropout_rate_secondary": s_info["school_dropout_rate_secondary"],
        "district": district,
        "district_mpi_pct": d_info["mpi_headcount_pct"],
        "is_aspirational_district": 1 if d_info["is_aspirational"] else 0,
        "district_rurality_pct": d_info["rurality_pct"],
        "district_ptr_ratio": d_info["ptr_ratio"]
    }

def generate_pre_10th(n: int, geo_data: Dict[str, Any], seed: int = 42) -> pd.DataFrame:
    rng = np.random.Generator(np.random.PCG64(seed))
    records = []
    
    for i in range(n):
        geo = sample_geo_context(geo_data, rng)
        gender = rng.choice(["Male", "Female"], p=[0.52, 0.48])
        age = int(rng.integers(9, 16))
        income_slab = rng.choice(["<2 LPA", "2-5 LPA", "5-10 LPA", ">10 LPA"], p=[0.45, 0.35, 0.15, 0.05])
        is_first_gen = int(rng.choice([1, 0], p=[0.48, 0.52]))
        
        # Domain Features
        distance_km = float(np.clip(rng.gamma(shape=2.5, scale=2.0) if geo["district_rurality_pct"] > 70 else rng.gamma(shape=1.5, scale=1.0), 0.5, 18.0))
        seasonal_migration = int(rng.choice([1, 0], p=[0.25 if geo["district_mpi_pct"] > 35 else 0.05, 0.75 if geo["district_mpi_pct"] > 35 else 0.95]))
        parental_literacy_yrs = int(np.clip(rng.normal(loc=geo["state_literacy_rate"]/10, scale=3.0), 0, 16))
        mid_day_meal_att = float(np.clip(rng.normal(loc=82.0 - 15.0 * seasonal_migration, scale=12.0), 20.0, 100.0))
        attendance_pct = float(np.clip(mid_day_meal_att * rng.uniform(0.85, 1.05) - (distance_km * 1.5), 15.0, 100.0))
        att_flag = 1 if attendance_pct < 75.0 else 0
        fln_score = float(np.clip(rng.normal(loc=65.0 - (0.3 * geo["district_ptr_ratio"]), scale=15.0) + (parental_literacy_yrs * 1.5), 10.0, 98.0))
        sibling_u5 = int(rng.choice([0, 1, 2, 3], p=[0.4, 0.35, 0.18, 0.07]))
        toilets_available = int(rng.choice([1, 0], p=[0.65 if geo["is_aspirational_district"] else 0.92, 0.35 if geo["is_aspirational_district"] else 0.08]))
        
        # Ground Truth Probability Formula
        logit = (
            -1.8
            + 0.045 * (75.0 - attendance_pct)
            + 0.035 * (60.0 - fln_score)
            + 0.12 * distance_km
            + 0.85 * seasonal_migration
            + 0.02 * geo["district_mpi_pct"]
            + 0.45 * (1 - toilets_available) * (1 if gender == "Female" else 0.2)
            + 0.30 * (sibling_u5 if gender == "Female" else sibling_u5 * 0.3)
            - 0.08 * parental_literacy_yrs
        )
        prob = 1.0 / (1.0 + np.exp(-logit))
        is_drop = int(prob > 0.50) if rng.uniform() > 0.08 else int(prob <= 0.50)
        
        records.append({
            "student_id": f"SCH_2026_{i+1:04d}",
            "educational_tier": "PRE_10TH",
            "gender": gender,
            "age": age,
            "family_income_slab": income_slab,
            "is_first_generation": is_first_gen,
            "distance_to_school_km": round(distance_km, 1),
            "attendance_percentage": round(attendance_pct, 1),
            "attendance_risk_flag": att_flag,
            "mid_day_meal_attendance_pct": round(mid_day_meal_att, 1),
            "parental_literacy_years": parental_literacy_yrs,
            "seasonal_migration_flag": seasonal_migration,
            "fln_score_pct": round(fln_score, 1),
            "sibling_count_under_5": sibling_u5,
            "toilets_available_flag": toilets_available,
            **geo,
            "ground_truth_risk_prob": round(prob, 4),
            "is_dropout": is_drop
        })
    return pd.DataFrame(records)

def generate_higher_secondary(n: int, geo_data: Dict[str, Any], seed: int = 43) -> pd.DataFrame:
    rng = np.random.Generator(np.random.PCG64(seed))
    records = []
    
    for i in range(n):
        geo = sample_geo_context(geo_data, rng)
        gender = rng.choice(["Male", "Female"], p=[0.51, 0.49])
        age = int(rng.integers(16, 19))
        income_slab = rng.choice(["<2 LPA", "2-5 LPA", "5-10 LPA", ">10 LPA"], p=[0.38, 0.40, 0.16, 0.06])
        is_first_gen = int(rng.choice([1, 0], p=[0.42, 0.58]))
        stream = rng.choice(["Science (PCM)", "Science (PCB)", "Commerce", "Arts / Humanities", "Vocational / ITI"])
        
        c10_board = float(np.clip(rng.normal(loc=72.0, scale=12.0), 38.0, 98.0))
        stream_match = int(rng.choice([1, 0], p=[0.72, 0.28]))
        coaching_strain = float(np.clip(rng.beta(2, 4) if income_slab in ["5-10 LPA", ">10 LPA"] else rng.beta(4, 2), 0.05, 0.98))
        att_pct = float(np.clip(rng.normal(loc=78.0, scale=14.0) - (20.0 * (1 - stream_match)), 20.0, 100.0))
        att_flag = 1 if att_pct < 75.0 else 0
        mock_score = float(np.clip(c10_board * 0.8 + rng.normal(0, 10) - (15.0 * (1 - stream_match)), 15.0, 96.0))
        vocational_att = float(np.clip(rng.normal(loc=att_pct, scale=8.0), 20.0, 100.0))
        debt_pressure = int(rng.choice([1, 0], p=[0.35 if income_slab == "<2 LPA" else 0.10, 0.65 if income_slab == "<2 LPA" else 0.90]))
        consec_abs = int(np.clip(rng.poisson(lam=2.5 + (1 - stream_match) * 3), 0, 20))
        
        logit = (
            -2.1
            + 0.040 * (75.0 - att_pct)
            + 0.038 * (65.0 - mock_score)
            + 0.80 * (1 - stream_match)
            + 0.75 * coaching_strain
            + 0.65 * debt_pressure
            + 0.015 * geo["district_mpi_pct"]
            + 0.08 * consec_abs
        )
        prob = 1.0 / (1.0 + np.exp(-logit))
        is_drop = int(prob > 0.50) if rng.uniform() > 0.08 else int(prob <= 0.50)
        
        records.append({
            "student_id": f"HSC_2026_{i+1:04d}",
            "educational_tier": "HIGHER_SECONDARY",
            "gender": gender,
            "age": age,
            "stream": stream,
            "family_income_slab": income_slab,
            "is_first_generation": is_first_gen,
            "class_10_board_pct": round(c10_board, 1),
            "attendance_percentage": round(att_pct, 1),
            "attendance_risk_flag": att_flag,
            "stream_preference_match": stream_match,
            "coaching_financial_strain": round(coaching_strain, 2),
            "mid_term_board_sim_score": round(mock_score, 1),
            "vocational_practical_att_pct": round(vocational_att, 1),
            "family_debt_pressure_flag": debt_pressure,
            "consecutive_absences": consec_abs,
            **geo,
            "ground_truth_risk_prob": round(prob, 4),
            "is_dropout": is_drop
        })
    return pd.DataFrame(records)

def generate_undergraduate(n: int, geo_data: Dict[str, Any], seed: int = 44) -> pd.DataFrame:
    """Uses the 37 established collegiate features with added geo context."""
    from ml.data_pipeline.generate_synthetic_indian import generate_indian_student_cohort
    df_raw = generate_indian_student_cohort(n_students=n, seed=seed)
    
    rng = np.random.Generator(np.random.PCG64(seed))
    geo_rows = [sample_geo_context(geo_data, rng) for _ in range(n)]
    df_geo = pd.DataFrame(geo_rows)
    
    df_combined = pd.concat([df_raw.drop(columns=["state", "district"], errors="ignore"), df_geo], axis=1)
    df_combined["student_id"] = [f"UG_2026_{i+1:04d}" for i in range(n)]
    df_combined["educational_tier"] = "UNDERGRADUATE"
    return df_combined

def generate_postgraduate(n: int, geo_data: Dict[str, Any], seed: int = 45) -> pd.DataFrame:
    rng = np.random.Generator(np.random.PCG64(seed))
    records = []
    
    for i in range(n):
        geo = sample_geo_context(geo_data, rng)
        gender = rng.choice(["Male", "Female"], p=[0.53, 0.47])
        age = int(rng.integers(21, 27))
        income_slab = rng.choice(["<2 LPA", "2-5 LPA", "5-10 LPA", ">10 LPA"], p=[0.25, 0.45, 0.22, 0.08])
        is_first_gen = int(rng.choice([1, 0], p=[0.35, 0.65]))
        
        ug_cgpa = float(np.clip(rng.normal(loc=7.4, scale=1.0), 5.5, 9.8))
        pg_cgpa = float(np.clip(ug_cgpa + rng.normal(loc=-0.3, scale=0.8), 4.0, 9.9))
        att_pct = float(np.clip(rng.normal(loc=76.0, scale=13.0), 25.0, 100.0))
        dissertation_progress = float(np.clip(rng.normal(loc=60.0, scale=22.0), 0.0, 100.0))
        loan_inr = float(rng.choice([0, 150000, 350000, 750000, 1200000], p=[0.40, 0.20, 0.20, 0.15, 0.05]))
        placement_ready = float(np.clip(rng.normal(loc=62.0, scale=18.0), 10.0, 95.0))
        part_time_hrs = int(np.clip(rng.poisson(lam=8.0 if loan_inr > 200000 else 2.0), 0, 35))
        backlogs = int(np.clip(rng.poisson(lam=0.4 if pg_cgpa > 6.5 else 1.8), 0, 5))
        fee_delay = int(np.clip(rng.exponential(scale=12.0), 0, 90))
        
        logit = (
            -2.4
            + 0.55 * (6.0 - pg_cgpa)
            + 0.035 * (50.0 - dissertation_progress)
            + 0.030 * (75.0 - att_pct)
            + 0.50 * backlogs
            + 0.0000015 * loan_inr
            + 0.035 * part_time_hrs
            + 0.02 * (50.0 - placement_ready)
        )
        prob = 1.0 / (1.0 + np.exp(-logit))
        is_drop = int(prob > 0.50) if rng.uniform() > 0.08 else int(prob <= 0.50)
        
        records.append({
            "student_id": f"PG_2026_{i+1:04d}",
            "educational_tier": "POSTGRADUATE",
            "gender": gender,
            "age": age,
            "family_income_slab": income_slab,
            "is_first_generation": is_first_gen,
            "ug_graduation_cgpa": round(ug_cgpa, 2),
            "current_pg_cgpa": round(pg_cgpa, 2),
            "attendance_percentage": round(att_pct, 1),
            "dissertation_milestone_pct": round(dissertation_progress, 1),
            "education_loan_burden_inr": loan_inr,
            "placement_readiness_score": round(placement_ready, 1),
            "part_time_work_hours_per_week": part_time_hrs,
            "backlog_count": backlogs,
            "fee_payment_delay_days": fee_delay,
            **geo,
            "ground_truth_risk_prob": round(prob, 4),
            "is_dropout": is_drop
        })
    return pd.DataFrame(records)

def generate_doctorate(n: int, geo_data: Dict[str, Any], seed: int = 46) -> pd.DataFrame:
    rng = np.random.Generator(np.random.PCG64(seed))
    records = []
    
    for i in range(n):
        geo = sample_geo_context(geo_data, rng)
        gender = rng.choice(["Male", "Female"], p=[0.54, 0.46])
        age = int(rng.integers(24, 34))
        income_slab = rng.choice(["<2 LPA", "2-5 LPA", "5-10 LPA", ">10 LPA"], p=[0.20, 0.40, 0.30, 0.10])
        is_first_gen = int(rng.choice([1, 0], p=[0.28, 0.72]))
        
        tenure_months = int(rng.integers(6, 72))
        fellowship_delay = int(np.clip(rng.exponential(scale=35.0), 0, 180))
        advisor_meetings = int(np.clip(rng.poisson(lam=3.2), 0, 10))
        peer_papers = int(np.clip(rng.poisson(lam=0.4 * (tenure_months / 12)), 0, 8))
        comp_attempts = int(rng.choice([1, 2, 3], p=[0.75, 0.20, 0.05]))
        stagnation_idx = float(np.clip(rng.beta(2, 3) + (0.05 * (fellowship_delay / 30)) - (0.05 * advisor_meetings), 0.0, 1.0))
        stipend_ratio = float(np.clip(rng.normal(loc=1.0 - (fellowship_delay / 120), scale=0.2), 0.2, 1.5))
        lab_isolation = float(np.clip(rng.beta(2, 2), 0.05, 0.98))
        
        logit = (
            -2.6
            + 2.2 * stagnation_idx
            + 0.015 * fellowship_delay
            + 0.85 * (1 if advisor_meetings <= 1 else 0)
            + 0.65 * (1 if comp_attempts >= 2 else 0)
            - 0.45 * peer_papers
            + 0.75 * lab_isolation
            - 0.60 * stipend_ratio
        )
        prob = 1.0 / (1.0 + np.exp(-logit))
        is_drop = int(prob > 0.50) if rng.uniform() > 0.08 else int(prob <= 0.50)
        
        records.append({
            "student_id": f"PHD_2026_{i+1:04d}",
            "educational_tier": "DOCTORATE",
            "gender": gender,
            "age": age,
            "family_income_slab": income_slab,
            "is_first_generation": is_first_gen,
            "phd_tenure_months": tenure_months,
            "fellowship_disbursement_delay_days": fellowship_delay,
            "advisor_meeting_frequency_per_month": advisor_meetings,
            "peer_reviewed_submissions": peer_papers,
            "comprehensive_exam_attempts": comp_attempts,
            "research_stagnation_index": round(stagnation_idx, 2),
            "stipend_adequacy_ratio": round(stipend_ratio, 2),
            "lab_isolation_index": round(lab_isolation, 2),
            **geo,
            "ground_truth_risk_prob": round(prob, 4),
            "is_dropout": is_drop
        })
    return pd.DataFrame(records)

def generate_all_universal_cohorts():
    logger.info("Generating Universal 5-Tier Cohort Data (N=10,000)...")
    geo_data = load_geo_index()
    
    df_pre10 = generate_pre_10th(N_PER_TIER, geo_data)
    df_hsc = generate_higher_secondary(N_PER_TIER, geo_data)
    df_ug = generate_undergraduate(N_PER_TIER, geo_data)
    df_pg = generate_postgraduate(N_PER_TIER, geo_data)
    df_phd = generate_doctorate(N_PER_TIER, geo_data)
    
    # Save individual tier files
    df_pre10.to_csv(UNIVERSAL_PROCESSED_DIR / "tier_pre_10th.csv", index=False)
    df_hsc.to_csv(UNIVERSAL_PROCESSED_DIR / "tier_higher_secondary.csv", index=False)
    df_ug.to_csv(UNIVERSAL_PROCESSED_DIR / "tier_undergraduate.csv", index=False)
    df_pg.to_csv(UNIVERSAL_PROCESSED_DIR / "tier_postgraduate.csv", index=False)
    df_phd.to_csv(UNIVERSAL_PROCESSED_DIR / "tier_doctorate.csv", index=False)
    
    logger.info(f"Saved all 5 tier datasets in {UNIVERSAL_PROCESSED_DIR.resolve()}")
    for name, df in [("PRE_10TH", df_pre10), ("HIGHER_SEC", df_hsc), ("UG", df_ug), ("PG", df_pg), ("PHD", df_phd)]:
        logger.info(f" -> Tier [{name}]: Rows={len(df)}, Cols={len(df.columns)}, Dropout Rate={df['is_dropout'].mean():.2%}")

if __name__ == "__main__":
    generate_all_universal_cohorts()
