"""
OULAD Dataset Source for Early-Warning Benchmark (UCI ID 349)
Dataset: Open University Learning Analytics Dataset (OULAD)
Focus: Time-based early-warning dropout prediction at horizons t in {14, 28, 56, 84} days.

Strict Rules:
- Expects raw CSVs in data/raw/oulad/
- Asserts studentInfo has 32,593 rows and final_result in {Pass, Fail, Withdrawn, Distinction}
- Converts studentVle once to data/interim/studentVle.parquet with explicit dtypes
- Population: date_unregistration is null OR > t (prior withdrawals strictly excluded)
- Label: final_result == "Withdrawn"
- Features: date <= t ONLY
- Static: num_of_prev_attempts, studied_credits, highest_education
- Audit: gender, age_band, imd_band, disability, region strictly in separate audit frame
- Split: train on 2013B + 2013J, test on 2014B + 2014J; secondary LOGO across code_module
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
OULAD_RAW_DIR = RAW_DATA_DIR / "oulad"
INTERIM_DATA_DIR = BASE_DIR / "data" / "interim"
STUDENT_VLE_PARQUET_PATH = INTERIM_DATA_DIR / "studentVle.parquet"

EXPECTED_ROWS = 32593
EXPECTED_RESULTS = {"Pass", "Fail", "Withdrawn", "Distinction"}
SNAPSHOT_DAYS = [14, 28, 56, 84]

REQUIRED_TABLES = [
    "studentInfo.csv",
    "studentRegistration.csv",
    "studentVle.csv",
    "vle.csv",
    "assessments.csv",
    "studentAssessment.csv",
    "courses.csv",
]

EDUCATION_MAPPING: Dict[str, int] = {
    "No Formal quals": 0,
    "Lower Than A Level": 1,
    "A Level or Equivalent": 2,
    "HE Qualification": 3,
    "Post Graduate Qualification": 4,
}

VLE_DTYPES = {
    "code_module": "category",
    "code_presentation": "category",
    "id_student": "int32",
    "id_site": "int32",
    "date": "int16",
    "sum_click": "int16",
}

TOP_ACTIVITY_TYPES = [
    "forumng",
    "oucontent",
    "resource",
    "subpage",
    "homepage",
    "quiz",
    "url",
]


def check_and_get_oulad_dir(data_dir: Optional[Path] = None) -> Path:
    """
    Validates presence of OULAD raw files.
    If missing, prints manual download instructions and raises FileNotFoundError.
    No synthetic fallback.
    """
    path = Path(data_dir) if data_dir else OULAD_RAW_DIR
    if not path.exists():
        raise FileNotFoundError(
            f"OULAD raw data directory not found at: {path}\n"
            "Manual download instructions:\n"
            "1. Download OULAD tables from https://analyse.kmi.open.ac.uk/open_dataset or UCI (ID 349).\n"
            "2. Extract the following 7 CSV tables into data/raw/oulad/:\n"
            f"   {', '.join(REQUIRED_TABLES)}\n"
            "No synthetic fallback is permitted in this benchmark phase."
        )

    missing = [t for t in REQUIRED_TABLES if not (path / t).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required OULAD tables in {path}: {missing}\n"
            "Manual download instructions:\n"
            "Download all tables from https://analyse.kmi.open.ac.uk/open_dataset or UCI (ID 349).\n"
            f"Expected tables: {', '.join(REQUIRED_TABLES)}"
        )

    return path


def load_raw_tables(data_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """
    Loads all 7 raw OULAD tables, asserting schema constraints and converting studentVle to parquet.
    """
    path = check_and_get_oulad_dir(data_dir)
    tables: Dict[str, pd.DataFrame] = {}

    # 1. studentInfo
    logger.info("Loading studentInfo from %s...", path / "studentInfo.csv")
    info_df = pd.read_csv(path / "studentInfo.csv")
    assert len(info_df) == EXPECTED_ROWS, (
        f"Integrity assertion error: Expected {EXPECTED_ROWS} rows in studentInfo, got {len(info_df)}."
    )
    actual_results = set(info_df["final_result"].astype(str).str.strip().unique())
    assert actual_results == EXPECTED_RESULTS, (
        f"Integrity assertion error: Expected final_result in {EXPECTED_RESULTS}, got {actual_results}."
    )
    tables["studentInfo"] = info_df

    # 2. studentRegistration
    logger.info("Loading studentRegistration...")
    tables["studentRegistration"] = pd.read_csv(path / "studentRegistration.csv")

    # 3. courses
    logger.info("Loading courses...")
    tables["courses"] = pd.read_csv(path / "courses.csv")

    # 4. assessments
    logger.info("Loading assessments...")
    tables["assessments"] = pd.read_csv(path / "assessments.csv")

    # 5. studentAssessment
    logger.info("Loading studentAssessment...")
    tables["studentAssessment"] = pd.read_csv(path / "studentAssessment.csv")

    # 6. vle
    logger.info("Loading vle...")
    tables["vle"] = pd.read_csv(path / "vle.csv")

    # 7. studentVle (~10.6M rows)
    # Check if cached interim parquet exists
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if STUDENT_VLE_PARQUET_PATH.exists() and data_dir is None:
        logger.info("Loading studentVle from interim parquet cache: %s", STUDENT_VLE_PARQUET_PATH)
        tables["studentVle"] = pd.read_parquet(STUDENT_VLE_PARQUET_PATH)
    else:
        logger.info("Loading studentVle from CSV with explicit dtypes...")
        svle_df = pd.read_csv(path / "studentVle.csv", dtype=VLE_DTYPES)
        if data_dir is None:
            logger.info("Converting studentVle to parquet at %s...", STUDENT_VLE_PARQUET_PATH)
            svle_df.to_parquet(STUDENT_VLE_PARQUET_PATH, index=False)
        tables["studentVle"] = svle_df

    return tables


def build_snapshot_dataset(
    t: int,
    tables: Optional[Dict[str, pd.DataFrame]] = None,
    data_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, np.ndarray, List[Tuple[np.ndarray, np.ndarray]], np.ndarray, pd.DataFrame, List[str]]:
    """
    Extracts time-bounded snapshot at t days from course start.

    Parameters:
    -----------
    t: Snapshot day from course start (e.g. 14, 28, 56, 84).
    tables: Pre-loaded raw tables dict (optional).
    data_dir: Path to directory containing raw CSVs (optional).

    Returns:
    --------
    (X, y, predefined_splits, groups, audit_df, feature_names)
    - X: Model feature matrix (dynamic interaction features <= t + static variables).
    - y: Binary outcome (1 for Withdrawn, 0 for Pass/Fail/Distinction).
    - predefined_splits: [(train_idx, test_idx)] where train is 2013B+2013J, test is 2014B+2014J.
    - groups: code_module values for Leave-One-Module-Out evaluation.
    - audit_df: Demographic and protected attributes (gender, age_band, imd_band, disability, region).
    - feature_names: Ordered list of feature column names in X.
    """
    if tables is None:
        tables = load_raw_tables(data_dir=data_dir)

    info_df = tables["studentInfo"]
    reg_df = tables["studentRegistration"]
    vle_df = tables["vle"]
    svle_df = tables["studentVle"]
    assessments_df = tables["assessments"]
    student_assess_df = tables["studentAssessment"]

    # 1. POPULATION FILTERING: Registrations active at day t
    # date_unregistration is null OR > t.
    # Students who already withdrew on or before day t must be excluded.
    reg_active = reg_df[reg_df["date_unregistration"].isna() | (reg_df["date_unregistration"] > t)].copy()

    # Join with studentInfo to get outcome and attributes
    pop = pd.merge(
        reg_active,
        info_df,
        on=["code_module", "code_presentation", "id_student"],
        how="inner",
    ).reset_index(drop=True)

    # 2. TARGET LABEL: final_result == "Withdrawn"
    y = (pop["final_result"].astype(str).str.strip() == "Withdrawn").astype(int).values

    # 3. AUDIT FRAME ISOLATION: Demographics strictly excluded from model features
    audit_cols = ["code_module", "code_presentation", "id_student", "gender", "age_band", "imd_band", "disability", "region"]
    audit_df = pop[audit_cols].copy()

    # 4. STATIC FEATURES
    pop["highest_education_encoded"] = pop["highest_education"].map(EDUCATION_MAPPING).fillna(1).astype(int)
    static_feats = pd.DataFrame({
        "num_of_prev_attempts": pop["num_of_prev_attempts"].fillna(0).astype(float),
        "studied_credits": pop["studied_credits"].fillna(60).astype(float),
        "highest_education": pop["highest_education_encoded"].astype(float),
    }, index=pop.index)

    # 5. DYNAMIC INTERACTION FEATURES: Rows with date <= t ONLY
    svle_t = svle_df[svle_df["date"] <= t].copy()

    # Merge studentVle with population keys so we only aggregate for active population
    keys_df = pop[["code_module", "code_presentation", "id_student"]].reset_index().rename(columns={"index": "pop_idx"})
    svle_active = pd.merge(
        svle_t,
        keys_df,
        on=["code_module", "code_presentation", "id_student"],
        how="inner",
    )

    n_pop = len(pop)

    # 5a. Total clicks <= t
    total_clicks = np.zeros(n_pop, dtype=float)
    if not svle_active.empty:
        agg_tot = svle_active.groupby("pop_idx")["sum_click"].sum()
        total_clicks[agg_tot.index.values] = agg_tot.values

    # 5b. Weekly clicks: clicks_week_1 to clicks_week_W where W = t // 7
    n_weeks = max(1, t // 7)
    weekly_clicks_dict: Dict[str, np.ndarray] = {}
    for w in range(1, n_weeks + 1):
        col_name = f"clicks_week_{w}"
        w_clicks = np.zeros(n_pop, dtype=float)
        if not svle_active.empty:
            if w == 1:
                # Week 1 includes pre-course activity (date <= 7)
                mask_w = svle_active["date"] <= 7
            else:
                start_d = (w - 1) * 7
                end_d = w * 7
                mask_w = (svle_active["date"] > start_d) & (svle_active["date"] <= end_d)
            if mask_w.any():
                agg_w = svle_active[mask_w].groupby("pop_idx")["sum_click"].sum()
                w_clicks[agg_w.index.values] = agg_w.values
        weekly_clicks_dict[col_name] = w_clicks

    # 5c. Days since last activity <= t
    # Default penalty = t + 30 for students with zero interactions
    days_since_last = np.full(n_pop, float(t + 30), dtype=float)
    if not svle_active.empty:
        max_d = svle_active.groupby("pop_idx")["date"].max()
        days_since_last[max_d.index.values] = float(t) - max_d.values

    # 5d. Active days <= t (count of distinct days with clicks)
    active_days = np.zeros(n_pop, dtype=float)
    if not svle_active.empty:
        act_d = svle_active.groupby("pop_idx")["date"].nunique()
        active_days[act_d.index.values] = act_d.values

    # 5e. Clicks by activity type (join vle on id_site)
    activity_clicks_dict: Dict[str, np.ndarray] = {}
    if not svle_active.empty and "id_site" in vle_df.columns:
        vle_sub = vle_df[["id_site", "activity_type"]].drop_duplicates("id_site")
        svle_with_type = pd.merge(svle_active, vle_sub, on="id_site", how="left")
        for atype in TOP_ACTIVITY_TYPES:
            col_name = f"clicks_{atype}"
            at_clicks = np.zeros(n_pop, dtype=float)
            mask_at = svle_with_type["activity_type"] == atype
            if mask_at.any():
                agg_at = svle_with_type[mask_at].groupby("pop_idx")["sum_click"].sum()
                at_clicks[agg_at.index.values] = agg_at.values
            activity_clicks_dict[col_name] = at_clicks
    else:
        for atype in TOP_ACTIVITY_TYPES:
            activity_clicks_dict[f"clicks_{atype}"] = np.zeros(n_pop, dtype=float)

    # 6. ASSESSMENT FEATURES
    # Explicitly handle assessments with null date (exams): they are NOT due by t
    valid_assess = assessments_df[assessments_df["date"].notna()].copy()
    valid_assess["date"] = valid_assess["date"].astype(float)
    due_assess = valid_assess[valid_assess["date"] <= t].copy()

    # Pre-compute total assessments due by t per (code_module, code_presentation)
    due_counts = due_assess.groupby(["code_module", "code_presentation"])["id_assessment"].count().rename("assessments_due").reset_index()
    pop_with_due = pd.merge(
        pop[["code_module", "code_presentation", "id_student"]].reset_index().rename(columns={"index": "pop_idx"}),
        due_counts,
        on=["code_module", "code_presentation"],
        how="left",
    )
    assessments_due = pop_with_due["assessments_due"].fillna(0).values.astype(float)

    # Filter studentAssessment: exclude is_banked rows
    valid_subm = student_assess_df[student_assess_df["is_banked"] != 1].copy()
    valid_subm = valid_subm[valid_subm["date_submitted"] <= t].copy()

    # Merge with due assessments to calculate submission count and lag
    if not due_assess.empty and not valid_subm.empty:
        due_subm = pd.merge(
            valid_subm,
            due_assess[["id_assessment", "date", "code_module", "code_presentation"]],
            on="id_assessment",
            how="inner",
        )
        due_subm = pd.merge(
            due_subm,
            keys_df,
            on=["code_module", "code_presentation", "id_student"],
            how="inner",
        )
        due_subm["lag"] = due_subm["date_submitted"] - due_subm["date"]

        # Number of submitted assessments due by t
        subm_counts = due_subm.groupby("pop_idx")["id_assessment"].count()
        assessments_submitted = np.zeros(n_pop, dtype=float)
        assessments_submitted[subm_counts.index.values] = subm_counts.values

        # Mean submission lag
        mean_lags = due_subm.groupby("pop_idx")["lag"].mean()
        mean_submission_lag = np.zeros(n_pop, dtype=float)
        mean_submission_lag[mean_lags.index.values] = mean_lags.values
    else:
        assessments_submitted = np.zeros(n_pop, dtype=float)
        mean_submission_lag = np.zeros(n_pop, dtype=float)

    # Mean score: counting only assessments whose deadline <= t - 7 (scores arrive after marking)
    # Exclude is_banked rows
    deadline_cutoff = t - 7
    scored_due_assess = valid_assess[valid_assess["date"] <= deadline_cutoff]
    mean_score = np.zeros(n_pop, dtype=float)

    if not scored_due_assess.empty and not valid_subm.empty:
        scored_subm = pd.merge(
            valid_subm[valid_subm["score"].notna()],
            scored_due_assess[["id_assessment", "code_module", "code_presentation"]],
            on="id_assessment",
            how="inner",
        )
        scored_subm = pd.merge(
            scored_subm,
            keys_df,
            on=["code_module", "code_presentation", "id_student"],
            how="inner",
        )
        if not scored_subm.empty:
            avg_scores = scored_subm.groupby("pop_idx")["score"].mean()
            mean_score[avg_scores.index.values] = avg_scores.values

    # Assemble complete feature matrix X
    feats_dict: Dict[str, np.ndarray] = {
        "total_clicks": total_clicks,
        "days_since_last_activity": days_since_last,
        "active_days": active_days,
        "assessments_due": assessments_due,
        "assessments_submitted": assessments_submitted,
        "mean_submission_lag": mean_submission_lag,
        "mean_score": mean_score,
        "num_of_prev_attempts": static_feats["num_of_prev_attempts"].values,
        "studied_credits": static_feats["studied_credits"].values,
        "highest_education": static_feats["highest_education"].values,
    }
    # Add weekly clicks
    feats_dict.update(weekly_clicks_dict)
    # Add activity type clicks
    feats_dict.update(activity_clicks_dict)

    X = pd.DataFrame(feats_dict, index=pop.index)
    feature_names = list(X.columns)

    # 7. VALIDATION SPLIT: Train on 2013B + 2013J, test on 2014B + 2014J
    presentations = pop["code_presentation"].astype(str).values
    train_mask = np.isin(presentations, ["2013B", "2013J"])
    test_mask = np.isin(presentations, ["2014B", "2014J"])

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    predefined_splits = [(train_idx, test_idx)]

    # Secondary: Leave-one-module-out across code_module
    groups = pop["code_module"].astype(str).values

    logger.info(
        "OULAD Snapshot t=%d created: N=%d (Train=%d, Test=%d), Features=%d, Prevalence=%.2f%%",
        t, n_pop, len(train_idx), len(test_idx), X.shape[1], float(np.mean(y) * 100) if len(y) > 0 else 0.0
    )

    return X, y, predefined_splits, groups, audit_df, feature_names
