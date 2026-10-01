"""
Shared UCI (ID 697) proxy builder and estimation/holdout split.

Used by both ml/simulation/estimate_parameters.py (effects estimated on the train rows only)
and ml/simulation/sim_to_real.py (evaluated on the holdout rows), so the sim-to-real check
never evaluates on rows that informed the simulator's parameters.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.sources.uci import load_uci_clean_df

HOLDOUT_FRACTION = 0.2
SPLIT_SEED = 42

# Portuguese higher-education qualification codes (parent holds a degree)
HE_QUALIFICATION_CODES = {2, 3, 4, 5, 6, 40, 41, 42, 43, 44}


def build_uci_proxies() -> Tuple[pd.DataFrame, pd.Series]:
    """
    UCI proxies for the simulated Indian features (Dropout vs Graduate; Enrolled excluded).
    Index = row index of the UCI clean frame, so splits can be checked row by row.
    - scholarship_holder -> has_scholarship
    - tuition_fees_up_to_date == 0 or debtor == 1 -> fee_payment_delay_days (BINARY 0/1 indicator)
    - neither parent with higher education -> is_first_generation
    - displaced -> is_hosteler
    - 1st-sem grade (0-20) rescaled to 0-10 -> current_cgpa
    - units enrolled - units approved -> backlog_count
    """
    df_raw = load_uci_clean_df()
    df = df_raw[df_raw["target"].isin(["Dropout", "Graduate"])].copy()
    y = (df["target"] == "Dropout").astype(int)

    mother_he = df["mothers_qualification"].isin(HE_QUALIFICATION_CODES)
    father_he = df["fathers_qualification"].isin(HE_QUALIFICATION_CODES)

    X = pd.DataFrame({
        "current_cgpa": (df["cu_1st_sem_grade"] / 2.0).clip(0.0, 10.0).astype(float),
        "backlog_count": np.maximum(0.0, df["cu_1st_sem_enrolled"] - df["cu_1st_sem_approved"]).astype(float),
        "has_scholarship": df["scholarship_holder"].astype(float),
        "fee_payment_delay_days": ((df["tuition_fees_up_to_date"] == 0) | (df["debtor"] == 1)).astype(float),
        "is_first_generation": (~mother_he & ~father_he).astype(float),
        "is_hosteler": df["displaced"].astype(float),
    }, index=df.index)
    return X, y


def split_uci_estimation_holdout(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.Index, pd.Index]:
    """
    Stratified split into estimation (train) rows and sim-to-real holdout rows.
    Returns index labels of X; the two sets are disjoint and together cover X.
    """
    train_X, holdout_X = train_test_split(
        X, test_size=HOLDOUT_FRACTION, random_state=SPLIT_SEED, stratify=y
    )
    return train_X.index, holdout_X.index
