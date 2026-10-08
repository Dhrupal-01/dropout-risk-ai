"""
Monotone attendance (audit M3, owner decision): at the same attendance trajectory, raising a
student's attendance never raises the production model's predicted risk.

Sweep ("attendance bundle"): the same shift is added to the 4 course and 3 monthly attendance
columns, each clipped to the generator's bounds; overall attendance is the generator's weighted
mean of the clipped months; the 75% flag, subject std and interactions are rebuilt by
build_engineered_features. attendance_3m_trend is held at the student's original value (owner
decision): near the 100% cap the capped months would otherwise shrink the trend, and an
improvement trend that shrinks is a separate risk signal pulling the other way. So where a month
is capped, the swept rows do not satisfy trend == (m3 - m1) / 2; the claim is about attendance
level at a fixed trajectory. Every other input (including consecutive_absences) is held fixed.
"""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

COURSE_COLS = ["att_core1", "att_core2", "att_lab", "att_elective"]
MONTH_COLS = ["attendance_month_1", "attendance_month_2", "attendance_month_3"]
REBUILT = ["attendance_risk_flag"]  # recomputed only when absent; attendance_3m_trend is held fixed
TARGETS = np.arange(40, 96)  # 40..95 inclusive, step 1
N_SAMPLED = 200
# Covers float32 rounding in sklearn's isotonic calibrator (it casts its output to float32, so a
# ~1e-16 interpolation error can surface as one float32 ulp); far below display precision.
# A step counts as an increase only when p[i+1] > p[i] + FLOAT32_TOLERANCE.
FLOAT32_TOLERANCE = 1e-6


def _bounds():
    from ml.data_pipeline.generate_synthetic_indian import get_param, load_simulation_assumptions

    a = load_simulation_assumptions()
    lo = float(get_param(a, "attendance_distribution", "attendance_min"))
    hi = float(get_param(a, "attendance_distribution", "attendance_max"))
    lab_lo = float(get_param(a, "attendance_distribution", "lab_attendance_min"))
    weights = list(get_param(a, "attendance_distribution", "month_weights"))
    return lo, hi, lab_lo, weights


def sweep_frame(raw: pd.DataFrame, targets: np.ndarray) -> pd.DataFrame:
    """One rebuilt row per (student, target). Raw columns only; engineered columns are rebuilt."""
    from ml.data_pipeline.feature_engineering import build_engineered_features

    lo, hi, lab_lo, weights = _bounds()
    rows = []
    for target in targets:
        df = raw.copy()
        delta = float(target) - df["attendance_percentage"].astype(float)
        for col in COURSE_COLS + MONTH_COLS:
            floor = lab_lo if col == "att_lab" else lo
            df[col] = np.clip(df[col].astype(float) + delta, floor, hi)
        df["attendance_percentage"] = np.round(sum(w * df[m] for w, m in zip(weights, MONTH_COLS, strict=True)), 1)
        df = df.drop(columns=[c for c in REBUILT if c in df.columns])
        df["sweep_target"] = int(target)
        rows.append(df)
    return build_engineered_features(pd.concat(rows, ignore_index=True))


def _stu03_profile() -> dict:
    """STU_03_BORDERLINE_COMMUTE exactly as defined in ml/validate_pipeline.py."""
    src = Path(__file__).resolve().parents[1] / "validate_pipeline.py"
    for node in ast.walk(ast.parse(src.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Dict):
            try:
                d = ast.literal_eval(node)
            except Exception:
                continue
            if isinstance(d, dict) and d.get("student_id") == "STU_03_BORDERLINE_COMMUTE":
                return d
    raise AssertionError("STU_03_BORDERLINE_COMMUTE not found in ml/validate_pipeline.py")


@pytest.fixture(scope="module")
def production(simulated_artifacts):
    import joblib
    from ml.config import FEATURE_NAMES_PATH, MODEL_ARTIFACT_PATH

    model = joblib.load(MODEL_ARTIFACT_PATH)
    feature_names = json.loads(Path(FEATURE_NAMES_PATH).read_text(encoding="utf-8"))
    return model, feature_names


def _predict(model, feature_names, frame: pd.DataFrame) -> np.ndarray:
    from ml.models.calibrate import predict_student_risk

    probs, _ = predict_student_risk(model, frame[feature_names])
    return probs


def _increases(frame: pd.DataFrame, probs: np.ndarray, id_col: str):
    """(id, from_target, to_target, p_from, p_to) wherever risk rises by more than FLOAT32_TOLERANCE
    as attendance rises, i.e. wherever p[i+1] <= p[i] + FLOAT32_TOLERANCE does not hold."""
    out = []
    df = frame[[id_col, "sweep_target"]].assign(p=probs).sort_values([id_col, "sweep_target"])
    for sid, g in df.groupby(id_col, sort=False):
        p, t = g["p"].to_numpy(), g["sweep_target"].to_numpy()
        for i in np.nonzero(np.diff(p) > FLOAT32_TOLERANCE)[0]:
            out.append((sid, int(t[i]), int(t[i + 1]), float(p[i]), float(p[i + 1])))
    return out


def test_every_base_model_carries_the_constraints(production):
    from ml.models.train import MONOTONE_CONSTRAINTS, monotone_constraints_for

    model, feature_names = production
    expected = monotone_constraints_for(feature_names)
    assert sum(1 for c in expected if c) == len(MONOTONE_CONSTRAINTS)
    for cc in model.calibrated_classifiers_:
        assert tuple(cc.estimator.get_params()["monotone_constraints"]) == expected


def test_calibration_is_monotone_non_decreasing(production):
    """Isotonic is non-decreasing by construction; Platt is increasing iff its slope a_ < 0
    (sklearn: p = 1 / (1 + exp(a * f + b)))."""
    model, _ = production
    assert model.method in ("isotonic", "sigmoid")
    for cc in model.calibrated_classifiers_:
        for calibrator in cc.calibrators:
            if model.method == "isotonic":
                assert calibrator.increasing_ is True
            else:
                assert calibrator.a_ < 0, f"Platt slope a_={calibrator.a_} would invert the ranking"


def test_risk_never_increases_as_attendance_rises(production):
    from ml.config import PROCESSED_DATA_PATH, RANDOM_SEED

    model, feature_names = production
    cohort = pd.read_csv(PROCESSED_DATA_PATH)
    sample = cohort.sample(n=N_SAMPLED, random_state=RANDOM_SEED)
    frame = sweep_frame(sample, TARGETS)
    probs = _predict(model, feature_names, frame)
    assert len(probs) == N_SAMPLED * len(TARGETS)
    bad = _increases(frame, probs, "student_id")
    assert not bad, (
        f"{len(bad)} increases in predicted risk as attendance rises "
        f"(student, from %, to %, p_from, p_to), first 20:\n" + "\n".join(map(str, bad[:20]))
    )


def test_stu03_sweep_table(production):
    model, feature_names = production
    raw = pd.DataFrame([_stu03_profile()])
    frame = sweep_frame(raw, TARGETS)
    probs = _predict(model, feature_names, frame)
    print("\nSTU_03_BORDERLINE_COMMUTE attendance sweep (additive shift, clipped, trend held, engineered rebuilt)")
    print(f"{'target':>6} {'overall':>8} {'trend':>7} {'flag':>4} {'subj_std':>8} {'p':>8}")
    for (_, r), p in zip(frame.iterrows(), probs, strict=True):
        if r["sweep_target"] % 5 == 0:
            print(f"{r['sweep_target']:>6} {r['attendance_percentage']:>8.1f} {r['attendance_3m_trend']:>7.2f} "
                  f"{int(r['attendance_risk_flag']):>4} {r['subject_attendance_std']:>8.2f} {p:>8.4f}")
    assert not _increases(frame, probs, "student_id")


def test_tolerance_still_catches_a_real_increase(production):
    """Guard for FLOAT32_TOLERANCE: a 0.001 rise injected into a copy of the STU_03 sweep is reported."""
    model, feature_names = production
    frame = sweep_frame(pd.DataFrame([_stu03_profile()]), TARGETS)
    probs = _predict(model, feature_names, frame).astype(np.float64)
    injected = probs.copy()
    i = len(injected) // 2
    injected[i + 1] = injected[i] + 0.001
    bad = _increases(frame, injected, "student_id")
    assert [(b[1], b[2]) for b in bad] == [(int(TARGETS[i]), int(TARGETS[i + 1]))], bad
