"""
Verification tests for V4 (Phase 1 — UCI benchmark).
"""

import json
from pathlib import Path
import pytest
from sklearn.pipeline import Pipeline

from verification.conftest import PROJECT_ROOT


class TestV4Phase1UCI:
    @pytest.mark.data
    def test_v4_1_uci_dataset_shape_and_targets(self):
        """V4.1: Exactly 4,424 rows; Target in {Dropout, Graduate, Enrolled}; no is_synthetic."""
        from ml.sources.uci import load_uci_clean_df

        df = load_uci_clean_df()
        assert len(df) == 4424, f"Expected 4424 rows, got {len(df)}"
        assert set(df["target"].unique()) == {"Dropout", "Graduate", "Enrolled"}
        assert "is_synthetic" not in df.columns

    def test_v4_2_temporal_horizons_feature_sets(self):
        """V4.2: ENROLMENT_TIME has no 1st/2nd-sem columns; END_OF_SEM1 has no 2nd-sem columns."""
        from ml.sources.uci import ENROLMENT_TIME, END_OF_SEM1, FULL

        for col in ENROLMENT_TIME:
            assert "1st_sem" not in col and "cu_1st" not in col, f"1st sem column in ENROLMENT_TIME: {col}"
            assert "2nd_sem" not in col and "cu_2nd" not in col, f"2nd sem column in ENROLMENT_TIME: {col}"

        for col in END_OF_SEM1:
            assert "2nd_sem" not in col and "cu_2nd" not in col, f"2nd sem column in END_OF_SEM1: {col}"

        # FULL has both
        has_1st = any("1st_sem" in col or "cu_1st" in col for col in FULL)
        has_2nd = any("2nd_sem" in col or "cu_2nd" in col for col in FULL)
        assert has_1st and has_2nd, "FULL must contain 1st and 2nd semester columns"

    def test_v4_3_models_are_sklearn_pipelines(self):
        """V4.3: Every model in harness is an sklearn Pipeline."""
        from ml.evaluation.harness import build_models

        models = build_models(seed=42)
        assert len(models) >= 3, "Expected at least 3 models in harness"
        for name, pipeline in models.items():
            assert isinstance(pipeline, Pipeline), f"Model {name} is not an sklearn Pipeline"

    @pytest.mark.data
    def test_v4_4_leave_one_course_out_isolation(self):
        """V4.4: Leave-one-course-out: no course in both train and test in any fold."""
        from ml.sources.uci import get_uci_benchmark_dataset
        from sklearn.model_selection import LeaveOneGroupOut

        X, y, groups, _ = get_uci_benchmark_dataset()
        logo = LeaveOneGroupOut()
        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups)):
            train_courses = set(groups[train_idx])
            test_courses = set(groups[test_idx])
            overlap = train_courses.intersection(test_courses)
            assert not overlap, f"Fold {fold_idx} has overlapping courses: {overlap}"

    def test_v4_6_label_variants(self):
        """V4.6: Both label variants exist; primary variant excludes Enrolled."""
        from ml.sources.uci import get_uci_benchmark_dataset

        _, y_prim, _, _ = get_uci_benchmark_dataset(label_variant="primary")
        assert len(y_prim) == 3630, f"Expected 3630 samples in primary variant, got {len(y_prim)}"
        assert set(y_prim) == {0, 1}

        _, y_sens, _, _ = get_uci_benchmark_dataset(label_variant="sensitivity")
        assert len(y_sens) == 4424, f"Expected 4424 samples in sensitivity variant, got {len(y_sens)}"
        assert set(y_sens) == {0, 1}

    @pytest.mark.data
    @pytest.mark.slow
    def test_v4_5_reproducibility_same_seed(self):
        """V4.5: Two runs with same seed produce identical benchmark metrics."""
        from ml.sources.uci import get_uci_benchmark_dataset
        from ml.evaluation.harness import build_models, evaluate_split_strategy
        from sklearn.model_selection import RepeatedStratifiedKFold

        X, y, _, _ = get_uci_benchmark_dataset(feature_set="enrolment_time")
        rskf = list(RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=42).split(X, y))

        models1 = build_models(seed=42)
        res1 = evaluate_split_strategy(X, y, rskf, models1, n_repeats=1, seed=42)

        models2 = build_models(seed=42)
        res2 = evaluate_split_strategy(X, y, rskf, models2, n_repeats=1, seed=42)

        for m in res1:
            assert res1[m]["metrics"]["roc_auc"]["point"] == res2[m]["metrics"]["roc_auc"]["point"]
            assert res1[m]["metrics"]["pr_auc"]["point"] == res2[m]["metrics"]["pr_auc"]["point"]

    def test_v4_7_metric_hierarchy_and_leakage_check(self):
        """V4.7: Majority baseline is worst; ENROLMENT_TIME <= END_OF_SEM1 <= FULL in PR-AUC; ROC-AUC < 0.97."""
        benchmarks_dir = PROJECT_ROOT / "ml" / "artifacts" / "benchmarks"
        enrol_path = benchmarks_dir / "uci_enrolment_time_primary.json"
        sem1_path = benchmarks_dir / "uci_end_of_sem1_primary.json"
        full_path = benchmarks_dir / "uci_full_primary.json"

        assert enrol_path.exists(), f"Missing {enrol_path}"
        assert sem1_path.exists(), f"Missing {sem1_path}"
        assert full_path.exists(), f"Missing {full_path}"

        with open(enrol_path, "r") as f:
            enrol_data = json.load(f)
        with open(sem1_path, "r") as f:
            sem1_data = json.load(f)
        with open(full_path, "r") as f:
            full_data = json.load(f)

        # In enrolment_time and sem1, check ROC-AUC < 0.97 for all models
        for d, name in [(enrol_data, "enrolment_time"), (sem1_data, "end_of_sem1")]:
            for strategy in d.get("results", {}).values():
                for model_name, m_res in strategy.items():
                    roc = m_res["metrics"]["roc_auc"]["point"]
                    assert roc < 0.97, f"Possible target leakage! {model_name} on {name} has ROC-AUC {roc} >= 0.97"
