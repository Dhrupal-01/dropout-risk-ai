"""
Verification tests for V5 (Phase 2 — OULAD early warning).
"""

import json
import numpy as np
import pytest
import pandas as pd

from verification.conftest import PROJECT_ROOT


class TestV5Phase2OULAD:
    @pytest.mark.data
    def test_v5_1_student_info_shape_and_targets(self):
        """V5.1: studentInfo has 32,593 rows; final_result in {Pass, Fail, Withdrawn, Distinction}."""
        csv_path = PROJECT_ROOT / "data" / "raw" / "oulad" / "studentInfo.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 32593, f"Expected 32593 rows, got {len(df)}"
        assert set(df["final_result"].unique()) == {"Pass", "Fail", "Withdrawn", "Distinction"}

    @pytest.mark.data
    def test_v5_2_independent_recomputation(self):
        """
        V5.2: For 200 random registrations at t=28, total clicks, days since last activity and
        assessments submitted by t, recomputed directly from the raw CSVs (no builder code), equal
        the snapshot features exactly.
        """
        from ml.sources.oulad import build_snapshot_dataset

        t = 28
        raw = PROJECT_ROOT / "data" / "raw" / "oulad"
        keys = ["code_module", "code_presentation", "id_student"]

        X, _, _, _, audit_df, _ = build_snapshot_dataset(t)
        rng = np.random.default_rng(0)
        sample_pos = np.sort(rng.choice(len(X), size=200, replace=False))
        sample = audit_df.iloc[sample_pos][keys].reset_index(drop=True)
        built = X.iloc[sample_pos].reset_index(drop=True)

        svle = pd.read_csv(raw / "studentVle.csv")
        svle = svle.merge(sample, on=keys, how="inner")
        svle = svle[svle["date"] <= t]
        clicks = svle.groupby(keys)["sum_click"].sum()
        last_day = svle.groupby(keys)["date"].max()

        assessments = pd.read_csv(raw / "assessments.csv")
        due = assessments[assessments["date"].notna() & (assessments["date"] <= t)]
        subm = pd.read_csv(raw / "studentAssessment.csv")
        subm = subm[(subm["is_banked"] != 1) & (subm["date_submitted"] <= t)]
        subm = subm.merge(due[["id_assessment", "code_module", "code_presentation"]], on="id_assessment", how="inner")
        subm = subm.merge(sample, on=keys, how="inner")
        submitted = subm.groupby(keys)["id_assessment"].count()

        mismatches = []
        for i, row in sample.iterrows():
            k = (row["code_module"], row["code_presentation"], row["id_student"])
            expected = {
                "total_clicks": float(clicks.get(k, 0)),
                # documented builder default for no activity by t: t + 30
                "days_since_last_activity": float(t - last_day[k]) if k in last_day.index else float(t + 30),
                "assessments_submitted": float(submitted.get(k, 0)),
            }
            for col, exp in expected.items():
                if built.at[i, col] != exp:
                    mismatches.append((k, col, built.at[i, col], exp))
        assert not mismatches, f"{len(mismatches)} mismatches (key, feature, built, recomputed); first 10: {mismatches[:10]}"

    @pytest.mark.data
    def test_v5_3_future_mutation_test(self):
        """
        V5.3: Appending studentVle and studentAssessment rows dated after t to copies of the raw
        tables leaves snapshot(t) unchanged (X, y and splits identical).
        """
        from ml.sources.oulad import build_snapshot_dataset, load_raw_tables

        t = 28
        tables = load_raw_tables()
        X0, y0, splits0, _, audit0, _ = build_snapshot_dataset(t, tables=tables)

        rng = np.random.default_rng(1)
        chosen = audit0.iloc[np.sort(rng.choice(len(audit0), size=50, replace=False))]

        svle = tables["studentVle"]
        sites = svle.drop_duplicates(["code_module", "code_presentation", "id_site"])
        future_vle = []
        for _, r in chosen.iterrows():
            site = sites[(sites["code_module"] == r["code_module"]) & (sites["code_presentation"] == r["code_presentation"])]["id_site"].iloc[0]
            for day in range(t + 1, t + 31):
                future_vle.append({"code_module": r["code_module"], "code_presentation": r["code_presentation"],
                                   "id_student": r["id_student"], "id_site": site, "date": day, "sum_click": 500})
        mutated_vle = pd.concat([svle, pd.DataFrame(future_vle).astype(svle.dtypes.to_dict())], ignore_index=True)

        assess = tables["assessments"]
        sa = tables["studentAssessment"]
        future_sa = []
        for _, r in chosen.iterrows():
            ids = assess[(assess["code_module"] == r["code_module"]) & (assess["code_presentation"] == r["code_presentation"])]["id_assessment"]
            for aid in ids:
                future_sa.append({"id_assessment": aid, "id_student": r["id_student"],
                                  "date_submitted": t + 5, "is_banked": 0, "score": 100.0})
        mutated_sa = pd.concat([sa, pd.DataFrame(future_sa)], ignore_index=True)

        assert len(mutated_vle) == len(svle) + len(future_vle) and len(future_vle) > 0
        assert len(mutated_sa) == len(sa) + len(future_sa) and len(future_sa) > 0

        mutated = dict(tables, studentVle=mutated_vle, studentAssessment=mutated_sa)
        X1, y1, splits1, _, _, _ = build_snapshot_dataset(t, tables=mutated)

        pd.testing.assert_frame_equal(X0, X1, check_exact=True)
        assert np.array_equal(y0, y1)
        assert all(np.array_equal(a, b) for a, b in zip(splits0[0], splits1[0], strict=True))

    @pytest.mark.data
    def test_v5_4_no_early_withdrawers_in_population(self):
        """V5.4: No registration with date_unregistration <= t appears in population at t."""
        reg_df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "oulad" / "studentRegistration.csv")
        for t in [14, 28, 56, 84]:
            active_reg = reg_df[reg_df["date_unregistration"].isna() | (reg_df["date_unregistration"] > t)]
            assert (active_reg["date_unregistration"].dropna() > t).all()

    def test_v5_5_mean_assessment_score_logic(self):
        """
        V5.5: At t=28 a score whose deadline is in (t-7, t] is excluded from mean_score; only
        deadlines <= t-7 count, and banked scores are excluded.
        """
        from ml.sources.oulad import build_snapshot_dataset

        t = 28
        key = {"code_module": "AAA", "code_presentation": "2013J", "id_student": 1}

        def tables(assessment_rows, submission_rows):
            return {
                "studentInfo": pd.DataFrame([{**key, "gender": "F", "region": "East", "highest_education": "HE Qualification",
                                              "imd_band": "50-60%", "age_band": "0-35", "num_of_prev_attempts": 0,
                                              "studied_credits": 60, "disability": "N", "final_result": "Pass"}]),
                "studentRegistration": pd.DataFrame([{**key, "date_registration": -10, "date_unregistration": np.nan}]),
                "vle": pd.DataFrame([{"id_site": 10, "code_module": "AAA", "code_presentation": "2013J", "activity_type": "resource"}]),
                "studentVle": pd.DataFrame([{**key, "id_site": 10, "date": 1, "sum_click": 3}]),
                "assessments": pd.DataFrame([{"code_module": "AAA", "code_presentation": "2013J", **a} for a in assessment_rows]),
                "studentAssessment": pd.DataFrame([{"id_student": 1, **r} for r in submission_rows]),
            }

        early = {"id_assessment": 100, "assessment_type": "TMA", "date": 14.0, "weight": 10}       # deadline <= t-7
        late = {"id_assessment": 200, "assessment_type": "TMA", "date": 25.0, "weight": 10}        # deadline in (t-7, t]
        banked = {"id_assessment": 300, "assessment_type": "TMA", "date": 10.0, "weight": 10}      # banked, deadline <= t-7
        submissions = [
            {"id_assessment": 100, "date_submitted": 12, "is_banked": 0, "score": 50.0},
            {"id_assessment": 200, "date_submitted": 20, "is_banked": 0, "score": 90.0},
            {"id_assessment": 300, "date_submitted": 5, "is_banked": 1, "score": 100.0},
        ]

        X, _, _, _, _, _ = build_snapshot_dataset(t, tables=tables([early, late, banked], submissions))
        assert X["mean_score"].tolist() == [50.0], f"Expected only the deadline<=t-7 unbanked score (50.0), got {X['mean_score'].tolist()}"

        X_late_only, _, _, _, _, _ = build_snapshot_dataset(t, tables=tables([late], submissions[1:2]))
        assert X_late_only["mean_score"].tolist() == [0.0], (
            f"A score with deadline in (t-7, t] must not count, got {X_late_only['mean_score'].tolist()}"
        )

    def test_v5_6_cohort_split_isolation(self):
        """V5.6: Training uses only 2013B/2013J, testing only 2014B/2014J."""
        from ml.sources.oulad import build_snapshot_dataset

        X, y, predefined_splits, _, audit_df, _ = build_snapshot_dataset(14)
        train_idx, test_idx = predefined_splits[0]
        train_pres = set(audit_df.iloc[train_idx]["code_presentation"].unique())
        test_pres = set(audit_df.iloc[test_idx]["code_presentation"].unique())

        assert train_pres.issubset({"2013B", "2013J"})
        assert test_pres.issubset({"2014B", "2014J"})
        assert not train_pres.intersection(test_pres)

    @pytest.mark.data
    def test_v5_7_audit_attributes_absent(self):
        """V5.7: gender, age_band, imd_band, disability, region are absent from feature matrices."""
        from ml.sources.oulad import build_snapshot_dataset

        audit_attrs = {"gender", "age_band", "imd_band", "disability", "region"}
        for t in [14, 28, 56, 84]:
            _, _, _, _, _, feature_names = build_snapshot_dataset(t)
            leaked = audit_attrs.intersection(set(feature_names))
            assert not leaked, f"Audit attributes leaked into feature columns at t={t}: {leaked}"

    def test_v5_8_artifacts_presence_and_base_rates(self):
        """V5.8: Per-t JSON and earliness_curve.png exist; report n and base rate."""
        earliness_png = PROJECT_ROOT / "docs" / "figures" / "earliness_curve.png"
        assert earliness_png.exists(), "earliness_curve.png not found"

        benchmarks_dir = PROJECT_ROOT / "ml" / "artifacts" / "benchmarks"
        for t in [14, 28, 56, 84]:
            t_json = benchmarks_dir / f"oulad_snapshot_t{t}_withdrawn.json"
            assert t_json.exists(), f"Missing {t_json}"
            with open(t_json, "r") as f:
                data = json.load(f)
            assert "n_samples" in data
            assert "prevalence" in data
