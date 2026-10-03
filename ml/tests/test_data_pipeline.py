"""
Unit & Integration Tests for DropoutGuard Data Pipeline
Tests:
- Schema correctness across all 4 pillars
- No unexpected nulls in required columns
- Statistical validity and ground-truth correlation directions and magnitudes
- Feature engineering transformations and interaction terms
- Real UCI 697 / OULAD loaders (ml/sources)
"""

import pytest
import numpy as np
import pandas as pd

from ml.data_pipeline.generate_synthetic_indian import generate_indian_student_cohort
from ml.data_pipeline.feature_engineering import (
    generate_processed_feature_dataset,
    PILLAR_COLUMNS
)
from ml.sources.oulad import SNAPSHOT_DAYS


class TestSyntheticIndianCohort:
    """Tests for the synthetic Indian college student cohort generation."""

    @pytest.fixture
    def cohort_df(self):
        return generate_indian_student_cohort(n_students=500, seed=42, output_path=None)

    def test_cohort_row_count_and_uniqueness(self, cohort_df):
        """Verify row count and unique student IDs."""
        assert len(cohort_df) == 500
        assert cohort_df["student_id"].nunique() == 500

    def test_cohort_value_ranges(self, cohort_df):
        """Verify numerical ranges are realistic and bounded."""
        assert cohort_df["attendance_percentage"].between(0.0, 100.0).all()
        assert cohort_df["current_cgpa"].between(0.0, 10.0).all()
        assert cohort_df["prev_sem_cgpa"].between(0.0, 10.0).all()
        assert cohort_df["backlog_count"].ge(0).all()
        assert cohort_df["fee_payment_delay_days"].ge(0).all()
        assert cohort_df["lms_logins_per_week"].ge(0).all()
        assert cohort_df["ground_truth_risk_prob"].between(0.0, 1.0).all()
        assert set(cohort_df["is_dropout"].unique()).issubset({0, 1})

    def test_mandatory_75_percent_attendance_flag(self, cohort_df):
        """Verify that attendance_risk_flag adheres strictly to the 75% rule."""
        expected_flag = (cohort_df["attendance_percentage"] < 75.0).astype(int)
        pd.testing.assert_series_equal(
            cohort_df["attendance_risk_flag"],
            expected_flag,
            check_names=False
        )


class TestGeneratorCoefficientRecovery:
    """
    The dropout labels must carry exactly the configured risk coefficients.

    Scaling: coefficients are compared in raw generator units on the logit scale (no
    standardisation, no rescaling factor). The generator draws
        is_dropout ~ Bernoulli(sigmoid(beta_0 + x.beta + eps)),  eps ~ N(0, sigma^2),
    so a plain logistic fit is attenuated by the noise. The fit here is the same logistic-normal
    model with sigma = idiosyncratic_noise_std from assumptions.yaml, integrated out by
    Gauss-Hermite quadrature, so its MLE targets the configured beta directly.
    Intervals are Bonferroni-adjusted over the k nonzero coefficients (familywise 95% for those k).
    Rule (owner decision): every configured coefficient, including the zero-valued protected ones,
    must lie inside its adjusted interval; the estimate's sign must match the configured sign only
    where the adjusted interval excludes 0 (otherwise the sign of the point estimate is noise).
    Seed, n and the interval width are fixed and must not be tuned to make the test pass.
    """

    N_STUDENTS = 200_000
    SEED = 42
    FAMILYWISE_ALPHA = 0.05
    QUADRATURE_NODES = 40

    @pytest.fixture(scope="class")
    @classmethod
    def recovery(cls):
        from scipy.optimize import minimize
        from scipy.special import expit
        from scipy.stats import norm

        from ml.data_pipeline.generate_synthetic_indian import load_simulation_assumptions

        assumptions = load_simulation_assumptions()
        coefficients = {k: v["value"] for k, v in assumptions["risk_coefficients"].items()}
        sigma = float(coefficients.pop("idiosyncratic_noise_std"))
        th = {k: float(v["value"]) for k, v in assumptions["regulations_and_thresholds"].items()}

        df = generate_indian_student_cohort(n_students=cls.N_STUDENTS, seed=cls.SEED, output_path=None)

        # Design columns mirror the generator's risk formula, built from its own output columns.
        columns = {
            name: df[name].astype(float).to_numpy()
            for name in [
                "current_cgpa", "backlog_count", "has_scholarship", "fee_payment_delay_days",
                "is_first_generation", "lms_logins_per_week", "days_since_last_lms_activity",
                "assignment_submission_lag_days", "attendance_3m_trend",
                "consecutive_absences", "cgpa_delta", "stem_core_fail_flag",
            ]
        }
        # Attendance enters the risk formula only as a hinge below the statutory threshold
        columns["attendance_deficit_slope"] = np.maximum(
            0.0, th["mandatory_attendance_threshold"] - df["attendance_percentage"].astype(float)
        ).to_numpy()
        columns["is_hosteler"] = (df["hostel_status"] == "Hosteler").astype(float).to_numpy()
        columns["gender_male"] = (df["gender"] == "Male").astype(float).to_numpy()
        for cat in ["obc", "sc", "st", "ews"]:
            columns[f"category_{cat}"] = (df["category"] == cat.upper()).astype(float).to_numpy()
        columns["interaction_low_att_high_fee"] = (
            (df["attendance_percentage"] < th["dual_crisis_att_threshold"])
            & (df["fee_payment_delay_days"] > th["dual_crisis_fee_threshold"])
        ).astype(float).to_numpy()
        columns["interaction_low_cgpa_high_backlogs"] = (
            (df["current_cgpa"] < th["academic_crisis_cgpa_threshold"])
            & (df["backlog_count"] >= th["academic_crisis_backlog_threshold"])
        ).astype(float).to_numpy()
        assert set(columns) == set(coefficients), (
            f"Configured coefficients without a design column: {sorted(set(coefficients) - set(columns))}; "
            f"design columns without a configured coefficient: {sorted(set(columns) - set(coefficients))}"
        )

        names = sorted(coefficients)
        X = np.column_stack([np.ones(len(df))] + [columns[n] for n in names])
        y = df["is_dropout"].to_numpy(dtype=float)

        # Start from a plain logistic fit (IRLS).
        beta = np.zeros(X.shape[1])
        for _ in range(50):
            p = expit(X @ beta)
            step = np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]), X.T @ (y - p))
            beta += step
            if np.abs(step).max() < 1e-10:
                break

        nodes, weights = np.polynomial.hermite.hermgauss(cls.QUADRATURE_NODES)
        eps = np.sqrt(2.0) * sigma * nodes
        weights = weights / np.sqrt(np.pi)

        def marginal(b):
            P = expit((X @ b)[:, None] + eps[None, :])
            p = np.clip(P @ weights, 1e-12, 1 - 1e-12)
            return p, (P * (1 - P)) @ weights

        def negloglik(b):
            p, dp = marginal(b)
            score = (y / p - (1 - y) / (1 - p)) * dp
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)), -(X.T @ score)

        res = minimize(negloglik, beta, jac=True, method="L-BFGS-B", options={"maxiter": 2000, "gtol": 1e-8})
        p, dp = marginal(res.x)
        fisher = X.T @ (X * (dp ** 2 / (p * (1 - p)))[:, None])
        se = np.sqrt(np.diag(np.linalg.inv(fisher)))

        nonzero = [n for n in names if coefficients[n] != 0.0]
        z = float(norm.ppf(1 - cls.FAMILYWISE_ALPHA / (2 * len(nonzero))))
        rows = {
            n: {"configured": float(coefficients[n]), "estimate": float(res.x[i]), "se": float(se[i]),
                "lo": float(res.x[i] - z * se[i]), "hi": float(res.x[i] + z * se[i])}
            for i, n in enumerate(names, start=1)
        }
        return {"result": res, "rows": rows, "nonzero": nonzero, "z": z}

    @staticmethod
    def _table(recovery):
        return "\n".join(
            f"  {n:36} configured={r['configured']:+.5f} estimate={r['estimate']:+.5f} "
            f"CI=[{r['lo']:+.5f}, {r['hi']:+.5f}]"
            for n, r in recovery["rows"].items()
        ) + f"\n  (Bonferroni z = {recovery['z']:.4f})"

    def test_fit_converged(self, recovery):
        assert recovery["result"].success, recovery["result"].message

    def test_generator_recovers_configured_coefficients(self, recovery):
        """Every coefficient lies in its adjusted 95% CI; sign must match where that CI excludes 0."""
        rows = recovery["rows"]
        assert len(rows) > 0 and len(recovery["nonzero"]) > 0
        not_covered = [n for n, r in rows.items() if not r["lo"] <= r["configured"] <= r["hi"]]
        sign_resolved = [n for n in recovery["nonzero"] if rows[n]["lo"] > 0 or rows[n]["hi"] < 0]
        assert sign_resolved, "no coefficient has an interval excluding 0; the sign rule would be vacuous"
        wrong_sign = [n for n in sign_resolved if np.sign(rows[n]["estimate"]) != np.sign(rows[n]["configured"])]
        assert not wrong_sign and not not_covered, (
            f"wrong sign (CI excludes 0): {wrong_sign}; CI misses configured value: {not_covered}\n"
            f"sign checked for {len(sign_resolved)} of {len(recovery['nonzero'])} nonzero coefficients\n"
            f"{self._table(recovery)}"
        )


class TestFeatureEngineering:
    """Tests for 4-pillar feature transformations and final processed dataset."""

    @pytest.fixture
    def processed_df(self, tmp_path):
        test_path = tmp_path / "test_features.csv"
        return generate_processed_feature_dataset(n_students=500, seed=42, output_path=test_path)

    def test_no_null_values_in_required_features(self, processed_df):
        """Ensure no missing or NaN values in processed feature set."""
        assert processed_df.isna().sum().sum() == 0, f"Found nulls: {processed_df.isna().sum()[processed_df.isna().sum() > 0]}"

    def test_pillar_columns_present(self, processed_df):
        """Ensure all required columns across the 4 pillars exist in the dataset."""
        for pillar_name, cols in PILLAR_COLUMNS.items():
            for col in cols:
                assert col in processed_df.columns, f"Missing column '{col}' for pillar '{pillar_name}'"

    def test_composite_indices(self, processed_df):
        """Verify behavior and financial stress indices are bounded between [0, 1]."""
        assert processed_df["behavioral_disengagement_index"].between(0.0, 1.0).all()
        assert processed_df["financial_stress_index"].between(0.0, 1.0).all()

    def test_interaction_features_non_negative(self, processed_df):
        """Verify interaction terms are computed properly and are non-negative."""
        assert (processed_df["interaction_att_x_fee"] >= 0.0).all()
        assert (processed_df["interaction_cgpa_x_backlog"] >= 0.0).all()
        assert (processed_df["interaction_firstgen_x_inactivity"] >= 0.0).all()
        assert (processed_df["interaction_att_x_cgpa_drop"] >= 0.0).all()

    def test_academic_crisis_flag_logic(self, processed_df):
        """Verify academic_crisis_flag = (CGPA < 5.0 | backlog >= 2)."""
        expected = ((processed_df["current_cgpa"] < 5.0) | (processed_df["backlog_count"] >= 2)).astype(int)
        pd.testing.assert_series_equal(
            processed_df["academic_crisis_flag"],
            expected,
            check_names=False
        )


@pytest.fixture(scope="module")
def oulad_tables():
    from ml.sources.oulad import load_raw_tables

    return load_raw_tables()


@pytest.mark.data
class TestRealDataLoaders:
    """
    The real UCI 697 and OULAD files load through ml/sources with a binary label and the feature
    columns the code defines. Replaces the loader tests deleted with the legacy
    ml/data_pipeline/load_uci.py and load_oulad.py (fea1a6e); refusal paths are in test_sources_integrity.
    """

    def test_uci_loader_schema_and_target(self):
        from ml.sources.uci import EXPECTED_TARGETS, FEATURE_SETS, get_uci_benchmark_dataset, load_uci_clean_df

        df = load_uci_clean_df()
        assert len(df) > 0
        assert "is_synthetic" not in df.columns
        assert set(df["target"].unique()) == EXPECTED_TARGETS

        n_dropout = int((df["target"] == "Dropout").sum())
        n_primary = int(df["target"].isin(["Dropout", "Graduate"]).sum())
        for feature_set, columns in FEATURE_SETS.items():
            X, y, groups, names = get_uci_benchmark_dataset(feature_set=feature_set, label_variant="primary")
            assert names == columns and list(X.columns) == columns, feature_set
            assert len(X) == len(y) == len(groups) == n_primary, feature_set
            assert set(np.unique(y)) == {0, 1}, feature_set
            assert int(y.sum()) == n_dropout, feature_set

        _, y_sensitivity, _, _ = get_uci_benchmark_dataset(label_variant="sensitivity")
        assert len(y_sensitivity) == len(df)
        assert set(np.unique(y_sensitivity)) == {0, 1}
        assert int(y_sensitivity.sum()) == n_dropout

    @pytest.mark.parametrize("t", SNAPSHOT_DAYS)
    def test_oulad_snapshot_schema_and_target(self, oulad_tables, t):
        from ml.sources.oulad import build_snapshot_dataset

        assert "is_synthetic" not in oulad_tables["studentInfo"].columns
        X, y, splits, groups, audit_df, names = build_snapshot_dataset(t=t, tables=oulad_tables)
        assert len(X) == len(y) == len(groups) == len(audit_df) > 0
        assert list(X.columns) == names
        assert set(np.unique(y)) == {0, 1}
        assert not X.isna().any().any()
        # Behavioural signals (the old loader's sum_click / submission lag) as the snapshot builder names them.
        weekly = [f"clicks_week_{w}" for w in range(1, max(1, t // 7) + 1)]
        for column in ["total_clicks", "days_since_last_activity", "mean_submission_lag", *weekly]:
            assert column in X.columns, column
        assert (X[["total_clicks", *weekly]] >= 0).all().all()
        train_idx, test_idx = splits[0]
        assert len(train_idx) > 0 and len(test_idx) > 0
