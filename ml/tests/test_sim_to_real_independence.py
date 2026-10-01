"""
Sim-to-real independence tests (Phase 6 item 5).

- UCI effects that parameterise the simulator are estimated on the estimation rows only.
- The sim-to-real UCI holdout is exactly the set of rows excluded from estimation.
- Simulated-cohort SDs used for effect transfer are computed from a seeded generated cohort.
"""

import pytest

import ml.simulation.estimate_parameters as estimate_parameters
from ml.data_pipeline.generate_synthetic_indian import generate_indian_student_cohort
from ml.simulation.sim_to_real import uci_train_holdout
from ml.simulation.uci_proxies import build_uci_proxies, split_uci_estimation_holdout


@pytest.mark.data
def test_uci_split_is_disjoint_complete_and_deterministic():
    X, y = build_uci_proxies()
    train_index, holdout_index = split_uci_estimation_holdout(X, y)

    assert set(train_index) & set(holdout_index) == set()
    assert set(train_index) | set(holdout_index) == set(X.index)

    train_again, holdout_again = split_uci_estimation_holdout(*build_uci_proxies())
    assert list(train_again) == list(train_index)
    assert list(holdout_again) == list(holdout_index)


@pytest.mark.data
def test_estimation_never_uses_holdout_rows(monkeypatch):
    seen_rows = []

    def recording_fit(X_df, y):
        seen_rows.append(set(X_df.index))
        assert len(X_df) == len(y)
        return {}

    monkeypatch.setattr(estimate_parameters, "compute_logistic_standardized_effects", recording_fit)
    estimate_parameters.estimate_uci_parameters()

    X, y = build_uci_proxies()
    train_index, holdout_index = split_uci_estimation_holdout(X, y)

    assert len(seen_rows) == 1, "estimate_uci_parameters must fit exactly once"
    estimation_rows = seen_rows[0]
    assert estimation_rows & set(holdout_index) == set(), "holdout rows reached parameter estimation"
    assert estimation_rows == set(train_index)

    # The sim-to-real evaluation set is exactly the set excluded from estimation
    _, X_holdout, _, _ = uci_train_holdout()
    assert set(X_holdout.index) == set(holdout_index)
    assert set(X_holdout.index) & estimation_rows == set()


def test_simulated_feature_stds_are_computed_from_seeded_cohort():
    stds = estimate_parameters.get_simulated_feature_stds()

    cohort = generate_indian_student_cohort(
        n_students=estimate_parameters.SIMULATED_SD_N_STUDENTS,
        seed=estimate_parameters.RANDOM_SEED,
        output_path=None,
    )
    cohort["is_hosteler"] = (cohort["hostel_status"] == "Hosteler").astype(int)

    assert list(stds) == estimate_parameters.SIMULATED_SD_FEATURES
    for feat, sd in stds.items():
        assert sd == cohort[feat].std(ddof=1), f"SD for {feat} is not computed from the seeded cohort"
    assert estimate_parameters.get_simulated_feature_stds() == stds
