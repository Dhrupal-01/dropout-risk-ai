# Phase 6: fix audit findings

Scope: correctness fixes only. No new features. Do not touch the universal engine or geo analytics.
Never edit `verification/REPORT.md` by hand. Work in this order; one commit per numbered item;
use plan mode and show the plan before editing.

## 1. Synthetic contamination (critical)

- `ml/tests/test_data_pipeline.py` calls the legacy loaders with `allow_synthetic_standin=True`, which
  write stand-in data to `data/raw/uci_dropout.csv`. Make `target_path` required whenever the flag is
  set, and pass pytest's `tmp_path` in tests.
- If `ml/data_pipeline/load_uci.py` and `load_oulad.py` are unused outside tests, delete them and their
  tests. `ml/sources/*` become the only loaders.
- `ml/sources/uci.py` and `ml/sources/oulad.py`: refuse any file containing an `is_synthetic` column.
  Record SHA-256 checksums of the official files in `ml/sources/checksums.json` and verify on load;
  on mismatch, raise with download instructions.
- Test: put a stand-in file at the real cache path inside a temporary project dir; the real loader
  must raise.

## 2. Protected vs audited attributes (critical)

- Create `ml/fairness/attributes.py` with two explicit lists per source:
  - PROTECTED (never model features): UCI `gender`, `age_at_enrollment`; OULAD `gender`, `age_band`,
    `disability`, `imd_band`, `region`; simulated `gender`, `category`.
  - AUDIT_GROUPS (grouping variables that may also be model features, documented as such): UCI
    `scholarship_holder`, `debtor`, `displaced`; OULAD `highest_education`; simulated `income_slab_idx`.
- Remove PROTECTED columns from every UCI feature set and the OULAD snapshot.
- Replace `ml/tests/test_fairness.py::test_audit_attributes_strictly_excluded_from_model_features` and
  verification V7.1 with tests that intersect the actual feature-matrix columns returned by each source
  with PROTECTED, without filtering or dropping either side first. Include UCI.
- Rerun UCI and OULAD benchmarks and fairness audits; regenerate docs through the renderers.

## 3. OULAD benchmark regeneration (critical)

- One shared feature builder for benchmarks, fairness audits and parameter estimation (they currently
  produce 19 vs 24 features).
- Rerun `python -m ml.evaluation.run --source oulad` on the full real data. In `run.py`, assert that the
  population at every t is at least the number of registrations whose `final_result` is not
  `Withdrawn`, computed from the data.
- Every benchmark and fairness JSON records input-file checksums, git commit and library versions.
  Renderers refuse to combine artifacts with different checksums.
- `scripts/render_benchmark_report.py`: remove the hardcoded "N = 32,593"; every number comes from JSON.

## 4. Make the verification report un-fakeable

- Delete `verification/REPORT.md`. Add `scripts/render_verification_report.py`: runs
  `python -m pytest verification --junitxml=<tmp>` and builds `REPORT.md` only from the XML (test id,
  outcome, skip reason, failure message) plus git HEAD and a hash of `pip freeze`. No free text.
- Rewrite hollow tests:
  - V5.2: recompute total clicks, days since last activity and submitted-by-t for 200 random
    registrations directly from the raw tables and compare exactly.
  - V5.3: append rows dated after t to copies of `studentVle` and `studentAssessment`; snapshot(t)
    must be unchanged.
  - V5.5: build a case with a score whose deadline is in (t-7, t] and assert it is excluded.
  - V1.1: use `sys.executable -m pytest`, never `.venv/bin/pytest`.
  - V1.2: diff changed assert lines and numeric thresholds, not only `def test_` lines, and list them.

## 5. Sim-to-real independence

- `ml/simulation/estimate_parameters.py` estimates UCI effects on a train split only; `sim_to_real.py`
  evaluates on the disjoint holdout via one shared split function. Add a test that holdout rows are
  never used in estimation.
- `get_simulated_feature_stds()`: compute SDs from a generated cohort with a fixed seed.
- `docs/simulation_mapping.md` (via its renderer): state that the UCI fee-delay proxy is binary.

## 6. Reproducibility

- Declare `pyyaml` and `python-dotenv` in `pyproject.toml`. Add a pinned lock file (uv or pip-compile)
  and document installing from it in the README.
- Tests that need generated artifacts build them in a session fixture into a temp dir, or are marked
  `@pytest.mark.artifacts` with a clear skip reason (register the marker). Tests that need UCI/OULAD
  files are marked `@pytest.mark.data`.
- `git rm --cached` the duplicate root `DROPOUTGUARD_PROJECT_DOCUMENTATION.txt`.

## 7. Grouped-CV metrics

Report mean ± SD of per-fold metrics as primary for leave-one-course/module-out; pooled out-of-fold
metrics as a labelled secondary.

## 8. README

Add a "Real-data benchmarks" section rendered from JSON (UCI tables now; OULAD after item 3).

## Acceptance (paste raw terminal output)

- In a fresh clone: `python -m pytest ml/tests -q`, then
  `python -c "from ml.sources.uci import load_uci_clean_df; load_uci_clean_df()"` must raise a
  download or checksum error, not return data.
- The OULAD benchmark `n_samples` at t=14 and the fairness test-set size come from the same run and the
  same checksums.
- `python -m scripts.render_verification_report` regenerates `REPORT.md` from junit XML.
- `pytest`, `pytest verification -rs`, `npm run build`, `npm run lint` output pasted in full summary form.
