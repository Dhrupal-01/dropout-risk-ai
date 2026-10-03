---
paths:
  - "ml/**"
  - "scripts/**"
  - "verification/**"
  - "data/**"
  - "docs/benchmarks.md"
  - "docs/ethics_and_fairness.md"
  - "docs/simulation*.md"
  - "docs/data_dictionary.md"
---

# ML and data rules

## Feature contract

- 28 raw inputs (`RAW_FEATURE_COLUMNS` in `backend/app/services/ml_service.py`) plus 9 engineered
  columns from `ml.data_pipeline.feature_engineering.build_engineered_features`. Training and
  serving call the same function. Never compute engineered features anywhere else.
- `ml/artifacts/feature_names.json` is the source of truth for model column order.
- `is_dropout` and `ground_truth_risk_prob` are labels. They must never reach `predict_proba`.
- `income_slab_idx` is a model input (need signal for financial support routing). Document it as
  such; never describe income as "excluded".

## Simulated Indian cohort

- Every generation constant lives in `ml/simulation/assumptions.yaml` with
  `source` ∈ {`estimated_from_uci`, `estimated_from_oulad`, `indian_regulation`, `assumption`,
  `TODO(citation)`}. No numeric literals for generation inside the Python file.
- Labels are Bernoulli: `is_dropout ~ Bernoulli(p)`. The intercept is solved with `brentq` so the
  expected rate equals `target_base_rate` (currently a 0.355 placeholder marked `TODO(citation)`).
- Gender and caste-category coefficients stay 0 unless the owner sets them in the YAML.
- Generator output schema must stay byte-for-byte the same column set; the backend depends on it.
- Transfer of effects from real data: Indian-unit coefficient = standardized coefficient ÷ SD of
  that feature in the simulated cohort. The SD must be computed from a generated cohort, not typed in.

## Real datasets

- Only `ml/sources/uci.py` and `ml/sources/oulad.py` read real data. They must refuse any file
  with an `is_synthetic` column and fail with download instructions when data is missing.
- UCI 697: 4,424 rows, Target ∈ {Dropout, Graduate, Enrolled}. Primary label = Dropout vs
  Graduate (Enrolled excluded); sensitivity label = Enrolled coded 0. Report both.
- UCI feature sets are explicit allow-lists: `ENROLMENT_TIME` (admission-time only),
  `END_OF_SEM1` (+ 1st-semester units), `FULL` (labelled "not early warning").
- OULAD point-in-time rules for `snapshot(t)`:
  - Population: registrations with `date_unregistration` null or `> t`.
  - Filter every raw table to `date <= t` **before** any groupby, merge or join.
  - Scores count only for assessments with deadline `<= t - 7`. Exclude `is_banked == 1`.
    Assessments with null date (exams) are never "due".
  - Split: train 2013B + 2013J, test 2014B + 2014J. Secondary: leave-one-module-out.
- One shared OULAD feature builder for benchmarks, fairness audits and parameter estimation.

## Evaluation

- Always include baselines: majority class and logistic regression.
- Metrics: ROC-AUC, PR-AUC, precision/recall at top 10% and 20% (mentor capacity), Brier, ECE,
  all with 95% bootstrap CIs (1,000 resamples, fixed seed). Never headline accuracy.
- Grouped CV (leave-one-course/module-out): report mean ± SD of per-fold metrics as primary;
  pooled out-of-fold metrics only as a labelled secondary.
- All preprocessing inside sklearn Pipelines, fit on training folds only.
- Leakage protocol when a result looks too good (e.g. ROC-AUC ≥ 0.97 on early-warning features):
  print top-15 importances, refit with shuffled labels (must fall to ~0.5), re-check time filters
  and group overlap. Report findings; never silently drop features.

## Fairness

- Two explicit lists per source (target design, see `docs/tasks/phase6_fixes.md`):
  PROTECTED (never features) and AUDIT_GROUPS (grouping variables that may also be features,
  documented). Tests must intersect the actual feature-matrix columns with PROTECTED without
  filtering either side first.
- Groups with n < 50 are flagged "insufficient sample" and get no reported gap.
- Reports state numbers and CIs only. No evaluative wording ("fair", "unbiased", "no bias").

## Artifacts and docs

- JSON in `ml/artifacts/` is the source of truth. Docs are rendered from it by `scripts/`.
- Every benchmark/fairness JSON should record input-file checksums, git commit and library versions.
- Never commit model binaries, parquet, raw data or generated CSVs.
