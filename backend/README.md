# DropoutGuard Backend (Phase 4 — FastAPI + PostgreSQL)

FastAPI service over the completed ML core. The ML pipeline is **not** modified by this
layer: the backend imports `ml.*` and treats `ml/artifacts/feature_names.json` plus the
trained model as the source of truth for the feature contract.

---

## The 37-feature contract

Verified empirically against `ml/artifacts/feature_names.json` **and** the trained
`XGBClassifier.feature_names_in_` — not against the Markdown docs.

```
37 model features = 28 RAW inputs + 9 ENGINEERED
```

Clients supply **raw inputs only**. The 9 engineered columns are derived server-side by
`ml.data_pipeline.feature_engineering.build_engineered_features` — the same function that
built the training set — then reordered to match the artifact exactly.

| Group | Columns |
|---|---|
| **Raw (28)** | `age`, `commute_distance_km`, `income_slab_idx`, `is_first_generation`, `has_scholarship`, `fee_payment_delay_days`, `att_core1`, `att_core2`, `att_lab`, `att_elective`, `attendance_month_1..3`, `attendance_percentage`, `attendance_3m_trend`, `consecutive_absences`, `attendance_risk_flag`, `prev_sem_cgpa`, `current_cgpa`, `cgpa_delta`, `backlog_count`, `internal_exam_score_pct`, `stem_core_fail_flag`, `lms_logins_per_week`, `assignment_submission_lag_days`, `resource_access_count`, `days_since_last_lms_activity`, `forum_participation_count` |
| **Engineered (9)** | `subject_attendance_std`, `academic_crisis_flag`, `behavioral_disengagement_index`, `is_hosteler`, `financial_stress_index`, `interaction_att_x_fee`, `interaction_cgpa_x_backlog`, `interaction_firstgen_x_inactivity`, `interaction_att_x_cgpa_drop` |
| **Also accepted** | `hostel_status` (`"Hosteler"` / `"Day Scholar"`) — the source for the engineered `is_hosteler` |

`attendance_3m_trend`, `attendance_risk_flag` and `cgpa_delta` are optional: when omitted
they are derived using the same formulas as the training pipeline.

### Never inference input

| Column | Why |
|---|---|
| `is_dropout`, `ground_truth_risk_prob` | **Labels.** Rejected by `MLService.build_feature_frame` and by `student_service.upsert_student`. |
| `gender`, `category`, `family_income_slab` | Excluded by `ml.config.EXCLUDED_FEATURES`. Only the ordinal `income_slab_idx` reaches the model. |
| `student_id`, `name`, `department` | Identifier and display metadata. |

> **Note:** `age` **is** a live model feature (index 0), despite being filed under
> `demographics_protected` in `data/processed/feature_metadata.json`.

---

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate on Unix
pip install -e ".[backend]"

cp .env.example .env              # then fill in DATABASE_URL
```

### ML artifacts

`*.joblib` and `features.csv` are gitignored, so a fresh clone has no model. Build them
once — the backend refuses to serve predictions without them:

```bash
python -m ml.data_pipeline.feature_engineering   # -> data/processed/features.csv
python -m ml.validate_pipeline --regenerate      # -> trains, calibrates, builds SHAP
```

### Migrations

```bash
alembic upgrade head
alembic revision --autogenerate -m "description"   # after changing models
```

`alembic.ini` holds no URL — `alembic/env.py` injects `DATABASE_URL` from the environment,
so credentials are never committed.

### Seeding

```bash
python -m backend.scripts.seed_students              # import the 2,000-student cohort
python -m backend.scripts.seed_students --limit 200  # subset
python -m backend.scripts.seed_students --predict    # also score each student
```

Idempotent — re-running upserts by `student_id`. `name`, `department` and
`assigned_mentor_id` do not exist in the dataset and are synthesised **deterministically**
from `student_id`, kept strictly out of the `features` payload.

### Run

```bash
uvicorn backend.app.main:app --reload --port 8000
```

- Swagger: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

### Tests

```bash
pytest                       # ml/tests + backend/tests
pytest backend/tests -v
```

Live-database tests skip unless `TEST_DATABASE_URL` is set. Point it at a throwaway Neon
**branch** — the suite creates and drops tables.

---

## Architecture notes

**ML artifacts load once.** The FastAPI lifespan handler calls `ml_service.load()` at
startup; handlers reuse that single instance. Nothing calls `joblib.load` per request.

**Startup degrades, it does not crash.** If artifacts are missing, the app still starts and
`/health` reports `model_loaded: false` with the reason, instead of a crash-loop with no
diagnostics.

**Predictions are append-only.** Each row stores the exact 37-feature snapshot plus the
`model_version` fingerprint, so any historical score is reproducible.

**Neon specifics.** `pool_pre_ping` discards connections the serverless endpoint closed;
`pool_recycle=280s` retires them first. Bare `postgresql://` URLs are rewritten to the
psycopg3 driver automatically.

---

## Status

Implemented: database schema, migrations, ML lifespan, seed script, `GET /health`, CORS.

Deferred to the next phase: the `/api/v1` endpoints. Their modules and routers exist and
are mounted, but carry no routes yet — the services behind them
(`MLService.predict` / `.explain` / `.recommend_interventions` /
`.generate_counterfactual`, `student_service`, `intervention_service`) are built and tested.
