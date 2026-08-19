# DropoutGuard — AI-Powered Academic Dropout Prediction & Intervention System

> **Smart India Hackathon 2026 (PSID 7-L)**  
> **UN Sustainable Development Goal 4 (SDG 4: Quality Education)**  
> Production-grade predictive intelligence, explainable SHAP attributions, prescriptive institutional interventions, and algorithmic fairness for higher education.

---

## Quick Links & Documentation

- [**Frontend API Handover**](docs/frontend_api_handover.md) - every endpoint, request/response, error codes
- [Backend Architecture & Feature Contract](backend/README.md)
- [ML Architecture, Feature Engineering & Pipeline Spec](docs/ml_architecture_and_pipeline.md)
- [Feature Data Dictionary](docs/data_dictionary.md)
- [Ethics, Responsible AI & Algorithmic Fairness Audit](docs/ethics_and_fairness.md)

---

## System Overview

DropoutGuard continuously ingests multi-source student data across **4 Core Pillars**:
1. **Attendance & Discipline**: Overall 3-month attendance %, recent-month trajectory, consecutive absence streaks, and the mandatory 75% AICTE/UGC debarment rule.
2. **Academic Performance**: Current & previous semester CGPA, semester grade velocity, uncleared active backlogs, continuous assessment marks, and core course failure status.
3. **Digital Learning Behavior (LMS)**: Weekly login frequency, assignment submission delays/lags, inactivity recency, and resource access volumes.
4. **Socio-Economic & Financial Resilience**: Tuition fee payment overdue days, family income brackets, scholarship buffers, and first-generation learner status.

```
                  ┌────────────────────────────────────────────────────────┐
                  │             DropoutGuard ML Core Pipeline             │
                  └────────────────────────────────────────────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
        [ Predictive Intelligence ]                       [ Prescriptive Recourse ]
  • Calibrated Probabilities (Platt / Sigmoid)       • 12 Codified Institutional Actions
  • Configurable Risk Tiers (Low/Med/High)           • Counterfactual Recourse Simulation
  • TreeSHAP Signed Local Attributions               • Prioritized Mentor Triage Queue
  • Counselor Plain-Language Statements              • Intervention Lifecycle Tracking
```

---

## Key Performance Highlights (Held-Out Test Set $N=300$)

- **At-Risk Recall (Sensitivity)**: `83.96%` (Minimizes missed vulnerable students)
- **At-Risk Precision**: `89.90%` (Prevents mentor alert fatigue)
- **Minority Class F1 Score**: `0.8683`
- **Macro-Averaged F1 Score**: `0.9000`
- **ROC-AUC**: `0.9752`
- **Overall Accuracy**: `91.00%`
- **Brier Calibration Score**: `0.0689`
- **Demographic Disparity**: Gender FNR gap $4.70\text{ pp}$, Economic proxy gap $3.11\text{ pp}$, First-Gen gap $2.35\text{ pp}$

---

## Quickstart & Local Setup

### Prerequisites

| Requirement | Version |
|---|---|
| Python | **3.12** (3.11+ works; pinned to 3.12.10 in development) |
| PostgreSQL | **16+** — a hosted [Neon](https://neon.tech) database is used in development |
| OS | Windows / macOS / Linux |

### 1. Environment & dependencies

```bash
git clone <repo-url>
cd dropout-risk-ai

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# ML core only
pip install -e .

# ML core + FastAPI backend (what you want for the API)
pip install -e ".[backend]"
```

### 2. Configuration

```bash
cp .env.example .env      # then edit
```

`DATABASE_URL` is **required** and has no default — the app refuses to start without it.

```bash
# Neon (note: sslmode=require is mandatory)
DATABASE_URL=postgresql://user:password@ep-xxx.region.aws.neon.tech/dbname?sslmode=require

# Optional: direct (non-pooled) endpoint used for Alembic DDL
MIGRATION_DATABASE_URL=postgresql://user:password@ep-xxx.region.aws.neon.tech/dbname?sslmode=require

# Optional: a THROWAWAY database for the test suite, which creates and drops tables
TEST_DATABASE_URL=postgresql://user:password@ep-xxx.region.aws.neon.tech/dropoutguard_test?sslmode=require

ENVIRONMENT=development
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

Bare `postgresql://` URLs are rewritten to the psycopg3 driver automatically.
`.env` is gitignored — never commit credentials.

### 3. Build the ML artifacts (required once)

Model binaries and `features.csv` are gitignored, so a fresh clone has none. The backend
serves no predictions until they exist.

```bash
python -m ml.data_pipeline.feature_engineering   # -> data/processed/features.csv (2000 x 44)
python -m ml.validate_pipeline --regenerate      # trains, calibrates, builds the SHAP explainer
```

Produces `ml/artifacts/`: `base_xgboost_model.joblib`, `calibrated_model.joblib`,
`shap_explainer.joblib`, `feature_names.json`, `interventions.json`.

> Re-training on a different machine can select a different champion imbalance strategy,
> so metrics may differ slightly from the committed `model_metrics.json`.

### 4. Database migration

```bash
alembic upgrade head
```

Creates `students`, `predictions`, `intervention_logs` with all indexes and constraints.
`alembic.ini` holds no URL — `alembic/env.py` injects it from the environment.

```bash
alembic revision --autogenerate -m "description"   # after changing models
alembic downgrade -1                               # roll back one revision
alembic current                                    # show applied revision
```

### 5. Seed the reference cohort

```bash
python -m backend.scripts.seed_students                # all 2,000 students
python -m backend.scripts.seed_students --limit 200    # a subset
python -m backend.scripts.seed_students --predict      # also score each student
```

Idempotent — re-running upserts by `student_id`. `name`/`department` do not exist in the
ML dataset and are synthesised deterministically for the dashboard.
`--predict` scores rows one at a time and takes roughly 15 minutes for the full cohort.

### 6. Run the API

```bash
# development, with auto-reload
uvicorn backend.app.main:app --reload --port 8000

# deployment entrypoint (binds $PORT, defaults to 8000)
python -m backend.app.main
```

| URL | Purpose |
|---|---|
| `http://localhost:8000/docs` | **Swagger UI** |
| `http://localhost:8000/redoc` | ReDoc |
| `http://localhost:8000/openapi.json` | OpenAPI schema |
| `http://localhost:8000/health` | Health probe |

### 7. Tests

```bash
pytest                      # everything (ml/tests + backend/tests)
pytest ml/tests/ -v         # ML core only  (27 tests)
pytest backend/tests/ -v    # backend only  (224 tests)
pytest -k "predict"         # by keyword
```

Database-backed tests skip unless `TEST_DATABASE_URL` is set. Point it at a **throwaway**
database — the suite creates and drops tables. Its schema is built by running the real
Alembic migration, so a broken migration fails the suite.

---

## Deployment

Deployable on any standard Python host (Render, Railway, Fly.io, Heroku).

```bash
# Procfile
web: python -m backend.app.main
```

The entrypoint binds `0.0.0.0` on `$PORT`, as platform routing requires.

**Required environment variables**

```bash
DATABASE_URL=postgresql://...      # managed PostgreSQL; SQLite is rejected in production
ENVIRONMENT=production
CORS_ORIGINS=https://your-frontend.example.com
```

- `ENVIRONMENT=production` **requires** a PostgreSQL `DATABASE_URL`. A file-backed database
  is refused at startup, because an ephemeral container filesystem would silently discard
  every prediction and intervention log on restart.
- Run `alembic upgrade head` as a release step.
- ML artifacts must be present in the image (they are gitignored — build them in CI or ship
  them as a build artifact). Artifact paths resolve from the package location, so the
  service works regardless of working directory.
- If artifacts are missing the app still starts and `/health` reports
  `model_loaded: false` rather than crash-looping.
- Connection pooling is tuned for Neon (`pool_pre_ping`, `pool_recycle=280s`).

---

## Repository Structure

```
dropout-risk-ai/
├── data/
│   ├── raw/                       # Raw benchmark datasets (UCI, OULAD)
│   └── processed/                 # Standardized features.csv & feature_metadata.json
├── docs/
│   ├── ml_architecture_and_pipeline.md  # Comprehensive ML & Backend handover guide
│   └── ethics_and_fairness.md           # Quantitative fairness audit report
├── ml/
│   ├── artifacts/                 # Serialized model, explainer, and intervention JSONs
│   │   ├── calibrated_model.joblib
│   │   ├── base_xgboost_model.joblib
│   │   ├── shap_explainer.joblib
│   │   ├── feature_names.json
│   │   └── interventions.json
│   ├── config.py                  # Central configuration & configurable risk thresholds
│   ├── data_pipeline/             # Ingestion & feature engineering scripts
│   ├── models/                    # Training, calibration, SHAP & fairness audit
│   ├── intervention/              # Prescriptive recommendation & counterfactual engine
│   ├── tests/                     # 27 comprehensive automated test cases
│   └── validate_pipeline.py       # End-to-end report generator and sanity assertions
├── pytest.ini
├── pyrightconfig.json
├── pyproject.toml
└── README.md
```

---

## License & Compliance
Built under the **MIT License** for **Smart India Hackathon 2026**. Designed in compliance with **UN SDG 4** and responsible AI fairness principles.