# DropoutGuard — AI-Powered Early-Warning and Student Support System

> **Smart India Hackathon 2026 (PSID 7-L)**  
> **UN Sustainable Development Goal 4 (SDG 4: Quality Education)**  
> Predictive intelligence, explainable SHAP attributions, prescriptive institutional interventions, and algorithmic fairness for higher education.

---

## Quick Links & Documentation

- [**Frontend API Handover**](docs/frontend_api_handover.md) — Endpoint specifications, schemas & error codes
- [Backend Architecture & Feature Contract](backend/README.md)
- [Feature Data Dictionary](docs/data_dictionary.md)
- [Real-Data Benchmarks](docs/benchmarks.md) (generated)
- [Ethics, Responsible AI & Algorithmic Fairness Audit](docs/ethics_and_fairness.md) (generated)
- [Simulated Cohort Specification](docs/simulation.md) and [Parameter Mapping](docs/simulation_mapping.md) (generated)

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

## Pipeline validation on simulated data (Held-Out Test Set $N=300$)

The evaluation cohort is simulated by `generate_synthetic_indian.py`, and target labels are derived from a known mathematical formula parameterized in that script. Consequently, these metrics demonstrate that the data ingestion, feature engineering, training, calibration, and fairness auditing pipelines function cohesively end-to-end, and they should not be construed as empirical evidence of real-world predictive accuracy.

<!-- METRICS:START -->
- **At-Risk Recall (Sensitivity)**: `78.10%` (Minimizes missed vulnerable students)
- **At-Risk Precision**: `90.11%` (Prevents mentor alert fatigue)
- **Minority Class F1 Score**: `0.8367`
- **Macro-Averaged F1 Score**: `0.8788`
- **ROC-AUC**: `0.9325`
- **Overall Accuracy**: `89.33%`
- **Brier Calibration Score**: `0.0902`
- **Demographic Disparity**: Gender FNR gap $1.42\text{ pp}$, Economic proxy gap $8.08\text{ pp}$, First-Gen gap $7.15\text{ pp}$
<!-- METRICS:END -->

---

## Real-data benchmarks

<!-- BENCHMARKS:START -->
Generated from `ml/artifacts/benchmarks/*.json` (commit `bef625a`; inputs `assessments.csv` `8cc738fb88ad`, `courses.csv` `4f16eee7454b`, `studentAssessment.csv` `fd5320786328`, `studentInfo.csv` `7e6f3e474a5e`, `studentRegistration.csv` `0d3267628537`, `studentVle.csv` `52668253d876`, `uci_dropout.csv` `3ef126de5cef`, `vle.csv` `d1b28303dea8`). Full results, all split strategies and metrics: [docs/benchmarks.md](docs/benchmarks.md). Values are point estimates with 95% bootstrap CIs (1,000 resamples).

#### UCI 697: Portuguese higher education

Primary label, Dropout vs Graduate (Enrolled excluded): n = 3,630, 1,421 dropouts, prevalence 0.3915. Repeated stratified 5-fold CV (3 repeats).

| Feature set | Features | Model | ROC-AUC [95% CI] | PR-AUC [95% CI] |
| :--- | ---: | :--- | :--- | :--- |
| ENROLMENT_TIME | 22 | Majority class | 0.5012 [0.4844, 0.5191] | 0.3924 [0.3747, 0.4103] |
| ENROLMENT_TIME | 22 | Logistic regression | 0.8044 [0.7896, 0.8183] | 0.7668 [0.7434, 0.7875] |
| ENROLMENT_TIME | 22 | XGBoost | 0.8516 [0.8382, 0.8644] | 0.8227 [0.8046, 0.8387] |
| END_OF_SEM1 | 28 | Majority class | 0.5012 [0.4844, 0.5191] | 0.3924 [0.3747, 0.4103] |
| END_OF_SEM1 | 28 | Logistic regression | 0.9334 [0.9242, 0.9422] | 0.9279 [0.9178, 0.9373] |
| END_OF_SEM1 | 28 | XGBoost | 0.9403 [0.9320, 0.9480] | 0.9343 [0.9250, 0.9427] |
| FULL (not early warning) | 34 | Majority class | 0.5012 [0.4844, 0.5191] | 0.3924 [0.3747, 0.4103] |
| FULL (not early warning) | 34 | Logistic regression | 0.9525 [0.9449, 0.9599] | 0.9503 [0.9427, 0.9574] |
| FULL (not early warning) | 34 | XGBoost | 0.9588 [0.9521, 0.9652] | 0.9550 [0.9480, 0.9612] |

> **Not validated on Indian college records.** This result is from a Portuguese polytechnic (UCI 697). The deployed model is trained on a simulated Indian cohort and has never been evaluated on real Indian student data.

#### OULAD: UK online learning

Label: Withdrawn. Temporal holdout: train 2013B + 2013J, test 2014B + 2014J. n = registrations still enrolled at day t.

| t (days) | n | Test n | Prevalence | Model | ROC-AUC [95% CI] | PR-AUC [95% CI] |
| ---: | ---: | ---: | ---: | :--- | :--- | :--- |
| 14 | 28,119 | 16,027 | 0.2024 | Majority class | 0.5000 [0.5000, 0.5000] | 0.2123 [0.2061, 0.2182] |
| 14 | 28,119 | 16,027 | 0.2024 | Logistic regression | 0.5909 [0.5798, 0.6016] | 0.2801 [0.2671, 0.2938] |
| 14 | 28,119 | 16,027 | 0.2024 | XGBoost | 0.5964 [0.5853, 0.6078] | 0.2832 [0.2714, 0.2978] |
| 14 | 28,119 | 16,027 | 0.2024 | GRU | 0.5963 [0.5853, 0.6071] | 0.2851 [0.2721, 0.2994] |
| 28 | 27,538 | 15,729 | 0.1855 | Majority class | 0.5000 [0.5000, 0.5000] | 0.1973 [0.1915, 0.2035] |
| 28 | 27,538 | 15,729 | 0.1855 | Logistic regression | 0.6555 [0.6442, 0.6662] | 0.3228 [0.3080, 0.3393] |
| 28 | 27,538 | 15,729 | 0.1855 | XGBoost | 0.6744 [0.6641, 0.6851] | 0.3572 [0.3405, 0.3749] |
| 28 | 27,538 | 15,729 | 0.1855 | GRU | 0.6603 [0.6494, 0.6708] | 0.3315 [0.3164, 0.3487] |
| 56 | 26,522 | 15,092 | 0.1543 | Majority class | 0.5000 [0.5000, 0.5000] | 0.1635 [0.1575, 0.1694] |
| 56 | 26,522 | 15,092 | 0.1543 | Logistic regression | 0.6645 [0.6529, 0.6756] | 0.2770 [0.2622, 0.2933] |
| 56 | 26,522 | 15,092 | 0.1543 | XGBoost | 0.6916 [0.6801, 0.7024] | 0.2863 [0.2714, 0.3018] |
| 56 | 26,522 | 15,092 | 0.1543 | GRU | 0.6884 [0.6773, 0.6988] | 0.2952 [0.2793, 0.3133] |
| 84 | 25,724 | 14,590 | 0.1281 | Majority class | 0.5000 [0.5000, 0.5000] | 0.1347 [0.1295, 0.1400] |
| 84 | 25,724 | 14,590 | 0.1281 | Logistic regression | 0.6769 [0.6641, 0.6887] | 0.2441 [0.2296, 0.2610] |
| 84 | 25,724 | 14,590 | 0.1281 | XGBoost | 0.6789 [0.6667, 0.6910] | 0.2302 [0.2171, 0.2459] |
| 84 | 25,724 | 14,590 | 0.1281 | GRU | 0.7036 [0.6919, 0.7154] | 0.2701 [0.2534, 0.2894] |

> **Not validated on Indian college records.** This result is from UK distance-learning students (OULAD). The deployed model is trained on a simulated Indian cohort and has never been evaluated on real Indian student data.
<!-- BENCHMARKS:END -->

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

#### Install from the lock file (exact versions)

`requirements.lock` pins every package (ML, backend and research extras, test tools) to the exact
versions the test suite was run with. It was produced with `python -m pip freeze --exclude-editable`
and has been verified on Python 3.14.5 (see its header). To reproduce that environment:

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.lock
pip install -e . --no-deps                            # the project itself, without re-resolving
pytest -m "not data" -q -rs                           # suite without the UCI / OULAD raw files
```

Regenerate the lock after an intentional upgrade with the same `pip freeze` command and commit it.

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

**Access control (first stage).** Every write (`/predict`, `/predict/batch`, `/predict/batch/csv`,
`/interventions/log`) needs `Authorization: Bearer <API_ADMIN_TOKEN>`; without the right token the
API returns 401 `unauthorized`. With no `API_ADMIN_TOKEN` set, all writes are refused, and
`ENVIRONMENT=production` will not start without one. Reads are public while `PUBLIC_READ_ONLY=true`
(the default) and need the same token when it is `false`. The frontend has an "Admin token" field
in the header, kept in memory only. This is one shared token, not per-mentor login.
**Do not load real student data until per-mentor authentication exists.**

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
pytest ml/tests/ -v         # ML core only
pytest backend/tests/ -v    # backend only
pytest -k "predict"         # by keyword
```

For the current number of tests, run `pytest --collect-only -q`.

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
│   ├── raw/                       # Reference benchmark datasets (UCI, OULAD)
│   └── processed/                 # Standardized features.csv & feature_metadata.json
├── docs/
│   ├── frontend_api_handover.md   # API contract: endpoints, schemas, error codes
│   ├── data_dictionary.md         # Feature definitions
│   ├── benchmarks.md              # Generated: real-data benchmark report
│   ├── ethics_and_fairness.md     # Generated: fairness audit report
│   ├── simulation.md              # Generated: simulated cohort parameters
│   ├── simulation_mapping.md      # Generated: parameter estimation from real data
│   └── figures/                   # Generated: benchmark figures
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
│   ├── tests/                     # ML test suite
│   └── validate_pipeline.py       # End-to-end report generator and sanity assertions
├── pytest.ini
├── pyrightconfig.json
├── pyproject.toml
└── README.md
```

---

## Limitations

- **No per-mentor authentication yet**: one shared admin token protects writes, and reads are public by default. Do not load real student data until per-mentor authentication exists.
- **Simulated Data Cohort**: The model is trained and evaluated exclusively on synthetic data generated via domain-informed rules and logistic formulas (`generate_synthetic_indian.py`). While the pipeline models Indian higher education patterns (attendance thresholds, backlogs, fee arrears), it has not been validated on real institutional student records.
- **Absence of Temporal Validation**: The current validation uses static held-out and cross-validation splits rather than time-split validation across sequential academic terms. Temporal degradation and concept drift have not yet been evaluated across multi-semester horizons.
- **Fairness Metrics Reflect Generator Assumptions**: The demographic parity and equalized opportunity evaluations reflect the distributions and functional dependencies programmed into the synthetic data generator rather than the systemic disparities observed in real-world educational institutions.

---

## License & Compliance
Built under the **MIT License** for **Smart India Hackathon 2026**. Designed in compliance with **UN SDG 4** and responsible AI fairness principles.