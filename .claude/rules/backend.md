---
paths:
  - "backend/**"
  - "alembic/**"
  - "alembic.ini"
  - "locustfile.py"
---

# Backend rules (FastAPI + SQLAlchemy 2 + Alembic + Postgres)

## Structure

- Endpoints in `backend/app/api/v1/endpoints/` stay thin: validate, call a service, return a schema.
- Business logic in `backend/app/services/`. ORM models in `backend/app/models/`.
  Pydantic v2 request/response models in `backend/app/schemas/`.
- All routes are under `/api/v1`: `/predict`, `/predict/batch`, `/predict/batch/csv`,
  `/students/{id}/explanation`, `/students/{id}/interventions[/history]`,
  `/interventions/catalog`, `/interventions/log`, `/mentors/queue`, `/mentors/filters`,
  `/stats/summary`. `/health` is outside the prefix.
- Error responses go through `backend/app/core/errors.py`. Error codes (`student_not_found`,
  `no_prediction_history`, `invalid_lifecycle_transition`, `validation_error`,
  `invalid_feature_payload`, `model_unavailable`, `internal_error`) are mirrored in
  `frontend/src/api/client.js` `errorMap`. Change both together.

## ML serving

- `MLService` loads artifacts once at startup. Never load a model or explainer per request.
- If artifacts are missing the app still starts and `/health` reports `model_loaded: false`.
- Batch scoring uses the vectorised path; SHAP for a batch is one TreeExplainer call.

## Data invariants

- Predictions are append-only history. Students use `ON DELETE RESTRICT` so audit history
  cannot be deleted by accident. Do not add cascades.
- `latest_predictions` is upserted in the same transaction as each prediction insert.
- `priority_score` is computed at write time in `services/priority_scoring.py`.
- The mentor queue filters, orders and paginates in SQL. No pandas in any request path.
  Keep `limit/offset` for compatibility; `cursor` is keyset pagination.
- Do not add new personal or sensitive fields (caste category, income, health, contact details)
  to storage or API responses without asking. There is no authentication yet.

## Migrations

- `alembic revision --autogenerate -m "..."`, then read and fix the generated file.
- Every migration has a working `downgrade()`. Postgres-only features (e.g. `pg_trgm`) must skip
  cleanly on SQLite.
- Before finishing: `alembic upgrade head`, `alembic downgrade base`, `alembic upgrade head`,
  `alembic check` on a throwaway database.

## Tests and config

- `TEST_DATABASE_URL` must be a throwaway database; the suite drops tables.
- `ENVIRONMENT=production` requires a Postgres `DATABASE_URL`. CORS origins are explicit; never `*`
  with credentials. No secrets in code or tests.
- `universal.py` and `geo_analytics.py` are pending an owner decision: do not extend them.
