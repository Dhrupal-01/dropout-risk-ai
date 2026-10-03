# DropoutGuard (dropout-risk-ai)

Early-warning system for colleges. It flags students who may need support, explains the flag
in plain language (SHAP drivers), suggests interventions, and tracks mentor outreach.
Built for Smart India Hackathon 2026 (PSID 7-L, SDG 4). Owner: @Dhrupal-01 (ML lead).

## Honest status (repeat this whenever you describe the project)

- The deployed model is trained on a **simulated Indian college cohort**
  (`ml/data_pipeline/generate_synthetic_indian.py`, parameters in `ml/simulation/assumptions.yaml`).
  It has not been validated on Indian college records.
- Real-data evidence comes from public benchmarks: UCI 697 (Portuguese polytechnic) and OULAD
  (UK online learning), both rebuilt from the official releases and verified against the
  checksums in `ml/sources/checksums.json`.
- Never describe metrics from the simulated cohort as real-world accuracy.

## Commands

```bash
# Python 3.12 (3.11+ works)
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[backend,research]"                    # research = torch, fairlearn, pyarrow
cp .env.example .env                                    # DATABASE_URL is required

# Simulated-cohort model artifacts (gitignored; the API cannot score without them)
python -m ml.data_pipeline.feature_engineering          # -> data/processed/features.csv
python -m ml.validate_pipeline --regenerate             # train, calibrate, SHAP (no README render)

# Database + API
alembic upgrade head
python -m backend.scripts.seed_students --limit 200 --predict
uvicorn backend.app.main:app --reload --port 8000       # Swagger at /docs, health at /health

# Frontend (API base URL from VITE_API_BASE_URL, default http://localhost:8000)
cd frontend && npm ci && npm run dev                    # http://localhost:5173
npm run build && npm run lint                           # both must pass before you finish

# Tests
pytest                                                  # ml/tests + backend/tests
pytest -m "not data and not slow"                       # no raw datasets needed
pytest verification -rs                                 # audit suite; -rs shows skip reasons
```

Real-data pipeline (needs `data/raw/uci_dropout.csv` and `data/raw/oulad/*.csv`):

```bash
python -m ml.pipeline run-all                           # all steps in order, then README/docs
```

`python -m ml.pipeline run-all --dry-run` lists the steps and the preflight result without running.

It checks once that no code/config file is modified (generated artifacts and README generated
blocks may be), stamps one run id + commit into every artifact's provenance, stops at the first
failing step, and renders README/docs only after every step succeeded. Renderers fail on mixed
commits or input checksums. Individual steps write JSON only; they never render docs.

Tests that hit Postgres need `TEST_DATABASE_URL` pointing to a **throwaway** database (the suite
drops tables). Without it those tests skip. A skipped test is not a passed test.

## Where things live

| Path | What it is |
|---|---|
| `ml/data_pipeline/` | Simulated cohort generator, feature engineering (28 raw → 37 model features) |
| `ml/sources/` | Real dataset loaders (UCI 697, OULAD). The only code allowed to read real data |
| `ml/evaluation/` | Benchmark harness: splits, baselines, metrics, bootstrap CIs |
| `ml/simulation/` | `ml/simulation/assumptions.yaml`, parameter estimation, sim-to-real check |
| `ml/pipeline.py` | `run-all`: the real-data pipeline runner |
| `ml/fairness/` | Audits, mitigations, 2013→2014 shift check, income ablation |
| `ml/models/`, `ml/intervention/` | XGBoost, calibration, SHAP, 12-action catalog, counterfactual recourse |
| `ml/artifacts/` | Metrics and benchmark/fairness JSON (tracked); model binaries (gitignored) |
| `backend/app/` | FastAPI: `backend/app/api/v1/endpoints/`, `services`, `models`, `schemas`, `core` |
| `alembic/versions/` | Schema source of truth |
| `frontend/src/` | React 19 + Vite + Tailwind 3.4 |
| `scripts/` | Doc renderers. Docs with numbers are generated from JSON |
| `verification/` | Audit test suite; `verification/REPORT.md` is generated from its junit XML by `python -m scripts.render_verification_report` |
| `docs/tasks/` | Task lists and audits (`docs/tasks/phase6_fixes.md`, `docs/tasks/audit_2026-10-03.md`) |

## Non-negotiable rules

1. **Never invent numbers.** No statistic, metric, dataset size, citation or coefficient may be
   typed in from memory. Compute it from data/code, read it from a JSON artifact, or take it from
   a cited source file. If a number is needed and unavailable, write `TODO(citation)` and tell me.
2. **Generated docs are never hand-edited:** the README block between `<!-- METRICS:START -->`
   and `<!-- METRICS:END -->`, `docs/benchmarks.md`, `docs/ethics_and_fairness.md`,
   `docs/simulation.md`, `docs/simulation_mapping.md`. Change the renderer or the data, then regenerate.
3. **No silent synthetic data.** Real-data code must fail loudly when data is missing. Stand-in
   generators may only write to temp paths in tests, never to `data/raw/`.
4. **Feature contract is frozen** unless the task says otherwise: `ml/artifacts/feature_names.json`,
   `RAW_FEATURE_COLUMNS` in `backend/app/services/ml_service.py`, and the generator's output
   schema. If it must change, update all of them, schemas, tests and `docs/data_dictionary.md` together.
5. **Protected attributes** (gender, age, caste category, disability, deprivation index, region)
   are never model features. Audit attributes live in a separate audit frame.
6. **Never commit** `.env`, `*.db`, `*.joblib`, `*.parquet`, `data/raw/*`, `data/interim/*`,
   generated CSVs, or `frontend/dist/`.
7. **Tests:** never delete, skip, xfail or loosen a test, tolerance or threshold to make something
   pass. If you believe a test is wrong, explain why and ask. Never assert on a list you just
   filtered to make the assert true.
8. **Evidence, not claims.** When you say something passed, paste the actual command output.
   Do not summarise results you did not see.
9. **Scope.** Do only what the task asks. Ask before adding pages, endpoints, features or
   dependencies. If instructions or code contradict each other, stop and ask.
10. **Product language.** Say "flags students who may need support", never "predicts who will
    drop out". Outputs are decision support for mentors; nothing is ever punitive or automatic.

## Open findings

Full list with evidence and fixes: `docs/tasks/audit_2026-10-03.md`. Still open: T1, V1–V3, W1–W6,
M1–M7, P1–P2, B1–B6, C1–C3, F1–F3, H1, H2, H4, D1, D2.

## Pending owner decision

`ml/data_pipeline/generate_universal_cohort.py`, `ml/models/universal_engine.py`,
`backend/app/api/v1/endpoints/universal.py`, `backend/app/api/v1/endpoints/geo_analytics.py`,
`data/macro/state_district_indices.json`, `frontend/src/pages/GeoAnalytics.jsx` and
`frontend/src/pages/UniversalPredictor.jsx` were added outside the agreed plan. They use
hand-typed coefficients, threshold labels and unsourced district figures. Do not extend them, link
them in navigation, or use them in any claim until the owner decides to keep or remove them.

## Workflow

- Use plan mode for any change under `ml/`, `alembic/` or the feature contract. Show the plan first.
- One branch per task; conventional commits (`feat(ml): …`, `fix(backend): …`, `docs: …`).
- Before finishing: run the relevant tests plus `npm run build && npm run lint` if the frontend changed.
- UI work: read `docs/ui_redesign_spec.md` first. App-screen framing rules are in
  `docs/frontend_ux_spec.md`.
- Detailed, path-scoped rules load automatically from `.claude/rules/ml-and-data.md`,
  `.claude/rules/backend.md` and `.claude/rules/frontend.md`.
