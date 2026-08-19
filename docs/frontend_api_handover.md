# DropoutGuard — Frontend API Handover

**Backend**: FastAPI + PostgreSQL (Neon) · **Model**: calibrated XGBoost + TreeSHAP
**Status**: Phase 4 complete — every endpoint below is implemented, tested and verified live.

> All examples in this document are **real captured responses**, not illustrations.

---

## 1. Base URL & conventions

| Environment | Base URL |
|---|---|
| Local development | `http://localhost:8000` |
| Deployed | set by the hosting platform (`$PORT`); ask the backend owner for the URL |

- All feature endpoints are under `/api/v1`. `/health` is at the root.
- Everything is JSON except the CSV upload, which is `multipart/form-data`.
- Interactive docs: **`/docs`** (Swagger UI) and **`/redoc`**. The OpenAPI schema is at `/openapi.json` — usable to generate a typed client.
- No authentication yet. Do not ship this publicly without adding it.

### CORS

Origins are environment-configured. `http://localhost:5173` (Vite), `http://localhost:3000` (CRA/Next) and `http://127.0.0.1:5173` are allowed by default, with credentials enabled.

To add a deployed frontend origin, the backend `.env` needs:

```bash
CORS_ORIGINS=http://localhost:5173,https://your-frontend.vercel.app
```

A wildcard `*` is never used together with credentials — the backend force-disables credentials if one is configured, because browsers reject that combination.

---

## 2. Core concepts

### Risk tiers

Assigned by `ml.config.get_risk_tier` from the calibrated probability. **The API never computes tiers itself**, and neither should the UI — always display the `risk_tier` field returned.

| Tier | Calibrated probability | Meaning |
|---|---|---|
| `Low` | `p < 0.33` | No urgent action; routine monitoring |
| `Medium` | `0.33 ≤ p ≤ 0.66` | Watch list; early supportive outreach |
| `High` | `p > 0.66` | Priority mentor outreach |

Thresholds are configurable via `RISK_THRESHOLD_LOW` / `RISK_THRESHOLD_HIGH`, so treat them as data, not constants.

> **Cohort note:** the `Medium` band is genuinely narrow — in the 2,000-student reference cohort it holds only 92 students (1,250 `Low`, 658 `High`). A tier filter returning few Medium results is expected, not a bug.

### Intervention lifecycle statuses

Taken from the real behaviour of `InterventionStatusTracker`.

**`status`** — moves forward only; `COMPLETED` is terminal:

```
ASSIGNED → IN_PROGRESS → APPLIED → COMPLETED
```

A backward move returns **409**. Re-assigning the same intervention after `COMPLETED` starts a new log.

**`outcome_status`** — derived from the risk delta (`baseline − post_intervention`):

| Value | Condition |
|---|---|
| `PENDING_EVALUATION` | No post-intervention probability recorded yet |
| `IMPROVED` | delta > +0.05 |
| `NO_CHANGE` | −0.05 ≤ delta ≤ +0.05 |
| `DETERIORATED` | delta < −0.05 |

> Two deliberate deviations from the ML docstring, which the code contradicts: `RESOLVED` is documented but **no code path can produce it**, so it is not offered; `DETERIORATED` **is** produced but undocumented, so it is included.

### Responsible AI — required UI framing

These outputs are **risk estimates and recommended support**, not verdicts:

- Never label a student as "will drop out". Use "estimated risk", "risk drivers", "recommended support".
- Counterfactual recourse is a **simulation**. Responses carry `is_projection: true` and a `disclaimer` — surface it near the projection.
- The mentor queue orders **outreach**, not punishment. Show its `disclaimer`.
- No endpoint triggers any academic action. Attendance debarment appears only as an existing statutory fact for a mentor to act on.

---

## 3. The feature contract

`POST /predict` requires **28 raw features** plus residency. The 9 engineered features
(`subject_attendance_std`, `academic_crisis_flag`, `behavioral_disengagement_index`,
`is_hosteler`, `financial_stress_index`, and four `interaction_*` terms) are **computed by
the backend** — sending them is rejected with 422.

**Required (28):**

`age`, `commute_distance_km`, `income_slab_idx`, `is_first_generation`, `has_scholarship`,
`fee_payment_delay_days`, `att_core1`, `att_core2`, `att_lab`, `att_elective`,
`attendance_month_1`, `attendance_month_2`, `attendance_month_3`, `attendance_percentage`,
`consecutive_absences`, `prev_sem_cgpa`, `current_cgpa`, `backlog_count`,
`internal_exam_score_pct`, `stem_core_fail_flag`, `lms_logins_per_week`,
`assignment_submission_lag_days`, `resource_access_count`, `days_since_last_lms_activity`,
`forum_participation_count`

**Residency — supply one:** `hostel_status` (`"Hosteler"` | `"Day Scholar"`) or `is_hosteler` (`0`/`1`).

**Optional — derived if omitted** (send them only if you have authoritative values):

| Field | Derived as |
|---|---|
| `cgpa_delta` | `current_cgpa − prev_sem_cgpa` |
| `attendance_3m_trend` | `(attendance_month_3 − attendance_month_1) / 2` |
| `attendance_risk_flag` | `1` if `attendance_percentage < 75` |

**Never send** `is_dropout` or `ground_truth_risk_prob` — these are training labels and are rejected with 422.

### Ranges

| Field | Range | Field | Range |
|---|---|---|---|
| `age` | 15–60 | `prev_sem_cgpa`, `current_cgpa` | 0–10 |
| `commute_distance_km` | 0–100 | `cgpa_delta` | −10–10 |
| `income_slab_idx` | 0–3 (0=`<2 LPA` … 3=`>8 LPA`) | `backlog_count` | 0–20 |
| `is_first_generation`, `has_scholarship`, `stem_core_fail_flag`, `attendance_risk_flag`, `is_hosteler` | 0 or 1 | `internal_exam_score_pct` | 0–100 |
| `fee_payment_delay_days` | 0–365 | `lms_logins_per_week` | 0–50 |
| all `att_*` and `attendance_*` percentages | 0–100 | `assignment_submission_lag_days` | −30–90 (negative = early) |
| `attendance_3m_trend` | −50–50 | `resource_access_count` | 0–1000 |
| `consecutive_absences` | 0–180 | `days_since_last_lms_activity` | 0–365 |
| | | `forum_participation_count` | 0–200 |

---

## 4. Error format

Every error shares one envelope. **Tracebacks are never returned.**

```json
{ "error": "student_not_found", "message": "Student 'ABC' not found", "student_id": "ABC" }
```

| Status | `error` | When |
|---|---|---|
| 404 | `student_not_found` | Unknown `student_id` |
| 404 | `no_prediction_history` | Student exists but was never scored — call `/predict` first |
| 409 | `invalid_lifecycle_transition` | Backward status move; includes `current_status` and `requested_status` |
| 422 | `validation_error` | Schema/range failure; includes a `details[]` array of `{field, type, message}` |
| 422 | `invalid_feature_payload` | Contract violation (label sent, unknown intervention id, bad CSV) |
| 503 | `model_unavailable` | ML artifacts not loaded — check `/health`; sends `Retry-After: 30` |
| 500 | `internal_error` | Unexpected fault; includes an `incident_id` to quote to the backend team |

`details[]` deliberately excludes the submitted values so student data is never reflected back.

---

## 5. Endpoints

### 5.1 `GET /health`

**Purpose:** liveness/readiness for a status badge. No credentials or connection strings are exposed.

**Request:** none.

```json
{
  "status": "healthy",
  "database": "connected",
  "model_loaded": true,
  "environment": "development",
  "model_version": "calibrated-14c199cc584b",
  "feature_count": 37,
  "detail": null
}
```

`status` is `"degraded"` if either the database or the model is unavailable, with the reason in `detail`. Always **200** — read the body, not the status code.

---

### 5.2 `POST /api/v1/predict`

**Purpose:** score one student, persist them, and append a new prediction to their history.

**Body:** `student_id` (required), `features` (required, see §3), optional `name`, `department`, `assigned_mentor_id`, and `include_explanation` (default `false`).

Top SHAP drivers are always stored with the prediction; `include_explanation: true` also returns them.

**Request**

```json
{
  "student_id": "E2E_DEMO_01",
  "name": "Meera Nair",
  "department": "Computer Science & Engineering",
  "assigned_mentor_id": "FAC_007",
  "include_explanation": false,
  "features": {
    "age": 20.5, "commute_distance_km": 28.0, "income_slab_idx": 0,
    "is_first_generation": 1, "has_scholarship": 0, "fee_payment_delay_days": 75,
    "hostel_status": "Day Scholar",
    "att_core1": 40.0, "att_core2": 42.0, "att_lab": 52.0, "att_elective": 41.0,
    "attendance_month_1": 56.0, "attendance_month_2": 45.0, "attendance_month_3": 36.0,
    "attendance_percentage": 44.2, "consecutive_absences": 12,
    "prev_sem_cgpa": 6.2, "current_cgpa": 5.05, "backlog_count": 2,
    "internal_exam_score_pct": 42.0, "stem_core_fail_flag": 1,
    "lms_logins_per_week": 1.5, "assignment_submission_lag_days": 5.8,
    "resource_access_count": 12, "days_since_last_lms_activity": 22,
    "forum_participation_count": 0
  }
}
```

**Response `201`**

```json
{
  "student_id": "E2E_DEMO_01",
  "calibrated_risk_probability": 0.9257,
  "risk_tier": "High",
  "risk_score_percentage": 92.57,
  "evaluation_timestamp": "2026-08-18T14:37:53.907520Z",
  "model_version": "calibrated-14c199cc584b",
  "prediction_id": "6c6fc2cb-96c5-4296-80e2-88ce4990267a",
  "top_drivers": null
}
```

**Errors:** 422 · 503 · 500.

> **No `confidence_interval`.** The ML package implements no interval estimator, so the field is deliberately absent rather than fabricated. Do not build UI expecting it.

---

### 5.3 `POST /api/v1/predict/batch`

**Purpose:** score many students in **one** inference call (~0.5 ms/student vs ~110 ms looping).

**Body:** `students` (1–1000 items, each shaped like a `/predict` body), `sort_by_risk_desc` (default `false`), `include_explanations` (default `false` — SHAP is per-row and slow in bulk).

Duplicate `student_id`s in one batch are rejected (422). One invalid row fails the whole batch — nothing is partially written.

**Response `201`**

```json
{
  "total_students_evaluated": 3,
  "high_risk_count": 2,
  "medium_risk_count": 0,
  "low_risk_count": 1,
  "model_version": "calibrated-14c199cc584b",
  "results": [ { "student_id": "...", "risk_tier": "High", "...": "as in 5.2" } ]
}
```

**Also:** `POST /api/v1/predict/batch/csv` — `multipart/form-data` with `file`, plus query params `sort_by_risk_desc`, `include_explanations`. Accepts the project's own `features.csv` column names; extra columns (labels, engineered, protected attributes) are ignored, never scored. Requires a `student_id` column. Same response shape.

---

### 5.4 `GET /api/v1/students/{student_id}/explanation`

**Purpose:** the top SHAP drivers behind the student's **latest** prediction.

**Query:** `top_k` (default `5`, 1–37).

Explanations come from the immutable snapshot that produced the stored score, so `risk_probability` here always matches the prediction shown elsewhere — even if the student's features changed since.

**Response `200`**

```json
{
  "student_id": "E2E_DEMO_01",
  "risk_probability": 0.9257,
  "risk_tier": "High",
  "top_drivers": [
    {
      "feature_name": "attendance_month_3",
      "display_name": "Current Month Attendance (Recent Month)",
      "feature_value": 36.0,
      "shap_value": 1.2258,
      "impact_direction": "RISK_INCREASING",
      "risk_delta_percentage_points": 26.4,
      "plain_language_explanation": "Low recent-month (Month 3) attendance (36.0%) is increasing risk by 26.4 percentage points."
    }
  ]
}
```

**Rendering:** `impact_direction` is `RISK_INCREASING` (positive `shap_value`) or `RISK_DECREASING`. Colour by direction; size bars by `|risk_delta_percentage_points|`. `plain_language_explanation` is counselor-ready — display it verbatim.

**Errors:** 404 (`student_not_found`, `no_prediction_history`) · 422 (bad `top_k`) · 503 · 500.

---

### 5.5 `GET /api/v1/students/{student_id}/interventions`

**Purpose:** recommended support actions plus a counterfactual projection, both built from the latest prediction.

**Query:** `max_recommendations` (default `3`), `top_k_drivers` (default `4`).

**Response `200`** (abridged)

```json
{
  "student_id": "E2E_DEMO_01",
  "current_risk_probability": 0.9257,
  "current_risk_tier": "High",
  "evaluated_at": "2026-08-18T14:37:53.907520Z",
  "model_version": "calibrated-14c199cc584b",
  "recommended_interventions": [
    {
      "intervention_id": "INT_ATT_01",
      "pillar": "attendance",
      "title": "Mandatory Attendance Counseling & Faculty Mentor Check-in",
      "description": "Schedule a 1-on-1 mentor session to identify root causes of absenteeism...",
      "action_type": "MENTOR_CHECKIN",
      "urgency": "HIGH",
      "suggested_duration_days": 14,
      "matched_driver_feature": "attendance_month_3",
      "rationale": "Triggered by Current Month Attendance (+26.4% risk impact): ..."
    }
  ],
  "counterfactual_recourse": {
    "intervention_plan_name": "Comprehensive Multi-Pillar Support Package",
    "current_risk_prob": 0.9257, "current_risk_tier": "High",
    "projected_risk_prob": 0.0436, "projected_risk_tier": "Low",
    "risk_reduction_pct": 88.2,
    "target_reached": true,
    "required_actions": [
      {
        "feature_name": "attendance_percentage",
        "current_value": 44.2, "target_value": 80.0,
        "plain_language_action": "Raise class attendance from 44.2% to 80.0% (+35.8% via regular attendance & lab make-up)."
      }
    ],
    "counselor_summary": "If the student executes the '...' plan, their dropout risk is projected to decrease...",
    "is_projection": true,
    "disclaimer": "Projection from a model simulation, not a causal guarantee..."
  },
  "disclaimer": "Risk estimate and recommended support only..."
}
```

`intervention_id` values come from the ML catalog artifact — the 12 valid ids are
`INT_ATT_01..03`, `INT_ACAD_01..03`, `INT_FIN_01..03`, `INT_BEH_01..03`.
Fetch the full catalog any time from **`GET /api/v1/interventions/catalog`**.

**Errors:** 404 · 503 · 500.

**Related:** `GET /api/v1/students/{student_id}/interventions/history` returns every logged intervention for the student.

---

### 5.6 `POST /api/v1/interventions/log`

**Purpose:** assign an intervention, or advance one already assigned. One call does both.

If the student has no open log for that `intervention_id`, a new one is created (`created: true`); otherwise the existing log is advanced in place (`created: false`), preserving `created_at` and **appending** to `notes`.

**Body**

```json
{
  "student_id": "E2E_DEMO_01",
  "intervention_id": "INT_ATT_01",
  "assigned_faculty_id": "FAC_007",
  "status": "ASSIGNED",
  "notes": "Mentor call scheduled with student and guardian",
  "scheduled_followup_date": "2026-09-05",
  "baseline_risk_probability": 0.9257,
  "post_intervention_risk_probability": null
}
```

Only `student_id` and `intervention_id` are required. Sending `post_intervention_risk_probability` recomputes `outcome_status` from the risk delta.

**Response `201`**

```json
{
  "created": false,
  "log": {
    "id": "6c6fc2cb-96c5-4296-80e2-88ce4990267a",
    "student_id": "1f2e...uuid",
    "intervention_id": "INT_ATT_01",
    "assigned_faculty_id": "FAC_007",
    "status": "APPLIED",
    "outcome_status": "IMPROVED",
    "baseline_risk_probability": 0.9257,
    "post_intervention_risk_probability": 0.31,
    "risk_delta": 0.6157,
    "notes": "Mentor call scheduled | Attendance recovered to 79%",
    "scheduled_followup_date": "2026-09-05",
    "created_at": "2026-08-18T14:37:55.101Z",
    "updated_at": "2026-08-18T14:37:56.442Z",
    "title": "Mandatory Attendance Counseling & Faculty Mentor Check-in",
    "pillar": "attendance",
    "urgency": "HIGH"
  },
  "disclaimer": "Risk estimate and recommended support only..."
}
```

**Errors:** 404 (unknown student) · 422 (unknown `intervention_id`, invalid `status` value) · **409** (backward transition) · 500.

**409 example**

```json
{
  "error": "invalid_lifecycle_transition",
  "message": "Cannot change status from 'APPLIED' to 'ASSIGNED' - lifecycle only moves forward: ASSIGNED -> IN_PROGRESS -> APPLIED -> COMPLETED.",
  "current_status": "APPLIED",
  "requested_status": "ASSIGNED"
}
```

Disable already-passed statuses in the UI rather than relying on the 409.

---

### 5.7 `GET /api/v1/mentors/queue`

**Purpose:** the prioritised support worklist — the main dashboard view.

**Query parameters**

| Param | Type | Default | Notes |
|---|---|---|---|
| `department` | string | — | Case-insensitive exact match |
| `risk_tier` | `Low`\|`Medium`\|`High` | — | 422 on any other value |
| `assigned_mentor_id` | string | — | e.g. `FAC_007` |
| `limit` | int 1–200 | `25` | |
| `offset` | int ≥ 0 | `0` | |

Each student appears **once**, ranked on their most recent prediction — historical predictions never create duplicate rows. `priority_rank` is cohort-wide across the filtered set, so page 2 continues `4, 5, 6…` rather than restarting.

**Response `200`**

```json
{
  "items": [
    {
      "priority_rank": 1,
      "student_id": "IND_2026_0778",
      "name": "Meera Desai",
      "department": "Computer Science & Engineering",
      "assigned_mentor_id": "FAC_007",
      "risk_probability": 0.9258,
      "risk_tier": "High",
      "risk_score_percentage": 92.58,
      "evaluated_at": "2026-08-17T23:58:12.114Z",
      "attendance": 37.1,
      "cgpa": 3.56,
      "backlogs": 1,
      "fee_delay_days": 48,
      "primary_intervention": "INT_ATT_01",
      "intervention_status": "APPLIED",
      "intervention_outcome_status": "IMPROVED"
    }
  ],
  "total": 660,
  "limit": 25,
  "offset": 0,
  "department": null,
  "risk_tier": "High",
  "assigned_mentor_id": null,
  "disclaimer": "Support triage order based on model risk estimates..."
}
```

`total` is the count **before** pagination — use it for the pager. `attendance`, `cgpa`, `backlogs` and `fee_delay_days` come from the same prediction snapshot as the score, so the numbers always agree with the risk shown. Any of them may be `null`; `primary_intervention` / `intervention_status` are `null` until something is logged.

Students who have never been scored do not appear.

---

## 6. Suggested UI flow

1. **Dashboard** → `GET /mentors/queue?risk_tier=High` — the triage table.
2. **Student detail** → `GET /students/{id}/explanation` (why) + `GET /students/{id}/interventions` (what to do).
3. **Mentor acts** → `POST /interventions/log` with `status: "ASSIGNED"`.
4. **Follow-up** → `POST /interventions/log` again with `IN_PROGRESS` / `APPLIED` / `COMPLETED`, adding `post_intervention_risk_probability` once re-scored to populate `outcome_status`.
5. **Re-score** → `POST /predict` appends new history; the queue and explanation follow the latest automatically.

**Bulk import** → `POST /predict/batch/csv`.

### Practical notes

- `/predict` takes roughly **150 ms** (model + SHAP + database). Show a spinner; don't debounce-spam it.
- Batch is far cheaper per student — prefer it for more than ~3 students.
- `evaluation_timestamp` / `evaluated_at` are UTC ISO-8601; convert for display.
- `prediction_id` and log `id` are UUIDs — treat as opaque strings.
- Probabilities are 0–1 (4 dp); `risk_score_percentage` is the same value 0–100 (2 dp). Use whichever suits the widget; don't recompute one from the other for display.
