# DropoutGuard — Frontend UX & Design Specification

**Audience**: the engineer building the React frontend
**Companion**: [`frontend_api_handover.md`](frontend_api_handover.md) — the API contract. This document covers *what to build and how it should look*; that one covers *what the server returns*.

---

## 1. Product framing

DropoutGuard is a **decision-support tool for a faculty mentor**, not an automated
enforcement system. Everything on screen exists to answer three questions in order:

| Question | Screen | Endpoint |
|---|---|---|
| *Who needs me most today?* | Triage queue | `GET /mentors/queue` |
| *Why is this student at risk?* | Student detail → drivers | `GET /students/{id}/explanation` |
| *What do I actually do about it?* | Student detail → actions | `GET /students/{id}/interventions` |

The primary user is a mentor with ~170 assigned students and about ten minutes between
classes. Optimise for **fast scanning and one obvious next action**, not for showing off
the model.

### Non-negotiable framing (Responsible AI)

This is an SIH project judged partly on ethics. The backend enforces this server-side; the
UI must not undo it.

| Never show | Show instead |
|---|---|
| "Will drop out", "predicted dropout", "failure" | "Estimated risk", "at-risk", "needs support" |
| "Flagged for debarment" | "Below the 75% attendance requirement" |
| Counterfactual as a promise | "Projected — model simulation, not a guarantee" |
| A bare risk score with no explanation | Score always adjacent to its top drivers |

Every response carrying a `disclaimer` field must render it. The counterfactual has
`is_projection: true` — label that panel **"Projected scenario"**, never "Outcome".

---

## 2. Screens

### 2.1 Triage Dashboard (`/`) — the money screen

Layout, top to bottom:

```
┌────────────────────────────────────────────────────────────────┐
│  DropoutGuard          [health dot] Model v14c199 · 2,000 students │
├────────────────────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │  658     │ │   92     │ │  1,250   │ │  2,000   │  KPI row   │
│  │ High     │ │ Medium   │ │  Low     │ │ Students │            │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘            │
├────────────────────────────────────────────────────────────────┤
│  [Dept ▾] [Risk tier ▾] [Mentor ▾]              [Search]  filters│
├────────────────────────────────────────────────────────────────┤
│  #  Student        Dept   Risk        Att  CGPA  Bk  Fee  Action│
│  1  Meera Desai    CSE    ▓ 92.6% High 37% 3.56  1   48   [View]│
│  2  Riya Verma     IT     ▓ 92.6% High 31% 4.20  4   33   [View]│
├────────────────────────────────────────────────────────────────┤
│  Showing 1–25 of 660                          [Prev] [Next]     │
└────────────────────────────────────────────────────────────────┘
```

- **KPI tiles** are hero numbers, not charts. Derive them from `total` on three
  `risk_tier`-filtered calls, or one unfiltered call at `limit=200`.
- **Filters sit in one row above the table.** Changing a filter resets `offset` to 0.
- **Row → student detail.** Whole row clickable, plus an explicit button for keyboard users.
- `priority_rank` is cohort-wide — display it verbatim. Page 2 starts at 26, not 1.

### 2.2 Student Detail (`/students/:studentId`)

Two columns on desktop, stacked on mobile:

```
┌─────────────────────────────┬──────────────────────────────┐
│ Meera Desai · IND_2026_0778 │  RECOMMENDED SUPPORT         │
│ CSE · Mentor FAC_007        │  ┌────────────────────────┐  │
│                             │  │ HIGH · Attendance      │  │
│    ▓▓▓▓▓▓▓▓▓░  92.6%        │  │ Mandatory Attendance   │  │
│      High risk              │  │ Counseling…            │  │
│                             │  │ Triggered by: Month-3  │  │
│ WHY (top drivers)           │  │ attendance             │  │
│ Month-3 attendance   ──────▶│  │        [Assign]        │  │
│ Absence × fee delay  ─────▶ │  └────────────────────────┘  │
│ Attendance × CGPA    ───▶   │  … 2 more                    │
│ Backlogs × CGPA      ──▶    │                              │
│                             │  PROJECTED SCENARIO          │
│ ⓘ Risk estimate — not a     │  92.6% ──▶ 4.7%  (−87.9%)    │
│   prediction of certain     │  • Raise attendance to 80%   │
│   dropout.                  │  • Clear fee arrears         │
│                             │  ⓘ Simulation, not a         │
│                             │    guarantee.                │
└─────────────────────────────┴──────────────────────────────┘
```

Fetch `explanation` and `interventions` **in parallel** — they're independent.

### 2.3 Bulk Scoring (`/import`) — optional, high demo value

Drag-drop CSV → `POST /predict/batch/csv` → show the returned tier counts and a link to
the queue. Accepts the project's own `features.csv`, which makes for a strong live demo.

---

## 3. Design system

### 3.1 Risk tier colors — **reserved status palette**

Risk tier is a *status*, not a category. Use these exact values and nothing else:

| Tier | Hex | Icon | Label |
|---|---|---|---|
| `Low` | `#0ca30c` | ● / check | "Low" |
| `Medium` | `#fab219` | ▲ / alert-triangle | "Medium" |
| `High` | `#d03b3b` | ■ / alert-octagon | "High" |

**Mandatory: never color-alone.** Each tier chip carries **icon + text label + color**.
This is not stylistic — `#fab219` measures 1.79:1 against a light surface, below the 3:1
threshold, and the icon+label pairing is the documented mitigation. A colorblind user,
a projector, or a printed judging sheet must still read the tier.

These three are validated for colorblind separation (worst adjacent ΔE 11.3 protan,
27.6 normal-vision) — do not substitute "nicer" greens/reds.

Never reuse these three for anything else (chart series, buttons, links). They mean
status and only status.

### 3.2 SHAP driver chart — **diverging, not categorical**

SHAP values are signed: positive raises risk, negative lowers it. That's a diverging
encoding — two poles and a neutral midpoint.

| Direction | Light | Dark |
|---|---|---|
| `RISK_INCREASING` (shap > 0) | `#e34948` | `#e66767` |
| `RISK_DECREASING` (shap < 0) | `#2a78d6` | `#3987e5` |
| Zero baseline | `#f0efec` | `#383835` |

Validated: all checks pass in both modes (CVD ΔE 21.6 light / 19.2 dark).

**Form**: horizontal diverging bar chart, one bar per driver, baseline at zero in the
middle, bars extending right for risk-increasing and left for risk-decreasing.

- Sort by `|risk_delta_percentage_points|` descending (the API already returns them ranked).
- Label each bar with `display_name` on the axis and the signed pp value at the bar end —
  **direct labels, no legend needed** for a single diverging series.
- Bars: 4px rounded ends on the data end only, anchored flat to the zero baseline.
  2px gap between bars.
- Recessive axis and gridlines — muted gray, never black.
- Hover tooltip shows `plain_language_explanation` in full.
- Cap at 5 drivers by default (`?top_k=5`); "Show more" bumps to 8.

### 3.3 Risk score display

The headline number is a **hero stat**, not a gauge or donut:

```
   92.6%          ← 48px, tabular figures, text-primary color
   High risk      ← tier chip: icon + label + tier color
```

A thin horizontal meter beneath is fine (filled proportion = probability, filled in the
tier color, track in a neutral). Do **not** use a speedometer/gauge — it implies precision
the calibrated probability doesn't have, and wastes space.

### 3.4 Foundations

**The app opens in LIGHT mode, always.** Do not auto-switch based on
`prefers-color-scheme` — a demo on an unknown laptop or projector must look identical
every time. Dark is opt-in via the toggle only.

| Token | Light (default) | Dark (toggled) |
|---|---|---|
| `--surface-1` (page background) | `#f7f7f5` | `#1a1a19` |
| `--surface-2` (cards, table) | `#ffffff` | `#232322` |
| `--surface-3` (hover, subtle fill) | `#f0efec` | `#2c2c2a` |
| `--text-primary` | `#0b0b0b` | `#ffffff` |
| `--text-secondary` | `#52514e` | `#c3c2b7` |
| `--text-muted` (axes, captions) | `#82817c` | `#8f8e88` |
| `--border` | `#e5e4e0` | `#38383a` |
| `--accent` (buttons, links, focus) | `#2a78d6` | `#3987e5` |
| `--accent-hover` | `#1f5fac` | `#5b9ded` |

The accent is the validated blue — deliberately distinct from the three reserved status
colors so a primary button never reads as a risk signal.

#### Theme toggle — required

- A sun/moon icon button in the **header, far right**, with `aria-label="Switch to dark mode"` / `"Switch to light mode"`.
- Persist the choice in `localStorage` under `dropoutguard-theme`.
- Apply by stamping `data-theme="dark"` on `<html>`; light is the absence of the attribute.
- On first load with nothing stored → **light**.
- Read the stored value in a tiny inline script in `index.html` *before* React mounts, so
  a returning dark-mode user never sees a white flash.

```css
:root {
  --surface-1: #f7f7f5;
  --text-primary: #0b0b0b;
  --accent: #2a78d6;
  /* …all light tokens */
}
:root[data-theme="dark"] {
  --surface-1: #1a1a19;
  --text-primary: #ffffff;
  --accent: #3987e5;
  /* …all dark tokens */
}
```

Every component reads tokens (`var(--surface-2)`), never raw hex. Both themes must be
checked — dark is a *selected* palette stepped for the dark surface, not an inversion
filter.

- **Type**: one sans stack (Inter / system-ui). Sizes 12 / 14 / 16 / 20 / 32 / 48.
  All numbers use `font-variant-numeric: tabular-nums` so columns align.
- **Spacing**: 4px base scale (4/8/12/16/24/32/48).
- **Radius**: 8px cards, 6px controls, 4px chips.
- **Elevation**: one subtle shadow for cards. No neumorphism, no glassmorphism.
- The three risk-tier status colors (§3.1) and the two SHAP poles (§3.2) are **fixed in
  both themes** — only the surfaces and text around them change.

### 3.5 Anti-patterns — do not ship these

- ❌ A gauge/speedometer for risk
- ❌ A pie or donut of tier counts (use the KPI tiles)
- ❌ Dual-axis charts (attendance and CGPA on one chart with two y-scales)
- ❌ Rainbow gradients across risk levels
- ❌ Color-only tier indication
- ❌ A number on every data point
- ❌ Red/green as the *only* difference between two adjacent things

---

## 4. Component states

Every data component ships four states. Judges notice missing ones.

| State | Requirement |
|---|---|
| **Loading** | Skeleton rows matching final layout. `/predict` takes ~150 ms; batch is longer. |
| **Empty** | Explain *why* it's empty and what to do: "No students match these filters. Clear filters." Never a bare "No data". |
| **Error** | Read `error` from the body and map it (see §5). Show a retry. Never print a raw stack or `[object Object]`. |
| **Success** | The real thing. |

Special empties worth handling explicitly:

- `no_prediction_history` (404) — the student exists but was never scored. Show
  **"Not yet scored"** with a **[Score now]** button calling `POST /predict`. This is a
  normal state, not an error.
- `model_loaded: false` from `/health` — show a persistent banner: "Scoring temporarily
  unavailable." Disable predict buttons; reads still work.

---

## 5. Error handling

Map `error` codes to human copy. Never surface `message` verbatim for 500s.

| Code | HTTP | UI copy |
|---|---|---|
| `student_not_found` | 404 | "Student not found." → back to queue |
| `no_prediction_history` | 404 | "Not yet scored." → [Score now] |
| `invalid_lifecycle_transition` | 409 | "This intervention is already at *{current_status}*." Refresh the log. |
| `validation_error` | 422 | Highlight the offending fields from `details[]` inline |
| `invalid_feature_payload` | 422 | Show `message` — it's specific and safe |
| `model_unavailable` | 503 | "Scoring temporarily unavailable. Retrying…" — honour `Retry-After` |
| `internal_error` | 500 | "Something went wrong. Reference: `{incident_id}`" |

**Prevent the 409 rather than catching it.** The lifecycle only moves forward
(`ASSIGNED → IN_PROGRESS → APPLIED → COMPLETED`); disable already-passed statuses in the
status control instead of letting the user pick one and fail.

---

## 6. Interaction rules

- **Optimistic UI only where safe.** Assigning an intervention may 409 — wait for the
  response before updating the chip.
- **Filters** live in one row, update the URL query string (shareable/back-button-able),
  and reset pagination.
- **Debounce** the search box at 300 ms. Never debounce-spam `/predict`.
- **Prefer batch** over looping `/predict` for more than ~3 students — it's ~200× cheaper
  per student server-side.
- **Timestamps** are UTC ISO-8601. Render relative ("2 hours ago") with the absolute value
  in a `title` attribute.
- **IDs** (`prediction_id`, log `id`) are opaque UUIDs — never parse them.

---

## 7. Accessibility

- Tier chips: icon + label + color (§3.1). Enforced, not optional.
- Chart has a **table view toggle** — required, because three light-mode chart colors sit
  below 3:1 contrast and the table is the documented relief.
- Keyboard: full tab order; queue rows reachable and activatable via Enter.
- `aria-live="polite"` on the risk score region so a screen reader announces re-scores.
- Focus rings visible — never `outline: none` without a replacement.
- Respect `prefers-reduced-motion`: no bar-grow animations when set.

---

## 8. Suggested stack

Pick boring, fast tools — the deadline is real.

| Concern | Recommendation |
|---|---|
| Build | **Vite + React 18 + TypeScript** (backend CORS already allows `:5173`) |
| Routing | React Router |
| Server state | **TanStack Query** — caching, loading/error states, parallel fetches for free |
| Styling | **Tailwind CSS** with the §3.4 tokens in `theme.extend` |
| Charts | **Recharts** (or hand-rolled SVG — the driver chart is simple bars) |
| Icons | lucide-react |
| Types | Generate from `/openapi.json` (`openapi-typescript`) — do not hand-write them |

> Generating the API types from `/openapi.json` is the single highest-leverage step. The
> backend schema is complete and accurate; hand-written interfaces will drift.

---

## 9. Build order (fastest path to a demo)

1. API client + generated types + `/health` badge.
2. Triage queue table with filters and pagination. **Demo-viable already.**
3. Student detail: risk hero + SHAP diverging chart.
4. Interventions panel + counterfactual "Projected scenario".
5. Assign/advance intervention with lifecycle-aware controls.
6. Loading/empty/error states everywhere.
7. Dark mode, responsive, accessibility pass.
8. CSV bulk import (optional, strong demo moment).

Stop at 6 if time runs short — 7 and 8 are polish.

---

## 10. Two-minute demo script

1. **Dashboard** — "2,000 students; 658 need attention now, ranked."
2. **Filter** to a department — "A mentor sees only their own list."
3. **Open rank 1** — "92.6% estimated risk. Here's *why*: Month-3 attendance collapsed to
   36%, compounded by a 75-day fee delay."
4. **Point at the drivers** — "This isn't a black box; every factor is quantified in
   percentage points."
5. **Interventions** — "The system proposes three concrete actions, each traced to a
   specific driver."
6. **Projected scenario** — "If attendance recovers to 80% and fees clear, risk drops to
   4.7%. A simulation, not a promise — we're explicit about that."
7. **Assign** → status chip flips to ASSIGNED — "The mentor acts; it's tracked."
8. **Back to queue** — the intervention badge now shows on the row. "Closed loop."

If the judges ask "is this real?" — open `/docs` and run a live request.
