# DropoutGuard UI redesign spec

Status: approved direction, not yet built. Owner: @Dhrupal-01.
Read with `CLAUDE.md` and `.claude/rules/frontend.md`. App-screen framing rules from
`docs/frontend_ux_spec.md` still apply.

## 1. Why

The current site opens straight into a dashboard. A first-time visitor (a professor reviewing the
project, a hackathon judge, a college administrator) never learns what problem this solves, why it
matters in India, or what the product actually does. The redesign splits the product into:

- a **public site** that tells the story: the problem, the evidence, how the product helps, and how it
  is kept responsible;
- the **app** (mentor workspace) with the existing features, restyled to match.

Primary job of the home page: in under a minute, a visitor understands that students leave college
gradually, that the signals are already in college records, and that DropoutGuard turns those
signals into early, explained outreach.

## 2. Routes and layouts

Two layouts: `SiteLayout` (top nav + footer with sources) and `AppLayout` (left rail + top bar).

| Route | Layout | Page |
|---|---|---|
| `/` | Site | Home |
| `/how-it-works` | Site | The four-step workflow in detail, with app screenshots |
| `/evidence` | Site | Real-data results, methodology, limitations |
| `/responsible-ai` | Site | Model card, principles, what the tool must never be used for |
| `/about` | Site | Origin (SIH 2026, SDG 4), team, repository link |
| `/app` | App | Redirect to `/app/worklist` |
| `/app/worklist` | App | Current `Dashboard.jsx` (student triage worklist) |
| `/app/students/:studentId` | App | Current `StudentDetail.jsx` |
| `/app/import` | App | Current `ImportQueue.jsx` |
| `/app/insights` | App | New: cohort overview built only from `GET /api/v1/stats/summary` |

- Redirect old routes: `/students/:studentId` → `/app/students/:studentId`, `/import` → `/app/import`.
- `/geo-analytics` and `/universal-predictor`: not linked anywhere until the owner decides. Leave the
  route files untouched.
- Unknown routes render a real 404 page on the site layout (not the dashboard).
- Lazy-load every route. Public pages must not pull in recharts or app code.

## 3. Design direction: "the register"

The most characteristic object in this subject's world is the college attendance register: ruled
paper, roll numbers down the side, a column per class, P and A marks in blue ink, a red margin line.
It is also exactly where dropout first becomes visible. The site borrows its materials (ruled lines,
register ink, margin red, handwritten marks) and uses them with restraint. The product itself stays
calm and typographic.

### Principles

1. One memorable thing: the hero register. Everything else is quiet and disciplined.
2. Sentences before numbers. Facts appear inside plain sentences with a citation, not as a wall of
   stat cards.
3. People, not scores. Copy talks about students and mentors, never "targets" or "cases".
4. Honest by default. Every number is sourced; every model claim states its data and its limits.
5. Structure encodes meaning. Numbering only for the real four-step sequence. Margin red only for
   absence and risk.

### Colour tokens (add to `src/index.css`, replace the current palette)

| Token | Light | Dark | Use |
|---|---|---|---|
| `--paper` | `#FAFBFD` | `#10151F` | Page background |
| `--rule` | `#D9E2F0` | `#26324A` | Ruled lines, borders, dividers |
| `--ink` | `#1D3A8A` | `#8FB0FF` | Primary actions, links, register marks |
| `--ink-wash` | `#E8EEFA` | `#1A2440` | Hover and selected backgrounds |
| `--graphite` | `#23272F` | `#E7EBF3` | Body text |
| `--slate` | `#5B6472` | `#A3ADBD` | Secondary text, captions |
| `--margin-red` | `#C0392B` | `#E0665A` | Absence marks, the register margin line. Site only |
| `--marigold` | `#E3A23A` | `#E8B45C` | The single highlight band on the flagged register row |

App risk tiers (reserved, always with icon + text; display labels only, API values stay `Low/Medium/High`):

| Tier (API) | Display label | Light | Dark |
|---|---|---|---|
| Low | On track | `#3F6FB0` | `#7FA6DE` |
| Medium | Monitor | `#B7791F` | `#D9A441` |
| High | Needs outreach | `#B4432F` | `#E07A62` |

Check every text/background pair for WCAG AA before shipping.

### Type

- **Newsreader** (variable, `@fontsource-variable/newsreader`): display and page titles only.
  Weight 500, slight negative tracking at large sizes.
- **Mukta** (`@fontsource/mukta`, weights 400/500/600): everything else, including the app. It
  supports tabular figures; use `font-variant-numeric: tabular-nums` in tables and numbers.
- **Kalam** (`@fontsource/kalam`, 400): only the handwritten P/A marks and the margin note in the
  hero register. Nowhere else.
- Scale (px): 13, 15, 17 (body), 20, 24, 32, 44, 60 (hero on desktop; 40 on mobile).
  Body line-height 1.55, measure ≤ 68ch. Sentence case everywhere.

### Layout and surface

- 12-column grid, max width 1200 px, left-aligned text. Generous vertical rhythm on the site
  (section padding ~ 96-128 px desktop, 64 px mobile).
- Panels use a 1 px `--rule` border and no shadow. Radius: 6 px on controls, 12 px on large panels.
  Different radii for different hierarchy levels, never one radius everywhere.
- Faint ruled-paper background (horizontal `--rule` lines every 32 px) appears only behind the hero
  register and the Sources section, like a sheet of paper, not across the whole site.

### Avoid (these read as generic templates)

All-caps eyebrow labels above headings; highlighting one word of a headline in another colour or
italic; grids of identical rounded cards with soft shadows; gradient washes; big-number-with-tiny-label
stat cards as the main device; "→" appended to button text; fade-and-slide-up on every section;
middle-dot meta strings; cream + terracotta; near-black + neon accent; stock photos of students;
any map of India (boundary requirements make this a liability).

## 4. Home page

```
┌──────────────────────────────────────────────────────────────────────────┐
│ DropoutGuard        How it works  Evidence  Responsible AI  About  [Open app]
├──────────────────────────────────────────────────────────────────────────┤
│ Most students don't drop out    │ ┌ Attendance register, Semester 3 ───┐ │
│ on one day.                     │ │ Roll no.  W1 W2 W3 W4 W5 W6 W7 W8   │ │
│                                 │ │ 24CE016   P  P  P  P  P  P  P  P    │ │
│ They drift. A missed week, a    │ │ 24CE017   P  P  A  P  P  P  P  P    │ │
│ fee left unpaid, an assignment  │ │▌24CE018   P  P  P  A  P  A  A  A   ◀── highlighted row
│ that never comes in. ...        │ │ 24CE019   P  P  P  P  P  P  A  P    │ │
│                                 │ │   margin note (Kalam)  + flag panel │ │
│ [Open the dashboard] [How it works] └────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────────────────────┤
│ The scale: 100-student cohort (OECD) with a timeframe toggle              │
│ India: enrolment sentence + exits by institution type (bar chart) + gap   │
├──────────────────────────────────────────────────────────────────────────┤
│ How DropoutGuard helps: 1 Bring records  2 See who and why  3 Reach out   │
│                        4 Record what happened                            │
├──────────────────────────────────────────────────────────────────────────┤
│ Evidence: one sentence from evidence.json + limitation + link             │
├──────────────────────────────────────────────────────────────────────────┤
│ Built to support, never to punish (four principles)                       │
├──────────────────────────────────────────────────────────────────────────┤
│ Closing call to action  │  Sources (numbered, on ruled paper)  │ footer   │
└──────────────────────────────────────────────────────────────────────────┘
```

Mobile: single column; register goes below the hero text and shows 6 weeks instead of 8.

### 4.1 Hero

- Headline (Newsreader): "Most students don't drop out on one day."
- Body: "They drift. A missed week, a fee left unpaid, an assignment that never comes in. The signs
  are already in your college's records. DropoutGuard brings them together, tells a mentor who may
  need support and why, while there is still time to help."
- Buttons: "Open the dashboard" (primary, to `/app/worklist`), "See how it works" (secondary).
- Register illustration (HTML table or SVG, `aria-hidden` with a text alternative in a visually hidden
  paragraph describing it). Fictional roll numbers. Rows 24CE016-24CE021, 8 weekly columns.
  Row 24CE018 accumulates A marks from week 4. A red margin line runs down the left.
- Margin note in Kalam next to 24CE018: "4 absences in 5 weeks, fee overdue, 2 assignments missing".
- Flag panel anchored to that row (Mukta): title "Flagged for outreach", body "Top reasons: attendance
  falling, fee overdue. Suggested: mentor call this week." This is illustrative; label the figure
  "Illustration" in the caption.

### 4.2 Hero motion (the only automatic animation on the site)

One sequence on first paint, about 2.4 s total:
1. Week columns fill left to right (about 150 ms per column); each mark fades in with a 2 px upward
   settle.
2. After week 8, the 24CE018 row gets the `--marigold` highlight band (300 ms).
3. The margin note appears (300 ms), then the flag panel slides 8 px in from the right (250 ms).

Use CSS keyframes with staggered `animation-delay`; add `motion` only if CSS cannot do it (ask first).
With `prefers-reduced-motion: reduce`, render the final state immediately. No scroll-triggered
animations anywhere else. Motion that responds to a user action (toggles, expanding panels) is fine.

### 4.3 The scale

Heading: "The problem is bigger than any one college."

**Global, as a 100-student unit chart.** A 10 x 10 grid of small circles. Three toggle buttons:
"On time", "One year late", "Three years late". Filled circles = `oecd_bachelor_on_time`,
`oecd_bachelor_plus_one_year`, `oecd_bachelor_plus_three_years` from `facts.json`. Filling animates
only when the user presses a toggle. Sentence under it, computed from facts:
"Of every 100 students who start a bachelor's degree across OECD countries, {plus3} have graduated
three years after the expected end. {100 - plus3} have not." Second line from
`us_some_college_no_credential`: "In the United States alone, {value} million people started college
and left without a credential."

**India.** Two parts, side by side on desktop:
- Sentence from `india_ger_2023_24` and `india_enrolment_2023_24`: "India enrols {enrolment} crore
  students in higher education, a gross enrolment ratio of {ger}. For every 100 people aged 18-23,
  that is about {ger} places in a college or university."
- Horizontal bar chart (plain SVG, no chart library on public pages) of
  `india_central_institution_exits_2019_2023.breakdown`, titled "{value} students left centrally
  funded institutions before finishing, 2019-2023". Caption paraphrases `context` so readers see that
  some exits are moves to jobs or other places, not all are failures.
- Closing paragraph: "Those counts cover only centrally funded institutions, a small part of the
  country's higher-education system. For most colleges there is no shared early-warning signal at
  all. The records that could provide one (attendance, internal marks, LMS activity, fee dues) sit in
  separate registers and spreadsheets." (Owner: confirm before publishing that no national
  higher-education dropout rate is published; adjust the wording if one exists.)

### 4.4 How DropoutGuard helps (a real sequence, so numbered 1-4)

1. **Bring the records you already have.** Upload a CSV of attendance, marks, LMS activity and fee
   status. No new data collection.
2. **See who may need support, and why.** Each student gets a risk level with the top reasons in plain
   language, not just a score.
3. **Reach out early.** Mentors get a worklist ordered by priority, with suggested actions from a
   catalogue of 12 institutional interventions.
4. **Record what happened.** Every outreach and outcome is logged, so the college learns what works.

Each step pairs with a cropped screenshot of the redesigned app (add after section 7 is built; use
neutral placeholders until then, clearly marked as placeholders).

### 4.5 Evidence

All numbers from `src/content/evidence.json` (section 6). Template:

"Tested on real records from {n} students at a Portuguese polytechnic (UCI dataset 697). Using only
what a college knows by the end of the first semester, the model ranked students who later dropped
out above those who graduated with a ROC-AUC of {point} (95% CI {lo}-{hi})."

Plain-language note under it: "ROC-AUC is the chance that the model ranks a randomly chosen student
who dropped out above one who graduated. 0.5 is a coin flip; 1.0 is perfect."

Limitation, same visual weight as the result: "What this does not show yet: how the model performs in
Indian colleges. The deployed demo runs on a simulated Indian cohort."

Link: "Read the full evaluation" to `/evidence`.

### 4.6 Built to support, never to punish

Four short principles in two columns (plain text, no cards): a person decides, the tool only suggests;
flags never trigger penalties, debarment or scholarship decisions; every flag comes with its reasons;
outcomes are audited across student groups. Link to `/responsible-ai`.

### 4.7 Close and sources

- Closing line: "Start with the records you already keep." Button: "Open the dashboard".
- Sources section on ruled paper: numbered list generated from `facts.json` (publisher, source title,
  year, link). Each in-text citation is a superscript number linking here.
- Footer: project name, "Smart India Hackathon 2026, SDG 4: Quality Education", GitHub link,
  "Demo data is simulated" note when `VITE_DEMO_MODE=true`.

## 5. Other public pages

- **How it works:** the four steps in depth; the four data pillars (attendance, academics, learning
  behaviour, fees and finances) described in plain language; what the mentor sees; what the tool
  never does.
- **Evidence:** full tables from `evidence.json` (UCI primary and sensitivity labels; enrolment-time,
  end-of-semester-1, full feature sets; majority baseline, logistic regression, XGBoost; CIs). Methods
  in plain language. OULAD section shows "Being re-run, results withheld until verified" until the
  owner marks the OULAD export as verified. Sim-to-real shown only with its caveat. Limitations list.
- **Responsible AI:** model card (intended use, out-of-scope uses such as admissions, penalties,
  debarment or scholarship decisions, training data, evaluation data, fairness audit summary from
  generated JSON once Phase 6 is done, human oversight, data-handling principles). No compliance
  claims.
- **About:** origin, team roles (owner fills in names), repository link, contact.

## 6. Content files and scripts

- `frontend/src/content/facts.json`: already drafted with sources. Every fact has `verified: false`
  until the owner checks the URL. Create `frontend/scripts/check-facts.mjs` and an npm script
  `check:facts` that fails if any fact lacks `publisher`, `source`, `year` or `url`, and lists
  unverified facts. In development, unverified facts render with a small dotted underline and an
  "unverified" tooltip; the owner runs `npm run check:facts` before any public deploy.
- `scripts/export_site_evidence.py` (new): reads `ml/artifacts/benchmarks/uci_*_{primary,sensitivity}.json`
  (keys: `n_samples`, `prevalence`, `results[split][model]["metrics"][metric] = {point, ci_lower, ci_upper}`)
  and writes `frontend/src/content/evidence.json` with dataset, feature set, label variant, split,
  model, metric values, CIs, n, the git commit and the export date. It includes OULAD only when run
  with `--include-oulad`. Run it after Phase 6 regenerates the benchmarks; until then the evidence
  block shows "Results being re-verified".
- Components read facts and evidence through a tiny helper (`src/content/index.js`) that throws in
  development if an id is missing.

## 7. App redesign

- `AppLayout`: left rail (Worklist, Insights, Import, a divider, "Back to site"), top bar with global
  student search (Ctrl/Cmd+K opens a palette that queries `GET /api/v1/mentors/queue?search=`), theme
  toggle, and the existing health indicator.
- Restyle existing pages with the new tokens. Keep all current behaviour, queries and endpoints.
- Tier display labels: On track / Monitor / Needs outreach, with icons. API values unchanged.
- Worklist rows show the top two reasons as short text chips when the data is available.
- Insights (new page): tier distribution overall and by department from `GET /api/v1/stats/summary`.
  Anything that needs a new endpoint (risk trend over time, driver prevalence) is a separate task:
  ask first.
- Page titles in Newsreader, everything else Mukta. Tables use tabular figures and 15 px text
  (the current 12 px `text-xs` is too small).

## 8. Quality bar (acceptance for every step)

- `npm run build && npm run lint` pass; `npm run check:facts` passes (unverified facts listed, none
  missing sources).
- Light and dark themes both correct; usable at 360 px; keyboard-only navigation works; focus visible.
- `prefers-reduced-motion: reduce` shows the hero's final state with no animation.
- No number in any `.jsx` file that comes from the outside world (grep for digits in copy strings
  during review).
- Lighthouse (mobile, home page): accessibility ≥ 95, performance ≥ 90. Report actual numbers.
- Fonts self-hosted with `font-display: swap`; no layout jump in the hero.

## 9. Build order (stop for review after each step)

1. Branch `feat/redesign`. Tokens, fonts, `SiteLayout`, `AppLayout`, routing with redirects and a 404.
   All existing app features reachable under `/app`.
2. `facts.json` wiring, content helper, `check:facts`, Sources section, citation component.
3. Home page sections 4.1 and 4.3 to 4.7, static (no motion yet).
4. Hero motion and the unit-chart toggle.
5. How it works, Responsible AI, About.
6. App restyle and Insights.
7. After Phase 6: `export_site_evidence.py`, the Evidence page and the home evidence block.
8. Final QA against section 8; list anything that could not be verified.
