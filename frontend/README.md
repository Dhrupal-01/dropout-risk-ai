# DropoutGuard frontend

The mentor-facing web app for DropoutGuard: a triage worklist, a student page (risk estimate,
plain-language risk drivers, rule-based alerts, recommended support, intervention log) and a CSV
batch import. It is decision support for mentors; nothing it shows is an automatic or punitive
decision.

Stack: React 19, Vite, Tailwind CSS 3.4, TanStack Query, React Router, axios. Lint: oxlint.

## Prerequisites

- Node.js and npm. `package-lock.json` is committed, so install with `npm ci`.
- The backend API running (see the repository `README.md`):
  `uvicorn backend.app.main:app --reload --port 8000`.

## Install and run

```bash
cd frontend
npm ci            # exact versions from package-lock.json
npm run dev       # dev server on http://localhost:5173
```

## Build and lint

```bash
npm run build     # production build into frontend/dist/ (gitignored, never committed)
npm run lint      # oxlint
npm run preview   # serve the production build locally
```

Both `npm run build` and `npm run lint` must pass before a change is finished.

## Configuration

Vite reads variables prefixed `VITE_` from the environment or from `frontend/.env.local`
(gitignored). They are fixed at build time.

| Variable | Default | Read in | Effect |
|---|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:8000` | `src/api/client.js` | Base URL of the backend API. Every request goes to this absolute URL. |
| `VITE_DEMO_MODE` | unset | `src/App.jsx` | When exactly `true`, shows a banner on every page: "Demo environment — all student records are simulated." It changes nothing else. |

Example `frontend/.env.local`:

```bash
VITE_API_BASE_URL=http://localhost:8000
VITE_DEMO_MODE=true
```

The backend must allow the frontend's origin in `CORS_ORIGINS` (repository `.env`).

`vite.config.js` also proxies `/api` and `/health` to `http://127.0.0.1:8000` on the dev server.
The app does not use that proxy as written: `src/api/client.js` always calls the absolute
`VITE_API_BASE_URL` (an empty value falls back to the default).

## Error messages

API errors arrive as `{ "error": <code>, "message": ... }`. `src/api/client.js` maps the codes in
`errorMap`; keep it in step with `backend/app/core/errors.py`.
