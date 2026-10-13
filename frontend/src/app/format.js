// Number formatting for the app (kept separate from src/content so app chunks don't pull in facts.json).
export const formatCount = (value) => Number(value).toLocaleString('en-IN');

export const formatShare = (part, whole) => (whole > 0 ? `${Math.round((part / whole) * 100)}%` : '–');

export const formatPercent = (value, digits = 1) => `${Number(value).toFixed(digits)}%`;
