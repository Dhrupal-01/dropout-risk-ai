import { AlertOctagon, AlertTriangle, CheckCircle2 } from 'lucide-react';

// Risk tiers: API values stay Low / Medium / High; the UI shows these labels, always with the icon.
// `color` is for text and icons (AA on paper); `mark` is for chart fills where tiers sit side by side.
export const TIERS = [
  { api: 'High', label: 'Needs outreach', icon: AlertOctagon, color: 'var(--tier-high)', mark: 'var(--tier-high-mark)' },
  { api: 'Medium', label: 'Monitor', icon: AlertTriangle, color: 'var(--tier-medium)', mark: 'var(--tier-medium-mark)' },
  { api: 'Low', label: 'On track', icon: CheckCircle2, color: 'var(--tier-low)', mark: 'var(--tier-low-mark)' },
];

const BY_API = Object.fromEntries(TIERS.map((tier) => [tier.api, tier]));

export const tierFor = (apiValue) => BY_API[apiValue];
