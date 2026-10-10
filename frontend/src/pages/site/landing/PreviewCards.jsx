import React from 'react';
import { AlertOctagon, AlertTriangle, CalendarX, CheckCircle2 } from 'lucide-react';

// Hero illustration: three cards in the style of the mentor dashboard. Everything here is invented
// for illustration (fictional roll numbers, no real students) and is labelled as simulated demo data.
// No statistic appears on these cards.

const TIERS = [
  { key: 'low', label: 'On track', icon: CheckCircle2, colorVar: '--tier-low' },
  { key: 'medium', label: 'Monitor', icon: AlertTriangle, colorVar: '--tier-medium' },
  { key: 'high', label: 'Needs outreach', icon: AlertOctagon, colorVar: '--tier-high' },
];

// Illustrative segment sizes for the donut. Not data, and never shown as numbers.
const ILLUSTRATIVE_SHARES = { low: 0.6, medium: 0.25, high: 0.15 };

const Card = ({ className = '', children }) => (
  <div className={`rounded-panel border border-rule bg-paper p-4 ${className}`}>{children}</div>
);

const RiskCard = () => (
  <Card>
    <div className="flex flex-wrap items-start justify-between gap-2">
      <div>
        <p className="text-13 text-slate">Student</p>
        <p className="text-17 font-semibold text-graphite tabular-nums">24CE018</p>
      </div>
      <span className="inline-flex items-center gap-1 rounded-control border border-rule px-2 py-0.5 text-13 font-medium text-tier-high">
        <AlertOctagon className="w-3.5 h-3.5" aria-hidden="true" />
        Needs outreach
      </span>
    </div>
    <p className="mt-3 text-13 text-slate">Top reasons</p>
    <ul className="mt-1 space-y-1 text-15 text-graphite">
      <li>Attendance falling</li>
      <li>Fee payment overdue</li>
      <li>Assignments missing</li>
    </ul>
    <p className="mt-3 border-t border-rule pt-2 text-13 text-slate">
      Suggested: <span className="text-graphite">mentor call this week</span>
    </p>
  </Card>
);

// Donut with a 2px surface gap between segments (drawn as gaps in the arc, not strokes).
const Donut = () => {
  const r = 34;
  const circumference = 2 * Math.PI * r;
  const gap = 2;
  const lengths = TIERS.map((tier) => ILLUSTRATIVE_SHARES[tier.key] * circumference);
  const offsets = lengths.map((_, i) => lengths.slice(0, i).reduce((sum, len) => sum + len, 0));
  return (
    <svg viewBox="0 0 88 88" className="w-24 h-24 shrink-0 -rotate-90" aria-hidden="true">
      {TIERS.map((tier, i) => {
        const length = lengths[i];
        return (
          <circle
            key={tier.key}
            cx="44"
            cy="44"
            r={r}
            fill="none"
            strokeWidth="12"
            style={{ stroke: `var(${tier.colorVar})` }}
            strokeDasharray={`${length - gap} ${circumference - length + gap}`}
            strokeDashoffset={-offsets[i]}
          />
        );
      })}
    </svg>
  );
};

const TierCard = () => (
  <Card>
    <p className="text-15 font-medium text-graphite">Cohort by risk level</p>
    <div className="mt-3 flex flex-wrap items-center gap-4">
      <Donut />
      <ul className="space-y-1.5 text-13 text-graphite">
        {TIERS.map(({ key, label, icon: Icon, colorVar }) => (
          <li key={key} className="flex items-center gap-1.5">
            <Icon className="w-3.5 h-3.5 shrink-0" style={{ color: `var(${colorVar})` }} aria-hidden="true" />
            {label}
          </li>
        ))}
      </ul>
    </div>
  </Card>
);

const AlertCard = () => (
  <Card>
    <div className="flex items-start gap-3">
      <CalendarX className="w-5 h-5 shrink-0 mt-0.5 text-slate" aria-hidden="true" />
      <div>
        <p className="text-15 font-medium text-graphite">Attendance alert</p>
        <p className="mt-1 text-13 text-slate">
          <span className="text-graphite tabular-nums">24CE021</span> has been below the attendance requirement for
          three weeks in a row.
        </p>
        <p className="mt-2 text-13 text-slate">Rule-based alert, separate from the risk model.</p>
      </div>
    </div>
  </Card>
);

const PreviewCards = () => (
  <figure className="relative">
    <div
      className="grid gap-4 sm:grid-cols-2"
      aria-label="Illustrative preview of the mentor dashboard: a student flagged for outreach with top reasons, the cohort by risk level, and an attendance alert"
      role="img"
    >
      <div className="motion-card sm:col-start-2 sm:row-start-1 sm:row-span-2" style={{ '--i': 0 }}>
        <RiskCard />
      </div>
      <div className="motion-card sm:col-start-1 sm:row-start-2 sm:row-span-2 sm:self-center" style={{ '--i': 1 }}>
        <TierCard />
      </div>
      <div className="motion-card sm:col-start-2 sm:row-start-3" style={{ '--i': 2 }}>
        <AlertCard />
      </div>
    </div>
    <figcaption className="mt-3 text-13 text-slate">Illustrative, simulated demo data</figcaption>
  </figure>
);

export default PreviewCards;
