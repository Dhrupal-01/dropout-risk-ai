import React from 'react';
import { tierFor } from '../app/tiers';

// Icon + display label in the tier colour. Never colour alone.
const TierLabel = ({ tier, className = '' }) => {
  const meta = tierFor(tier);
  if (!meta) return <span className={className}>{tier}</span>;
  const Icon = meta.icon;
  return (
    <span className={`inline-flex items-center gap-1.5 font-medium ${className}`} style={{ color: meta.color }}>
      <Icon className="w-4 h-4 shrink-0" aria-hidden="true" />
      {meta.label}
    </span>
  );
};

export default TierLabel;
