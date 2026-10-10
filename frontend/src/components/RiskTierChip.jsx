import React from 'react';
import TierLabel from './TierLabel';

// Bordered chip around the tier label: icon + display label in the tier colour (API value unchanged).
const RiskTierChip = ({ tier, className = '' }) => (
  <span className={`inline-flex items-center rounded-control border border-rule bg-paper px-2 py-0.5 whitespace-nowrap ${className}`}>
    <TierLabel tier={tier} className="text-15" />
  </span>
);

export default RiskTierChip;
