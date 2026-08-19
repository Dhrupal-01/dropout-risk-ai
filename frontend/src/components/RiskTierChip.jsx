import React from 'react';
import { CheckCircle2, AlertTriangle, AlertOctagon } from 'lucide-react';

/**
 * RiskTierChip component
 * Displays a student's risk tier with a specific icon, text label, and color as per the UX spec.
 * 
 * @param {Object} props
 * @param {string} props.tier - Low, Medium, or High
 * @param {string} [props.className] - Optional extra class names
 */
const RiskTierChip = ({ tier, className = '' }) => {
  const normalizedTier = (tier || '').trim().toLowerCase();

  let config = {
    color: '#0ca30c', // default low
    bg: 'rgba(12, 163, 12, 0.08)',
    text: 'Low',
    icon: <CheckCircle2 className="w-4 h-4 mr-1.5 shrink-0" style={{ color: '#0ca30c' }} />,
  };

  if (normalizedTier === 'medium') {
    config = {
      color: '#fab219',
      bg: 'rgba(250, 178, 25, 0.08)',
      text: 'Medium',
      icon: <AlertTriangle className="w-4 h-4 mr-1.5 shrink-0" style={{ color: '#fab219' }} />,
    };
  } else if (normalizedTier === 'high') {
    config = {
      color: '#d03b3b',
      bg: 'rgba(208, 59, 59, 0.08)',
      text: 'High',
      icon: <AlertOctagon className="w-4 h-4 mr-1.5 shrink-0" style={{ color: '#d03b3b' }} />,
    };
  }

  return (
    <span
      className={`inline-flex items-center px-2.5 py-1 rounded-md text-xs font-semibold select-none border border-transparent ${className}`}
      style={{
        color: config.color,
        backgroundColor: config.bg,
        borderColor: `${config.color}25`, // 15% opacity border
      }}
    >
      {config.icon}
      <span>{config.text}</span>
    </span>
  );
};

export default RiskTierChip;
