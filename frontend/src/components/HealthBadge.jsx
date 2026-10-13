import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { AlertTriangle, CheckCircle2, RefreshCw, WifiOff } from 'lucide-react';
import { checkHealth } from '../api/endpoints';

// Backend health indicator for the app top bar; polls /health.
const HealthBadge = () => {
  // Health check query (polls every 30 seconds)
  const { data: health, status: healthQueryStatus } = useQuery({
    queryKey: ['health'],
    queryFn: checkHealth,
    refetchInterval: 30000,
    retry: 2,
    refetchOnWindowFocus: true,
  });

  // Identity by icon + text (status colours stay neutral; tier colours are reserved for risk tiers).
  const badge = (Icon, label, title, extra = '') => (
    <span
      className={`inline-flex items-center gap-1.5 rounded-control border border-rule px-2 py-0.5 text-13 font-medium text-graphite select-none ${extra}`}
      title={title}
    >
      <Icon className="w-3.5 h-3.5 shrink-0" aria-hidden="true" />
      {label}
    </span>
  );

  const getHealthBadge = () => {
    if (healthQueryStatus === 'pending') {
      return badge(RefreshCw, 'Checking…', 'Checking the backend', 'text-slate');
    }
    if (healthQueryStatus === 'error' || !health) {
      return badge(WifiOff, 'Offline', 'The API server cannot be reached. Check that it is running.');
    }
    if (health.status === 'healthy' && health.model_loaded) {
      return badge(CheckCircle2, 'Ready', `Model version: ${health.model_version || 'unknown'}`);
    }
    // Degraded or model failed loading: reads still work, scoring does not.
    return badge(AlertTriangle, 'Scoring unavailable', health.detail || 'Database degraded or model artifacts missing.');
  };

  return getHealthBadge();
};

export default HealthBadge;
