import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { RefreshCw } from 'lucide-react';
import { checkHealth } from '../api/endpoints';

// Backend health indicator for the app top bar (moved unchanged from the old Header).
const HealthBadge = () => {
  // Health check query (polls every 30 seconds)
  const { data: health, status: healthQueryStatus } = useQuery({
    queryKey: ['health'],
    queryFn: checkHealth,
    refetchInterval: 30000,
    retry: 2,
    refetchOnWindowFocus: true,
  });

  const getHealthBadge = () => {
    if (healthQueryStatus === 'pending') {
      return (
        <span className="inline-flex items-center text-xs text-muted select-none">
          <RefreshCw className="w-3.5 h-3.5 mr-1.5 animate-spin" />
          Checking status...
        </span>
      );
    }

    if (healthQueryStatus === 'error' || !health) {
      return (
        <span
          className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-red-100 dark:bg-red-950 text-red-700 dark:text-red-300 border border-red-200 dark:border-red-900 select-none animate-pulse"
          title="FastAPI server is unreachable. Check local console or network connections."
        >
          <span className="w-2 h-2 rounded-full bg-red-600 dark:bg-red-400 mr-1.5" />
          Offline
        </span>
      );
    }

    if (health.status === 'healthy' && health.model_loaded) {
      return (
        <span
          className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-green-100 dark:bg-green-950/30 text-green-700 dark:text-green-400 border border-green-200 dark:border-green-900 select-none"
          title={`Model Version: ${health.model_version || 'unknown'}`}
        >
          <span className="w-2 h-2 rounded-full bg-risk-low mr-1.5" />
          ● Ready
        </span>
      );
    }

    // Degraded or model failed loading
    return (
      <span
        className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-amber-100 dark:bg-amber-950 text-amber-700 dark:text-amber-300 border border-amber-200 dark:border-amber-900 select-none"
        title={health.detail || 'Database degraded or model artifacts missing.'}
      >
        <span className="w-2 h-2 rounded-full bg-risk-medium mr-1.5" />
        Scoring Unavailable
      </span>
    );
  };

  return getHealthBadge();
};

export default HealthBadge;
