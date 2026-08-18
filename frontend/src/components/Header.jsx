import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { Sun, Moon, Database, Activity, RefreshCw } from 'lucide-react';
import { checkHealth } from '../api/endpoints';

const Header = () => {
  const [theme, setTheme] = useState(
    () => localStorage.getItem('dropoutguard-theme') || 'light'
  );

  // Sync theme to HTML dataset attribute
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem('dropoutguard-theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prevTheme) => (prevTheme === 'light' ? 'dark' : 'light'));
  };

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

  return (
    <header className="sticky top-0 z-50 flex items-center justify-between px-6 py-4 bg-card border-b border-border shadow-sm">
      {/* Brand Logo & Name */}
      <div className="flex items-center space-x-3">
        <Link to="/" className="flex flex-col group select-none">
          <span className="text-xl font-bold tracking-tight text-primary transition-colors group-hover:text-accent">
            DropoutGuard
          </span>
          <span className="text-[10px] uppercase tracking-wider text-muted font-medium">
            Academic Early-Warning & Intervention
          </span>
        </Link>
      </div>

      {/* Middle Health Probe Indicators */}
      <div className="hidden sm:flex items-center space-x-6">
        {/* API Connection Indicator */}
        <div className="flex items-center space-x-2">
          <Activity className="w-4 h-4 text-muted" />
          <span className="text-xs font-medium text-secondary">Backend Status:</span>
          {getHealthBadge()}
        </div>

        {/* Database Connected Indicator */}
        {health?.database === 'connected' && (
          <div className="flex items-center space-x-1.5 border-l border-border pl-4">
            <Database className="w-3.5 h-3.5 text-muted" />
            <span className="text-xs text-secondary font-medium">PostgreSQL Connected</span>
          </div>
        )}
      </div>

      {/* Action Buttons & Theme Toggler */}
      <div className="flex items-center space-x-4">


        <button
          onClick={toggleTheme}
          className="p-2 rounded-md hover:bg-hover text-secondary border border-border transition-colors focus:ring-2 focus:ring-accent"
          aria-label={theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
          title={theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
        >
          {theme === 'light' ? (
            <Moon className="w-4 h-4" />
          ) : (
            <Sun className="w-4 h-4" />
          )}
        </button>
      </div>
    </header>
  );
};

export default Header;
