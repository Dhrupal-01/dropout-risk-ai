import React, { useState } from 'react';
import { Table, BarChart2, Info } from 'lucide-react';

/**
 * ShapChart Component
 * Visualizes the TreeSHAP local attribution values as a diverging horizontal bar chart.
 * 
 * @param {Object} props
 * @param {Array} props.drivers - List of SHAP drivers from the API response
 * @param {boolean} [props.isLoading] - Loading state
 */
const ShapChart = ({ drivers = [], isLoading = false }) => {
  const [viewMode, setViewMode] = useState('chart'); // 'chart' | 'table'
  const [limit, setLimit] = useState(5);

  if (isLoading) {
    return (
      <div className="space-y-4 animate-pulse select-none">
        <div className="h-6 w-1/3 bg-subtle rounded" />
        {[...Array(5)].map((_, i) => (
          <div key={i} className="flex space-x-3 items-center">
            <div className="h-4 w-1/4 bg-subtle rounded" />
            <div className="h-6 flex-1 bg-subtle rounded-full" />
            <div className="h-4 w-12 bg-subtle rounded" />
          </div>
        ))}
      </div>
    );
  }

  if (!drivers || drivers.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center p-6 border border-dashed border-border rounded-lg bg-subtle/30 text-center">
        <Info className="w-5 h-5 text-muted mb-2" />
        <p className="text-sm font-medium text-secondary">No explanation data available.</p>
      </div>
    );
  }

  // Slice based on the user-selected limit
  const activeDrivers = drivers.slice(0, limit);
  
  // Find maximum absolute value to scale the bars dynamically
  const maxVal = Math.max(
    ...drivers.map((d) => Math.abs(d.risk_delta_percentage_points || 0)),
    1 // fallback to prevent division by 0
  );

  return (
    <div className="space-y-4">
      {/* Chart controls & Toggle */}
      <div className="flex items-center justify-between border-b border-border pb-2.5">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-secondary">
          Why this student needs support
        </h3>
        
        {/* Toggle Button for accessibility (Chart / Table view) */}
        <div className="flex items-center border border-border rounded-md bg-subtle p-0.5 select-none">
          <button
            onClick={() => setViewMode('chart')}
            className={`flex items-center px-2 py-1 rounded text-xs font-medium transition-colors ${
              viewMode === 'chart'
                ? 'bg-card text-primary shadow-sm'
                : 'text-secondary hover:text-primary'
            }`}
            aria-label="Show chart view"
          >
            <BarChart2 className="w-3.5 h-3.5 mr-1" />
            Chart
          </button>
          <button
            onClick={() => setViewMode('table')}
            className={`flex items-center px-2 py-1 rounded text-xs font-medium transition-colors ${
              viewMode === 'table'
                ? 'bg-card text-primary shadow-sm'
                : 'text-secondary hover:text-primary'
            }`}
            aria-label="Show table view"
          >
            <Table className="w-3.5 h-3.5 mr-1" />
            Table
          </button>
        </div>
      </div>

      {viewMode === 'chart' ? (
        /* 1. CHART VIEW (DIVERGING BARS) */
        <div className="space-y-4 pt-2">
          {activeDrivers.map((driver) => {
            const isIncreasing = driver.impact_direction === 'RISK_INCREASING';
            const value = driver.risk_delta_percentage_points || 0;
            const percentage = Math.min((Math.abs(value) / maxVal) * 100, 100);

            // Positioning calculations for zero-centered diverging chart
            // Left column is 40% width, Right column is 50% width, text value is 10%
            return (
              <div
                key={driver.feature_name}
                className="group flex items-center min-h-[32px] text-xs transition-colors hover:bg-hover/40 px-2 py-1 rounded"
                title={driver.plain_language_explanation}
              >
                {/* Feature Label (40% width, aligned right) */}
                <div className="w-[35%] pr-4 text-right font-medium text-secondary truncate" title={driver.display_name}>
                  {driver.display_name || driver.feature_name}
                </div>

                {/* Bar Container (55% width) */}
                <div className="w-[53%] h-6 relative flex items-center bg-subtle/50 rounded overflow-hidden">
                  {/* Zero midpoint line */}
                  <div className="absolute top-0 bottom-0 left-[50%] w-[1px] bg-border z-10" />

                  {isIncreasing ? (
                    // RISK INCREASING: grow right
                    <div
                      className="absolute h-4 left-[50%] bg-shap-increase rounded-r transition-all duration-500 ease-out"
                      style={{ width: `${percentage * 0.5}%` }}
                    />
                  ) : (
                    // RISK DECREASING: grow left
                    <div
                      className="absolute h-4 bg-shap-decrease rounded-l transition-all duration-500 ease-out"
                      style={{
                        width: `${percentage * 0.5}%`,
                        left: `${50 - percentage * 0.5}%`,
                      }}
                    />
                  )}
                </div>

                {/* Signed Percentage impact (12% width) */}
                <div
                  className={`w-[12%] pl-3 font-semibold text-right tabular-nums ${
                    isIncreasing ? 'text-shap-increase' : 'text-shap-decrease'
                  }`}
                >
                  {isIncreasing ? '+' : ''}
                  {value.toFixed(1)} pp
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        /* 2. ACCESSIBLE TABLE VIEW */
        <div className="overflow-x-auto border border-border rounded-lg bg-card">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="bg-subtle text-secondary font-semibold border-b border-border">
                <th className="p-3">Risk Driver</th>
                <th className="p-3 text-right">Value</th>
                <th className="p-3 text-right">Impact</th>
                <th className="p-3">Attribution Reason</th>
              </tr>
            </thead>
            <tbody>
              {activeDrivers.map((driver) => {
                const isIncreasing = driver.impact_direction === 'RISK_INCREASING';
                return (
                  <tr
                    key={driver.feature_name}
                    className="border-b border-border last:border-0 hover:bg-hover/20"
                  >
                    <td className="p-3 font-medium text-primary">
                      {driver.display_name || driver.feature_name}
                    </td>
                    <td className="p-3 text-right text-secondary font-mono">
                      {driver.feature_value !== null ? driver.feature_value.toFixed(2) : '-'}
                    </td>
                    <td
                      className={`p-3 text-right font-semibold ${
                        isIncreasing ? 'text-shap-increase' : 'text-shap-decrease'
                      }`}
                    >
                      {isIncreasing ? '+' : ''}
                      {driver.risk_delta_percentage_points.toFixed(1)} pp
                    </td>
                    <td className="p-3 text-secondary italic">
                      {driver.plain_language_explanation}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Show more/less toggle buttons */}
      {drivers.length > 5 && (
        <div className="flex justify-center pt-2">
          {limit === 5 ? (
            <button
              onClick={() => setLimit(8)}
              className="text-xs font-semibold text-accent hover:text-accent-hover transition-colors border border-border px-3 py-1.5 rounded-md hover:bg-hover focus:ring-2 focus:ring-accent"
            >
              Show More Drivers (8)
            </button>
          ) : (
            <button
              onClick={() => setLimit(5)}
              className="text-xs font-semibold text-accent hover:text-accent-hover transition-colors border border-border px-3 py-1.5 rounded-md hover:bg-hover focus:ring-2 focus:ring-accent"
            >
              Show Less Drivers
            </button>
          )}
        </div>
      )}
    </div>
  );
};

export default ShapChart;
