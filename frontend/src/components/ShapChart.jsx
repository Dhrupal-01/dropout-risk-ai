import React, { useState } from 'react';
import { BarChart2, Info, Table } from 'lucide-react';

const DEFAULT_LIMIT = 5;
const EXPANDED_LIMIT = 8;

const signed = (value) => `${value > 0 ? '+' : ''}${value.toFixed(1)} pp`;

/**
 * The SHAP drivers behind a student's estimate, as a diverging bar chart (bars right raise the
 * estimate, bars left lower it) with a table view. Values are percentage points of estimated risk.
 */
const ShapChart = ({ drivers = [] }) => {
  const [viewMode, setViewMode] = useState('chart');
  const [limit, setLimit] = useState(DEFAULT_LIMIT);

  const toggle = (
    <div role="group" aria-label="View as" className="inline-flex rounded-control border border-control p-0.5">
      {[
        { mode: 'chart', label: 'Chart', icon: BarChart2 },
        { mode: 'table', label: 'Table', icon: Table },
      ].map(({ mode, label, icon: Icon }) => (
        <button
          key={mode}
          type="button"
          aria-pressed={viewMode === mode}
          onClick={() => setViewMode(mode)}
          className={`inline-flex items-center gap-1 rounded-[4px] px-2.5 py-1 text-13 ${
            viewMode === mode ? 'bg-ink text-on-ink' : 'text-graphite hover:bg-ink-wash'
          }`}
        >
          <Icon className="w-3.5 h-3.5" aria-hidden="true" />
          {label}
        </button>
      ))}
    </div>
  );

  const header = (
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div>
        <h2 className="text-17 font-semibold text-graphite">Why this student may need support</h2>
        <p className="mt-0.5 text-13 text-slate">
          The factors that moved this estimate most. They explain the model; they are not causes.
        </p>
      </div>
      {drivers.length > 0 && toggle}
    </div>
  );

  if (!drivers.length) {
    return (
      <div className="space-y-4">
        {header}
        <p className="flex items-center gap-2 rounded-control border border-dashed border-rule p-4 text-15 text-slate">
          <Info className="w-4 h-4 shrink-0" aria-hidden="true" />
          Reasons are not available for this estimate.
        </p>
      </div>
    );
  }

  const shown = drivers.slice(0, limit);
  const maxAbs = Math.max(...drivers.map((d) => Math.abs(d.risk_delta_percentage_points || 0)), 1);

  return (
    <div className="space-y-4">
      {header}

      {viewMode === 'chart' ? (
        <ul className="space-y-1">
          {shown.map((driver) => {
            const value = driver.risk_delta_percentage_points || 0;
            const raises = driver.impact_direction === 'RISK_INCREASING';
            const width = `${(Math.abs(value) / maxAbs) * 50}%`;
            return (
              <li
                key={driver.feature_name}
                title={driver.plain_language_explanation}
                className="grid grid-cols-[minmax(0,2fr)_minmax(0,3fr)_4.5rem] items-center gap-3 rounded-control px-1 py-1.5 text-15 hover:bg-ink-wash"
              >
                <span className="text-right text-graphite leading-snug">{driver.display_name || driver.feature_name}</span>
                <span className="relative h-5" aria-hidden="true">
                  <span className="absolute inset-y-0 left-1/2 w-px bg-rule" />
                  <span
                    className={`absolute top-0.5 bottom-0.5 ${raises ? 'left-1/2 rounded-r-[4px]' : 'right-1/2 rounded-l-[4px]'}`}
                    style={{ width, background: raises ? 'var(--shap-increase)' : 'var(--shap-decrease)' }}
                  />
                </span>
                <span className="text-right tabular-nums text-graphite">
                  {signed(value)}
                  <span className="sr-only">
                    {raises ? ', raises the estimate. ' : ', lowers the estimate. '}
                    {driver.plain_language_explanation}
                  </span>
                </span>
              </li>
            );
          })}
        </ul>
      ) : (
        <div className="relative overflow-x-auto">
          <table className="w-full border-collapse text-left text-15 tabular-nums">
            <thead>
              <tr className="border-b border-rule text-13 text-slate">
                <th scope="col" className="py-2 pr-3 font-medium">Factor</th>
                <th scope="col" className="py-2 pr-3 text-right font-medium">Value</th>
                <th scope="col" className="py-2 pr-3 text-right font-medium">Effect</th>
                <th scope="col" className="py-2 font-medium">Explanation</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((driver) => (
                <tr key={driver.feature_name} className="border-b border-rule last:border-0 align-top">
                  <td className="py-2 pr-3 text-graphite">{driver.display_name || driver.feature_name}</td>
                  <td className="py-2 pr-3 text-right text-graphite">
                    {driver.feature_value != null ? driver.feature_value.toFixed(2) : '—'}
                  </td>
                  <td className="py-2 pr-3 text-right text-graphite whitespace-nowrap">{signed(driver.risk_delta_percentage_points)}</td>
                  <td className="py-2 text-slate">{driver.plain_language_explanation}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="flex flex-wrap items-center justify-between gap-3 text-13 text-slate">
        <span className="inline-flex items-center gap-4">
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-[2px]" style={{ background: 'var(--shap-increase)' }} aria-hidden="true" />
            Raises the estimate
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-[2px]" style={{ background: 'var(--shap-decrease)' }} aria-hidden="true" />
            Lowers it
          </span>
        </span>
        {drivers.length > DEFAULT_LIMIT && (
          <button
            type="button"
            onClick={() => setLimit(limit === DEFAULT_LIMIT ? EXPANDED_LIMIT : DEFAULT_LIMIT)}
            className="btn btn-secondary py-1 text-13"
          >
            {limit === DEFAULT_LIMIT ? 'Show more factors' : 'Show fewer factors'}
          </button>
        )}
      </div>
    </div>
  );
};

export default ShapChart;
