import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Bar, BarChart, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { getStatsDrivers } from '../../../api/endpoints';
import { formatCount, formatShare } from '../../../app/format';
import { BAR_SIZE } from '../../../components/charts/chartTheme';
import TooltipBox from '../../../components/charts/TooltipBox';
import WrappedCategoryTick from '../../../components/charts/WrappedCategoryTick';
import OverviewCard from './OverviewCard';

const LIMIT = 8;
const ROW_HEIGHT = 40;
const SCOPES = [
  { value: 'High', label: 'Needs outreach' },
  { value: '', label: 'All students' },
];

const DriverTooltip = ({ active, payload, base }) => {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <TooltipBox title={row.display_name}>
      {formatCount(row.student_count)} of {formatCount(base)} students ({formatShare(row.student_count, base)})
    </TooltipBox>
  );
};

const TopDrivers = ({ className = '' }) => {
  const [scope, setScope] = useState(SCOPES[0].value);
  const drivers = useQuery({
    queryKey: ['stats', 'drivers', LIMIT, scope],
    queryFn: () => getStatsDrivers({ limit: LIMIT, risk_tier: scope }),
  });
  const data = drivers.data;
  const base = data?.students_with_drivers ?? 0;

  const toggle = (
    <div role="group" aria-label="Which students" className="inline-flex rounded-control border border-control p-0.5">
      {SCOPES.map((s) => (
        <button
          key={s.label}
          type="button"
          aria-pressed={scope === s.value}
          onClick={() => setScope(s.value)}
          className={`rounded-[4px] px-2.5 py-1 text-13 ${scope === s.value ? 'bg-ink text-on-ink' : 'text-graphite hover:bg-ink-wash'}`}
        >
          {s.label}
        </button>
      ))}
    </div>
  );

  return (
    <OverviewCard
      className={className}
      title="Most common reasons for a raised risk score"
      description="How often each factor appears among students' top reasons. Reasons explain the model's estimate; they are not causes."
      headerExtra={toggle}
      status={drivers.status}
      error={drivers.error}
      onRetry={() => drivers.refetch()}
      isEmpty={!data?.drivers.length}
      emptyMessage={
        data?.students_considered
          ? 'No stored reasons for these students yet. Reasons are saved when students are scored.'
          : 'No students in this group yet.'
      }
      loadingClassName="h-72"
    >
      <figure aria-label={`Most common reasons for a raised risk score: ${SCOPES.find((s) => s.value === scope).label.toLowerCase()}`}>
        <div aria-hidden="true" style={{ height: (data?.drivers.length ?? 0) * ROW_HEIGHT + 8 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart accessibilityLayer={false} data={data?.drivers} layout="vertical" margin={{ top: 0, right: 48, bottom: 0, left: 0 }}>
              <XAxis type="number" hide />
              <YAxis
                type="category"
                dataKey="display_name"
                width={170}
                interval={0}
                tickLine={false}
                axisLine={false}
                tick={<WrappedCategoryTick maxChars={22} />}
              />
              <Tooltip content={<DriverTooltip base={base} />} cursor={{ fill: 'var(--ink-wash)' }} isAnimationActive={false} />
              <Bar dataKey="student_count" barSize={BAR_SIZE} radius={[0, 4, 4, 0]} fill="var(--ink)" isAnimationActive={false}>
                <LabelList dataKey="student_count" position="right" formatter={formatCount} style={{ fill: 'var(--graphite)', fontSize: 13 }} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="mt-2 text-13 text-slate">
          Out of {formatCount(base)} students with stored reasons{scope ? ' who need outreach' : ''}.
        </p>
        <div className="sr-only">
          <table>
            <caption>Most common reasons for a raised risk score</caption>
            <thead>
              <tr>
                <th scope="col">Reason</th>
                <th scope="col">Students</th>
              </tr>
            </thead>
            <tbody>
              {data?.drivers.map((d) => (
                <tr key={d.feature_name}>
                  <th scope="row">{d.display_name}</th>
                  <td>{formatCount(d.student_count)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </figure>
    </OverviewCard>
  );
};

export default TopDrivers;
