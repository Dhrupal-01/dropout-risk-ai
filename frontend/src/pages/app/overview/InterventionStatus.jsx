import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { Bar, BarChart, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { getStatsInterventions } from '../../../api/endpoints';
import { formatCount } from '../../../app/format';
import { BAR_SIZE, axisTick } from '../../../components/charts/chartTheme';
import TooltipBox from '../../../components/charts/TooltipBox';
import OverviewCard from './OverviewCard';

// Lifecycle order (backend LIFECYCLE_ORDER); labels for display only.
const STAGES = [
  { status: 'ASSIGNED', label: 'Assigned' },
  { status: 'IN_PROGRESS', label: 'In progress' },
  { status: 'APPLIED', label: 'Applied' },
  { status: 'COMPLETED', label: 'Completed' },
];
const OUTCOMES = [
  { status: 'IMPROVED', label: 'improved' },
  { status: 'NO_CHANGE', label: 'no change' },
  { status: 'DETERIORATED', label: 'deteriorated' },
  { status: 'PENDING_EVALUATION', label: 'awaiting evaluation' },
];
const ROW_HEIGHT = 40;

const StageTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return <TooltipBox title={row.label}>{formatCount(row.count)} interventions</TooltipBox>;
};

const InterventionStatus = () => {
  const interventions = useQuery({ queryKey: ['stats', 'interventions'], queryFn: getStatsInterventions });
  const data = interventions.data;
  const rows = STAGES.map((stage) => ({ ...stage, count: data?.by_status[stage.status] ?? 0 }));

  return (
    <OverviewCard
      title="Interventions by stage"
      description="The current stage of every logged intervention."
      status={interventions.status}
      error={interventions.error}
      onRetry={() => interventions.refetch()}
      isEmpty={data?.total === 0}
      emptyMessage={
        <>
          No interventions logged yet. Mentors assign them from a{' '}
          <Link to="/app/students" className="text-ink underline underline-offset-4">
            student's page
          </Link>
          .
        </>
      }
      loadingClassName="h-44"
    >
      <figure aria-label="Interventions by stage">
        <div aria-hidden="true" style={{ height: rows.length * ROW_HEIGHT + 8 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart accessibilityLayer={false} data={rows} layout="vertical" margin={{ top: 0, right: 48, bottom: 0, left: 0 }}>
              <XAxis type="number" hide />
              <YAxis type="category" dataKey="label" width={96} tick={axisTick} tickLine={false} axisLine={false} />
              <Tooltip content={<StageTooltip />} cursor={{ fill: 'var(--ink-wash)' }} isAnimationActive={false} />
              <Bar dataKey="count" barSize={BAR_SIZE} radius={[0, 4, 4, 0]} fill="var(--ink)" isAnimationActive={false}>
                <LabelList dataKey="count" position="right" formatter={formatCount} style={{ fill: 'var(--graphite)', fontSize: 13 }} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="mt-2 text-13 text-slate">
          Outcomes so far:{' '}
          {OUTCOMES.map((o) => `${formatCount(data?.by_outcome_status[o.status] ?? 0)} ${o.label}`).join(', ')}.
        </p>
        <div className="sr-only">
          <table>
            <caption>Interventions by stage</caption>
            <tbody>
              {rows.map((row) => (
                <tr key={row.status}>
                  <th scope="row">{row.label}</th>
                  <td>{formatCount(row.count)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </figure>
    </OverviewCard>
  );
};

export default InterventionStatus;
