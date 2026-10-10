import React from 'react';
import { useQueries, useQuery } from '@tanstack/react-query';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { getMentorQueue, getStatsSummary } from '../../../api/endpoints';
import { TIERS } from '../../../app/tiers';
import { formatCount } from '../../../app/format';
import { axisTick, BAR_SIZE, gridStroke } from '../../../components/charts/chartTheme';
import TooltipBox from '../../../components/charts/TooltipBox';
import WrappedCategoryTick from '../../../components/charts/WrappedCategoryTick';
import TierLabel from '../../../components/TierLabel';
import OverviewCard from './OverviewCard';

const ROW_HEIGHT = 48;

// Counts per department and tier come from the queue's pre-pagination `total` (limit=1), one
// request per pair, so no new endpoint is needed.
const useTierByDepartment = () => {
  const summary = useQuery({ queryKey: ['stats', 'summary'], queryFn: getStatsSummary });
  const departments = Object.keys(summary.data?.by_department ?? {});
  const pairs = departments.flatMap((department) => TIERS.map((tier) => ({ department, tier: tier.api })));

  const counts = useQueries({
    queries: pairs.map(({ department, tier }) => ({
      queryKey: ['queue-count', department, tier],
      queryFn: () => getMentorQueue({ department, risk_tier: tier, limit: 1 }).then((page) => page.total),
    })),
    combine: (results) => ({
      status: results.some((r) => r.status === 'error') ? 'error' : results.some((r) => r.status === 'pending') ? 'pending' : 'success',
      error: results.find((r) => r.error)?.error,
      values: results.map((r) => r.data),
      refetch: () => results.forEach((r) => r.refetch()),
    }),
  });

  const status = summary.status !== 'success' ? summary.status : counts.status;
  const rows = departments
    .map((department) => {
      const row = { department };
      TIERS.forEach((tier) => {
        row[tier.api] = counts.values[pairs.findIndex((p) => p.department === department && p.tier === tier.api)] ?? 0;
      });
      row.total = TIERS.reduce((sum, tier) => sum + row[tier.api], 0);
      return row;
    })
    .sort((a, b) => b.High - a.High || a.department.localeCompare(b.department));

  return {
    status,
    error: summary.error || counts.error,
    refetch: () => (summary.status === 'error' ? summary.refetch() : counts.refetch()),
    rows,
  };
};

const DepartmentTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <TooltipBox title={row.department}>
      {TIERS.map((tier) => (
        <p key={tier.api}>
          {tier.label}: {formatCount(row[tier.api])}
        </p>
      ))}
    </TooltipBox>
  );
};

const TierByDepartment = () => {
  const { status, error, refetch, rows } = useTierByDepartment();

  return (
    <OverviewCard
      title="Risk level by department"
      description="Departments with the most students needing outreach first."
      status={status}
      error={error}
      onRetry={refetch}
      isEmpty={rows.length === 0}
      emptyMessage="No department information yet. Departments appear once scored students have one."
      loadingClassName="h-72"
    >
      <ul className="mb-3 flex flex-wrap gap-x-5 gap-y-1 text-13">
        {TIERS.map((tier) => (
          <li key={tier.api} className="flex items-center gap-1.5">
            <span className="h-3 w-3 rounded-[3px]" style={{ background: tier.mark }} aria-hidden="true" />
            <TierLabel tier={tier.api} />
          </li>
        ))}
      </ul>
      <figure aria-label="Risk level by department">
        <div aria-hidden="true" style={{ height: rows.length * ROW_HEIGHT + 40 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart accessibilityLayer={false} data={rows} layout="vertical" margin={{ top: 0, right: 16, bottom: 0, left: 0 }}>
              <CartesianGrid horizontal={false} stroke={gridStroke} />
              <XAxis type="number" allowDecimals={false} tick={axisTick} stroke={gridStroke} />
              <YAxis
                type="category"
                dataKey="department"
                width={130}
                interval={0}
                tickLine={false}
                axisLine={false}
                tick={<WrappedCategoryTick maxChars={16} />}
              />
              <Tooltip content={<DepartmentTooltip />} cursor={{ fill: 'var(--ink-wash)' }} isAnimationActive={false} />
              {TIERS.map((tier) => (
                <Bar
                  key={tier.api}
                  dataKey={tier.api}
                  stackId="tiers"
                  barSize={BAR_SIZE}
                  fill={tier.mark}
                  stroke="var(--paper)"
                  strokeWidth={2}
                  isAnimationActive={false}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="sr-only">
          <table>
            <caption>Students by risk level in each department</caption>
            <thead>
              <tr>
                <th scope="col">Department</th>
                {TIERS.map((tier) => (
                  <th key={tier.api} scope="col">
                    {tier.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.department}>
                  <th scope="row">{row.department}</th>
                  {TIERS.map((tier) => (
                    <td key={tier.api}>{formatCount(row[tier.api])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </figure>
    </OverviewCard>
  );
};

export default TierByDepartment;
