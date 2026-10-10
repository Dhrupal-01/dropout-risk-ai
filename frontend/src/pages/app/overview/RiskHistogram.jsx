import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { getStatsDistribution } from '../../../api/endpoints';
import { formatCount } from '../../../app/format';
import { axisTick, gridStroke } from '../../../components/charts/chartTheme';
import TooltipBox from '../../../components/charts/TooltipBox';
import OverviewCard from './OverviewCard';

const percent = (fraction) => `${Math.round(fraction * 100)}%`;

const BinTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const bin = payload[0].payload;
  return (
    <TooltipBox title={`${percent(bin.lower)}–${percent(bin.upper)} estimated risk`}>
      {formatCount(bin.count)} students
    </TooltipBox>
  );
};

const RiskHistogram = () => {
  const distribution = useQuery({ queryKey: ['stats', 'distribution', 10], queryFn: () => getStatsDistribution(10) });
  const data = distribution.data;
  const bins = (data?.bins ?? []).map((bin) => ({ ...bin, label: percent(bin.lower) }));

  return (
    <OverviewCard
      title="Spread of risk estimates"
      description={
        data ? `Students by latest estimated risk, in steps of ${percent(1 / data.bin_count)}.` : 'Students by latest estimated risk.'
      }
      status={distribution.status}
      error={distribution.error}
      onRetry={() => distribution.refetch()}
      isEmpty={data?.total === 0}
      emptyMessage="No risk estimates yet. Scores appear here once students are imported and scored."
      loadingClassName="h-60"
    >
      <figure aria-label="Spread of risk estimates">
        <div aria-hidden="true" className="h-60">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart accessibilityLayer={false} data={bins} margin={{ top: 8, right: 8, bottom: 0, left: -16 }} barCategoryGap={2}>
              <CartesianGrid vertical={false} stroke={gridStroke} />
              <XAxis dataKey="label" tick={axisTick} stroke={gridStroke} tickLine={false} />
              <YAxis allowDecimals={false} tick={axisTick} axisLine={false} tickLine={false} />
              <Tooltip content={<BinTooltip />} cursor={{ fill: 'var(--ink-wash)' }} isAnimationActive={false} />
              <Bar dataKey="count" fill="var(--ink)" radius={[4, 4, 0, 0]} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="mt-2 text-13 text-slate">Estimated risk (calibrated probability), left to right.</p>
        <div className="sr-only">
          <table>
            <caption>Students by estimated risk band</caption>
            <thead>
              <tr>
                <th scope="col">Estimated risk</th>
                <th scope="col">Students</th>
              </tr>
            </thead>
            <tbody>
              {bins.map((bin) => (
                <tr key={bin.lower}>
                  <th scope="row">
                    {percent(bin.lower)} to {percent(bin.upper)}
                  </th>
                  <td>{formatCount(bin.count)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </figure>
    </OverviewCard>
  );
};

export default RiskHistogram;
