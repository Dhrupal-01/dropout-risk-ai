import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { getStatsSummary } from '../../../api/endpoints';
import { TIERS } from '../../../app/tiers';
import { formatCount, formatShare } from '../../../app/format';
import TierLabel from '../../../components/TierLabel';
import OverviewCard from './OverviewCard';

// Part-to-whole as one 100% bar rather than a donut of tier counts. 2px surface gaps
// separate the segments; every segment is also named with icon, label and count below.
const TierSplit = () => {
  const summary = useQuery({ queryKey: ['stats', 'summary'], queryFn: getStatsSummary });
  const total = summary.data?.total ?? 0;
  const parts = TIERS.map((tier) => ({ ...tier, count: summary.data?.by_tier[tier.api] ?? 0 }));
  const description = parts.map((p) => `${p.label} ${formatCount(p.count)} (${formatShare(p.count, total)})`).join(', ');

  return (
    <OverviewCard
      title="Students by risk level"
      description="Each student's latest model estimate."
      status={summary.status}
      error={summary.error}
      onRetry={() => summary.refetch()}
      isEmpty={total === 0}
      emptyMessage={
        <>
          No students have been scored yet.{' '}
          <Link to="/app/import" className="text-ink underline underline-offset-4">
            Import a CSV
          </Link>{' '}
          to see the split.
        </>
      }
      loadingClassName="h-24"
    >
      <div role="img" aria-label={`Students by risk level: ${description}`} className="flex h-8 w-full gap-0.5">
        {parts
          .filter((p) => p.count > 0)
          .map((p) => (
            <div
              key={p.api}
              className="h-full first:rounded-l-control last:rounded-r-control"
              style={{ width: `${(p.count / total) * 100}%`, background: p.mark }}
            />
          ))}
      </div>
      <ul className="mt-4 flex flex-wrap gap-x-6 gap-y-2">
        {parts.map((p) => (
          <li key={p.api} className="flex items-center gap-2 whitespace-nowrap text-15">
            <span className="h-3 w-3 shrink-0 rounded-[3px]" style={{ background: p.mark }} aria-hidden="true" />
            <TierLabel tier={p.api} />
            <span className="tabular-nums text-graphite">
              {formatCount(p.count)} <span className="text-slate">({formatShare(p.count, total)})</span>
            </span>
          </li>
        ))}
      </ul>
    </OverviewCard>
  );
};

export default TierSplit;
