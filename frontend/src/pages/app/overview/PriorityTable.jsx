import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { getMentorQueue } from '../../../api/endpoints';
import { formatPercent } from '../../../app/format';
import TierLabel from '../../../components/TierLabel';
import OverviewCard from './OverviewCard';

const TOP_N = 10;

const PriorityTable = ({ className = '' }) => {
  const queue = useQuery({ queryKey: ['mentor-queue', 'top', TOP_N], queryFn: () => getMentorQueue({ limit: TOP_N }) });
  const items = queue.data?.items ?? [];

  return (
    <OverviewCard
      className={className}
      title="Top 10 students to contact first"
      description="Ordered by priority: estimated risk first, then backlogs and attendance."
      headerExtra={
        <Link to="/app/students" className="text-15 text-ink underline underline-offset-4 hover:no-underline">
          View all students
        </Link>
      }
      status={queue.status}
      error={queue.error}
      onRetry={() => queue.refetch()}
      isEmpty={items.length === 0}
      emptyMessage={
        <>
          No students have been scored yet.{' '}
          <Link to="/app/import" className="text-ink underline underline-offset-4">
            Import a CSV
          </Link>{' '}
          to build the list.
        </>
      }
      loadingClassName="h-80"
    >
      <table className="w-full border-collapse text-15 tabular-nums">
        <thead>
          <tr className="border-b border-rule text-left text-13 text-slate">
            <th scope="col" className="py-2 pr-3 font-medium">#</th>
            <th scope="col" className="py-2 pr-3 font-medium">Student</th>
            <th scope="col" className="hidden md:table-cell py-2 pr-3 font-medium">Department</th>
            <th scope="col" className="py-2 pr-3 font-medium">Risk level</th>
            <th scope="col" className="py-2 pr-3 text-right font-medium">Estimated risk</th>
            <th scope="col" className="hidden sm:table-cell py-2 text-right font-medium">Attendance</th>
          </tr>
        </thead>
        <tbody>
          {items.map((s) => (
            <tr key={s.student_id} className="border-b border-rule last:border-0">
              <td className="py-2.5 pr-3 text-slate">{s.priority_rank}</td>
              <td className="py-2.5 pr-3">
                <Link
                  to={`/app/students/${encodeURIComponent(s.student_id)}`}
                  className="font-medium text-ink underline-offset-4 hover:underline"
                >
                  {s.name || s.student_id}
                </Link>
                {s.name && <div className="text-13 text-slate">{s.student_id}</div>}
              </td>
              <td className="hidden md:table-cell py-2.5 pr-3 text-graphite">{s.department || '—'}</td>
              <td className="py-2.5 pr-3">
                <TierLabel tier={s.risk_tier} className="whitespace-nowrap text-13 sm:text-15" />
              </td>
              <td className="py-2.5 pr-3 text-right text-graphite">{formatPercent(s.risk_score_percentage)}</td>
              <td className="hidden sm:table-cell py-2.5 text-right text-graphite">
                {s.attendance == null ? '—' : formatPercent(s.attendance)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {queue.data?.disclaimer && <p className="mt-3 text-13 text-slate max-w-measure">{queue.data.disclaimer}</p>}
    </OverviewCard>
  );
};

export default PriorityTable;
