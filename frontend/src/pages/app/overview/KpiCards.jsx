import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { CalendarX, ClipboardList, RotateCw, Users } from 'lucide-react';
import { getStatsAlerts, getStatsInterventions, getStatsSummary } from '../../../api/endpoints';
import { formatCount, formatShare } from '../../../app/format';
import TierLabel from '../../../components/TierLabel';
import { Skeleton } from './OverviewCard';

const KpiCard = ({ label, query, value, sub }) => (
  <div className="min-w-0 rounded-panel border border-rule bg-paper p-4">
    <div className="text-15 text-slate">{label}</div>
    {query.status === 'pending' && <Skeleton className="mt-2 h-9 w-20" />}
    {query.status === 'error' && (
      <div className="mt-2 flex items-center gap-2 text-15 text-graphite">
        Unavailable
        <button
          type="button"
          onClick={() => query.refetch()}
          className="inline-flex items-center justify-center w-8 h-8 rounded-control border border-control hover:bg-ink-wash"
          aria-label="Try loading again"
          title="Try again"
        >
          <RotateCw className="w-3.5 h-3.5" aria-hidden="true" />
        </button>
      </div>
    )}
    {query.status === 'success' && (
      <>
        <p className="mt-1 text-32 font-semibold leading-tight text-graphite">{value(query.data)}</p>
        <p className="mt-1 text-13 text-slate">{sub(query.data)}</p>
      </>
    )}
  </div>
);

const IconLabel = ({ icon: Icon, children }) => (
  <span className="inline-flex items-center gap-1.5">
    <Icon className="w-4 h-4 shrink-0" aria-hidden="true" />
    {children}
  </span>
);

const KpiCards = () => {
  const summary = useQuery({ queryKey: ['stats', 'summary'], queryFn: getStatsSummary });
  const alerts = useQuery({ queryKey: ['stats', 'alerts'], queryFn: getStatsAlerts });
  const interventions = useQuery({ queryKey: ['stats', 'interventions'], queryFn: getStatsInterventions });

  const attendanceCount = (data) => data.alerts.find((a) => a.code === 'ATTENDANCE_BELOW_REQUIREMENT')?.student_count ?? 0;
  const tierCard = (tier) => ({
    label: <TierLabel tier={tier} />,
    query: summary,
    value: (d) => formatCount(d.by_tier[tier] ?? 0),
    sub: (d) => (d.total ? `${formatShare(d.by_tier[tier] ?? 0, d.total)} of students` : 'No students scored yet'),
  });

  const cards = [
    {
      key: 'total',
      label: <IconLabel icon={Users}>Students scored</IconLabel>,
      query: summary,
      value: (d) => formatCount(d.total),
      sub: (d) => (d.total ? 'with a current risk estimate' : 'Import a CSV to get started'),
    },
    { key: 'High', ...tierCard('High') },
    { key: 'Medium', ...tierCard('Medium') },
    { key: 'Low', ...tierCard('Low') },
    {
      key: 'attendance',
      label: (
        <IconLabel icon={CalendarX}>
          {alerts.data ? `Below ${formatCount(alerts.data.attendance_threshold)}% attendance` : 'Attendance alerts'}
        </IconLabel>
      ),
      query: alerts,
      value: (d) => formatCount(attendanceCount(d)),
      sub: () => 'rule-based, separate from the model',
    },
    {
      key: 'open',
      label: <IconLabel icon={ClipboardList}>Open interventions</IconLabel>,
      query: interventions,
      value: (d) => formatCount(d.open),
      sub: (d) =>
        d.total ? `across ${formatCount(d.students_with_open_interventions)} students` : 'None logged yet',
    },
  ];

  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
      {cards.map(({ key, ...card }) => (
        <KpiCard key={key} {...card} />
      ))}
    </div>
  );
};

export default KpiCards;
