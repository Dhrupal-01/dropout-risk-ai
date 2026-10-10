import React, { useState, useEffect } from 'react';
import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { useSearchParams, useNavigate, Link } from 'react-router-dom';
import { AlertCircle, CalendarX, ChevronLeft, ChevronRight, FilterX, Search, Upload, Users } from 'lucide-react';
import { getMentorQueue, getMentorFilters, getStatsAlerts, getStatsSummary } from '../api/endpoints';
import { TIERS } from '../app/tiers';
import { formatCount, formatPercent } from '../app/format';
import RiskTierChip from '../components/RiskTierChip';

// Student worklist: every scored student, ordered by who to contact first. Filters live in the URL.

const PAGE_SIZE = 25;
const STATUS_LABELS = { ASSIGNED: 'Assigned', IN_PROGRESS: 'In progress', APPLIED: 'Applied', COMPLETED: 'Completed' };

// Helper hook to debounce input search queries
function useDebounce(value, delay) {
  const [debouncedValue, setDebouncedValue] = useState(value);
  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);
    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);
  return debouncedValue;
}

const controlClass =
  'w-full rounded-control border border-control bg-paper px-3 py-2 text-15 text-graphite disabled:opacity-50';

const FilterSelect = ({ label, value, onChange, disabled, children }) => (
  <label className="flex min-w-0 flex-col gap-1 text-13 text-slate">
    {label}
    <select value={value} onChange={onChange} disabled={disabled} className={controlClass}>
      {children}
    </select>
  </label>
);

const Dashboard = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();

  // Local state for search box (debounced)
  const initialSearch = searchParams.get('search') || '';
  const [searchInput, setSearchInput] = useState(initialSearch);
  const debouncedSearch = useDebounce(searchInput, 300);

  // Extract filters from URL search params
  const department = searchParams.get('department') || '';
  const riskTier = searchParams.get('risk_tier') || '';
  const assignedMentorId = searchParams.get('assigned_mentor_id') || '';
  const limit = parseInt(searchParams.get('limit') || String(PAGE_SIZE), 10);
  const offset = parseInt(searchParams.get('offset') || '0', 10);

  // Synchronise debounced search back to searchParams, resetting offset to 0
  useEffect(() => {
    const params = new URLSearchParams(searchParams);
    if (debouncedSearch) {
      params.set('search', debouncedSearch);
    } else {
      params.delete('search');
    }
    params.set('offset', '0'); // reset pagination on filter change
    setSearchParams(params);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedSearch]);

  // Sync initial URL search param to search input on history navigation
  useEffect(() => {
    // eslint-disable-next-line react/set-state-in-effect
    setSearchInput(searchParams.get('search') || '');
  }, [searchParams]);

  // Update other filters in URL, resetting offset to 0
  const handleFilterChange = (key, value) => {
    const params = new URLSearchParams(searchParams);
    if (value) {
      params.set(key, value);
    } else {
      params.delete(key);
    }
    params.set('offset', '0'); // reset pagination
    setSearchParams(params);
  };

  // Reset all filters in URL
  const clearFilters = () => {
    setSearchInput('');
    setSearchParams({ limit: String(PAGE_SIZE), offset: '0' });
  };

  // Main worklist query
  const { data: queueData, isPending, isError, error, refetch } = useQuery({
    queryKey: ['queue', { department, riskTier, assignedMentorId, limit, offset, debouncedSearch }],
    queryFn: () => getMentorQueue({
      department,
      risk_tier: riskTier || undefined,
      assigned_mentor_id: assignedMentorId || undefined,
      search: debouncedSearch || undefined,
      limit,
      offset,
    }),
    placeholderData: keepPreviousData,
  });

  // Cohort counts (shared cache with the Overview page)
  const { data: statsData } = useQuery({ queryKey: ['stats', 'summary'], queryFn: getStatsSummary });

  // Attendance threshold for the rule-based alert, from server configuration
  const { data: alertsData } = useQuery({ queryKey: ['stats', 'alerts'], queryFn: getStatsAlerts });
  const attendanceThreshold = alertsData?.attendance_threshold;

  // Dynamic departments and mentors list from backend filters endpoint
  const { data: filtersData, isPending: isFiltersLoading } = useQuery({
    queryKey: ['mentor-filters'],
    queryFn: getMentorFilters,
  });
  const departments = filtersData?.departments || [];
  const mentorIds = filtersData?.mentor_ids || [];

  // Pagination page count helpers
  const totalItems = queueData?.total || 0;
  const currentPage = Math.floor(offset / limit) + 1;
  const totalPages = Math.ceil(totalItems / limit) || 1;

  const goToOffset = (nextOffset) => {
    const params = new URLSearchParams(searchParams);
    params.set('offset', String(nextOffset));
    setSearchParams(params);
  };
  const handlePrevPage = () => offset > 0 && goToOffset(Math.max(0, offset - limit));
  const handleNextPage = () => offset + limit < totalItems && goToOffset(offset + limit);

  const hasFilters = Boolean(department || riskTier || assignedMentorId || searchInput);
  const isBelowThreshold = (attendance) =>
    attendance != null && attendanceThreshold != null && attendance < attendanceThreshold;

  return (
    <div className="px-4 sm:px-6 py-8 max-w-[1400px] space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-display font-medium text-32 tracking-display text-graphite">Students</h1>
          <p className="mt-2 max-w-measure text-15 text-slate">
            Everyone with a current risk estimate, ordered by whom to contact first.
          </p>
        </div>
        <Link to="/app/import" className="btn btn-secondary">
          <Upload className="w-4 h-4" aria-hidden="true" />
          Import CSV
        </Link>
      </div>

      {/* Cohort counts */}
      <dl className="grid grid-cols-2 gap-3 md:grid-cols-4">
        {[...TIERS.map((t) => ({ key: t.api, tier: t.api, value: statsData?.by_tier?.[t.api] })),
          { key: 'total', value: statsData?.total }].map(({ key, tier, value }) => (
          <div key={key} className="min-w-0 rounded-panel border border-rule bg-paper p-4">
            <dt className="text-15 text-slate">
              {tier ? (
                <RiskTierChip tier={tier} className="border-0 px-0 py-0" />
              ) : (
                <span className="inline-flex items-center gap-1.5">
                  <Users className="w-4 h-4" aria-hidden="true" />
                  All scored students
                </span>
              )}
            </dt>
            <dd className="mt-1 text-32 font-semibold leading-tight text-graphite">
              {value === undefined ? '…' : formatCount(value)}
            </dd>
          </div>
        ))}
      </dl>

      {/* Filters and search */}
      <div className="rounded-panel border border-rule bg-paper p-4">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-[repeat(3,minmax(0,12rem))_minmax(0,1fr)] lg:items-end">
          <FilterSelect
            label="Department"
            value={department}
            onChange={(e) => handleFilterChange('department', e.target.value)}
            disabled={isFiltersLoading}
          >
            <option value="">All departments</option>
            {departments.map((dept) => (
              <option key={dept} value={dept}>{dept}</option>
            ))}
          </FilterSelect>

          <FilterSelect label="Risk level" value={riskTier} onChange={(e) => handleFilterChange('risk_tier', e.target.value)}>
            <option value="">All risk levels</option>
            {TIERS.map((t) => (
              <option key={t.api} value={t.api}>{t.label}</option>
            ))}
          </FilterSelect>

          <FilterSelect
            label="Mentor"
            value={assignedMentorId}
            onChange={(e) => handleFilterChange('assigned_mentor_id', e.target.value)}
            disabled={isFiltersLoading}
          >
            <option value="">All mentors</option>
            {mentorIds.map((mId) => (
              <option key={mId} value={mId}>{mId}</option>
            ))}
          </FilterSelect>

          <label className="flex min-w-0 flex-col gap-1 text-13 text-slate sm:col-span-2 lg:col-span-1">
            Search
            <span className="relative block">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate" aria-hidden="true" />
              <input
                type="search"
                placeholder="Student ID or name"
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                className={`${controlClass} pl-9`}
              />
            </span>
          </label>
        </div>
        {hasFilters && (
          <button type="button" onClick={clearFilters} className="mt-3 inline-flex items-center gap-1.5 text-15 text-ink hover:underline">
            <FilterX className="w-4 h-4" aria-hidden="true" />
            Clear filters
          </button>
        )}
      </div>

      {/* Worklist table */}
      <div className="rounded-panel border border-rule bg-paper">
        {isPending ? (
          <div role="status" aria-label="Loading students" className="space-y-3 p-4">
            {[...Array(8)].map((_, i) => (
              <div key={i} className="h-9 rounded-control bg-ink-wash motion-safe:animate-pulse" />
            ))}
          </div>
        ) : isError ? (
          <div className="space-y-3 p-6">
            <p className="flex items-start gap-2 text-15 text-graphite">
              <AlertCircle className="mt-0.5 w-4 h-4 shrink-0 text-slate" aria-hidden="true" />
              Couldn't load the student list. {error?.message || 'Check the backend status in the top bar.'}
            </p>
            <button type="button" onClick={() => refetch()} className="btn btn-secondary">
              Try again
            </button>
          </div>
        ) : !queueData || queueData.items.length === 0 ? (
          <div className="space-y-3 p-6">
            <p className="text-15 text-graphite">
              {hasFilters ? 'No students match these filters.' : 'No students have been scored yet.'}
            </p>
            {hasFilters ? (
              <button type="button" onClick={clearFilters} className="btn btn-secondary">
                Clear filters
              </button>
            ) : (
              <Link to="/app/import" className="btn btn-primary">
                Import a CSV
              </Link>
            )}
          </div>
        ) : (
          <div className="relative overflow-x-auto">
            <table className="w-full border-collapse text-left text-15 tabular-nums">
              <thead>
                <tr className="border-b border-rule text-13 text-slate">
                  <th scope="col" className="px-4 py-3 font-medium">#</th>
                  <th scope="col" className="px-4 py-3 font-medium">Student</th>
                  <th scope="col" className="hidden md:table-cell px-4 py-3 font-medium">Department</th>
                  <th scope="col" className="px-4 py-3 text-right font-medium">Estimated risk</th>
                  <th scope="col" className="px-4 py-3 font-medium">Risk level</th>
                  <th scope="col" className="hidden sm:table-cell px-4 py-3 text-right font-medium">Attendance</th>
                  <th scope="col" className="hidden lg:table-cell px-4 py-3 text-right font-medium">CGPA</th>
                  <th scope="col" className="hidden lg:table-cell px-4 py-3 text-right font-medium">Backlogs</th>
                  <th scope="col" className="hidden lg:table-cell px-4 py-3 text-right font-medium">Fee delay</th>
                  <th scope="col" className="hidden xl:table-cell px-4 py-3 font-medium">Intervention</th>
                  <th scope="col" className="px-4 py-3"><span className="sr-only">Open</span></th>
                </tr>
              </thead>
              <tbody>
                {queueData.items.map((student) => {
                  const openStudent = () => navigate(`/app/students/${encodeURIComponent(student.student_id)}`);
                  const lowAttendance = isBelowThreshold(student.attendance);
                  return (
                    <tr
                      key={student.student_id}
                      onClick={openStudent}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') openStudent();
                      }}
                      tabIndex={0}
                      className="cursor-pointer border-b border-rule last:border-0 hover:bg-ink-wash focus-visible:bg-ink-wash"
                    >
                      <td className="px-4 py-3 text-slate">{student.priority_rank}</td>
                      <td className="px-4 py-3">
                        <div className="font-medium text-graphite">{student.name || student.student_id}</div>
                        {student.name && <div className="text-13 text-slate">{student.student_id}</div>}
                      </td>
                      <td className="hidden md:table-cell px-4 py-3 text-graphite">{student.department || '—'}</td>
                      <td className="px-4 py-3 text-right text-graphite">{formatPercent(student.risk_score_percentage)}</td>
                      <td className="px-4 py-3">
                        <RiskTierChip tier={student.risk_tier} className="border-0 px-0 py-0" />
                      </td>
                      <td className="hidden sm:table-cell px-4 py-3 text-right">
                        {student.attendance == null ? (
                          '—'
                        ) : (
                          <span
                            className={`inline-flex items-center gap-1 ${lowAttendance ? 'font-semibold text-graphite' : 'text-graphite'}`}
                            title={lowAttendance ? `Below the ${attendanceThreshold}% attendance requirement` : undefined}
                          >
                            {lowAttendance && <CalendarX className="w-3.5 h-3.5 text-slate" aria-hidden="true" />}
                            {formatPercent(student.attendance)}
                            {lowAttendance && <span className="sr-only"> (below the requirement)</span>}
                          </span>
                        )}
                      </td>
                      <td className="hidden lg:table-cell px-4 py-3 text-right text-graphite">
                        {student.cgpa == null ? '—' : student.cgpa.toFixed(2)}
                      </td>
                      <td className="hidden lg:table-cell px-4 py-3 text-right text-graphite">
                        {student.backlogs == null ? '—' : student.backlogs}
                      </td>
                      <td className="hidden lg:table-cell px-4 py-3 text-right text-graphite">
                        {student.fee_delay_days == null ? '—' : `${student.fee_delay_days} days`}
                      </td>
                      <td className="hidden xl:table-cell px-4 py-3">
                        {student.intervention_status ? (
                          <>
                            <div className="text-graphite">{student.primary_intervention}</div>
                            <div className="text-13 text-slate">{STATUS_LABELS[student.intervention_status] || student.intervention_status}</div>
                          </>
                        ) : (
                          <span className="whitespace-nowrap text-slate">None logged</span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <Link
                          to={`/app/students/${encodeURIComponent(student.student_id)}`}
                          onClick={(e) => e.stopPropagation()}
                          tabIndex={-1}
                          className="text-ink underline-offset-4 hover:underline"
                          aria-label={`Open ${student.name || student.student_id}`}
                        >
                          Open
                        </Link>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Pagination */}
      {!isPending && !isError && totalItems > 0 && (
        <nav aria-label="Pages" className="flex flex-wrap items-center justify-between gap-3 text-15 text-slate">
          <p>
            Showing <span className="tabular-nums text-graphite">{formatCount(offset + 1)}</span>–
            <span className="tabular-nums text-graphite">{formatCount(Math.min(offset + limit, totalItems))}</span> of{' '}
            <span className="tabular-nums text-graphite">{formatCount(totalItems)}</span> students
          </p>
          <div className="flex items-center gap-2">
            <button type="button" onClick={handlePrevPage} disabled={offset === 0} className="btn btn-secondary disabled:opacity-50" aria-label="Previous page">
              <ChevronLeft className="w-4 h-4" aria-hidden="true" />
              Previous
            </button>
            <span className="px-2 tabular-nums text-graphite">
              Page {currentPage} of {totalPages}
            </span>
            <button type="button" onClick={handleNextPage} disabled={offset + limit >= totalItems} className="btn btn-secondary disabled:opacity-50" aria-label="Next page">
              Next
              <ChevronRight className="w-4 h-4" aria-hidden="true" />
            </button>
          </div>
        </nav>
      )}

      {queueData?.disclaimer && <p className="max-w-measure text-13 text-slate">{queueData.disclaimer}</p>}
    </div>
  );
};

export default Dashboard;
