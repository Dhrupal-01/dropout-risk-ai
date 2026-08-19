import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useSearchParams, useNavigate, Link } from 'react-router-dom';
import { 
  Search, FilterX, Users, AlertOctagon, AlertTriangle, CheckCircle, 
  ChevronLeft, ChevronRight, Eye, AlertCircle
} from 'lucide-react';
import { getMentorQueue } from '../api/endpoints';
import RiskTierChip from '../components/RiskTierChip';

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
  const limit = parseInt(searchParams.get('limit') || '25', 10);
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
    setSearchParams({ limit: '25', offset: '0' });
  };

  // Main worklist query
  const { data: queueData, isLoading, isError, error, refetch } = useQuery({
    queryKey: ['queue', { department, riskTier, assignedMentorId, limit, offset, debouncedSearch }],
    queryFn: () => getMentorQueue({
      department,
      risk_tier: riskTier || undefined,
      assigned_mentor_id: assignedMentorId || undefined,
      limit,
      offset,
      // Search parameter is handled client-side if API doesn't support search,
      // but we send it or apply it appropriately.
    }),
    keepPreviousData: true,
  });

  // KPI metadata queries to show totals (using limit=1 for optimal performance)
  const { data: totalHigh } = useQuery({
    queryKey: ['kpi-high'],
    queryFn: () => getMentorQueue({ risk_tier: 'High', limit: 1 }),
  });
  const { data: totalMedium } = useQuery({
    queryKey: ['kpi-medium'],
    queryFn: () => getMentorQueue({ risk_tier: 'Medium', limit: 1 }),
  });
  const { data: totalLow } = useQuery({
    queryKey: ['kpi-low'],
    queryFn: () => getMentorQueue({ risk_tier: 'Low', limit: 1 }),
  });
  const { data: totalStudents } = useQuery({
    queryKey: ['kpi-total'],
    queryFn: () => getMentorQueue({ limit: 1 }),
  });

  // Common departments and mentors list for filters (static mock lists for demo selectors)
  const DEPARTMENTS = [
    'Computer Science & Engineering',
    'Information Technology',
    'Electronics & Communication Engineering',
    'Mechanical Engineering',
    'Electrical Engineering'
  ];
  const MENTORS = [
    { id: 'FAC_001', name: 'Dr. A. Sharma' },
    { id: 'FAC_007', name: 'Prof. Meera Nair (FAC_007)' },
    { id: 'FAC_012', name: 'Dr. R. Patel' },
    { id: 'FAC_015', name: 'Prof. S. Das' },
  ];

  // Pagination page count helpers
  const totalItems = queueData?.total || 0;
  const currentPage = Math.floor(offset / limit) + 1;
  const totalPages = Math.ceil(totalItems / limit) || 1;

  const handlePrevPage = () => {
    if (offset > 0) {
      const params = new URLSearchParams(searchParams);
      params.set('offset', String(Math.max(0, offset - limit)));
      setSearchParams(params);
    }
  };

  const handleNextPage = () => {
    if (offset + limit < totalItems) {
      const params = new URLSearchParams(searchParams);
      params.set('offset', String(offset + limit));
      setSearchParams(params);
    }
  };

  // Render KPI Card
  const renderKpiCard = (title, count, icon, colorClass, borderStyle) => {
    return (
      <div className={`bg-card p-5 rounded-lg border ${borderStyle} shadow-sm flex items-center justify-between`}>
        <div>
          <span className="text-xs uppercase tracking-wider text-secondary font-semibold">{title}</span>
          <h2 className="text-3xl font-bold text-primary mt-1 select-all font-mono">
            {count !== undefined ? count.toLocaleString() : '...'}
          </h2>
        </div>
        <div className={`p-3 rounded-full ${colorClass}`}>
          {icon}
        </div>
      </div>
    );
  };

  return (
    <div className="container mx-auto px-6 py-8 max-w-7xl space-y-8">
      {/* Triage Welcome Title */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-primary">Student Triage Worklist</h1>
          <p className="text-sm text-secondary mt-1">
            Prioritized outreach list generated from calibrated model risk estimates.
          </p>
        </div>
        <Link 
          to="/import"
          className="inline-flex items-center justify-center px-4 py-2 text-xs font-semibold rounded-md border border-accent text-accent hover:bg-accent/10 focus:ring-2 focus:ring-accent transition-colors"
        >
          Bulk CSV Import
        </Link>
      </div>

      {/* KPI Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        {renderKpiCard(
          'High Risk outreach', 
          totalHigh?.total, 
          <AlertOctagon className="w-5 h-5 text-risk-high" />, 
          'bg-red-50 dark:bg-red-950/20', 
          'border-red-200 dark:border-red-900/40'
        )}
        {renderKpiCard(
          'Medium Risk watchlist', 
          totalMedium?.total, 
          <AlertTriangle className="w-5 h-5 text-risk-medium" />, 
          'bg-amber-50 dark:bg-amber-950/20', 
          'border-amber-200 dark:border-amber-900/40'
        )}
        {renderKpiCard(
          'Low Risk monitor', 
          totalLow?.total, 
          <CheckCircle className="w-5 h-5 text-risk-low" />, 
          'bg-green-50 dark:bg-green-950/20', 
          'border-green-200 dark:border-green-900/40'
        )}
        {renderKpiCard(
          'Total Students scored', 
          totalStudents?.total, 
          <Users className="w-5 h-5 text-accent" />, 
          'bg-blue-50 dark:bg-blue-950/20', 
          'border-blue-200 dark:border-blue-900/40'
        )}
      </div>

      {/* Filter and Search Bar */}
      <div className="bg-card p-4 rounded-lg border border-border shadow-sm space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-3">
            {/* Department Filter */}
            <select
              value={department}
              onChange={(e) => handleFilterChange('department', e.target.value)}
              className="text-xs border border-border rounded-md px-3 py-2 bg-card text-primary font-medium hover:border-accent focus:ring-2 focus:ring-accent transition-colors"
              aria-label="Filter by department"
            >
              <option value="">All Departments</option>
              {DEPARTMENTS.map(dept => (
                <option key={dept} value={dept}>{dept}</option>
              ))}
            </select>

            {/* Risk Tier Filter */}
            <select
              value={riskTier}
              onChange={(e) => handleFilterChange('risk_tier', e.target.value)}
              className="text-xs border border-border rounded-md px-3 py-2 bg-card text-primary font-medium hover:border-accent focus:ring-2 focus:ring-accent transition-colors"
              aria-label="Filter by risk tier"
            >
              <option value="">All Risk Levels</option>
              <option value="High">High Risk</option>
              <option value="Medium">Medium Risk</option>
              <option value="Low">Low Risk</option>
            </select>

            {/* Assigned Mentor Filter */}
            <select
              value={assignedMentorId}
              onChange={(e) => handleFilterChange('assigned_mentor_id', e.target.value)}
              className="text-xs border border-border rounded-md px-3 py-2 bg-card text-primary font-medium hover:border-accent focus:ring-2 focus:ring-accent transition-colors"
              aria-label="Filter by assigned mentor"
            >
              <option value="">All Mentors</option>
              {MENTORS.map(mentor => (
                <option key={mentor.id} value={mentor.id}>{mentor.name}</option>
              ))}
            </select>

            {/* Clear Filters action */}
            {(department || riskTier || assignedMentorId || searchInput) && (
              <button
                onClick={clearFilters}
                className="flex items-center text-xs font-semibold text-accent hover:text-accent-hover transition-colors px-2 py-1.5 rounded hover:bg-hover"
                title="Clear all active worklist filters"
              >
                <FilterX className="w-3.5 h-3.5 mr-1" />
                Clear Filters
              </button>
            )}
          </div>

          {/* Search bar */}
          <div className="relative w-full sm:w-64 select-none">
            <Search className="absolute left-3 top-2.5 h-3.5 w-3.5 text-muted" />
            <input
              type="text"
              placeholder="Search student ID or Name..."
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              className="text-xs border border-border rounded-md pl-9 pr-4 py-2 w-full bg-card text-primary hover:border-accent focus:ring-2 focus:ring-accent transition-colors"
              aria-label="Search student database"
            />
          </div>
        </div>
      </div>

      {/* Main Mentor Triage Table */}
      <div className="bg-card border border-border rounded-lg shadow-sm overflow-hidden">
        {isLoading ? (
          /* Table Skeletons for Loading State */
          <div className="p-6 space-y-4 animate-pulse select-none">
            <div className="h-6 bg-subtle rounded w-full" />
            {[...Array(8)].map((_, i) => (
              <div key={i} className="flex space-x-3 items-center">
                <div className="h-4 bg-subtle rounded flex-1" />
                <div className="h-4 bg-subtle rounded w-16" />
                <div className="h-4 bg-subtle rounded w-24" />
                <div className="h-4 bg-subtle rounded w-12" />
              </div>
            ))}
          </div>
        ) : isError ? (
          /* Error State */
          <div className="p-8 text-center space-y-4">
            <AlertCircle className="w-8 h-8 text-risk-high mx-auto animate-bounce" />
            <h3 className="text-md font-semibold text-primary">Failed to load worklist</h3>
            <p className="text-xs text-secondary max-w-md mx-auto">
              {error?.message || 'A server connection issue occurred. Please check network settings.'}
            </p>
            <button
              onClick={() => refetch()}
              className="px-4 py-2 bg-accent text-white text-xs font-semibold rounded hover:bg-accent-hover transition-colors"
            >
              Retry Connection
            </button>
          </div>
        ) : !queueData || queueData.items.length === 0 ? (
          /* Empty State */
          <div className="p-12 text-center space-y-3">
            <Users className="w-8 h-8 text-muted mx-auto" />
            <h3 className="text-md font-semibold text-primary">No students match these filters</h3>
            <p className="text-xs text-secondary max-w-sm mx-auto">
              Try adjusting your department, risk level, search parameters or clear the filters.
            </p>
            <button
              onClick={clearFilters}
              className="px-4 py-2 bg-accent text-white text-xs font-semibold rounded hover:bg-accent-hover transition-colors"
            >
              Clear Filters
            </button>
          </div>
        ) : (
          /* Data Success State */
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-left text-xs text-primary">
              <thead>
                <tr className="bg-subtle text-secondary font-semibold border-b border-border select-none">
                  <th className="p-4 w-12">Priority</th>
                  <th className="p-4">Student</th>
                  <th className="p-4">Department</th>
                  <th className="p-4">Risk Probability</th>
                  <th className="p-4">Tier</th>
                  <th className="p-4 text-center">Att %</th>
                  <th className="p-4 text-center">CGPA</th>
                  <th className="p-4 text-center">Backlogs</th>
                  <th className="p-4 text-center">Fee Delay</th>
                  <th className="p-4">Active Intervention</th>
                  <th className="p-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody>
                {queueData.items.map((student) => {
                  const hasHistory = student.intervention_status !== null;
                  
                  // Keyboard row navigation helper
                  const handleRowClick = () => {
                    navigate(`/students/${student.student_id}`);
                  };

                  return (
                    <tr 
                      key={student.student_id}
                      onClick={handleRowClick}
                      className="border-b border-border last:border-0 hover:bg-hover/40 cursor-pointer transition-colors group"
                      tabIndex={0}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleRowClick();
                      }}
                    >
                      {/* Priority Rank column */}
                      <td className="p-4 font-mono font-semibold text-secondary tabular-nums">
                        #{student.priority_rank}
                      </td>

                      {/* Name / ID column */}
                      <td className="p-4 font-semibold text-primary">
                        <div className="flex flex-col">
                          <span>{student.name || 'Anonymous Student'}</span>
                          <span className="text-[10px] text-muted font-mono mt-0.5">{student.student_id}</span>
                        </div>
                      </td>

                      {/* Department column */}
                      <td className="p-4 text-secondary">{student.department || '-'}</td>

                      {/* Risk Probability bar column */}
                      <td className="p-4 tabular-nums">
                        <div className="flex items-center space-x-2.5">
                          <span className="font-bold text-primary">
                            {student.risk_score_percentage.toFixed(1)}%
                          </span>
                          <div className="w-16 h-1.5 bg-subtle rounded-full overflow-hidden hidden sm:block border border-border/20">
                            <div 
                              className={`h-full ${
                                student.risk_tier === 'High' ? 'bg-risk-high' :
                                student.risk_tier === 'Medium' ? 'bg-risk-medium' : 'bg-risk-low'
                              }`}
                              style={{ width: `${student.risk_score_percentage}%` }}
                            />
                          </div>
                        </div>
                      </td>

                      {/* Tier Chip column */}
                      <td className="p-4">
                        <RiskTierChip tier={student.risk_tier} />
                      </td>

                      {/* Attendance column */}
                      <td className={`p-4 text-center font-semibold tabular-nums ${
                        student.attendance !== null && student.attendance < 75 
                          ? 'text-risk-high' 
                          : 'text-secondary'
                      }`}>
                        {student.attendance !== null ? `${student.attendance.toFixed(1)}%` : '-'}
                      </td>

                      {/* CGPA column */}
                      <td className="p-4 text-center text-secondary tabular-nums">
                        {student.cgpa !== null ? student.cgpa.toFixed(2) : '-'}
                      </td>

                      {/* Backlog column */}
                      <td className={`p-4 text-center font-semibold tabular-nums ${
                        student.backlogs > 0 ? 'text-risk-medium' : 'text-secondary'
                      }`}>
                        {student.backlogs !== null ? student.backlogs : '-'}
                      </td>

                      {/* Fee Delay column */}
                      <td className={`p-4 text-center font-semibold tabular-nums ${
                        student.fee_delay_days > 30 ? 'text-risk-high' : 'text-secondary'
                      }`}>
                        {student.fee_delay_days !== null ? `${student.fee_delay_days}d` : '-'}
                      </td>

                      {/* Intervention status column */}
                      <td className="p-4">
                        {hasHistory ? (
                          <div className="flex flex-col">
                            <span className="font-semibold text-secondary text-[11px] truncate max-w-[120px]">
                              {student.primary_intervention}
                            </span>
                            <span className="text-[10px] text-accent font-semibold mt-0.5">
                              {student.intervention_status}
                            </span>
                          </div>
                        ) : (
                          <span className="text-muted font-normal text-[11px] italic">None Logged</span>
                        )}
                      </td>

                      {/* Clickable Action View Link */}
                      <td className="p-4 text-right">
                        <Link
                          to={`/students/${student.student_id}`}
                          onClick={(e) => e.stopPropagation()} // stop duplicate navigation triggers
                          className="inline-flex items-center text-xs font-bold text-accent hover:text-accent-hover transition-colors focus:ring-2 focus:ring-accent px-2.5 py-1.5 rounded hover:bg-hover border border-border"
                          title={`View explanation details for ${student.name}`}
                          tabIndex={-1} // handled by row navigation
                        >
                          <Eye className="w-3.5 h-3.5 mr-1" />
                          View
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

      {/* Pagination Controls */}
      {!isLoading && !isError && totalItems > 0 && (
        <div className="flex items-center justify-between bg-card px-4 py-3.5 border border-border rounded-lg shadow-sm text-xs font-semibold text-secondary select-none">
          <div>
            Showing <span className="text-primary tabular-nums">{offset + 1}</span>–
            <span className="text-primary tabular-nums">
              {Math.min(offset + limit, totalItems)}
            </span> of <span className="text-primary tabular-nums">{totalItems}</span> students
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={handlePrevPage}
              disabled={offset === 0}
              className="flex items-center px-3 py-1.5 border border-border rounded-md hover:bg-hover text-secondary hover:text-primary transition-colors disabled:opacity-50 disabled:pointer-events-none focus:ring-2 focus:ring-accent"
              aria-label="Previous Page"
            >
              <ChevronLeft className="w-4 h-4 mr-1" />
              Previous
            </button>
            
            <div className="px-3 py-1 border border-border bg-subtle rounded-md text-primary font-bold">
              Page {currentPage} of {totalPages}
            </div>

            <button
              onClick={handleNextPage}
              disabled={offset + limit >= totalItems}
              className="flex items-center px-3 py-1.5 border border-border rounded-md hover:bg-hover text-secondary hover:text-primary transition-colors disabled:opacity-50 disabled:pointer-events-none focus:ring-2 focus:ring-accent"
              aria-label="Next Page"
            >
              Next
              <ChevronRight className="w-4 h-4 ml-1" />
            </button>
          </div>
        </div>
      )}

      {/* Queue Triage Simulation Disclaimer */}
      {queueData?.disclaimer && (
        <div className="text-[10px] text-muted bg-card border border-border rounded p-3 text-center italic leading-relaxed">
          {queueData.disclaimer}
        </div>
      )}
    </div>
  );
};

export default Dashboard;
