import React, { useEffect, useId, useRef, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  AlertCircle, ArrowLeft, Calendar, CalendarX, CheckCircle2, CircleDot, ClipboardList, Info, RefreshCw, X,
} from 'lucide-react';
import {
  getMentorQueue,
  getStudentExplanation,
  getStudentInterventions,
  getStudentInterventionsHistory,
  logIntervention,
  rescoreStudent,
} from '../api/endpoints';
import { tierFor } from '../app/tiers';
import { formatPercent } from '../app/format';
import RiskTierChip from '../components/RiskTierChip';
import ShapChart from '../components/ShapChart';
import AdminTokenNotice from '../components/AdminTokenNotice';

const STATUS_ORDER = ['ASSIGNED', 'IN_PROGRESS', 'APPLIED', 'COMPLETED'];
const STATUS_LABELS = { ASSIGNED: 'Assigned', IN_PROGRESS: 'In progress', APPLIED: 'Applied', COMPLETED: 'Completed' };
const OUTCOME_LABELS = { IMPROVED: 'Improved', NO_CHANGE: 'No change', DETERIORATED: 'Deteriorated', PENDING_EVALUATION: 'Awaiting evaluation' };
// Drivers fetched for the chart: 5 shown by default, "Show more factors" reveals the rest.
const EXPLANATION_TOP_K = 8;

const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

// Keeps Tab and Shift+Tab inside `container`, wrapping from the last control to the first and back.
const trapTab = (e, container) => {
  const items = [...container.querySelectorAll(FOCUSABLE)].filter((el) => el.getClientRects().length > 0);
  if (!items.length) return;
  const first = items[0];
  const last = items[items.length - 1];
  const active = document.activeElement;
  if (e.shiftKey && (active === first || !container.contains(active))) {
    e.preventDefault();
    last.focus();
  } else if (!e.shiftKey && (active === last || !container.contains(active))) {
    e.preventDefault();
    first.focus();
  }
};

const formatDate = (value) =>
  new Date(value).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });

const Panel = ({ title, description, children, className = '' }) => {
  const headingId = useId();
  return (
    <section aria-labelledby={title ? headingId : undefined} className={`min-w-0 rounded-panel border border-rule bg-paper p-4 sm:p-5 ${className}`}>
      {title && (
        <div className="mb-4">
          <h2 id={headingId} className="text-17 font-semibold text-graphite">{title}</h2>
          {description && <p className="mt-0.5 text-13 text-slate">{description}</p>}
        </div>
      )}
      {children}
    </section>
  );
};

const fieldClass = 'w-full rounded-control border border-control bg-paper px-3 py-2 text-15 text-graphite';
const labelClass = 'block text-13 font-medium text-graphite';

const StudentDetail = () => {
  const { studentId } = useParams();
  const queryClient = useQueryClient();

  // Log drawer state
  const [isLogOpen, setIsLogOpen] = useState(false);
  const [selectedIntervention, setSelectedIntervention] = useState(null);
  const [assignedFaculty, setAssignedFaculty] = useState('');
  const [notes, setNotes] = useState('');
  const [followupDate, setFollowupDate] = useState('');
  const [postRiskPct, setPostRiskPct] = useState('');
  const [logStatus, setLogStatus] = useState('ASSIGNED');
  const [statusFloor, setStatusFloor] = useState('ASSIGNED');
  const openerRef = useRef(null);
  const firstFieldRef = useRef(null);
  const drawerRef = useRef(null);
  const drawerTitleId = useId();

  const explanationQuery = useQuery({
    queryKey: ['explanation', studentId],
    queryFn: () => getStudentExplanation(studentId, EXPLANATION_TOP_K),
    retry: 1,
  });
  const interventionsQuery = useQuery({
    queryKey: ['interventions', studentId],
    queryFn: () => getStudentInterventions(studentId),
    retry: 1,
  });
  const { data: historyLogs } = useQuery({
    queryKey: ['history', studentId],
    queryFn: () => getStudentInterventionsHistory(studentId),
  });
  // Name, department and mentor are not part of the explanation; read them from the queue entry.
  const { data: identity } = useQuery({
    queryKey: ['student-identity', studentId],
    queryFn: () =>
      getMentorQueue({ search: studentId, limit: 5 }).then((page) => page.items.find((s) => s.student_id === studentId) ?? null),
  });

  const explanation = explanationQuery.data;
  const recoData = interventionsQuery.data;

  // Any write changes the queue, the overview and this student's panels.
  const refreshAfterWrite = () => queryClient.invalidateQueries();

  const closeDrawer = () => {
    setIsLogOpen(false);
    openerRef.current?.focus();
  };

  const logMutation = useMutation({
    mutationFn: logIntervention,
    onSuccess: () => {
      refreshAfterWrite();
      setSelectedIntervention(null);
      setAssignedFaculty('');
      setNotes('');
      setFollowupDate('');
      setPostRiskPct('');
      closeDrawer();
    },
  });

  // "Score now" for a student who exists but was never scored. The server scores the inputs it
  // already holds for the student; nothing typed in the browser reaches the model.
  const reScoreMutation = useMutation({
    mutationFn: () => rescoreStudent(studentId),
    onSuccess: refreshAfterWrite,
  });

  // Drawer: focus the first field on open; Tab stays inside it; Escape closes it from anywhere.
  useEffect(() => {
    if (!isLogOpen) return undefined;
    firstFieldRef.current?.focus();
    const onKey = (e) => {
      if (e.key === 'Escape') closeDrawer();
      else if (e.key === 'Tab' && drawerRef.current) trapTab(e, drawerRef.current);
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [isLogOpen]);

  const isPending = explanationQuery.isPending || interventionsQuery.isPending;
  const isError = explanationQuery.isError || interventionsQuery.isError;
  const activeError = explanationQuery.error || interventionsQuery.error;

  if (isPending) {
    return (
      <div role="status" aria-label="Loading student" className="px-4 sm:px-6 py-8 max-w-[1400px] space-y-4">
        <div className="h-5 w-32 rounded-control bg-ink-wash motion-safe:animate-pulse" />
        <div className="h-28 rounded-panel bg-ink-wash motion-safe:animate-pulse" />
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="h-80 rounded-panel bg-ink-wash motion-safe:animate-pulse" />
          <div className="h-80 rounded-panel bg-ink-wash motion-safe:animate-pulse" />
        </div>
      </div>
    );
  }

  const backLink = (
    <Link to="/app/students" className="inline-flex items-center gap-1.5 text-15 text-ink hover:underline">
      <ArrowLeft className="w-4 h-4" aria-hidden="true" />
      All students
    </Link>
  );

  // The student exists but was never scored: a normal state, not an error.
  if (isError && activeError?.code === 'no_prediction_history') {
    return (
      <div className="px-4 sm:px-6 py-8 max-w-2xl space-y-6">
        {backLink}
        <Panel>
          <h1 className="font-display font-medium text-32 tracking-display text-graphite">Not yet scored</h1>
          <p className="mt-2 text-15 text-slate">
            {identity?.name || studentId} is in the database but has no risk estimate yet. Scoring runs the model
            and stores the reasons behind the estimate.
          </p>
          {activeError.has_stored_features === false ? (
            <p className="mt-4 text-15 text-graphite">
              There are no stored records for this student to score from, so scoring is not available here.{' '}
              <Link to="/app/import" className="text-ink underline underline-offset-4 hover:no-underline">
                Import their records
              </Link>{' '}
              to score them.
            </p>
          ) : (
            <div className="mt-4 space-y-3">
              <AdminTokenNotice />
              {reScoreMutation.isError && (
                <p className="text-15 text-graphite">Scoring failed. {reScoreMutation.error?.message}</p>
              )}
              <button
                type="button"
                onClick={() => reScoreMutation.mutate()}
                disabled={reScoreMutation.isPending}
                className="btn btn-primary disabled:opacity-60"
              >
                {reScoreMutation.isPending && <RefreshCw className="w-4 h-4 motion-safe:animate-spin" aria-hidden="true" />}
                Score now
              </button>
            </div>
          )}
        </Panel>
      </div>
    );
  }

  if (isError || !explanation) {
    return (
      <div className="px-4 sm:px-6 py-8 max-w-2xl space-y-6">
        {backLink}
        <Panel>
          <h1 className="font-display font-medium text-32 tracking-display text-graphite">Student not found</h1>
          <p className="mt-2 text-15 text-slate">
            {activeError?.message || 'The requested student record could not be loaded.'}
          </p>
        </Panel>
      </div>
    );
  }

  const riskPct = explanation.risk_probability * 100;
  const riskTier = explanation.risk_tier;
  const drivers = explanation.top_drivers || [];
  const recommendations = recoData?.recommended_interventions || [];
  const counterfactual = recoData?.counterfactual_recourse;
  const alerts = counterfactual?.rule_based_alerts || [];
  const reasonsUnavailable = counterfactual?.drivers_available === false;
  const driverName = (feature) => drivers.find((d) => d.feature_name === feature)?.display_name || feature;

  const openLogModal = (intervention, opener) => {
    openerRef.current = opener;
    // An open log for this intervention is advanced in place; otherwise this is a new assignment.
    const existingActiveLog = historyLogs?.find(
      (log) => log.intervention_id === intervention.intervention_id && log.status !== 'COMPLETED'
    );
    setSelectedIntervention(intervention);
    logMutation.reset();
    if (existingActiveLog) {
      setAssignedFaculty(existingActiveLog.assigned_faculty_id || '');
      setLogStatus(existingActiveLog.status);
      setStatusFloor(existingActiveLog.status);
      setFollowupDate(existingActiveLog.scheduled_followup_date || '');
      setNotes(''); // notes are appended, so start empty
    } else {
      setAssignedFaculty(identity?.assigned_mentor_id || '');
      setLogStatus('ASSIGNED');
      setStatusFloor('ASSIGNED');
      setFollowupDate('');
      setNotes('');
    }
    setPostRiskPct('');
    setIsLogOpen(true);
  };

  const handleLogSubmit = (e) => {
    e.preventDefault();
    if (!selectedIntervention) return;
    logMutation.mutate({
      student_id: studentId,
      intervention_id: selectedIntervention.intervention_id,
      assigned_faculty_id: assignedFaculty || undefined,
      status: logStatus,
      notes: notes || undefined,
      scheduled_followup_date: followupDate || undefined,
      baseline_risk_probability: explanation.risk_probability,
      post_intervention_risk_probability: postRiskPct ? parseFloat(postRiskPct) / 100 : null,
    });
  };

  const tier = tierFor(riskTier);

  return (
    <div className="px-4 sm:px-6 py-8 max-w-[1400px] space-y-6">
      {backLink}

      {/* Who */}
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div className="min-w-0">
          <h1 className="font-display font-medium text-32 tracking-display text-graphite">
            {identity?.name || studentId}
          </h1>
          <p className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-15 text-slate">
            {identity?.name && <span className="tabular-nums">{studentId}</span>}
            {identity?.department && <span>{identity.department}</span>}
            {identity?.assigned_mentor_id && <span>Mentor {identity.assigned_mentor_id}</span>}
          </p>
        </div>
        {recoData?.evaluated_at && (
          <p className="text-13 text-slate" title={recoData.model_version ? `Model ${recoData.model_version}` : undefined}>
            Scored {formatDate(recoData.evaluated_at)}
          </p>
        )}
      </header>

      {/* Rule-based alerts: for every tier, separate from the model estimate */}
      {alerts.length > 0 && (
        <section aria-labelledby="alerts-heading" className="rounded-panel border border-rule bg-ink-wash p-4">
          <h2 id="alerts-heading" className="flex items-center gap-2 text-15 font-semibold text-graphite">
            <CalendarX className="w-4 h-4 shrink-0 text-ink" aria-hidden="true" />
            Rule-based alerts
            <span className="font-normal text-slate">(separate from the model's estimate)</span>
          </h2>
          <ul className="mt-2 space-y-1.5 text-15 text-graphite">
            {alerts.map((alert) => (
              <li key={alert.code}>
                <span className="font-medium">{alert.message.charAt(0).toUpperCase() + alert.message.slice(1)}.</span>
                {alert.recommended_intervention_title && (
                  <span className="text-slate"> Suggested: {alert.recommended_intervention_title}.</span>
                )}
              </li>
            ))}
          </ul>
        </section>
      )}

      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-2">
        <div className="min-w-0 space-y-4">
          {/* Estimated risk */}
          <Panel>
            <h2 className="text-15 text-slate">Estimated risk</h2>
            <div className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-2">
              <span className="text-[3rem] font-semibold leading-none text-graphite" aria-live="polite">
                {formatPercent(riskPct)}
              </span>
              <RiskTierChip tier={riskTier} />
            </div>
            <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-ink-wash" aria-hidden="true">
              <div className="h-full rounded-full" style={{ width: `${riskPct}%`, background: tier?.mark }} />
            </div>
            {recoData?.disclaimer && (
              <p className="mt-4 flex items-start gap-2 text-13 text-slate">
                <Info className="mt-0.5 w-3.5 h-3.5 shrink-0" aria-hidden="true" />
                {recoData.disclaimer}
              </p>
            )}
          </Panel>

          {/* Reasons */}
          <Panel>
            <ShapChart drivers={drivers} />
          </Panel>
        </div>

        <div className="min-w-0 space-y-4">
          {/* Recommended support */}
          <Panel title="Suggested support" description="Matched to this student's reasons from the intervention catalogue.">
            {reasonsUnavailable && (
              <p className="mb-3 rounded-control border border-rule p-3 text-15 text-slate">
                Reasons are unavailable for this estimate, so suggestions are not matched to them.
              </p>
            )}
            {recommendations.length === 0 ? (
              <p className="text-15 text-slate">No specific interventions are suggested for this student.</p>
            ) : (
              <ul className="space-y-3">
                {recommendations.map((rec) => {
                  const openLog = historyLogs?.find((l) => l.intervention_id === rec.intervention_id && l.status !== 'COMPLETED');
                  const isActive = Boolean(openLog);
                  const isCompleted = historyLogs?.some((l) => l.intervention_id === rec.intervention_id && l.status === 'COMPLETED');
                  return (
                    <li key={rec.intervention_id} className="rounded-control border border-rule p-4">
                      <div className="flex flex-wrap items-start justify-between gap-2">
                        <div className="min-w-0">
                          <p className="text-13 text-slate">
                            <span className="capitalize">{rec.pillar}</span>
                            {rec.urgency && <span>, {rec.urgency.toLowerCase()} urgency</span>}
                          </p>
                          <h3 className="mt-0.5 text-15 font-semibold text-graphite">{rec.title}</h3>
                        </div>
                        {isActive && (
                          <span className="inline-flex items-center gap-1 rounded-control border border-rule px-2 py-0.5 text-13 text-ink">
                            <CircleDot className="w-3.5 h-3.5" aria-hidden="true" /> {STATUS_LABELS[openLog.status] || openLog.status}
                          </span>
                        )}
                        {!isActive && isCompleted && (
                          <span className="inline-flex items-center gap-1 rounded-control border border-rule px-2 py-0.5 text-13 text-graphite">
                            <CheckCircle2 className="w-3.5 h-3.5" aria-hidden="true" /> Completed
                          </span>
                        )}
                      </div>
                      <p className="mt-2 text-15 text-slate">{rec.description}</p>
                      <p className="mt-2 text-13 text-slate">
                        Matched reason: <span className="text-graphite">{rec.matched_driver_feature ? driverName(rec.matched_driver_feature) : 'General'}</span>
                        {rec.suggested_duration_days != null && (
                          <>
                            {'. '}Suggested duration: <span className="tabular-nums text-graphite">{rec.suggested_duration_days} days</span>
                          </>
                        )}
                      </p>
                      <div className="mt-3">
                        <button
                          type="button"
                          onClick={(e) => openLogModal(rec, e.currentTarget)}
                          className={`btn ${isActive ? 'btn-primary' : 'btn-secondary'}`}
                        >
                          <ClipboardList className="w-4 h-4" aria-hidden="true" />
                          {isActive ? 'Update status' : 'Assign outreach'}
                        </button>
                      </div>
                    </li>
                  );
                })}
              </ul>
            )}
          </Panel>

          {/* Projected scenario: a simulation, never an outcome */}
          {counterfactual && (
            <Panel title="Projected scenario" description="What the model projects if the suggested changes happened. A simulation, not a promise.">
              <dl className="grid grid-cols-3 gap-3 rounded-control border border-rule p-3 text-center">
                <div>
                  <dt className="text-13 text-slate">Current</dt>
                  <dd className="mt-1 text-20 font-semibold tabular-nums text-graphite">{formatPercent(counterfactual.current_risk_prob * 100)}</dd>
                </div>
                <div>
                  <dt className="text-13 text-slate">Projected</dt>
                  <dd className="mt-1 text-20 font-semibold tabular-nums text-graphite">{formatPercent(counterfactual.projected_risk_prob * 100)}</dd>
                </div>
                <div>
                  <dt className="text-13 text-slate">Relative change</dt>
                  <dd className="mt-1 text-20 font-semibold tabular-nums text-graphite">−{formatPercent(counterfactual.risk_reduction_pct, 0)}</dd>
                </div>
              </dl>
              {counterfactual.required_actions?.length > 0 && (
                <>
                  <h3 className="mt-4 text-13 font-medium text-graphite">Changes in the projection</h3>
                  <ul className="mt-2 space-y-1.5 text-15 text-slate">
                    {counterfactual.required_actions.map((action) => (
                      <li key={action.feature_name || action.plain_language_action} className="flex gap-2">
                        <span aria-hidden="true" className="text-ink">–</span>
                        {action.plain_language_action}
                      </li>
                    ))}
                  </ul>
                </>
              )}
              {counterfactual.disclaimer && <p className="mt-4 text-13 text-slate">{counterfactual.disclaimer}</p>}
            </Panel>
          )}

          {/* History */}
          <Panel title="Outreach history">
            {!historyLogs || historyLogs.length === 0 ? (
              <p className="text-15 text-slate">Nothing logged for this student yet.</p>
            ) : (
              <ol className="relative space-y-5 border-l border-rule pl-5">
                {historyLogs.map((log) => {
                  const completed = log.status === 'COMPLETED';
                  return (
                    <li key={log.id} className="relative">
                      <span className="absolute -left-[1.6rem] top-0.5 rounded-full bg-paper p-0.5 text-ink" aria-hidden="true">
                        {completed ? <CheckCircle2 className="w-3.5 h-3.5" /> : <CircleDot className="w-3.5 h-3.5" />}
                      </span>
                      <div className="flex flex-wrap items-baseline justify-between gap-2">
                        <h3 className="text-15 font-semibold text-graphite">{log.title || log.intervention_id}</h3>
                        <span className="text-13 tabular-nums text-slate">{formatDate(log.created_at)}</span>
                      </div>
                      <p className="mt-0.5 text-13 text-slate">
                        {STATUS_LABELS[log.status] || log.status}
                        {log.assigned_faculty_id && `, assigned to ${log.assigned_faculty_id}`}
                        {log.outcome_status && log.outcome_status !== 'PENDING_EVALUATION' && `, outcome: ${(OUTCOME_LABELS[log.outcome_status] || log.outcome_status).toLowerCase()}`}
                      </p>
                      {log.notes && <p className="mt-1.5 rounded-control bg-ink-wash p-2.5 text-15 text-graphite whitespace-pre-line">{log.notes}</p>}
                      {log.scheduled_followup_date && (
                        <p className="mt-1.5 flex items-center gap-1.5 text-13 text-slate">
                          <Calendar className="w-3.5 h-3.5" aria-hidden="true" />
                          Follow-up {formatDate(log.scheduled_followup_date)}
                        </p>
                      )}
                    </li>
                  );
                })}
              </ol>
            )}
          </Panel>
        </div>
      </div>

      {/* Log drawer */}
      {isLogOpen && selectedIntervention && (
        <div className="fixed inset-0 z-50 flex justify-end bg-black/40" onClick={closeDrawer}>
          <div
            ref={drawerRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby={drawerTitleId}
            onClick={(e) => e.stopPropagation()}
            className="flex h-full w-full max-w-lg flex-col border-l border-rule bg-paper"
          >
            <div className="flex items-start justify-between gap-3 border-b border-rule p-5">
              <div>
                <h2 id={drawerTitleId} className="text-20 font-semibold text-graphite">Log outreach</h2>
                <p className="mt-0.5 text-15 text-slate">{selectedIntervention.title}</p>
              </div>
              <button type="button" onClick={closeDrawer} className="rounded-control p-1.5 text-slate hover:bg-ink-wash" aria-label="Close">
                <X className="w-5 h-5" aria-hidden="true" />
              </button>
            </div>

            <form onSubmit={handleLogSubmit} id="log-form" className="flex-1 space-y-4 overflow-y-auto p-5">
              <div className="space-y-1.5">
                <label htmlFor="assigned_faculty" className={labelClass}>Assigned faculty</label>
                <input
                  ref={firstFieldRef}
                  type="text"
                  id="assigned_faculty"
                  value={assignedFaculty}
                  onChange={(e) => setAssignedFaculty(e.target.value)}
                  placeholder="e.g. FAC_007"
                  className={fieldClass}
                  required
                />
              </div>

              <div className="space-y-1.5">
                <label htmlFor="status" className={labelClass}>Status</label>
                <select id="status" value={logStatus} onChange={(e) => setLogStatus(e.target.value)} className={fieldClass}>
                  {STATUS_ORDER.map((status) => {
                    const passed = STATUS_ORDER.indexOf(status) < STATUS_ORDER.indexOf(statusFloor);
                    return (
                      <option key={status} value={status} disabled={passed}>
                        {STATUS_LABELS[status]}{passed ? ' (already passed)' : ''}
                      </option>
                    );
                  })}
                </select>
                <p className="text-13 text-slate">Status only moves forward: assigned, in progress, applied, completed.</p>
              </div>

              <div className="space-y-1.5">
                <label htmlFor="notes" className={labelClass}>Notes</label>
                <textarea
                  id="notes"
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="What happened in the call or meeting, what was agreed"
                  rows={4}
                  className={`${fieldClass} resize-y`}
                />
              </div>

              <div className="space-y-1.5">
                <label htmlFor="followup" className={labelClass}>Follow-up date (optional)</label>
                <input type="date" id="followup" value={followupDate} onChange={(e) => setFollowupDate(e.target.value)} className={fieldClass} />
              </div>

              <div className="space-y-1.5 border-t border-rule pt-4">
                <label htmlFor="post_risk" className={labelClass}>Estimated risk after re-scoring, % (optional)</label>
                <input
                  type="number"
                  id="post_risk"
                  min="0"
                  max="100"
                  step="0.01"
                  value={postRiskPct}
                  onChange={(e) => setPostRiskPct(e.target.value)}
                  className={`${fieldClass} tabular-nums`}
                />
                <p className="text-13 text-slate">
                  Compared with the current estimate ({formatPercent(riskPct)}) to record whether the outcome improved.
                </p>
              </div>
            </form>

            <div className="space-y-3 border-t border-rule p-5">
              <AdminTokenNotice />
              {logMutation.isError && (
                <p role="alert" className="flex items-start gap-2 text-15 text-graphite">
                  <AlertCircle className="mt-0.5 w-4 h-4 shrink-0 text-slate" aria-hidden="true" />
                  {logMutation.error?.message || 'The log could not be saved.'}
                </p>
              )}
              <div className="flex gap-3">
                <button type="button" onClick={closeDrawer} className="btn btn-secondary flex-1">
                  Cancel
                </button>
                <button type="submit" form="log-form" disabled={logMutation.isPending} className="btn btn-primary flex-1 disabled:opacity-60">
                  {logMutation.isPending && <RefreshCw className="w-4 h-4 motion-safe:animate-spin" aria-hidden="true" />}
                  Save
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default StudentDetail;
