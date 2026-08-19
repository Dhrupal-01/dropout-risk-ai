import React, { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  ArrowLeft, Calendar, ShieldAlert, TrendingDown, 
  Play, CheckCircle2, User, AlertCircle, 
  HelpCircle, CheckSquare, ListPlus, RefreshCw
} from 'lucide-react';
import { 
  getStudentExplanation, 
  getStudentInterventions, 
  getStudentInterventionsHistory, 
  logIntervention,
  predictStudent
} from '../api/endpoints';
import RiskTierChip from '../components/RiskTierChip';
import ShapChart from '../components/ShapChart';

const StudentDetail = () => {
  const { studentId } = useParams();
  const queryClient = useQueryClient();

  // Modal log state
  const [isLogOpen, setIsLogOpen] = useState(false);
  const [selectedIntervention, setSelectedIntervention] = useState(null);
  
  // Form input state
  const [assignedFaculty, setAssignedFaculty] = useState('');
  const [notes, setNotes] = useState('');
  const [followupDate, setFollowupDate] = useState('');
  const [postRiskPct, setPostRiskPct] = useState('');
  const [logStatus, setLogStatus] = useState('ASSIGNED');

  // 1. Fetch explanation data in parallel
  const { 
    data: explanation, 
    isLoading: isExplLoading, 
    isError: isExplError,
    error: explError
  } = useQuery({
    queryKey: ['explanation', studentId],
    queryFn: () => getStudentExplanation(studentId, 5),
    retry: 1,
  });

  // 2. Fetch interventions and projected counterfactual recourse data
  const { 
    data: recoData, 
    isLoading: isRecoLoading, 
    isError: isRecoError,
    error: recoError
  } = useQuery({
    queryKey: ['interventions', studentId],
    queryFn: () => getStudentInterventions(studentId),
    retry: 1,
  });

  // 3. Fetch logged intervention history for student
  const { 
    data: historyLogs
  } = useQuery({
    queryKey: ['history', studentId],
    queryFn: () => getStudentInterventionsHistory(studentId),
  });



  // Log mutation to update/assign intervention logs
  const logMutation = useMutation({
    mutationFn: logIntervention,
    onSuccess: () => {
      // Invalidate queries to trigger re-fetches
      queryClient.invalidateQueries(['interventions', studentId]);
      queryClient.invalidateQueries(['explanation', studentId]);
      queryClient.invalidateQueries(['history', studentId]);
      queryClient.invalidateQueries(['queue']);
      
      // Close modal and reset fields
      setIsLogOpen(false);
      setSelectedIntervention(null);
      setAssignedFaculty('');
      setNotes('');
      setFollowupDate('');
      setPostRiskPct('');
    },
    onError: (err) => {
      alert(err.message || 'Failed to update intervention log.');
    }
  });

  // Re-score mutation to trigger model predict endpoint
  const reScoreMutation = useMutation({
    mutationFn: (featuresPayload) => predictStudent({
      student_id: studentId,
      name: explanation?.name,
      department: explanation?.department,
      features: featuresPayload
    }),
    onSuccess: () => {
      queryClient.invalidateQueries(['interventions', studentId]);
      queryClient.invalidateQueries(['explanation', studentId]);
      queryClient.invalidateQueries(['history', studentId]);
      queryClient.invalidateQueries(['queue']);
    },
    onError: (err) => {
      alert(err.message || 'Manual scoring run failed.');
    }
  });

  const isLoading = isExplLoading || isRecoLoading;
  const isError = isExplError || isRecoError;
  const activeError = explError || recoError;

  if (isLoading) {
    return (
      <div className="container mx-auto px-6 py-8 max-w-7xl space-y-6 animate-pulse select-none">
        <div className="h-6 w-24 bg-subtle rounded" />
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="space-y-6">
            <div className="h-40 bg-subtle rounded-lg" />
            <div className="h-64 bg-subtle rounded-lg" />
          </div>
          <div className="space-y-6">
            <div className="h-80 bg-subtle rounded-lg" />
            <div className="h-48 bg-subtle rounded-lg" />
          </div>
        </div>
      </div>
    );
  }

  // Map 404 "no_prediction_history" error to counselor view
  if (isError && activeError?.code === 'no_prediction_history') {
    return (
      <div className="container mx-auto px-6 py-12 max-w-lg text-center space-y-6">
        <AlertCircle className="w-12 h-12 text-risk-medium mx-auto animate-pulse" />
        <h2 className="text-xl font-bold text-primary">Student Not Scored Yet</h2>
        <p className="text-xs text-secondary leading-relaxed">
          This student exists in the database but has no predictive scoring records. 
          Run a manual scoring evaluation now to generate the risk calibration and TreeSHAP attribution profiles.
        </p>
        <button
          onClick={() => {
            // Trigger dummy initial predict payload matching features schema
            const dummyFeatures = {
              age: 20.0, commute_distance_km: 10.0, income_slab_idx: 1,
              is_first_generation: 0, has_scholarship: 0, fee_payment_delay_days: 0,
              hostel_status: "Day Scholar",
              att_core1: 75.0, att_core2: 75.0, att_lab: 80.0, att_elective: 80.0,
              attendance_month_1: 78.0, attendance_month_2: 76.0, attendance_month_3: 75.0,
              attendance_percentage: 76.5, consecutive_absences: 2,
              prev_sem_cgpa: 7.0, current_cgpa: 6.8, backlog_count: 0,
              internal_exam_score_pct: 65.0, stem_core_fail_flag: 0,
              lms_logins_per_week: 5.0, assignment_submission_lag_days: 0.5,
              resource_access_count: 150, days_since_last_lms_activity: 3,
              forum_participation_count: 2
            };
            reScoreMutation.mutate(dummyFeatures);
          }}
          disabled={reScoreMutation.isLoading}
          className="px-4 py-2 bg-accent text-white text-xs font-semibold rounded hover:bg-accent-hover transition-colors inline-flex items-center space-x-2"
        >
          {reScoreMutation.isLoading && <RefreshCw className="w-3 h-3 animate-spin mr-1.5" />}
          Score Now
        </button>
      </div>
    );
  }

  if (isError || !explanation) {
    return (
      <div className="container mx-auto px-6 py-12 max-w-lg text-center space-y-4">
        <ShieldAlert className="w-12 h-12 text-risk-high mx-auto" />
        <h2 className="text-xl font-bold text-primary">Student Not Found</h2>
        <p className="text-xs text-secondary">
          {activeError?.message || 'The requested student record could not be loaded.'}
        </p>
        <Link to="/" className="inline-flex items-center text-xs font-semibold text-accent hover:underline">
          <ArrowLeft className="w-3.5 h-3.5 mr-1" />
          Back to Triage Queue
        </Link>
      </div>
    );
  }

  // Extract variables from data
  const riskPct = explanation.risk_probability * 100;
  const riskTier = explanation.risk_tier;
  const drivers = explanation.top_drivers || [];
  const recommendations = recoData?.recommended_interventions || [];
  const counterfactual = recoData?.counterfactual_recourse;

  const openLogModal = (intervention) => {
    // Check if there is an active log for this intervention already in history
    const existingActiveLog = historyLogs?.find(
      (log) => log.intervention_id === intervention.intervention_id && log.status !== 'COMPLETED'
    );

    setSelectedIntervention(intervention);
    if (existingActiveLog) {
      // Prepopulate fields to advance status in place
      setAssignedFaculty(existingActiveLog.assigned_faculty_id || '');
      setLogStatus(existingActiveLog.status);
      setFollowupDate(existingActiveLog.scheduled_followup_date || '');
      setNotes(''); // Clear notes so user only types fresh appends
    } else {
      // Set default values for fresh assignment
      setAssignedFaculty(recoData?.assigned_mentor_id || '');
      setLogStatus('ASSIGNED');
      setFollowupDate('');
      setNotes('');
    }
    
    setIsLogOpen(true);
  };

  const handleLogSubmit = (e) => {
    e.preventDefault();
    if (!selectedIntervention) return;

    const payload = {
      student_id: studentId,
      intervention_id: selectedIntervention.intervention_id,
      assigned_faculty_id: assignedFaculty || undefined,
      status: logStatus,
      notes: notes || undefined,
      scheduled_followup_date: followupDate || undefined,
      baseline_risk_probability: explanation.risk_probability,
      post_intervention_risk_probability: postRiskPct ? parseFloat(postRiskPct) / 100 : null,
    };

    logMutation.mutate(payload);
  };

  // Helper function to render status transition lists
  const renderStatusOption = (statusValue, currentStatus) => {
    const statusOrder = ['ASSIGNED', 'IN_PROGRESS', 'APPLIED', 'COMPLETED'];
    const currentIndex = statusOrder.indexOf(currentStatus);
    const targetIndex = statusOrder.indexOf(statusValue);
    
    // Disable backward transitions
    const isDisabled = targetIndex < currentIndex;

    return (
      <option key={statusValue} value={statusValue} disabled={isDisabled}>
        {statusValue.replace('_', ' ')} {isDisabled ? '(Passed)' : ''}
      </option>
    );
  };

  return (
    <div className="container mx-auto px-6 py-8 max-w-7xl space-y-6">
      {/* Back button */}
      <div>
        <Link 
          to="/" 
          className="inline-flex items-center text-xs font-semibold text-secondary hover:text-primary transition-colors focus:ring-2 focus:ring-accent"
        >
          <ArrowLeft className="w-4 h-4 mr-1.5" />
          Back to Triage Worklist
        </Link>
      </div>

      {/* Main Student Header Info */}
      <div className="bg-card border border-border p-6 rounded-lg shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-6">
        <div className="flex items-start space-x-3.5">
          <div className="p-3 bg-subtle rounded-full text-secondary">
            <User className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-primary">
              {explanation.name || 'Anonymous Student'}
            </h1>
            <p className="text-xs text-secondary mt-0.5 flex flex-wrap items-center gap-2">
              <span className="font-mono bg-subtle border border-border px-1.5 py-0.5 rounded">{studentId}</span>
              <span>•</span>
              <span>{explanation.department || 'General Science'}</span>
              {recoData?.assigned_mentor_id && (
                <>
                  <span>•</span>
                  <span>Mentor: <strong className="font-semibold">{recoData.assigned_mentor_id}</strong></span>
                </>
              )}
            </p>
          </div>
        </div>
      </div>

      {/* 2 Column Layout Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
        {/* LEFT COLUMN: Risk Hero & SHAP Explainability */}
        <div className="space-y-6">
          {/* Risk Hero Card */}
          <div className="bg-card border border-border p-6 rounded-lg shadow-sm space-y-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-secondary">
              Current Estimated Risk Status
            </h3>
            
            <div className="flex items-baseline space-x-4">
              {/* Tabular numbers for clean digit rendering */}
              <span 
                className="text-5xl font-black tracking-tight text-primary font-mono select-all"
                aria-live="polite"
              >
                {riskPct.toFixed(1)}%
              </span>
              <RiskTierChip tier={riskTier} className="scale-110" />
            </div>

            {/* Flat Probability Bar (replacing gauge speedometers) */}
            <div className="space-y-1">
              <div className="w-full h-2.5 bg-subtle rounded-full overflow-hidden border border-border/25">
                <div 
                  className={`h-full transition-all duration-700 ease-out ${
                    riskTier === 'High' ? 'bg-risk-high' :
                    riskTier === 'Medium' ? 'bg-risk-medium' : 'bg-risk-low'
                  }`}
                  style={{ width: `${riskPct}%` }}
                />
              </div>
              <div className="flex items-center justify-between text-[10px] text-muted font-mono select-none">
                <span>0% Low</span>
                <span>33% Med</span>
                <span>66% High</span>
                <span>100%</span>
              </div>
            </div>

            {/* Responsible AI Disclaimer rendered visibly */}
            {recoData?.disclaimer && (
              <div className="text-[10px] text-muted bg-subtle/50 p-3 rounded leading-relaxed border border-border/10 flex items-start">
                <HelpCircle className="w-3.5 h-3.5 mr-2 shrink-0 text-muted mt-0.5" />
                <p className="italic">{recoData.disclaimer}</p>
              </div>
            )}
          </div>

          {/* SHAP Chart panel */}
          <div className="bg-card border border-border p-6 rounded-lg shadow-sm">
            <ShapChart drivers={drivers} />
          </div>
        </div>

        {/* RIGHT COLUMN: Recommended Support & Scenario Simulations */}
        <div className="space-y-6">
          {/* Recommended support package */}
          <div className="bg-card border border-border p-6 rounded-lg shadow-sm space-y-4">
            <div className="border-b border-border pb-2.5">
              <h3 className="text-sm font-semibold uppercase tracking-wider text-secondary">
                Recommended Support Interventions
              </h3>
            </div>

            {recommendations.length === 0 ? (
              <p className="text-xs text-muted italic">No specific interventions recommended by the system.</p>
            ) : (
              <div className="space-y-4">
                {recommendations.map((rec) => {
                  const isActive = historyLogs?.some(
                    (log) => log.intervention_id === rec.intervention_id && log.status !== 'COMPLETED'
                  );
                  const isCompleted = historyLogs?.some(
                    (log) => log.intervention_id === rec.intervention_id && log.status === 'COMPLETED'
                  );

                  return (
                    <div 
                      key={rec.intervention_id}
                      className="border border-border rounded-lg p-4 bg-subtle/20 space-y-3 flex flex-col justify-between hover:border-accent/40 transition-colors"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="space-y-1">
                          <div className="flex items-center gap-2 flex-wrap select-none">
                            {/* Urgency tag */}
                            <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${
                              rec.urgency === 'HIGH' 
                                ? 'bg-red-50 dark:bg-red-950/20 text-risk-high border-red-200 dark:border-red-900/40' 
                                : 'bg-blue-50 dark:bg-blue-950/20 text-accent border-blue-200 dark:border-blue-900/40'
                            }`}>
                              {rec.urgency}
                            </span>
                            {/* Pillar */}
                            <span className="text-[10px] font-semibold uppercase text-secondary tracking-wider font-mono">
                              {rec.pillar}
                            </span>
                          </div>
                          <h4 className="text-xs font-bold text-primary pt-0.5">{rec.title}</h4>
                        </div>
                        
                        {/* Status indicators */}
                        {isActive && (
                          <span className="text-[10px] font-bold text-accent px-1.5 py-0.5 rounded border border-accent/20 bg-accent-soft">
                            Active
                          </span>
                        )}
                        {isCompleted && (
                          <span className="text-[10px] font-bold text-risk-low px-1.5 py-0.5 rounded border border-risk-low/20 bg-green-50 dark:bg-green-950/20">
                            Completed
                          </span>
                        )}
                      </div>

                      <p className="text-xs text-secondary leading-relaxed">{rec.description}</p>

                      <div className="flex flex-wrap items-center justify-between gap-3 text-[10px] text-muted border-t border-border/50 pt-2 font-medium">
                        <span>Matched Driver: <strong className="text-secondary font-semibold font-mono">{rec.matched_driver_feature || 'General'}</strong></span>
                        <span>Duration: <strong className="text-secondary font-semibold">{rec.suggested_duration_days} days</strong></span>
                      </div>

                      <div className="pt-1 text-right">
                        <button
                          onClick={() => openLogModal(rec)}
                          className={`inline-flex items-center text-xs font-semibold px-3 py-1.5 rounded-md transition-colors border ${
                            isActive 
                              ? 'bg-accent/10 border-accent/30 text-accent hover:bg-accent/20' 
                              : 'bg-card border-border text-primary hover:bg-hover'
                          } focus:ring-2 focus:ring-accent`}
                        >
                          <ListPlus className="w-3.5 h-3.5 mr-1" />
                          {isActive ? 'Update Status' : 'Assign outreach'}
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Scenario prediction Recourse panel */}
          {counterfactual && (
            <div className="bg-card border border-border p-6 rounded-lg shadow-sm space-y-4">
              <div className="border-b border-border pb-2.5">
                <h3 className="text-sm font-semibold uppercase tracking-wider text-secondary">
                  Projected Scenario
                </h3>
              </div>

              {/* Recourse delta calculations */}
              <div className="flex items-center justify-between p-4 rounded-lg bg-accent-soft/40 border border-accent/10">
                <div className="text-center flex-1">
                  <span className="text-[10px] text-secondary uppercase font-semibold">Current Risk</span>
                  <div className="text-lg font-bold text-primary font-mono mt-1">
                    {(counterfactual.current_risk_prob * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="text-muted shrink-0 px-2 font-mono font-black select-none">→</div>
                <div className="text-center flex-1">
                  <span className="text-[10px] text-secondary uppercase font-semibold">Projected Risk</span>
                  <div className="text-lg font-bold text-risk-low font-mono mt-1">
                    {(counterfactual.projected_risk_prob * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="text-center flex-1 border-l border-border/80 pl-3">
                  <span className="text-[10px] text-secondary uppercase font-semibold">Risk Reduction</span>
                  <div className="text-lg font-bold text-accent font-mono mt-1 flex items-center justify-center gap-0.5">
                    <TrendingDown className="w-4 h-4 text-accent" />
                    -{counterfactual.risk_reduction_pct.toFixed(0)}%
                  </div>
                </div>
              </div>

              {/* Target Actions list */}
              <div className="space-y-3 pt-1">
                <span className="text-xs font-semibold text-secondary">Target Recourse Requirements:</span>
                <ul className="space-y-2 select-all">
                  {counterfactual.required_actions?.map((action, i) => (
                    <li 
                      key={i} 
                      className="text-xs text-secondary pl-6 relative leading-relaxed"
                    >
                      <CheckSquare className="w-3.5 h-3.5 text-accent absolute left-0.5 top-0.5 shrink-0" />
                      {action.plain_language_action}
                    </li>
                  ))}
                </ul>
              </div>

              {/* Simulation Disclaimer rendered visibly */}
              {counterfactual.disclaimer && (
                <div className="text-[10px] text-muted bg-subtle/50 p-3 rounded leading-relaxed border border-border/10">
                  <strong className="text-secondary select-none font-bold">Simulation Disclaimer:</strong> {counterfactual.disclaimer}
                </div>
              )}
            </div>
          )}

          {/* Intervention Tracking Logs Timeline */}
          <div className="bg-card border border-border p-6 rounded-lg shadow-sm space-y-4">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-secondary border-b border-border pb-2.5">
              Outreach Logs &amp; Progress History
            </h3>

            {(!historyLogs || historyLogs.length === 0) ? (
              <p className="text-xs text-muted italic">No intervention history logged yet for this student.</p>
            ) : (
              <div className="relative pl-6 border-l border-border/80 space-y-6 pt-2">
                {historyLogs.map((log) => {
                  const isCompleted = log.status === 'COMPLETED';
                  const dateString = new Date(log.created_at).toLocaleDateString(undefined, {
                    month: 'short',
                    day: 'numeric',
                    year: 'numeric'
                  });

                  return (
                    <div key={log.id} className="relative select-text">
                      {/* Timeline dot */}
                      <span className={`absolute -left-[31px] top-1 p-1 rounded-full border bg-card ${
                        isCompleted 
                          ? 'border-risk-low text-risk-low' 
                          : 'border-accent text-accent'
                      }`}>
                        {isCompleted ? <CheckCircle2 className="w-3 h-3" /> : <Play className="w-3 h-3" />}
                      </span>

                      <div className="space-y-1">
                        <div className="flex items-center justify-between gap-3 flex-wrap">
                          <h4 className="text-xs font-bold text-primary">{log.title || log.intervention_id}</h4>
                          <span className="text-[10px] text-muted font-mono">{dateString}</span>
                        </div>

                        <div className="flex items-center space-x-3 text-[10px] font-semibold text-secondary select-none">
                          <span className="text-accent uppercase font-mono">{log.status}</span>
                          <span>•</span>
                          <span className="font-mono">Faculty: {log.assigned_faculty_id || 'unassigned'}</span>
                          {log.outcome_status !== 'PENDING_EVALUATION' && (
                            <>
                              <span>•</span>
                              <span className={
                                log.outcome_status === 'IMPROVED' ? 'text-risk-low' : 'text-risk-medium'
                              }>
                                {log.outcome_status}
                              </span>
                            </>
                          )}
                        </div>

                        {log.notes && (
                          <p className="text-xs text-secondary bg-subtle/30 p-2.5 rounded border border-border/15 font-serif italic mt-1 leading-relaxed">
                            "{log.notes}"
                          </p>
                        )}

                        {log.scheduled_followup_date && (
                          <div className="flex items-center space-x-1.5 text-[10px] text-muted select-none">
                            <Calendar className="w-3 h-3" />
                            <span>Follow-up scheduled: {log.scheduled_followup_date}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* INTERVENTION LOGGER MODAL DRAWER */}
      {isLogOpen && selectedIntervention && (
        <div className="fixed inset-0 z-50 flex items-center justify-end bg-black/40 backdrop-blur-xs select-none">
          <div 
            className="w-full max-w-lg h-full bg-card border-l border-border shadow-2xl p-6 flex flex-col justify-between select-text"
            onKeyDown={(e) => {
              if (e.key === 'Escape') setIsLogOpen(false);
            }}
          >
            <div className="space-y-6 overflow-y-auto pr-1">
              {/* Modal header */}
              <div className="flex items-center justify-between border-b border-border pb-4">
                <div>
                  <h2 className="text-md font-bold text-primary">Log Support Outreach</h2>
                  <p className="text-xs text-secondary mt-0.5">Assign or update progress trackers</p>
                </div>
                <button
                  onClick={() => setIsLogOpen(false)}
                  className="text-secondary hover:text-primary p-1 rounded hover:bg-hover transition-colors"
                  aria-label="Close modal"
                >
                  <ArrowLeft className="w-4 h-4 rotate-180" />
                </button>
              </div>

              {/* Modal Form */}
              <form onSubmit={handleLogSubmit} id="log-form" className="space-y-4">
                {/* Intervention Title display */}
                <div className="space-y-1">
                  <span className="text-[10px] uppercase font-bold text-muted tracking-wider">Support Intervention</span>
                  <div className="text-xs font-bold text-primary p-3 bg-subtle/40 border border-border rounded-md">
                    {selectedIntervention.title}
                  </div>
                </div>

                {/* Assigned Faculty Input */}
                <div className="space-y-1.5">
                  <label htmlFor="assigned_faculty" className="text-[10px] uppercase font-bold text-secondary tracking-wider">
                    Assigned Faculty
                  </label>
                  <input
                    type="text"
                    id="assigned_faculty"
                    value={assignedFaculty}
                    onChange={(e) => setAssignedFaculty(e.target.value)}
                    placeholder="e.g. FAC_007"
                    className="text-xs border border-border bg-card text-primary rounded-md p-2.5 w-full hover:border-accent focus:ring-2 focus:ring-accent transition-colors"
                    required
                  />
                </div>

                {/* Status Dropdown */}
                <div className="space-y-1.5">
                  <label htmlFor="status" className="text-[10px] uppercase font-bold text-secondary tracking-wider">
                    Outreach Status
                  </label>
                  <select
                    id="status"
                    value={logStatus}
                    onChange={(e) => setLogStatus(e.target.value)}
                    className="text-xs border border-border bg-card text-primary rounded-md p-2.5 w-full hover:border-accent focus:ring-2 focus:ring-accent transition-colors"
                  >
                    {/* Render status options, enforcing forward-only rules */}
                    {['ASSIGNED', 'IN_PROGRESS', 'APPLIED', 'COMPLETED'].map((opt) => 
                      renderStatusOption(opt, logStatus)
                    )}
                  </select>
                  <span className="text-[10px] text-muted leading-relaxed block">
                    * Status flows forward only: ASSIGNED → IN_PROGRESS → APPLIED → COMPLETED. Backward moves are disabled.
                  </span>
                </div>

                {/* Notes Input */}
                <div className="space-y-1.5">
                  <label htmlFor="notes" className="text-[10px] uppercase font-bold text-secondary tracking-wider">
                    Outreach Notes / Updates
                  </label>
                  <textarea
                    id="notes"
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    placeholder="Provide details about mentor call meetings, resolved delays, or recommendations..."
                    rows={4}
                    className="text-xs border border-border bg-card text-primary rounded-md p-2.5 w-full hover:border-accent focus:ring-2 focus:ring-accent transition-colors font-serif resize-none"
                  />
                </div>

                {/* Follow-up Date Input */}
                <div className="space-y-1.5">
                  <label htmlFor="followup" className="text-[10px] uppercase font-bold text-secondary tracking-wider">
                    Scheduled Follow-up Date (Optional)
                  </label>
                  <input
                    type="date"
                    id="followup"
                    value={followupDate}
                    onChange={(e) => setFollowupDate(e.target.value)}
                    className="text-xs border border-border bg-card text-primary rounded-md p-2.5 w-full hover:border-accent focus:ring-2 focus:ring-accent transition-colors"
                  />
                </div>

                {/* Optional Post-intervention Risk Score (recomputes outcome status) */}
                <div className="space-y-1.5 border-t border-border/80 pt-4">
                  <label htmlFor="post_risk" className="text-[10px] uppercase font-bold text-secondary tracking-wider block">
                    Post-Intervention Risk Probability % (Optional)
                  </label>
                  <input
                    type="number"
                    id="post_risk"
                    min="0"
                    max="100"
                    step="0.01"
                    value={postRiskPct}
                    onChange={(e) => setPostRiskPct(e.target.value)}
                    placeholder="e.g. 31.00"
                    className="text-xs border border-border bg-card text-primary rounded-md p-2.5 w-full hover:border-accent focus:ring-2 focus:ring-accent transition-colors font-mono"
                  />
                  <span className="text-[10px] text-muted leading-relaxed block">
                    * If provided, this value is compared against baseline risk ({riskPct.toFixed(1)}%) to compute outcome improvement indicators.
                  </span>
                </div>
              </form>
            </div>

            {/* Modal footer actions */}
            <div className="flex items-center space-x-3 border-t border-border pt-4 select-none">
              <button
                type="button"
                onClick={() => setIsLogOpen(false)}
                className="flex-1 px-4 py-2 border border-border rounded-md hover:bg-hover text-xs font-semibold text-secondary hover:text-primary transition-colors focus:ring-2 focus:ring-accent"
              >
                Cancel
              </button>
              <button
                type="submit"
                form="log-form"
                disabled={logMutation.isLoading}
                className="flex-1 px-4 py-2 bg-accent text-white text-xs font-semibold rounded-md hover:bg-accent-hover focus:ring-2 focus:ring-accent transition-colors inline-flex items-center justify-center"
              >
                {logMutation.isLoading && <RefreshCw className="w-3.5 h-3.5 mr-1.5 animate-spin" />}
                Log Status Change
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default StudentDetail;
