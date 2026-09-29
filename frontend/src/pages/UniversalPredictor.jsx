import React, { useState } from 'react';
import { Layers, UserCheck, ArrowRight, RefreshCw } from 'lucide-react';

const TIERS = [
  { id: 'PRE_10TH', label: 'School (Pre-10th)', sub: 'K-10 Primary & Secondary' },
  { id: 'HIGHER_SECONDARY', label: 'Higher Secondary (11th-12th)', sub: 'Board Exam & ITI/Diploma' },
  { id: 'UNDERGRADUATE', label: 'Undergraduate (UG)', sub: 'B.Tech, B.Sc, B.Com, B.A.' },
  { id: 'POSTGRADUATE', label: 'Postgraduate (PG)', sub: 'M.Tech, M.Sc, MBA, M.A.' },
  { id: 'DOCTORATE', label: 'Doctorate (PhD)', sub: 'Doctoral Research & Thesis' },
];

const PRESETS = {
  PRE_10TH: {
    student_id: 'SCH_PURNEA_001',
    educational_tier: 'PRE_10TH',
    state: 'Bihar',
    district: 'Purnia',
    features: {
      distance_to_school_km: 6.5,
      attendance_percentage: 52.0,
      attendance_risk_flag: 1,
      mid_day_meal_attendance_pct: 45.0,
      parental_literacy_years: 2,
      seasonal_migration_flag: 1,
      fln_score_pct: 32.0,
      sibling_count_under_5: 2,
      toilets_available_flag: 0,
      state_literacy_rate: 61.8,
      district_mpi_pct: 50.4,
      is_aspirational_district: 1,
      district_rurality_pct: 89.2,
      district_ptr_ratio: 42.1,
    },
  },
  HIGHER_SECONDARY: {
    student_id: 'HSC_BAHRAICH_002',
    educational_tier: 'HIGHER_SECONDARY',
    state: 'Uttar Pradesh',
    district: 'Bahraich',
    features: {
      class_10_board_pct: 54.0,
      attendance_percentage: 46.0,
      attendance_risk_flag: 1,
      stream_preference_match: 0,
      coaching_financial_strain: 0.85,
      mid_term_board_sim_score: 35.0,
      vocational_practical_att_pct: 40.0,
      family_debt_pressure_flag: 1,
      consecutive_absences: 9,
      state_literacy_rate: 67.7,
      district_mpi_pct: 54.4,
      is_aspirational_district: 1,
      district_rurality_pct: 92.1,
    },
  },
  UNDERGRADUATE: {
    student_id: 'UG_PUNE_003',
    educational_tier: 'UNDERGRADUATE',
    state: 'Maharashtra',
    district: 'Pune',
    features: {
      attendance_percentage: 44.0,
      attendance_month_1: 60.0,
      attendance_month_2: 45.0,
      attendance_month_3: 28.0,
      attendance_3m_trend: -16.0,
      consecutive_absences: 6,
      attendance_risk_flag: 1,
      subject_attendance_std: 8.0,
      att_core1: 35.0,
      att_core2: 40.0,
      att_lab: 50.0,
      att_elective: 45.0,
      current_cgpa: 4.8,
      prev_sem_cgpa: 6.4,
      cgpa_delta: -1.6,
      backlog_count: 3,
      internal_exam_score_pct: 45.0,
      stem_core_fail_flag: 1,
      academic_crisis_flag: 1,
      lms_logins_per_week: 1.5,
      assignment_submission_lag_days: 6.0,
      resource_access_count: 10,
      days_since_last_lms_activity: 24,
      forum_participation_count: 0,
      behavioral_disengagement_index: 0.85,
      income_slab_idx: 1,
      is_first_generation: 1,
      fee_payment_delay_days: 50,
      has_scholarship: 0,
      is_hosteler: 0,
      commute_distance_km: 15.0,
      financial_stress_index: 0.70,
      interaction_att_x_fee: 93.3,
      interaction_cgpa_x_backlog: 15.6,
      interaction_firstgen_x_inactivity: 1.0,
      interaction_att_x_cgpa_drop: 1.6,
      age: 20,
      state_literacy_rate: 82.3,
      district_mpi_pct: 3.9,
      is_aspirational_district: 0,
    },
  },
  POSTGRADUATE: {
    student_id: 'PG_NUH_004',
    educational_tier: 'POSTGRADUATE',
    state: 'Haryana',
    district: 'Nuh / Mewat',
    features: {
      ug_graduation_cgpa: 6.5,
      current_pg_cgpa: 4.9,
      attendance_percentage: 55.0,
      dissertation_milestone_pct: 20.0,
      education_loan_burden_inr: 750000,
      placement_readiness_score: 30.0,
      part_time_work_hours_per_week: 25,
      backlog_count: 2,
      fee_payment_delay_days: 45,
      state_literacy_rate: 75.6,
      district_mpi_pct: 44.8,
      is_aspirational_district: 1,
    },
  },
  DOCTORATE: {
    student_id: 'PHD_RAMANATH_005',
    educational_tier: 'DOCTORATE',
    state: 'Tamil Nadu',
    district: 'Ramanathapuram',
    features: {
      phd_tenure_months: 44,
      fellowship_disbursement_delay_days: 120,
      advisor_meeting_frequency_per_month: 0,
      peer_reviewed_submissions: 0,
      comprehensive_exam_attempts: 2,
      research_stagnation_index: 0.90,
      stipend_adequacy_ratio: 0.30,
      lab_isolation_index: 0.95,
      state_literacy_rate: 80.1,
      district_mpi_pct: 7.8,
      is_aspirational_district: 1,
    },
  },
};

const UniversalPredictor = () => {
  const [selectedTier, setSelectedTier] = useState('PRE_10TH');
  const [payload, setPayload] = useState(PRESETS.PRE_10TH);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleTierChange = (tierId) => {
    setSelectedTier(tierId);
    setPayload(PRESETS[tierId]);
    setResult(null);
  };

  const handlePredict = async () => {
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch('http://127.0.0.1:8000/api/v1/universal/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error('Inference error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-6 py-8 space-y-8">
      {/* Header */}
      <div>
        <div className="flex items-center space-x-2 text-xs font-semibold uppercase tracking-wider text-sky-500 mb-1">
          <Layers className="w-4 h-4" />
          <span>Universal 5-Stage Machine Learning Engine</span>
        </div>
        <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-slate-900 dark:text-slate-100">
          Multi-Tier Dropout Risk &amp; Recourse Simulator
        </h1>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
          Select an educational stage to evaluate specialized domain features and regional socio-economic context.
        </p>
      </div>

      {/* Tier Selector Pills */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {TIERS.map((t) => (
          <button
            key={t.id}
            onClick={() => handleTierChange(t.id)}
            className={`p-3.5 rounded-xl border text-left transition-all ${
              selectedTier === t.id
                ? 'bg-sky-50 dark:bg-sky-950/50 border-sky-500 ring-2 ring-sky-500/20'
                : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 hover:border-slate-300'
            }`}
          >
            <div
              className={`text-xs font-bold ${
                selectedTier === t.id ? 'text-sky-600 dark:text-sky-400' : 'text-slate-800 dark:text-slate-200'
              }`}
            >
              {t.label}
            </div>
            <div className="text-[10px] text-slate-500 dark:text-slate-400 mt-0.5">{t.sub}</div>
          </button>
        ))}
      </div>

      {/* Main Split: Form & Raw Input vs. Prediction Output */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left: Input Payload & Actions */}
        <div className="lg:col-span-6 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6 space-y-4 shadow-sm">
          <div className="flex justify-between items-center pb-3 border-b border-slate-200 dark:border-slate-700">
            <div>
              <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold">Active Archetype</span>
              <h3 className="text-base font-bold text-slate-900 dark:text-slate-100">{payload.student_id}</h3>
            </div>
            <span className="text-xs px-2.5 py-1 bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-md font-mono">
              {payload.district}, {payload.state}
            </span>
          </div>

          <div className="space-y-2">
            <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
              Domain Features Payload (JSON Preview):
            </label>
            <pre className="p-4 rounded-lg bg-slate-900 text-slate-100 text-xs font-mono overflow-auto max-h-72 border border-slate-700">
              {JSON.stringify(payload.features, null, 2)}
            </pre>
          </div>

          <button
            onClick={handlePredict}
            disabled={loading}
            className="w-full py-3 px-4 rounded-lg bg-sky-600 hover:bg-sky-500 text-white font-semibold text-sm flex items-center justify-center space-x-2 transition-colors shadow-sm disabled:opacity-50"
          >
            {loading ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Running Multi-Tier Model &amp; TreeSHAP...</span>
              </>
            ) : (
              <>
                <span>Run Calibrated Dropout Prediction</span>
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </div>

        {/* Right: Results Panel */}
        <div className="lg:col-span-6 space-y-4">
          {result ? (
            <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6 space-y-6 shadow-sm">
              {/* Risk Gauge Header */}
              <div className="flex justify-between items-center pb-4 border-b border-slate-200 dark:border-slate-700">
                <div>
                  <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold">
                    Calibrated Inference Result
                  </span>
                  <div className="flex items-center space-x-2 mt-1">
                    <span className="text-3xl font-extrabold text-slate-900 dark:text-slate-100">
                      {result.risk_score_percentage}%
                    </span>
                    <span
                      className={`text-xs px-2.5 py-1 rounded-full font-bold uppercase ${
                        result.risk_tier === 'High'
                          ? 'bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300'
                          : result.risk_tier === 'Medium'
                          ? 'bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300'
                          : 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300'
                      }`}
                    >
                      {result.risk_tier} Risk
                    </span>
                  </div>
                </div>
                <div className="text-right">
                  <span className="text-xs text-slate-400 block">Educational Tier</span>
                  <span className="text-xs font-semibold px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 font-mono">
                    {result.educational_tier}
                  </span>
                </div>
              </div>

              {/* TreeSHAP Drivers */}
              <div className="space-y-3">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                  Top SHAP Root Cause Drivers:
                </h4>
                <div className="space-y-2">
                  {result.top_drivers.map((d, i) => (
                    <div
                      key={i}
                      className="p-3 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 text-xs text-slate-800 dark:text-slate-200 space-y-1"
                    >
                      <div className="font-medium">{d.plain_language_explanation}</div>
                      <div className="text-[10px] text-slate-400 font-mono">
                        Impact: {d.risk_delta_percentage_points}pp | Shap: {d.shap_value}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Mapped Interventions */}
              <div className="space-y-3 pt-2">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                  Targeted Tier-Specific Interventions:
                </h4>
                <div className="space-y-2">
                  {result.recommended_interventions.map((intv) => (
                    <div
                      key={intv.intervention_id}
                      className="p-3.5 rounded-lg border border-sky-100 dark:border-sky-950 bg-sky-50/50 dark:bg-sky-950/20 space-y-1"
                    >
                      <div className="flex justify-between items-center text-xs font-bold text-sky-900 dark:text-sky-200">
                        <span>{intv.title}</span>
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-sky-200 dark:bg-sky-900 text-sky-800 dark:text-sky-300 font-mono">
                          {intv.intervention_id}
                        </span>
                      </div>
                      <p className="text-xs text-slate-600 dark:text-slate-300">{intv.action_description}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center p-12 border-2 border-dashed border-slate-200 dark:border-slate-700 rounded-xl text-center space-y-3 bg-slate-50/50 dark:bg-slate-900/20">
              <UserCheck className="w-10 h-10 text-slate-300 dark:text-slate-600" />
              <div className="text-sm font-medium text-slate-600 dark:text-slate-300">
                Ready for Multi-Tier Inference
              </div>
              <p className="text-xs text-slate-400 max-w-sm">
                Click 'Run Calibrated Dropout Prediction' to test this archetype through the specialized model and TreeSHAP explainer.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default UniversalPredictor;
