import client from './client';
import { withAdminToken } from './adminToken';

/**
 * Health probe API endpoint
 * @returns {Promise<Object>} Status of backend database, model, environment
 */
export const checkHealth = () => {
  return client.get('/health');
};

// Drop empty filters ('' from an "All" select, null, undefined) so the API applies no filter for them.
const withoutEmptyParams = (params) =>
  Object.fromEntries(Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== ''));

/**
 * Fetch prioritized mentor triage queue
 * @param {Object} params - Query filters { department, risk_tier, assigned_mentor_id, limit, offset }
 * @returns {Promise<Object>} Page of students ranked by risk priority
 */
export const getMentorQueue = (params = {}) => {
  return client.get('/api/v1/mentors/queue', { params: withoutEmptyParams(params) });
};

/**
 * Fetch distinct filter values (departments and assigned mentor IDs)
 * @returns {Promise<Object>} { departments: string[], mentor_ids: string[] }
 */
export const getMentorFilters = () => {
  return client.get('/api/v1/mentors/filters');
};

/**
 * Fetch cohort statistics summary (totals, risk tier distribution, department breakdown)
 * @returns {Promise<Object>} { total: number, by_tier: Object, by_department: Object }
 */
export const getStatsSummary = () => {
  return client.get('/api/v1/stats/summary');
};

/**
 * Histogram of latest calibrated risk probabilities
 * @param {number} [bins=10] - Equal-width bins over [0, 1] (2-50)
 * @returns {Promise<Object>} { total, bin_count, bins: [{ lower, upper, count }] }
 */
export const getStatsDistribution = (bins = 10) => {
  return client.get('/api/v1/stats/distribution', { params: { bins } });
};

/**
 * Most common risk-increasing SHAP drivers among students' latest predictions
 * @param {Object} [params] - { limit?: number, risk_tier?: 'High'|'Medium'|'Low' }
 * @returns {Promise<Object>} { risk_tier, students_considered, students_with_drivers, drivers: [{ feature_name, display_name, student_count }] }
 */
export const getStatsDrivers = (params = {}) => {
  return client.get('/api/v1/stats/drivers', { params: withoutEmptyParams(params) });
};

/**
 * Intervention log counts by lifecycle and outcome status
 * @returns {Promise<Object>} { total, open, students_with_open_interventions, by_status, by_outcome_status }
 */
export const getStatsInterventions = () => {
  return client.get('/api/v1/stats/interventions');
};

/**
 * Students triggering each rule-based alert, with the configured attendance threshold
 * @returns {Promise<Object>} { students_considered, students_with_any_alert, attendance_threshold, alerts: [{ code, student_count }] }
 */
export const getStatsAlerts = () => {
  return client.get('/api/v1/stats/alerts');
};


/**
 * Fetch top SHAP driver explainability markers for a student
 * @param {string} studentId - Student identifier
 * @param {number} topK - Number of drivers to return (default 5)
 * @returns {Promise<Object>} Drivers list
 */
export const getStudentExplanation = (studentId, topK = 5) => {
  return client.get(`/api/v1/students/${studentId}/explanation`, {
    params: { top_k: topK },
  });
};

/**
 * Fetch recommended interventions & counterfactual projected scenarios
 * @param {string} studentId - Student identifier
 * @returns {Promise<Object>} Recommendations and projected outcome data
 */
export const getStudentInterventions = (studentId) => {
  return client.get(`/api/v1/students/${studentId}/interventions`);
};

/**
 * Fetch all logged interventions history for a student
 * @param {string} studentId - Student identifier
 * @returns {Promise<Array>} List of intervention logs
 */
export const getStudentInterventionsHistory = (studentId) => {
  return client.get(`/api/v1/students/${studentId}/interventions/history`);
};

/**
 * Fetch full intervention catalog
 * @returns {Promise<Array>} List of pre-defined system interventions
 */
export const getInterventionsCatalog = () => {
  return client.get('/api/v1/interventions/catalog');
};

/**
 * Log or update an intervention state
 * @param {Object} payload - { student_id, intervention_id, assigned_faculty_id, status, notes, scheduled_followup_date, baseline_risk_probability, post_intervention_risk_probability }
 * @returns {Promise<Object>} Created/updated log response
 */
export const logIntervention = (payload) => {
  return withAdminToken(() => client.post('/api/v1/interventions/log', payload));
};

/**
 * Score a single student and persist/append prediction to their history
 * @param {Object} payload - { student_id, name, department, assigned_mentor_id, features }
 * @returns {Promise<Object>} Calibrated risk and tier
 */
export const predictStudent = (payload) => {
  return withAdminToken(() => client.post('/api/v1/predict', payload));
};

/**
 * Re-score a student from the inputs already stored on the server (no body is sent)
 * @param {string} studentId - Student ID
 * @returns {Promise<Object>} Calibrated risk and tier
 */
export const rescoreStudent = (studentId) => {
  return withAdminToken(() => client.post(`/api/v1/students/${encodeURIComponent(studentId)}/rescore`));
};

/**
 * Batch score students using a CSV file upload
 * @param {File} file - CSV File object containing student rows matching feature headers
 * @param {boolean} sortByRiskDesc - Whether to sort returned rows by highest risk (default true)
 * @returns {Promise<Object>} Evaluation results with totals and scores
 */
export const uploadBatchCsv = (file, sortByRiskDesc = true) => {
  const formData = new FormData();
  formData.append('file', file);
  return withAdminToken(() => client.post('/api/v1/predict/batch/csv', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    params: {
      sort_by_risk_desc: sortByRiskDesc,
      include_explanations: false, // Default false to optimize speed
    }
  }));
};
