import client from './client';

/**
 * Health probe API endpoint
 * @returns {Promise<Object>} Status of backend database, model, environment
 */
export const checkHealth = () => {
  return client.get('/health');
};

/**
 * Fetch prioritized mentor triage queue
 * @param {Object} params - Query filters { department, risk_tier, assigned_mentor_id, limit, offset }
 * @returns {Promise<Object>} Page of students ranked by risk priority
 */
export const getMentorQueue = (params = {}) => {
  // Clean up empty params
  const cleanParams = {};
  Object.keys(params).forEach(key => {
    if (params[key] !== undefined && params[key] !== null && params[key] !== '') {
      cleanParams[key] = params[key];
    }
  });
  return client.get('/api/v1/mentors/queue', { params: cleanParams });
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
  return client.post('/api/v1/interventions/log', payload);
};

/**
 * Score a single student and persist/append prediction to their history
 * @param {Object} payload - { student_id, name, department, assigned_mentor_id, features }
 * @returns {Promise<Object>} Calibrated risk and tier
 */
export const predictStudent = (payload) => {
  return client.post('/api/v1/predict', payload);
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
  return client.post('/api/v1/predict/batch/csv', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    params: {
      sort_by_risk_desc: sortByRiskDesc,
      include_explanations: false, // Default false to optimize speed
    }
  });
};
