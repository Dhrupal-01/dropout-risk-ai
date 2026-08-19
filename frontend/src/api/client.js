import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const client = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 15000, // 15s timeout
});

// Centralized error mapping from frontend UX spec
export const errorMap = {
  student_not_found: 'Student not found.',
  no_prediction_history: 'Not yet scored.',
  invalid_lifecycle_transition: 'This intervention is already in this status or cannot transition backward.',
  validation_error: 'Please verify the fields. Schema or value validation failure.',
  invalid_feature_payload: 'Invalid data format or feature contract violation.',
  model_unavailable: 'Scoring is temporarily unavailable. Retrying shortly...',
  internal_error: 'Something went wrong. Reference:',
};

// Response interceptor to handle and standardise error responses
client.interceptors.response.use(
  (response) => response.data,
  (error) => {
    let standardError = {
      code: 'unknown_error',
      message: 'An unexpected connection error occurred.',
      details: [],
      status: error.response?.status,
    };

    if (error.response) {
      const data = error.response.data;
      const status = error.response.status;

      // Extract error envelope from FastAPI backend response
      if (data && typeof data === 'object') {
        standardError.code = data.error || 'internal_error';
        standardError.message = data.message || errorMap[standardError.code] || 'Server error';
        standardError.details = data.details || [];
        standardError.student_id = data.student_id;
        standardError.incident_id = data.incident_id;

        // Custom mappings as per UX spec
        if (standardError.code === 'student_not_found') {
          standardError.message = 'Student not found.';
        } else if (standardError.code === 'no_prediction_history') {
          standardError.message = 'Not yet scored.';
        } else if (standardError.code === 'invalid_lifecycle_transition') {
          standardError.message = `This intervention is already at ${data.current_status || 'this state'}.`;
        } else if (standardError.code === 'model_unavailable') {
          standardError.message = 'Scoring temporarily unavailable.';
          const retryAfter = error.response.headers['retry-after'];
          if (retryAfter) {
            standardError.retryAfter = parseInt(retryAfter, 10);
          }
        } else if (standardError.code === 'internal_error') {
          standardError.message = `Something went wrong. Reference: ${data.incident_id || 'unknown'}`;
        }
      } else {
        // Fallback for non-JSON or other HTML error status codes
        if (status === 404) {
          standardError.code = 'not_found';
          standardError.message = 'Requested resource not found.';
        } else if (status === 503) {
          standardError.code = 'model_unavailable';
          standardError.message = 'Scoring temporarily unavailable.';
        } else {
          standardError.code = 'server_error';
          standardError.message = `Server returned status code ${status}.`;
        }
      }
    } else if (error.request) {
      // The request was made but no response was received
      standardError.code = 'network_error';
      standardError.message = 'Cannot connect to the DropoutGuard backend. Please check if the server is running.';
    } else {
      // Something happened in setting up the request that triggered an Error
      standardError.message = error.message;
    }

    return Promise.reject(standardError);
  }
);

export default client;
