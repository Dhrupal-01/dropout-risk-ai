import { useSyncExternalStore } from 'react';

// Admin token for API writes (first-stage access control; per-mentor login comes later).
// Kept in memory only, in this module variable: never localStorage, sessionStorage or cookies.
// A page reload clears it.
let token = '';
const listeners = new Set();

export const ADMIN_TOKEN_MISSING_MESSAGE =
  'This action changes data and needs the admin token. Enter it in the "Admin token" field in the ' +
  'header. It is kept in memory only, so you need to enter it again after reloading the page.';

export const getAdminToken = () => token.trim();

export const setAdminToken = (value) => {
  token = value || '';
  listeners.forEach((listener) => listener());
};

const subscribe = (listener) => {
  listeners.add(listener);
  return () => listeners.delete(listener);
};

// Raw value for the input field (so typing is not trimmed mid-edit).
export const useAdminToken = () => useSyncExternalStore(subscribe, () => token);

// Write calls go through this: without a token they fail before any request, with a clear message.
export const withAdminToken = (request) => {
  if (!getAdminToken()) {
    return Promise.reject({
      code: 'admin_token_missing',
      message: ADMIN_TOKEN_MISSING_MESSAGE,
      details: [],
    });
  }
  return request();
};
