import React from 'react';

// Suspense fallback while a lazily loaded route chunk downloads. It fills the viewport so the
// footer (and anything else below the outlet) starts off-screen and does not jump when the page arrives.
const PageLoading = () => (
  <p role="status" className="min-h-screen px-4 sm:px-6 py-16 text-15 text-slate">
    Loading…
  </p>
);

export default PageLoading;
