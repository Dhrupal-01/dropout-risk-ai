import React from 'react';

// Suspense fallback while a lazily loaded route chunk downloads.
const PageLoading = () => (
  <p role="status" className="px-4 sm:px-6 py-16 text-15 text-slate">
    Loading…
  </p>
);

export default PageLoading;
