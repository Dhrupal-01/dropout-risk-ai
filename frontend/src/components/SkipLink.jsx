import React from 'react';

const SkipLink = () => (
  <a
    href="#main"
    className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-[100] focus:px-4 focus:py-2 focus:rounded-control focus:bg-paper focus:text-ink focus:border focus:border-control"
  >
    Skip to content
  </a>
);

export default SkipLink;
