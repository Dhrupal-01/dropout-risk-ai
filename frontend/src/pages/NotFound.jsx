import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const NotFound = () => {
  const { pathname } = useLocation();

  return (
    <section className="max-w-page mx-auto px-4 sm:px-6 py-16 md:py-28">
      <h1 className="font-display font-medium text-40 md:text-44 tracking-display text-graphite">
        Page not found
      </h1>
      <p className="mt-4 max-w-measure text-17 text-slate">
        There is no page at <code className="text-graphite break-all">{pathname}</code>. It may have
        moved when the site was reorganised.
      </p>
      <div className="mt-8 flex flex-wrap gap-3">
        <Link to="/" className="btn btn-primary">
          Go to the home page
        </Link>
        <Link to="/app/overview" className="btn btn-secondary">
          Open dashboard
        </Link>
      </div>
    </section>
  );
};

export default NotFound;
