import React from 'react';
import { Link } from 'react-router-dom';

// Placeholder until the Overview dashboard is built (build step 7).
const Overview = () => (
  <div className="px-4 sm:px-6 py-8 max-w-page">
    <h1 className="font-display font-medium text-32 tracking-display text-graphite">Overview</h1>
    <p className="mt-3 max-w-measure text-17 text-slate">
      The cohort overview is not built yet. The student list already shows every student, their risk
      level and the reasons behind it.
    </p>
    <div className="mt-6">
      <Link to="/app/students" className="btn btn-primary">
        Go to students
      </Link>
    </div>
  </div>
);

export default Overview;
