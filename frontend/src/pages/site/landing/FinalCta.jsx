import React from 'react';
import { Link } from 'react-router-dom';
import { REPO_URL } from '../../../content/site';

const FinalCta = () => (
  <section aria-labelledby="cta-heading" className="border-t border-rule">
    <div className="max-w-page mx-auto px-4 sm:px-6 py-16 md:py-24">
      <h2 id="cta-heading" className="font-display font-medium text-32 md:text-44 tracking-display text-graphite max-w-[22ch]">
        Start with the records you already keep.
      </h2>
      <div className="mt-8 flex flex-wrap gap-3">
        <Link to="/app/overview" className="btn btn-primary">
          Open dashboard
        </Link>
        <a href={REPO_URL} className="btn btn-secondary">
          View the code on GitHub
        </a>
      </div>
    </div>
  </section>
);

export default FinalCta;
