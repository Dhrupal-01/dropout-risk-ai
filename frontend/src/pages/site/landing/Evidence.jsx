import React, { lazy } from 'react';
import LoadWhenNear from '../../../components/LoadWhenNear';

const EvidenceContent = lazy(() => import('./EvidenceContent'));

// The heading stays in the page so #evidence always resolves; results and charts load lazily.
const Evidence = () => (
  <section id="evidence" aria-labelledby="evidence-heading" className="border-t border-rule scroll-mt-4">
    <div className="max-w-page mx-auto px-4 sm:px-6 py-16 md:py-24">
      <h2 id="evidence-heading" className="font-display font-medium text-32 md:text-44 tracking-display text-graphite">
        What the evidence shows, and what it doesn't.
      </h2>
      <div className="mt-8">
        <LoadWhenNear minHeight="60rem">
          <EvidenceContent />
        </LoadWhenNear>
      </div>
    </div>
  </section>
);

export default Evidence;
