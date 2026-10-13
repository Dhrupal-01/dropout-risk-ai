import React from 'react';
import { FAIRNESS_DOC_URL } from '../../../content/site';

// Plain text in two columns, no cards (spec 4.6).
const PRINCIPLES = [
  {
    title: 'A person decides; the tool only suggests.',
    body: 'Every flag is a prompt for a mentor to look and talk to the student. Nothing happens automatically.',
  },
  {
    title: 'Flags never punish.',
    body: 'They must never be used for penalties, debarment, admissions or scholarship decisions.',
  },
  {
    title: 'Every flag comes with its reasons.',
    body: 'A mentor sees why a student was flagged, in plain language, and can disagree.',
  },
  {
    title: 'Outcomes are audited across student groups.',
    body: 'Gender, age, caste category and disability are never model inputs; results are checked across groups and the gaps are reported.',
  },
];

const ResponsibleAI = () => (
  <section id="responsible-ai" aria-labelledby="responsible-heading" className="border-t border-rule scroll-mt-4">
    <div className="max-w-page mx-auto px-4 sm:px-6 py-16 md:py-24">
      <h2 id="responsible-heading" className="font-display font-medium text-32 md:text-44 tracking-display text-graphite">
        Built to support, never to punish.
      </h2>
      <dl className="mt-10 grid gap-x-12 gap-y-8 md:grid-cols-2">
        {PRINCIPLES.map((p) => (
          <div key={p.title}>
            <dt className="text-20 font-semibold text-graphite">{p.title}</dt>
            <dd className="mt-2 text-17 text-slate max-w-measure">{p.body}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-10 text-15">
        <a href={FAIRNESS_DOC_URL} className="text-ink underline underline-offset-4 hover:no-underline">
          Read the fairness audit
        </a>
      </p>
    </div>
  </section>
);

export default ResponsibleAI;
