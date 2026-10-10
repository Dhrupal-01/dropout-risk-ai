import React from 'react';
import Citation from '../../../components/Citation';
import { formatFactValue, getFact } from '../../../content';

// Values and their wording come from facts.json; nothing numeric is typed here.
const STAT_IDS = [
  'oecd_bachelor_on_time',
  'us_some_college_no_credential',
  'india_enrolment_2023_24',
  'india_central_institution_exits_2019_2023',
];

const ProblemNumbers = () => (
  <section id="problem" aria-labelledby="problem-heading" className="border-t border-rule scroll-mt-4">
    <div className="max-w-page mx-auto px-4 sm:px-6 py-16 md:py-24">
      <h2 id="problem-heading" className="font-display font-medium text-32 md:text-44 tracking-display text-graphite max-w-[22ch]">
        Many students who start college do not finish on time.
      </h2>
      <dl className="mt-10 grid gap-x-8 gap-y-10 sm:grid-cols-2 lg:grid-cols-4">
        {STAT_IDS.map((id) => {
          const fact = getFact(id);
          return (
            <div key={id} className="border-t-2 border-graphite pt-4">
              <dt className="sr-only">{fact.label}</dt>
              <dd>
                <p className="text-40 font-semibold text-graphite">
                  {formatFactValue(fact)}
                  <Citation factId={id} />
                </p>
                <p className="mt-2 text-15 text-slate" aria-hidden="true">
                  {fact.label}
                </p>
              </dd>
            </div>
          );
        })}
      </dl>
    </div>
  </section>
);

export default ProblemNumbers;
