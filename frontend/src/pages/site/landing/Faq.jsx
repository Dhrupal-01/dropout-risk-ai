import React from 'react';

// Honest answers, including what is not done yet. Native <details> keeps it keyboard accessible with no script.
const QUESTIONS = [
  {
    q: 'Does DropoutGuard predict who will drop out?',
    a: 'No. It flags students who may need support and explains why, so a mentor can decide whether and how to reach out. It is decision support; nothing about a student changes automatically.',
  },
  {
    q: 'Is it trained on Indian student data?',
    a: 'No, not yet. The model in the demo is trained on a simulated Indian college cohort, built from Indian regulations (such as attendance and pass-mark rules) and stated assumptions. It has not been validated on records from Indian colleges. The real-data results above come from a Portuguese polytechnic and a UK distance-learning university.',
  },
  {
    q: 'What data does a college need?',
    a: 'The records most colleges already keep: attendance, internal marks, learning-platform activity and fee status, uploaded as a CSV file. No new data collection.',
  },
  {
    q: "Can a flag affect a student's marks, admission or scholarship?",
    a: 'It must not. Flags are a prompt for a supportive conversation and must never be used for penalties, debarment, admissions or scholarship decisions.',
  },
  {
    q: 'Who can see student records?',
    a: 'Access control is at an early stage. In this demo, viewing is open and every change needs one shared admin token. Sign-in for individual mentors is not built yet, so real student records should not be loaded until it is.',
  },
  {
    q: 'Is the model fair to every group of students?',
    a: 'Not always, and we report where it is not. Protected attributes such as gender, age, caste category and disability are never model inputs, but the audits on the public datasets still find differences between groups in how often students who later left were missed. Those differences are published with confidence intervals in the fairness audit.',
  },
];

const Faq = () => (
  <section aria-labelledby="faq-heading" className="border-t border-rule">
    <div className="max-w-page mx-auto px-4 sm:px-6 py-16 md:py-24">
      <h2 id="faq-heading" className="font-display font-medium text-32 md:text-44 tracking-display text-graphite">
        Questions, answered plainly.
      </h2>
      <div className="mt-8 max-w-[52rem] divide-y divide-rule border-y border-rule">
        {QUESTIONS.map(({ q, a }) => (
          <details key={q} className="group py-4">
            <summary className="flex cursor-pointer list-none items-start justify-between gap-4 text-17 font-semibold text-graphite [&::-webkit-details-marker]:hidden">
              {q}
              <span aria-hidden="true" className="mt-0.5 text-20 leading-none text-ink group-open:rotate-45">
                +
              </span>
            </summary>
            <p className="mt-3 max-w-measure text-17 text-slate">{a}</p>
          </details>
        ))}
      </div>
    </div>
  </section>
);

export default Faq;
