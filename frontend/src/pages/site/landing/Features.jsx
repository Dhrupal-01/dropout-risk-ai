import React from 'react';
import { CalendarX, ClipboardList, ListOrdered, MessageSquareText, Scale, Upload } from 'lucide-react';

// Ruled columns, not boxed cards: one icon, a title and one sentence each.
const FEATURES = [
  {
    icon: MessageSquareText,
    title: 'Risk level with reasons',
    body: 'Each student is shown as on track, monitor or needs outreach, with the top reasons in plain language, not just a score.',
  },
  {
    icon: ListOrdered,
    title: 'Mentor worklist',
    body: 'Students ordered by priority and filterable by department, risk level and mentor, so the calls that matter this week come first.',
  },
  {
    icon: ClipboardList,
    title: 'Interventions and logging',
    body: 'Suggested actions from a catalogue of institutional interventions, and a log of every outreach and its outcome.',
  },
  {
    icon: Upload,
    title: 'CSV import',
    body: 'Upload the attendance, marks, learning-platform and fee records the college already keeps. No new data collection.',
  },
  {
    icon: CalendarX,
    title: 'Rule-based alerts',
    body: 'Simple rules, separate from the model, raise an alert when attendance falls below the required minimum.',
  },
  {
    icon: Scale,
    title: 'Fairness audits',
    body: 'Results are audited across student groups, and the differences are reported with confidence intervals, not hidden.',
  },
];

const Features = () => (
  <section aria-labelledby="features-heading" className="border-t border-rule">
    <div className="max-w-page mx-auto px-4 sm:px-6 py-16 md:py-24">
      <h2 id="features-heading" className="font-display font-medium text-32 md:text-44 tracking-display text-graphite max-w-[22ch]">
        What a mentor gets.
      </h2>
      <ul className="mt-10 grid gap-x-8 gap-y-10 sm:grid-cols-2 lg:grid-cols-3">
        {FEATURES.map(({ icon: Icon, title, body }) => (
          <li key={title} className="border-t border-rule pt-4">
            <Icon className="w-5 h-5 text-ink" aria-hidden="true" />
            <h3 className="mt-3 text-20 font-semibold text-graphite">{title}</h3>
            <p className="mt-2 text-15 text-slate max-w-measure">{body}</p>
          </li>
        ))}
      </ul>
    </div>
  </section>
);

export default Features;
