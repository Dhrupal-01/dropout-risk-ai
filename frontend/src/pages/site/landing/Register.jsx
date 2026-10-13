import React from 'react';
import '@fontsource/kalam/400.css';
import { useInView } from '../../../hooks/useInView';

// Illustration only: fictional roll numbers and marks, no real students. The flagged row's margin note
// is computed from these marks so the picture and the note can never disagree.
const ROWS = [
  { roll: '24CE016', marks: 'PPPPPPPP' },
  { roll: '24CE017', marks: 'PPAPPPPP' },
  { roll: '24CE018', marks: 'PPPAPAAA', flagged: true, feeOverdue: true, assignmentsMissing: 2 },
  { roll: '24CE019', marks: 'PPPPPPAP' },
  { roll: '24CE020', marks: 'PPPPPPPP' },
  { roll: '24CE021', marks: 'PAPPPPPP' },
];
const WEEKS = ROWS[0].marks.length;
// Small screens show the most recent weeks only (spec: 6 instead of 8).
const HIDDEN_ON_MOBILE = WEEKS - 6;

const flagged = ROWS.find((row) => row.flagged);
const firstAbsence = flagged.marks.indexOf('A');
const absences = [...flagged.marks].filter((mark) => mark === 'A').length;
const weeksSinceFirstAbsence = WEEKS - firstAbsence;
const marginNote = [
  `${absences} absences in ${weeksSinceFirstAbsence} weeks`,
  flagged.feeOverdue && 'fee overdue',
  flagged.assignmentsMissing && `${flagged.assignmentsMissing} assignments missing`,
]
  .filter(Boolean)
  .join(', ');

const description =
  `Illustration of a college attendance register with ${ROWS.length} fictional students over ${WEEKS} weeks. ` +
  `Student ${flagged.roll} is marked present for the first weeks, then accumulates absences ` +
  `(${absences} in the last ${weeksSinceFirstAbsence} weeks). A handwritten margin note reads "${marginNote}", ` +
  'and a panel flags the student for outreach: top reasons attendance falling and fee overdue; suggested action, a mentor call this week.';

const weekCellClass = (index) => (index < HIDDEN_ON_MOBILE ? 'hidden sm:table-cell' : '');

const Register = () => {
  const [ref, inView] = useInView({ threshold: 0.35 });
  return (
    <section aria-labelledby="register-heading" className="border-t border-rule">
      <div className="max-w-page mx-auto px-4 sm:px-6 py-16 md:py-24 grid gap-10 lg:grid-cols-12 lg:items-start">
        <div className="lg:col-span-4">
          <h2 id="register-heading" className="font-display font-medium text-32 md:text-44 tracking-display text-graphite">
            Dropout doesn't happen on one day.
          </h2>
          <p className="mt-6 max-w-measure text-17 text-graphite">
            On a register page, drifting away looks like this: a few absences, then more. No single week stands
            out. Put together with an overdue fee and missing assignments, the marks become a clear signal that
            a student may need support.
          </p>
        </div>

        <figure className="lg:col-span-8">
          <p className="sr-only">{description}</p>
          <div
            ref={ref}
            aria-hidden="true"
            data-revealed={inView ? '' : undefined}
            style={{ '--weeks': WEEKS }}
            className="register ruled-paper rounded-panel border border-rule overflow-hidden"
          >
            <div className="border-l-2 border-margin-red ml-4 sm:ml-8 py-4 pr-4">
              <p className="pl-4 text-15 text-slate">Attendance register, semester three</p>
              <table className="mt-2 border-collapse text-15">
                <thead>
                  <tr className="h-8 text-13 text-slate">
                    <th className="pl-4 pr-3 text-left font-medium">Roll no.</th>
                    {Array.from({ length: WEEKS }, (_, i) => (
                      <th key={i} className={`w-10 text-center font-medium ${weekCellClass(i)}`}>
                        W{i + 1}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {ROWS.map((row) => (
                    <tr key={row.roll} className={`h-8 ${row.flagged ? 'register-band' : ''}`}>
                      <td className="pl-4 pr-3 tabular-nums text-graphite">{row.roll}</td>
                      {[...row.marks].map((mark, i) => (
                        <td key={i} className={`text-center font-hand text-20 leading-none ${weekCellClass(i)}`}>
                          <span
                            className={`register-mark inline-block ${mark === 'A' ? 'text-margin-red' : 'text-ink'}`}
                            style={{ '--col': i }}
                          >
                            {mark}
                          </span>
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>

              <div className="mt-4 pl-4 grid gap-4 sm:grid-cols-2 sm:items-start">
                <p className="register-note font-hand text-20 text-margin-red leading-snug">
                  {flagged.roll}: {marginNote}
                </p>
                <div className="register-panel rounded-control border border-rule bg-paper p-3">
                  <p className="text-15 font-semibold text-graphite">Flagged for outreach</p>
                  <p className="mt-1 text-15 text-graphite">Top reasons: attendance falling, fee overdue.</p>
                  <p className="mt-1 text-15 text-slate">Suggested: mentor call this week.</p>
                </div>
              </div>
            </div>
          </div>
          <figcaption className="mt-3 text-13 text-slate">Illustration. Fictional roll numbers.</figcaption>
        </figure>
      </div>
    </section>
  );
};

export default Register;
