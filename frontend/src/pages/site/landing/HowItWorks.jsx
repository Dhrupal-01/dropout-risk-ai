import React from 'react';
import importLight1x from '../../../assets/how-it-works/import-light-576.webp';
import importLight2x from '../../../assets/how-it-works/import-light-1152.webp';
import importDark1x from '../../../assets/how-it-works/import-dark-576.webp';
import importDark2x from '../../../assets/how-it-works/import-dark-1152.webp';
import studentLight1x from '../../../assets/how-it-works/student-light-576.webp';
import studentLight2x from '../../../assets/how-it-works/student-light-1152.webp';
import studentDark1x from '../../../assets/how-it-works/student-dark-576.webp';
import studentDark2x from '../../../assets/how-it-works/student-dark-1152.webp';
import worklistLight1x from '../../../assets/how-it-works/worklist-light-576.webp';
import worklistLight2x from '../../../assets/how-it-works/worklist-light-1152.webp';
import worklistDark1x from '../../../assets/how-it-works/worklist-dark-576.webp';
import worklistDark2x from '../../../assets/how-it-works/worklist-dark-1152.webp';
import historyLight1x from '../../../assets/how-it-works/history-light-576.webp';
import historyLight2x from '../../../assets/how-it-works/history-light-1152.webp';
import historyDark1x from '../../../assets/how-it-works/history-dark-576.webp';
import historyDark2x from '../../../assets/how-it-works/history-dark-1152.webp';

// The real four-step sequence, so it is numbered. Each step shows a cropped screenshot of the app
// running on the simulated demo cohort, in a light and a dark version that follow the site theme.
const STEPS = [
  {
    title: 'Bring the records you already have.',
    body: 'Upload a CSV of attendance, marks, learning-platform activity and fee status. No new data collection.',
    shot: {
      light: [importLight1x, importLight2x],
      dark: [importDark1x, importDark2x],
      alt: 'The Import students screen after a CSV upload: every row has been scored, with how many students need outreach, should be monitored or are on track, and buttons to import another file or open the student list.',
    },
  },
  {
    title: 'See who may need support, and why.',
    body: 'Each student gets a risk level with the top reasons in plain language, not just a score.',
    shot: {
      light: [studentLight1x, studentLight2x],
      dark: [studentDark1x, studentDark2x],
      alt: "A student's estimated risk, labelled needs outreach, above a bar chart of the factors that moved the estimate most, such as uncleared backlogs and recent attendance, each marked as raising or lowering it.",
    },
  },
  {
    title: 'Reach out early.',
    body: 'Mentors get a worklist ordered by priority, with suggested actions from a catalogue of institutional interventions.',
    shot: {
      light: [worklistLight1x, worklistLight2x],
      dark: [worklistDark1x, worklistDark2x],
      alt: 'The Students worklist: counts of students who need outreach, should be monitored or are on track, filters for department, risk level and mentor, and a table of students ordered by whom to contact first.',
    },
  },
  {
    title: 'Record what happened.',
    body: 'Every outreach and outcome is logged, so the college learns what works.',
    shot: {
      light: [historyLight1x, historyLight2x],
      dark: [historyDark1x, historyDark2x],
      alt: "The outreach history on a student's page: a completed attendance counselling session with the mentor's note, peer tutoring in progress and newly assigned support actions, each with its status, notes and follow-up date.",
    },
  },
];

// Images are 1152 x 960 at 2x (a 768 x 640 crop of the app). Grid columns are at most 552px wide.
const SIZES = '(min-width: 1200px) 552px, (min-width: 768px) calc(50vw - 48px), calc(100vw - 32px)';

const Screenshot = ({ light, dark, alt }) => (
  <figure>
    <div className="overflow-hidden rounded-panel border border-rule bg-paper">
      {[
        { srcs: light, className: 'dark:hidden' },
        { srcs: dark, className: 'hidden dark:block' },
      ].map(({ srcs: [src1x, src2x], className }) => (
        <img
          key={src1x}
          src={src1x}
          srcSet={`${src1x} 576w, ${src2x} 1152w`}
          sizes={SIZES}
          width={1152}
          height={960}
          alt={alt}
          loading="lazy"
          decoding="async"
          className={`block h-auto w-full ${className}`}
        />
      ))}
    </div>
    <figcaption className="mt-2 text-13 text-slate">Simulated demo data</figcaption>
  </figure>
);

const HowItWorks = () => (
  <section id="how-it-works" aria-labelledby="how-heading" className="border-t border-rule scroll-mt-4">
    <div className="max-w-page mx-auto px-4 sm:px-6 py-16 md:py-24">
      <h2 id="how-heading" className="font-display font-medium text-32 md:text-44 tracking-display text-graphite">
        How DropoutGuard helps.
      </h2>
      <ol className="mt-10 grid gap-12 md:grid-cols-2">
        {STEPS.map((step, i) => (
          <li key={step.title}>
            <Screenshot {...step.shot} />
            <div className="mt-4 flex gap-4">
              <span className="font-display text-32 leading-none text-ink" aria-hidden="true">
                {i + 1}
              </span>
              <div>
                <h3 className="text-20 font-semibold text-graphite">
                  <span className="sr-only">Step {i + 1}: </span>
                  {step.title}
                </h3>
                <p className="mt-1 text-15 text-slate max-w-measure">{step.body}</p>
              </div>
            </div>
          </li>
        ))}
      </ol>
    </div>
  </section>
);

export default HowItWorks;
