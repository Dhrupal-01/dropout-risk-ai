import React from 'react';
import { ImageOff } from 'lucide-react';

// The real four-step sequence, so it is numbered. Screenshots of the redesigned app replace the
// placeholders once the dashboard is built (build phase 9).
const STEPS = [
  {
    title: 'Bring the records you already have.',
    body: 'Upload a CSV of attendance, marks, learning-platform activity and fee status. No new data collection.',
    placeholder: 'the CSV import screen',
  },
  {
    title: 'See who may need support, and why.',
    body: 'Each student gets a risk level with the top reasons in plain language, not just a score.',
    placeholder: 'a student with their risk level and top reasons',
  },
  {
    title: 'Reach out early.',
    body: 'Mentors get a worklist ordered by priority, with suggested actions from a catalogue of institutional interventions.',
    placeholder: 'the mentor worklist',
  },
  {
    title: 'Record what happened.',
    body: 'Every outreach and outcome is logged, so the college learns what works.',
    placeholder: 'the intervention log',
  },
];

const ScreenshotPlaceholder = ({ subject }) => (
  <div className="flex aspect-[16/10] flex-col items-center justify-center gap-2 rounded-panel border-2 border-dashed border-control bg-ink-wash p-4 text-center">
    <ImageOff className="w-5 h-5 text-slate" aria-hidden="true" />
    <p className="text-15 font-semibold text-graphite">Screenshot placeholder</p>
    <p className="text-13 text-slate">Will show {subject} once the redesigned dashboard is built.</p>
  </div>
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
            <ScreenshotPlaceholder subject={step.placeholder} />
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
