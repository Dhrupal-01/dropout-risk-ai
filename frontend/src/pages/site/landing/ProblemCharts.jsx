import React, { lazy, useState } from 'react';
import Citation from '../../../components/Citation';
import LoadWhenNear from '../../../components/LoadWhenNear';
import { formatFactValue, formatNumber, getFact } from '../../../content';

const IndiaExitsChart = lazy(() => import('./IndiaExitsChart'));

// A percentage read as "of every 100 students": one dot per student in a 10 x 10 grid.
const UNIT = 100;
const GRID = 10;

const OECD_VIEWS = [
  { id: 'oecd_bachelor_on_time', label: 'On time', clause: "graduate within the programme's expected duration" },
  { id: 'oecd_bachelor_plus_one_year', label: 'One year late', clause: 'have graduated one year after the expected end' },
  { id: 'oecd_bachelor_plus_three_years', label: 'Three years late', clause: 'have graduated three years after the expected end' },
];

const UnitChart = ({ filled, label }) => (
  <svg viewBox={`0 0 ${GRID * 12} ${GRID * 12}`} className="w-full max-w-[16rem]" role="img" aria-label={label}>
    {Array.from({ length: UNIT }, (_, i) => {
      const isFilled = i < filled;
      return (
        <circle
          key={i}
          cx={(i % GRID) * 12 + 6}
          cy={Math.floor(i / GRID) * 12 + 6}
          r="4.5"
          style={isFilled ? { fill: 'var(--ink)' } : { fill: 'none', stroke: 'var(--slate)', strokeWidth: 1 }}
        />
      );
    })}
  </svg>
);

const OecdCompletion = () => {
  const [viewId, setViewId] = useState(OECD_VIEWS[2].id);
  const view = OECD_VIEWS.find((v) => v.id === viewId);
  const fact = getFact(view.id);
  const graduated = fact.value;
  const sentence = `Of every ${UNIT} students who start a bachelor's degree, ${formatNumber(graduated)} ${view.clause}. ${formatNumber(UNIT - graduated)} have not.`;

  return (
    <div>
      <h3 className="text-20 font-semibold text-graphite">Bachelor's completion, average across OECD and partner countries</h3>
      <div role="group" aria-label="Time since the expected end of the degree" className="mt-4 inline-flex flex-wrap gap-1 rounded-control border border-control p-1">
        {OECD_VIEWS.map((v) => (
          <button
            key={v.id}
            type="button"
            aria-pressed={v.id === viewId}
            onClick={() => setViewId(v.id)}
            className={`rounded-[4px] px-3 py-1.5 text-15 transition-colors ${
              v.id === viewId ? 'bg-ink text-on-ink' : 'text-graphite hover:bg-ink-wash'
            }`}
          >
            {v.label}
          </button>
        ))}
      </div>
      <div className="mt-6 grid gap-6 sm:grid-cols-[16rem_1fr] sm:items-center">
        <UnitChart filled={graduated} label={sentence} />
        <div>
          <p className="text-17 text-graphite" aria-live="polite">
            Of every {UNIT} students who start a bachelor's degree,{' '}
            <strong className="font-semibold">{formatNumber(graduated)}</strong> {view.clause}.
            <Citation factId={view.id} /> {formatNumber(UNIT - graduated)} have not.
          </p>
          <p className="mt-3 flex items-center gap-4 text-13 text-slate">
            <span className="inline-flex items-center gap-1.5">
              <svg viewBox="0 0 10 10" className="w-2.5 h-2.5" aria-hidden="true"><circle cx="5" cy="5" r="4.5" style={{ fill: 'var(--ink)' }} /></svg>
              Graduated
            </span>
            <span className="inline-flex items-center gap-1.5">
              <svg viewBox="0 0 10 10" className="w-2.5 h-2.5" aria-hidden="true"><circle cx="5" cy="5" r="4" style={{ fill: 'none', stroke: 'var(--slate)', strokeWidth: 1 }} /></svg>
              Not yet graduated
            </span>
          </p>
        </div>
      </div>
    </div>
  );
};

const IndiaExits = () => {
  const enrolment = getFact('india_enrolment_2023_24');
  const ger = getFact('india_ger_2023_24');
  const exits = getFact('india_central_institution_exits_2019_2023');

  return (
    <div>
      <h3 className="text-20 font-semibold text-graphite">India</h3>
      <p className="mt-4 text-17 text-graphite max-w-measure">
        India enrols <strong className="font-semibold">{formatFactValue(enrolment)}</strong> students in higher
        education
        <Citation factId={enrolment.id} />, with a gross enrolment ratio of{' '}
        <strong className="font-semibold">{formatFactValue(ger)}</strong>
        <Citation factId={ger.id} />.
      </p>
      <p className="mt-2 text-13 text-slate max-w-measure">{ger.label}.</p>

      <figure className="mt-8">
        <figcaption className="text-15 font-medium text-graphite">
          {formatFactValue(exits)} {exits.label}
          <Citation factId={exits.id} />
        </figcaption>
        <LoadWhenNear minHeight="20rem">
          <IndiaExitsChart breakdown={exits.breakdown} label={`${formatFactValue(exits)} ${exits.label}`} />
        </LoadWhenNear>
        <p className="mt-3 text-13 text-slate max-w-measure">{exits.context}</p>
      </figure>
    </div>
  );
};

const ProblemCharts = () => (
  <section aria-labelledby="scale-heading" className="border-t border-rule">
    <div className="max-w-page mx-auto px-4 sm:px-6 py-16 md:py-24">
      <h2 id="scale-heading" className="font-display font-medium text-32 md:text-44 tracking-display text-graphite max-w-[22ch]">
        The problem is bigger than any one college.
      </h2>
      <div className="mt-10 grid gap-16 lg:grid-cols-2">
        <OecdCompletion />
        <IndiaExits />
      </div>
      <p className="mt-16 max-w-measure text-17 text-graphite">
        Those counts cover only centrally funded institutions, a small part of the country's
        higher-education system. For most colleges there is no shared early-warning signal at all. The
        records that could provide one (attendance, internal marks, LMS activity, fee dues) sit in separate
        registers and spreadsheets.
      </p>
    </div>
  </section>
);

export default ProblemCharts;
