import React from 'react';
import {
  Area,
  CartesianGrid,
  ComposedChart,
  ErrorBar,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { formatNumber } from '../../../content';
import { ciLevelPct, evidenceProvenance, findResult, findResults, formatScore } from '../../../content/evidence';
import { BENCHMARKS_DOC_URL } from '../../../content/site';
import Reveal from '../../../components/Reveal';

// Lazy chunk: evidence.json and recharts load only when the Evidence section nears the screen.
// Splits match the README: UCI repeated stratified CV, OULAD temporal holdout (predefined_split).

const UCI = { dataset: 'uci_697', label_variant: 'primary', split: 'repeated_stratified_cv' };
const OULAD = { dataset: 'oulad_349', split: 'predefined_split', model: 'xgboost' };
const MODELS = [
  { key: 'majority_class', label: 'Majority baseline' },
  { key: 'logistic_regression', label: 'Logistic regression' },
  { key: 'xgboost', label: 'XGBoost' },
];
const FEATURE_SETS = [
  { key: 'enrolment_time', label: 'Known at enrolment' },
  { key: 'end_of_sem1', label: 'Known by the end of semester one' },
];
// ROC-AUC of a coin flip: the reference every model is read against.
const CHANCE = 0.5;
const UCI_DOMAIN = [0.4, 1];

const axisTick = { fill: 'var(--slate)', fontSize: 13 };
const ciText = (m) => `${formatScore(m.point)} (${ciLevelPct}% CI ${formatScore(m.ci_lower)}–${formatScore(m.ci_upper)})`;

const ScoreTooltip = ({ active, payload, labelFor }) => {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <div className="rounded-control border border-rule bg-paper px-3 py-2 text-13 text-graphite">
      <p className="font-medium">{labelFor(row)}</p>
      <p className="text-slate tabular-nums">ROC-AUC {ciText(row.metric)}</p>
    </div>
  );
};

const uciRows = (featureSet) =>
  MODELS.map(({ key, label }) => {
    const metric = findResult({ ...UCI, feature_set: featureSet, model: key }).metrics.roc_auc;
    return { label, metric, point: metric.point, err: [metric.point - metric.ci_lower, metric.ci_upper - metric.point] };
  });

// One small multiple per feature set: a single series each, so no legend and no second colour.
const UciPanel = ({ featureSet }) => {
  const rows = uciRows(featureSet.key);
  return (
    <Reveal>
      <h4 className="text-15 font-semibold text-graphite">{featureSet.label}</h4>
      <div aria-hidden="true" className="h-52">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 24, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid horizontal={false} stroke="var(--rule)" />
            <XAxis
              type="number"
              dataKey="point"
              domain={UCI_DOMAIN}
              ticks={[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
              tickFormatter={(v) => v.toFixed(1)}
              tick={axisTick}
              stroke="var(--rule)"
            />
            <YAxis type="category" dataKey="label" width={128} tick={axisTick} tickLine={false} axisLine={false} />
            <ReferenceLine x={CHANCE} stroke="var(--slate)" strokeDasharray="0" label={{ value: 'Coin flip', position: 'top', fill: 'var(--slate)', fontSize: 12 }} />
            <Tooltip content={<ScoreTooltip labelFor={(row) => row.label} />} cursor={false} isAnimationActive={false} />
            <Scatter data={rows} fill="var(--ink)" isAnimationActive={false}>
              <ErrorBar dataKey="err" direction="x" width={6} strokeWidth={2} stroke="var(--ink)" />
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <div className="sr-only">
        <table>
          <caption>ROC-AUC with {ciLevelPct}% confidence interval, {featureSet.label.toLowerCase()}</caption>
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col">ROC-AUC ({ciLevelPct}% CI)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.label}>
                <th scope="row">{row.label}</th>
                <td>{ciText(row.metric)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Reveal>
  );
};

const OuladChart = ({ rows }) => {
  const lows = rows.map((r) => r.metric.ci_lower);
  const highs = rows.map((r) => r.metric.ci_upper);
  // One step below the coin-flip line so the line and its label sit clear of the x-axis.
  const domain = [Math.round((Math.min(CHANCE, Math.floor(Math.min(...lows) * 10) / 10) - 0.1) * 10) / 10, Math.ceil(Math.max(...highs) * 10) / 10];
  // Ticks every 0.1 across the domain, so the axis labels are exact (auto ticks rounded to one decimal repeat).
  const yTicks = Array.from({ length: Math.round((domain[1] - domain[0]) * 10) + 1 }, (_, i) => Math.round((domain[0] + i / 10) * 10) / 10);
  return (
    <Reveal>
      <div aria-hidden="true" className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={rows} margin={{ top: 16, right: 16, bottom: 24, left: 0 }}>
            <CartesianGrid vertical={false} stroke="var(--rule)" />
            <XAxis
              type="number"
              dataKey="day"
              domain={['dataMin', 'dataMax']}
              padding={{ left: 16, right: 16 }}
              ticks={rows.map((r) => r.day)}
              tick={axisTick}
              stroke="var(--rule)"
              label={{ value: 'Day of the course', position: 'insideBottom', offset: -16, fill: 'var(--slate)', fontSize: 13 }}
            />
            <YAxis domain={domain} ticks={yTicks} tickFormatter={(v) => v.toFixed(1)} tick={axisTick} width={40} axisLine={false} tickLine={false} />
            <ReferenceLine y={CHANCE} stroke="var(--slate)" label={{ value: 'Coin flip', position: 'insideTopRight', fill: 'var(--slate)', fontSize: 12 }} />
            <Tooltip content={<ScoreTooltip labelFor={(row) => `Day ${row.day}`} />} cursor={{ stroke: 'var(--rule)' }} isAnimationActive={false} />
            <Area dataKey="band" stroke="none" fill="var(--ink)" fillOpacity={0.1} isAnimationActive={false} />
            <Line
              dataKey="point"
              stroke="var(--ink)"
              strokeWidth={2}
              dot={{ r: 4, fill: 'var(--ink)', stroke: 'var(--paper)', strokeWidth: 2 }}
              activeDot={{ r: 5, fill: 'var(--ink)', stroke: 'var(--paper)', strokeWidth: 2 }}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="sr-only">
        <table>
          <caption>XGBoost ROC-AUC with {ciLevelPct}% confidence interval by day of the course, OULAD</caption>
          <thead>
            <tr>
              <th scope="col">Day</th>
              <th scope="col">Students evaluated</th>
              <th scope="col">ROC-AUC ({ciLevelPct}% CI)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.day}>
                <th scope="row">{row.day}</th>
                <td>{formatNumber(row.n)}</td>
                <td>{ciText(row.metric)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Reveal>
  );
};

const EvidenceContent = () => {
  const headline = findResult({ ...UCI, feature_set: 'end_of_sem1', model: 'xgboost' });
  const atEnrolment = findResult({ ...UCI, feature_set: 'enrolment_time', model: 'xgboost' });
  const ouladRows = findResults(OULAD)
    .sort((a, b) => a.snapshot_t - b.snapshot_t)
    .map((r) => {
      const m = r.metrics.roc_auc;
      return { day: r.snapshot_t, n: r.n_evaluated, metric: m, point: m.point, band: [m.ci_lower, m.ci_upper] };
    });
  const first = ouladRows[0];
  const last = ouladRows[ouladRows.length - 1];

  return (
    <div>
      <div className="grid gap-6 lg:grid-cols-2">
        <p className="text-20 text-graphite">
          Tested on real records from {formatNumber(headline.n_samples)} students at a Portuguese polytechnic
          (UCI dataset 697) who either dropped out or graduated. Using only what a college knows by the end of the
          first semester, the model ranked students who later dropped out above those who graduated with a
          ROC-AUC of <strong className="font-semibold">{ciText(headline.metrics.roc_auc)}</strong>.
        </p>
        <p className="text-20 text-graphite">
          What this does not show yet: how the model performs in Indian colleges. The deployed demo runs on a
          simulated Indian cohort.
        </p>
      </div>
      <p className="mt-4 max-w-measure text-15 text-slate">
        ROC-AUC is the chance that the model ranks a randomly chosen student who dropped out above one who
        graduated. 0.5 is a coin flip; 1.0 is perfect.
      </p>

      <figure className="mt-12">
        <figcaption>
          <h3 className="text-20 font-semibold text-graphite">UCI 697: ROC-AUC with {ciLevelPct}% confidence intervals</h3>
          <p className="mt-1 text-15 text-slate max-w-measure">
            With only what is known at enrolment, the same model reaches {formatScore(atEnrolment.metrics.roc_auc.point)}.
            A majority-class baseline sits at the coin-flip line.
          </p>
        </figcaption>
        <div className="mt-6 grid gap-8 md:grid-cols-2">
          {FEATURE_SETS.map((fs) => (
            <UciPanel key={fs.key} featureSet={fs} />
          ))}
        </div>
      </figure>

      <figure className="mt-12">
        <figcaption>
          <h3 className="text-20 font-semibold text-graphite">OULAD: how early the signal appears</h3>
          <p className="mt-1 text-15 text-slate max-w-measure">
            UK distance-learning records, trained on earlier course presentations and tested on a later year.
            XGBoost ROC-AUC is {formatScore(first.point)} at day {first.day} and {formatScore(last.point)} by day{' '}
            {last.day}: early signals are weaker, but better than chance. The band is the {ciLevelPct}% confidence interval.
          </p>
        </figcaption>
        <div className="mt-6">
          <OuladChart rows={ouladRows} />
        </div>
      </figure>

      <p className="mt-10 text-15 text-slate">
        <a href={BENCHMARKS_DOC_URL} className="text-ink underline underline-offset-4 hover:no-underline">
          Read the full evaluation
        </a>{' '}
        (all metrics and split strategies). Exported from benchmark run{' '}
        <span className="tabular-nums">{evidenceProvenance.run_id}</span>.
      </p>
    </div>
  );
};

export default EvidenceContent;
