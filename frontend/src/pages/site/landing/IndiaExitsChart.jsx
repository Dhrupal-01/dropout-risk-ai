import React from 'react';
import { Bar, BarChart, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { formatNumber } from '../../../content';

// Lazy chunk: the only landing-page module that imports recharts.
// Single series, so no legend: the figure caption names it. Values sit at the bar tips; a table
// carries the same numbers for screen readers.

const BAR_SIZE = 20;
const ROW_HEIGHT = 40;
const LABEL_CHARS_PER_LINE = 18;

const wrapWords = (text) => {
  const lines = [];
  for (const word of text.split(' ')) {
    const last = lines[lines.length - 1];
    if (last && `${last} ${word}`.length <= LABEL_CHARS_PER_LINE) lines[lines.length - 1] = `${last} ${word}`;
    else lines.push(word);
  }
  return lines;
};

const InstitutionTick = ({ x, y, payload }) => {
  const lines = wrapWords(payload.value);
  const lineHeight = 15;
  const top = y - ((lines.length - 1) * lineHeight) / 2;
  return (
    <text x={x - 8} y={top} textAnchor="end" dominantBaseline="middle" style={{ fill: 'var(--graphite)', fontSize: 13 }}>
      {lines.map((line, i) => (
        <tspan key={line} x={x - 8} dy={i === 0 ? 0 : lineHeight}>
          {line}
        </tspan>
      ))}
    </text>
  );
};

const ChartTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const { institution, count } = payload[0].payload;
  return (
    <div className="rounded-control border border-rule bg-paper px-3 py-2 text-13 text-graphite">
      <p className="font-medium">{institution}</p>
      <p className="text-slate tabular-nums">{formatNumber(count)} students</p>
    </div>
  );
};

const IndiaExitsChart = ({ breakdown, label }) => (
  <div>
    <div aria-hidden="true" style={{ height: breakdown.length * ROW_HEIGHT + 16 }} className="mt-4">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={breakdown} layout="vertical" margin={{ top: 8, right: 64, bottom: 8, left: 0 }}>
          <XAxis type="number" hide />
          <YAxis type="category" dataKey="institution" width={140} tickLine={false} axisLine={false} tick={<InstitutionTick />} interval={0} />
          <Tooltip content={<ChartTooltip />} cursor={{ fill: 'var(--ink-wash)' }} isAnimationActive={false} />
          <Bar dataKey="count" barSize={BAR_SIZE} radius={[0, 4, 4, 0]} fill="var(--ink)" isAnimationActive={false}>
            <LabelList dataKey="count" position="right" formatter={formatNumber} style={{ fill: 'var(--graphite)', fontSize: 13 }} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
    {/* sr-only on a wrapper: a <table> ignores the 1px width and would widen the page */}
    <div className="sr-only">
      <table>
        <caption>{label}</caption>
        <thead>
          <tr>
            <th scope="col">Institution type</th>
            <th scope="col">Students who left</th>
          </tr>
        </thead>
        <tbody>
          {breakdown.map((row) => (
            <tr key={row.institution}>
              <th scope="row">{row.institution}</th>
              <td>{formatNumber(row.count)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </div>
);

export default IndiaExitsChart;
