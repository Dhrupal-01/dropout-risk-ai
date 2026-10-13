import React from 'react';

// Recharts category-axis tick that wraps long labels onto several lines instead of clipping them.
const LINE_HEIGHT = 15;

const wrapWords = (text, maxChars) => {
  const lines = [];
  for (const word of String(text).split(' ')) {
    const last = lines[lines.length - 1];
    if (last && `${last} ${word}`.length <= maxChars) lines[lines.length - 1] = `${last} ${word}`;
    else lines.push(word);
  }
  return lines;
};

const WrappedCategoryTick = ({ x, y, payload, maxChars = 18 }) => {
  const lines = wrapWords(payload.value, maxChars);
  const top = y - ((lines.length - 1) * LINE_HEIGHT) / 2;
  return (
    <text x={x - 8} y={top} textAnchor="end" dominantBaseline="middle" style={{ fill: 'var(--graphite)', fontSize: 13 }}>
      {lines.map((line, i) => (
        <tspan key={`${line}-${i}`} x={x - 8} dy={i === 0 ? 0 : LINE_HEIGHT}>
          {line}
        </tspan>
      ))}
    </text>
  );
};

export default WrappedCategoryTick;
