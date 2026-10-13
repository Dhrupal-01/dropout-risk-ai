import React, { useEffect, useState } from 'react';
import { formatFactValue } from '../../../content';
import { usePrefersReducedMotion } from '../../../hooks/usePrefersReducedMotion';

const DURATION_MS = 1200;
const easeOutCubic = (t) => 1 - (1 - t) ** 3;

// Counts a fact's value up from zero once `start` is true. Screen readers get the final value only;
// an invisible copy of the final value reserves the width so nothing moves while the digits change.
// With prefers-reduced-motion: reduce the final value is shown straight away.
const CountUp = ({ fact, start }) => {
  const reducedMotion = usePrefersReducedMotion();
  const [progress, setProgress] = useState(0);
  const finalText = formatFactValue(fact);
  const decimals = (String(fact.value).split('.')[1] || '').length;

  useEffect(() => {
    if (!start || reducedMotion) return undefined;
    let frame;
    const startedAt = performance.now();
    const tick = (now) => {
      const t = Math.min(1, (now - startedAt) / DURATION_MS);
      setProgress(t);
      if (t < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [start, reducedMotion]);

  const done = reducedMotion || progress >= 1;
  const current = Number((fact.value * easeOutCubic(progress)).toFixed(decimals));
  const shown = done ? finalText : formatFactValue({ ...fact, value: current });

  return (
    <>
      <span className="sr-only">{finalText}</span>
      <span aria-hidden="true" className="relative inline-block">
        <span className="invisible">{finalText}</span>
        <span className="absolute inset-0 whitespace-nowrap">{shown}</span>
      </span>
    </>
  );
};

export default CountUp;
