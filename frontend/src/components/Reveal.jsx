import React from 'react';
import { useInView } from '../hooks/useInView';

// Fades a chart in once, the first time it scrolls into view. The motion itself is CSS (.reveal in
// index.css) and only runs under prefers-reduced-motion: no-preference; otherwise it is simply shown.
const Reveal = ({ children, className = '' }) => {
  const [ref, inView] = useInView({ threshold: 0.2 });
  return (
    <div ref={ref} className={`reveal ${className}`} data-revealed={inView ? '' : undefined}>
      {children}
    </div>
  );
};

export default Reveal;
