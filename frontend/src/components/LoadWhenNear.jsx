import React, { Suspense } from 'react';
import { useInView } from '../hooks/useInView';

// Renders `children` (typically a React.lazy component) only once the placeholder comes within
// `margin` of the viewport, so heavy chunks such as recharts are not fetched on first paint.
// The placeholder keeps `minHeight` so the page does not jump when the chunk arrives.
const LoadWhenNear = ({ children, minHeight, margin = '400px' }) => {
  const [ref, near] = useInView({ rootMargin: margin });
  const placeholder = <div style={{ minHeight }} aria-hidden="true" />;

  return (
    <div ref={ref} style={{ minHeight }}>
      {near ? <Suspense fallback={placeholder}>{children}</Suspense> : placeholder}
    </div>
  );
};

export default LoadWhenNear;
