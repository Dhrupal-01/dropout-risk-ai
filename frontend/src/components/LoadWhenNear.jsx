import React, { Suspense, useEffect, useRef, useState } from 'react';

// Renders `children` (typically a React.lazy component) only once the placeholder comes within
// `margin` of the viewport, so heavy chunks such as recharts are not fetched on first paint.
// The placeholder keeps `minHeight` so the page does not jump when the chunk arrives.
const LoadWhenNear = ({ children, minHeight, margin = '400px' }) => {
  const ref = useRef(null);
  // Without IntersectionObserver there is no way to wait, so load straight away.
  const [near, setNear] = useState(() => !('IntersectionObserver' in window));

  useEffect(() => {
    const el = ref.current;
    if (!el || near) return undefined;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setNear(true);
          observer.disconnect();
        }
      },
      { rootMargin: margin }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [near, margin]);

  const placeholder = <div style={{ minHeight }} aria-hidden="true" />;

  return (
    <div ref={ref} style={{ minHeight }}>
      {near ? <Suspense fallback={placeholder}>{children}</Suspense> : placeholder}
    </div>
  );
};

export default LoadWhenNear;
