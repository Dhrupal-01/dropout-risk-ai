import { useEffect, useRef, useState } from 'react';

// True once the element has entered the viewport (expanded by `rootMargin`), then stays true.
// Without IntersectionObserver it is true from the start, so content is never stuck hidden.
export const useInView = ({ threshold = 0, rootMargin = '0px' } = {}) => {
  const ref = useRef(null);
  const [inView, setInView] = useState(() => !('IntersectionObserver' in window));

  useEffect(() => {
    const el = ref.current;
    if (!el || inView) return undefined;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setInView(true);
          observer.disconnect();
        }
      },
      { threshold, rootMargin }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [inView, threshold, rootMargin]);

  return [ref, inView];
};
