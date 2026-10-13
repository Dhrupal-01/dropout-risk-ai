import { useSyncExternalStore } from 'react';

const QUERY = '(prefers-reduced-motion: reduce)';

const subscribe = (onChange) => {
  const mq = window.matchMedia(QUERY);
  mq.addEventListener('change', onChange);
  return () => mq.removeEventListener('change', onChange);
};

// For motion driven from JavaScript (the count-up, isAnimationActive on the landing charts). CSS animations use the media query directly.
export const usePrefersReducedMotion = () => useSyncExternalStore(subscribe, () => window.matchMedia(QUERY).matches);
