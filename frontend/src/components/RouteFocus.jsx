import { useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';

const HEADING = 'h1, h2, h3, h4, h5, h6';

// Focuses an element that is not normally focusable without adding it to the tab order.
const focusElement = (el) => {
  if (!el.hasAttribute('tabindex')) el.setAttribute('tabindex', '-1');
  el.focus({ preventScroll: true });
};

// A target that is already focusable (the skip link's <main tabIndex=-1>) takes focus itself; a
// section hands focus to its heading (aria-labelledby, else its first heading).
const focusTargetFor = (target) => {
  if (target.hasAttribute('tabindex')) return target;
  const labelledBy = target.getAttribute('aria-labelledby');
  return (labelledBy && document.getElementById(labelledBy)) || target.querySelector(HEADING) || target;
};

// Calls onFound with find()'s element once it exists: pages load lazily and some headings wait for
// data, so there is no time limit; the returned cleanup (next navigation) stops waiting. If the user
// has moved focus somewhere else by then, it is left alone.
const whenPresent = (find, onFound) => {
  const startFocus = document.activeElement;
  const handle = (el) => {
    const active = document.activeElement;
    if (active === startFocus || active === document.body) onFound(el);
  };
  const found = find();
  if (found) {
    handle(found);
    return undefined;
  }
  const observer = new MutationObserver(() => {
    const el = find();
    if (!el) return;
    observer.disconnect();
    handle(el);
  });
  observer.observe(document.body, { childList: true, subtree: true });
  return () => observer.disconnect();
};

// Deferred a frame: the browser's own fragment navigation runs its focusing steps after the click
// (a section is not focusable, so focus would fall back to <body>), and must not undo ours.
const focusHash = (hash) => {
  let frame;
  const stop = whenPresent(
    () => document.getElementById(decodeURIComponent(hash.slice(1))),
    (target) => {
      frame = requestAnimationFrame(() => {
        target.scrollIntoView();
        focusElement(focusTargetFor(target));
      });
    },
  );
  return () => {
    stop?.();
    cancelAnimationFrame(frame);
  };
};

// Moves keyboard and screen-reader focus on navigation: to the new page's <h1> when the path
// changes, or to the target section's heading when an in-page link changes the hash. Query-string
// changes (worklist filters) leave focus alone. On first load focus stays with the browser, except
// for a #hash, which the browser cannot scroll to before the lazy page renders.
const RouteFocus = () => {
  const { pathname, hash } = useLocation();
  const initialPath = useRef(`${pathname}${hash}`);
  const navigated = useRef(false);

  useEffect(() => {
    if (`${pathname}${hash}` !== initialPath.current) navigated.current = true;
    if (hash) return focusHash(hash);
    if (!navigated.current) return undefined;
    return whenPresent(
      () => document.querySelector('#main h1'),
      (heading) => {
        window.scrollTo(0, 0);
        focusElement(heading);
      },
    );
  }, [pathname, hash]); // not search: filter changes in the query string do not move focus

  // Activating a link to the hash already in the URL fires no navigation, so handle that click here.
  useEffect(() => {
    const onClick = (e) => {
      if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      const link = e.target.closest?.('a[href*="#"]');
      if (!link) return;
      const url = new URL(link.href);
      const here = window.location;
      if (url.hash && url.origin === here.origin && url.pathname === here.pathname && url.search === here.search && url.hash === here.hash) {
        focusHash(url.hash);
      }
    };
    document.addEventListener('click', onClick);
    return () => document.removeEventListener('click', onClick);
  }, []);

  return null;
};

export default RouteFocus;
