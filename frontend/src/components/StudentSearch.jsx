import React, { useEffect, useRef, useState } from 'react';
import { Search } from 'lucide-react';
import SearchPalette from './SearchPalette';

const isMac = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
const SHORTCUT_LABEL = isMac ? '⌘K' : 'Ctrl K';

// Top-bar trigger for the student search palette; Ctrl/Cmd+K opens it from anywhere in the app.
const StudentSearch = () => {
  const [open, setOpen] = useState(false);
  const triggerRef = useRef(null);
  const returnFocusRef = useRef(null);

  useEffect(() => {
    const onKey = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        returnFocusRef.current = document.activeElement;
        setOpen(true);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const close = () => {
    setOpen(false);
    (returnFocusRef.current || triggerRef.current)?.focus?.();
  };

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        onClick={() => {
          returnFocusRef.current = triggerRef.current;
          setOpen(true);
        }}
        className="inline-flex min-w-0 items-center gap-2 rounded-control border border-control px-3 py-1.5 text-15 text-slate hover:bg-ink-wash"
        aria-keyshortcuts="Control+K Meta+K"
      >
        <Search className="w-4 h-4 shrink-0" aria-hidden="true" />
        <span className="truncate">Search students</span>
        <kbd className="hidden md:inline rounded-[4px] border border-rule px-1.5 text-13">{SHORTCUT_LABEL}</kbd>
      </button>
      <SearchPalette open={open} onClose={close} />
    </>
  );
};

export default StudentSearch;
