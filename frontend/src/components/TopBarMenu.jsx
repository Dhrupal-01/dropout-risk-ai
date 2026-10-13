import React, { useEffect, useId, useRef, useState } from 'react';
import { SlidersHorizontal } from 'lucide-react';
import AdminTokenField from './AdminTokenField';
import HealthBadge from './HealthBadge';

const LABEL = 'Admin token and backend status';

// Small screens: the admin token field and backend health sit behind one button so the top bar is one row.
// Disclosure pattern, not role="menu": the panel holds a text field. The panel follows the button in tab
// order. Escape closes it and returns focus to the button; a tap outside or tabbing past it also closes it.
// The wrapper is not positioned, so the panel anchors to the sticky header and spans it on phones.
const TopBarMenu = ({ className = '' }) => {
  const [open, setOpen] = useState(false);
  const panelId = useId();
  const wrapperRef = useRef(null);
  const buttonRef = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    const onPointerDown = (e) => {
      if (!wrapperRef.current?.contains(e.target)) setOpen(false);
    };
    document.addEventListener('pointerdown', onPointerDown);
    return () => document.removeEventListener('pointerdown', onPointerDown);
  }, [open]);

  const onKeyDown = (e) => {
    if (e.key === 'Escape' && open) {
      e.stopPropagation();
      setOpen(false);
      buttonRef.current?.focus();
    }
  };

  // relatedTarget is null when focus goes nowhere (a tap on panel text); outside taps are handled above.
  const onBlur = (e) => {
    if (e.relatedTarget && !wrapperRef.current?.contains(e.relatedTarget)) setOpen(false);
  };

  return (
    <div ref={wrapperRef} className={className} onKeyDown={onKeyDown} onBlur={onBlur}>
      <button
        ref={buttonRef}
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-controls={panelId}
        aria-label={LABEL}
        title={LABEL}
        className="inline-flex items-center justify-center w-9 h-9 shrink-0 rounded-control border border-control text-graphite hover:bg-ink-wash transition-colors"
      >
        <SlidersHorizontal className="w-4 h-4" aria-hidden="true" />
      </button>
      {open && (
        <div
          id={panelId}
          role="group"
          aria-label={LABEL}
          className="absolute inset-x-4 sm:left-auto sm:right-6 sm:w-80 top-full mt-2 rounded-control border border-rule bg-paper p-3 flex flex-col gap-3"
        >
          <AdminTokenField />
          <div className="flex items-center gap-2 text-13 text-slate">
            <span>Backend:</span>
            <HealthBadge />
          </div>
        </div>
      )}
    </div>
  );
};

export default TopBarMenu;
