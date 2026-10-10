import React, { useEffect, useId, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { Search } from 'lucide-react';
import { getMentorQueue } from '../api/endpoints';
import TierLabel from './TierLabel';

const RESULT_LIMIT = 8;
const DEBOUNCE_MS = 200;

// Student search over GET /mentors/queue?search= (substring match on id or name).
// Combobox pattern: the input keeps focus, arrow keys move the active option, Enter opens it.
const SearchPalette = ({ open, onClose }) => {
  const navigate = useNavigate();
  const inputRef = useRef(null);
  const listId = useId();
  const [term, setTerm] = useState('');
  const [debounced, setDebounced] = useState('');
  const [active, setActive] = useState(0);

  useEffect(() => {
    const handle = setTimeout(() => setDebounced(term.trim()), DEBOUNCE_MS);
    return () => clearTimeout(handle);
  }, [term]);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  const results = useQuery({
    queryKey: ['student-search', debounced],
    queryFn: () => getMentorQueue({ search: debounced, limit: RESULT_LIMIT }),
    enabled: open && debounced.length > 0,
  });
  const items = results.data?.items ?? [];

  if (!open) return null;

  const close = () => {
    setTerm('');
    setDebounced('');
    setActive(0);
    onClose();
  };

  const openStudent = (student) => {
    close();
    navigate(`/app/students/${encodeURIComponent(student.student_id)}`);
  };

  const onKeyDown = (e) => {
    if (e.key === 'Escape') {
      e.preventDefault();
      close();
    } else if (e.key === 'ArrowDown' && items.length) {
      e.preventDefault();
      setActive((i) => (i + 1) % items.length);
    } else if (e.key === 'ArrowUp' && items.length) {
      e.preventDefault();
      setActive((i) => (i - 1 + items.length) % items.length);
    } else if (e.key === 'Enter' && items[active]) {
      e.preventDefault();
      openStudent(items[active]);
    } else if (e.key === 'Tab') {
      // Only the input is focusable inside the dialog; keep focus here.
      e.preventDefault();
    }
  };

  let status = null;
  if (!debounced) status = 'Type a student ID or name.';
  else if (results.isPending) status = 'Searching…';
  else if (results.isError) status = `Search failed. ${results.error?.message || ''}`;
  else if (!items.length) status = `No students match “${debounced}”.`;

  return (
    <div className="fixed inset-0 z-[60] flex items-start justify-center bg-black/40 px-4 pt-[12vh]" onClick={close}>
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Search students"
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-xl overflow-hidden rounded-panel border border-rule bg-paper"
      >
        <div className="flex items-center gap-2 border-b border-rule px-4">
          <Search className="w-4 h-4 shrink-0 text-slate" aria-hidden="true" />
          <input
            ref={inputRef}
            type="text"
            role="combobox"
            aria-expanded={items.length > 0}
            aria-controls={listId}
            aria-activedescendant={items[active] ? `${listId}-${active}` : undefined}
            aria-autocomplete="list"
            placeholder="Search students by ID or name"
            value={term}
            onChange={(e) => {
              setTerm(e.target.value);
              setActive(0);
            }}
            onKeyDown={onKeyDown}
            className="w-full bg-transparent py-3.5 text-17 text-graphite placeholder:text-slate focus:outline-none"
          />
          <kbd className="hidden sm:inline rounded-[4px] border border-rule px-1.5 text-13 text-slate">Esc</kbd>
        </div>

        <ul id={listId} role="listbox" aria-label="Matching students" className="max-h-[50vh] overflow-y-auto py-1">
          {items.map((student, i) => (
            <li
              key={student.student_id}
              id={`${listId}-${i}`}
              role="option"
              aria-selected={i === active}
              onMouseEnter={() => setActive(i)}
              onClick={() => openStudent(student)}
              className={`flex cursor-pointer items-center justify-between gap-3 px-4 py-2.5 ${i === active ? 'bg-ink-wash' : ''}`}
            >
              <span className="min-w-0">
                <span className="block truncate text-15 font-medium text-graphite">{student.name || student.student_id}</span>
                <span className="block truncate text-13 text-slate">
                  {student.name ? `${student.student_id}${student.department ? `, ${student.department}` : ''}` : student.department}
                </span>
              </span>
              <TierLabel tier={student.risk_tier} className="shrink-0 whitespace-nowrap text-13" />
            </li>
          ))}
        </ul>
        {status && (
          <p role="status" className="px-4 py-3 text-15 text-slate">
            {status}
          </p>
        )}
      </div>
    </div>
  );
};

export default SearchPalette;
