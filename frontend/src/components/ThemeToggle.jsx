import React from 'react';
import { Sun, Moon } from 'lucide-react';
import { useTheme } from '../theme';

const ThemeToggle = () => {
  const { theme, toggleTheme } = useTheme();
  const label = theme === 'light' ? 'Switch to dark theme' : 'Switch to light theme';

  return (
    <button
      type="button"
      onClick={toggleTheme}
      className="inline-flex items-center justify-center w-9 h-9 shrink-0 rounded-control border border-control text-graphite hover:bg-ink-wash transition-colors"
      aria-label={label}
      title={label}
    >
      {theme === 'light' ? (
        <Moon className="w-4 h-4" aria-hidden="true" />
      ) : (
        <Sun className="w-4 h-4" aria-hidden="true" />
      )}
    </button>
  );
};

export default ThemeToggle;
