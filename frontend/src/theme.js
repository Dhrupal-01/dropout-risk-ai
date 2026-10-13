import { useEffect, useState } from 'react';

// index.html sets data-theme before first paint from this key; localStorage holds only this preference.
const THEME_KEY = 'dropoutguard-theme';

const currentTheme = () =>
  document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light';

export const useTheme = () => {
  const [theme, setTheme] = useState(currentTheme);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      // Storage blocked (private mode etc.): the theme still applies for this page view.
    }
  }, [theme]);

  const toggleTheme = () => setTheme((prev) => (prev === 'light' ? 'dark' : 'light'));
  return { theme, toggleTheme };
};
