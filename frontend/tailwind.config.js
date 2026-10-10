/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: ['selector', '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        paper: 'var(--paper)',
        rule: 'var(--rule)',
        ink: {
          DEFAULT: 'var(--ink)',
          hover: 'var(--ink-hover)',
          wash: 'var(--ink-wash)',
        },
        graphite: 'var(--graphite)',
        // Merged with Tailwind's slate scale, so slate-100 etc. keep working
        slate: { DEFAULT: 'var(--slate)' },
        'margin-red': 'var(--margin-red)',
        marigold: 'var(--marigold)',
        'on-ink': 'var(--on-ink)',
        'on-marigold': 'var(--on-marigold)',
        control: 'var(--control-border)',
        tier: {
          low: 'var(--tier-low)',
          medium: 'var(--tier-medium)',
          high: 'var(--tier-high)',
        },

        // Legacy names used by the current app pages until their restyle
        page: 'var(--surface-page)',
        card: 'var(--surface-card)',
        subtle: 'var(--surface-subtle)',
        hover: 'var(--surface-hover)',
        primary: 'var(--text-primary)',
        secondary: 'var(--text-secondary)',
        muted: 'var(--text-muted)',
        border: 'var(--border)',
        accent: {
          DEFAULT: 'var(--accent)',
          hover: 'var(--accent-hover)',
          soft: 'var(--accent-soft)',
        },
        risk: {
          low: 'var(--tier-low)',
          medium: 'var(--tier-medium)',
          high: 'var(--tier-high)',
        },
        shap: {
          increase: 'var(--shap-increase)',
          decrease: 'var(--shap-decrease)',
          neutral: 'var(--shap-neutral)',
        }
      },
      fontFamily: {
        sans: ['Mukta', 'system-ui', '-apple-system', '"Segoe UI"', 'sans-serif'],
        display: ['"Newsreader Variable"', 'Georgia', 'serif'],
        hand: ['Kalam', 'cursive'],
      },
      // Spec type scale (px): 13, 15, 17 body, 20, 24, 32, 44, 60; hero is 40 on mobile
      fontSize: {
        13: ['0.8125rem', { lineHeight: '1.5' }],
        15: ['0.9375rem', { lineHeight: '1.5' }],
        17: ['1.0625rem', { lineHeight: '1.55' }],
        20: ['1.25rem', { lineHeight: '1.45' }],
        24: ['1.5rem', { lineHeight: '1.3' }],
        32: ['2rem', { lineHeight: '1.2' }],
        40: ['2.5rem', { lineHeight: '1.1' }],
        44: ['2.75rem', { lineHeight: '1.1' }],
        60: ['3.75rem', { lineHeight: '1.05' }],
      },
      letterSpacing: {
        display: '-0.015em',
      },
      borderRadius: {
        control: '6px',
        panel: '12px',
      },
      maxWidth: {
        page: '1200px',
        measure: '68ch',
      },
    },
  },
  plugins: [],
}
