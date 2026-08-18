/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
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
          low: '#0ca30c',
          medium: '#fab219',
          high: '#d03b3b',
        },
        shap: {
          increase: 'var(--shap-increase)',
          decrease: 'var(--shap-decrease)',
          neutral: 'var(--shap-neutral)',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
