import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Fonts used above the fold on the landing page: Mukta 400 (body text) and Newsreader (headings).
// Vite hashes asset names, so the preload links are added to index.html after bundling.
const PRELOAD_FONTS = [/^mukta-latin-400-normal-.+\.woff2$/, /^newsreader-latin-wght-normal-.+\.woff2$/]

const preloadFonts = () => {
  let base = '/'
  return {
    name: 'preload-fonts',
    apply: 'build',
    configResolved(config) {
      base = config.base
    },
    transformIndexHtml: {
      order: 'post',
      handler(html, { bundle }) {
        return PRELOAD_FONTS.map((pattern) => {
          const file = Object.keys(bundle).find((name) => pattern.test(name.split('/').pop()))
          if (!file) throw new Error(`preload-fonts: no bundled font matches ${pattern}`)
          return {
            tag: 'link',
            attrs: { rel: 'preload', href: `${base}${file}`, as: 'font', type: 'font/woff2', crossorigin: true },
            injectTo: 'head',
          }
        })
      },
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), preloadFonts()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
