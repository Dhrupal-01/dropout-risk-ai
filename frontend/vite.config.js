import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Fonts used above the fold on first paint of every route: Mukta 400 (body text) and Newsreader (headings).
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
          // The href must equal the @font-face src URL, or the browser fetches the font twice.
          const href = `${base}${file}`
          const css = Object.values(bundle).filter((asset) => asset.fileName.endsWith('.css'))
          if (!css.some((asset) => String(asset.source).includes(`url(${href})`))) {
            throw new Error(`preload-fonts: no @font-face src is url(${href})`)
          }
          return {
            tag: 'link',
            // Fonts are always fetched in CORS anonymous mode, so the preload needs crossorigin too.
            attrs: { rel: 'preload', href, as: 'font', type: 'font/woff2', crossorigin: true },
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
