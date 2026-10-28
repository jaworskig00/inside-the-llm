import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// GitHub Pages serves the app from /<repo>/, every other target from the root.
// Swapping host (Cloudflare Pages, Hetzner, S3) is a one-line change here - see README.
const base = process.env.VITE_BASE ?? '/inside-the-llm/';

export default defineConfig({
  base,
  plugins: [react(), tailwindcss()],
  build: {
    target: 'esnext',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          d3: ['d3', 'd3-sankey'],
        },
      },
    },
  },
  worker: {
    format: 'es',
  },
  // onnxruntime-web ships prebuilt .wasm/.mjs artifacts; pre-bundling them breaks the
  // dynamic import of the WebGPU backend, so it stays out of optimizeDeps.
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  server: {
    headers: {
      // Lets onnxruntime-web use multi-threaded WASM locally. GitHub Pages cannot send
      // these headers, so the deployed build runs single-threaded WASM (WebGPU is unaffected).
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'credentialless',
    },
  },
});
