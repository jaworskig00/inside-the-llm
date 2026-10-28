import { createReadStream, existsSync, statSync } from 'node:fs';
import { resolve } from 'node:path';
import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

/**
 * Serves a local copy of the weights during `npm run dev` from `.devmodel/model.onnx`.
 *
 * The file lives outside `public/` on purpose: the deployed app fetches it from a GitHub
 * Release, and an 89 MB blob has no business being copied into `dist/` on every build.
 * Drop the exported `model.onnx` into `.devmodel/` to develop without the network.
 */
function devModelServer(): Plugin {
  const file = resolve(import.meta.dirname, '.devmodel/model.onnx');
  return {
    name: 'inside-the-llm:dev-model',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        if (!request.url?.endsWith('/model/model.onnx')) return next();
        if (!existsSync(file)) return next();
        response.setHeader('content-type', 'application/octet-stream');
        response.setHeader('content-length', String(statSync(file).size));
        createReadStream(file).pipe(response);
      });
    },
  };
}

// GitHub Pages serves the app from /<repo>/, every other target from the root.
// Swapping host (Cloudflare Pages, Hetzner, S3) is a one-line change here - see README.
const base = process.env.VITE_BASE ?? '/inside-the-llm/';

export default defineConfig({
  base,
  plugins: [react(), tailwindcss(), devModelServer()],
  build: {
    target: 'esnext',
    sourcemap: true,
    rollupOptions: {
      output: {
        // d3 barely changes between deploys; its own chunk stays in the browser cache
        // while the app code around it is replaced.
        manualChunks(id) {
          if (/node_modules[\\/]d3(-[a-z-]+)?[\\/]/.test(id)) return 'd3';
          return undefined;
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
