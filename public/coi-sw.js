/**
 * Cross-origin isolation on a static host.
 *
 * onnxruntime-web can only use multi-threaded WASM when `SharedArrayBuffer` exists, and
 * that needs COOP/COEP response headers. GitHub Pages serves static files and sends no
 * custom headers, so the deployed app was stuck on a single thread — and it is not only
 * the WASM backend that suffers: ORT assigns shape-related ops to CPU even on the WebGPU
 * path, so those kernels were single-threaded too. Measured on the deployed site, that was
 * 3.1 tok/s against 14.5 locally, where the dev server does send the headers.
 *
 * This worker re-serves every response with the isolation headers, which the browser then
 * honours. It caches nothing and rewrites no bodies.
 *
 * `require-corp` rather than `credentialless`: every asset this app loads — the model, the
 * tokenizer, the ONNX runtime, the recorded run — is same-origin, so there is nothing to
 * relax for, and `require-corp` is the variant Safari supports.
 */

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.cache === 'only-if-cached' && request.mode !== 'same-origin') return;

  event.respondWith(
    fetch(request)
      .then((response) => {
        if (response.status === 0) return response;

        const headers = new Headers(response.headers);
        headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
        headers.set('Cross-Origin-Opener-Policy', 'same-origin');
        headers.set('Cross-Origin-Resource-Policy', 'same-origin');

        return new Response(response.body, {
          status: response.status,
          statusText: response.statusText,
          headers,
        });
      })
      .catch((error) => {
        console.error('[coi-sw]', error);
        throw error;
      }),
  );
});
