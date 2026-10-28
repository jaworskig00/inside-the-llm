/**
 * Registers the cross-origin isolation service worker (see `public/coi-sw.js`).
 *
 * Without isolation `SharedArrayBuffer` is absent, onnxruntime-web is pinned to one thread
 * and the deployed app runs roughly five times slower than it does behind a dev server
 * that sends the headers itself. The worker supplies those headers on a static host.
 *
 * Isolation only takes effect once the worker controls the page, so the first visit
 * reloads exactly once. The attempt is recorded in `sessionStorage`, so a browser that
 * refuses to isolate (or a host that already sends the headers) never ends up in a loop.
 */
const ATTEMPT_KEY = 'inside-the-llm:coi-reload';

export function ensureCrossOriginIsolation(): void {
  if (typeof window === 'undefined') return;

  // Already isolated: either a host that sends the headers, or the worker is live.
  if (window.crossOriginIsolated) {
    safeRemove(ATTEMPT_KEY);
    return;
  }

  // Service workers need a secure context; file:// and plain http are out.
  if (!window.isSecureContext || !('serviceWorker' in navigator)) return;

  void navigator.serviceWorker
    .register(new URL('coi-sw.js', document.baseURI).href, {
      scope: new URL('./', document.baseURI).pathname,
    })
    .then((registration) => {
      // One reload, and only one: if isolation still fails afterwards the app carries on
      // single-threaded rather than reloading forever.
      if (registration.active && !navigator.serviceWorker.controller) return reloadOnce();
      registration.addEventListener('updatefound', () => {
        const installing = registration.installing;
        if (!installing) return;
        installing.addEventListener('statechange', () => {
          if (installing.state === 'activated') reloadOnce();
        });
      });
    })
    .catch((error) => {
      console.warn('[inside-the-llm] cross-origin isolation unavailable', error);
    });
}

function reloadOnce(): void {
  if (safeGet(ATTEMPT_KEY)) return;
  safeSet(ATTEMPT_KEY, '1');
  window.location.reload();
}

function safeGet(key: string): string | null {
  try {
    return window.sessionStorage.getItem(key);
  } catch {
    return null;
  }
}

function safeSet(key: string, value: string): void {
  try {
    window.sessionStorage.setItem(key, value);
  } catch {
    // Private windows can refuse storage; a missed reload only costs performance.
  }
}

function safeRemove(key: string): void {
  try {
    window.sessionStorage.removeItem(key);
  } catch {
    // ignore
  }
}
