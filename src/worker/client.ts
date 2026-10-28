/**
 * Creates the inference worker, or the recorded stand-in when the app runs in demo mode.
 *
 * Demo mode is on when the URL carries `?demo=1`, or when the viewport is too small to
 * justify an 89 MB download. Playwright uses the same switch, which is why CI never has to
 * fetch the weights.
 */
import type { MainToWorker, WorkerToMain } from '../lib/protocol';

export interface WorkerLike {
  postMessage(message: MainToWorker, transfer?: Transferable[]): void;
  addEventListener(listener: (event: MessageEvent<WorkerToMain>) => void): void;
  terminate(): void;
}

export interface WorkerHandle {
  worker: WorkerLike;
  demoMode: boolean;
}

/**
 * Where the weights live. `VITE_MODEL_URL` overrides it, which is how you would point the
 * app at a Hugging Face repo, an R2 bucket or your own server instead. In dev the Vite
 * middleware in `vite.config.ts` serves them from `.devmodel/`.
 */
const RELEASE_MODEL_URL =
  'https://github.com/jaworskig00/inside-the-llm/releases/download/model-v1/model.onnx';

/** Bumped whenever the exported graph changes, so stale weights are never reused. */
const CACHE_KEY = 'inside-the-llm-model-v1';

export function isDemoRequested(): boolean {
  const params = new URLSearchParams(window.location.search);
  if (params.get('demo') === '1') return true;
  if (params.get('demo') === '0') return false;
  return window.matchMedia('(max-width: 1023px)').matches;
}

export async function createWorker(): Promise<WorkerHandle> {
  const base = import.meta.env.BASE_URL;
  const absolute = (path: string) =>
    new URL(path.replace(/^\//, ''), new URL(base, window.location.href)).href;

  if (isDemoRequested()) {
    const { createMockWorker } = await import('./mockWorker');
    return { worker: createMockWorker(absolute('demo-run.json')), demoMode: true };
  }

  const native = new Worker(new URL('./inference.worker.ts', import.meta.url), {
    type: 'module',
    name: 'inside-the-llm',
  });

  const worker: WorkerLike = {
    postMessage: (message, transfer) =>
      transfer ? native.postMessage(message, transfer) : native.postMessage(message),
    addEventListener: (listener) => native.addEventListener('message', listener),
    terminate: () => native.terminate(),
  };

  return { worker, demoMode: false };
}

/** `?backend=wasm` forces the CPU path, so the fallback can be tried on purpose. */
function forcedBackend(): 'wasm' | undefined {
  return new URLSearchParams(window.location.search).get('backend') === 'wasm' ? 'wasm' : undefined;
}

export function initMessage(): MainToWorker {
  const base = import.meta.env.BASE_URL;
  const absolute = (path: string) => new URL(path, new URL(base, window.location.href)).href;
  const modelUrl =
    import.meta.env.VITE_MODEL_URL ??
    (import.meta.env.DEV ? absolute('model/model.onnx') : RELEASE_MODEL_URL);

  const force = forcedBackend();

  return {
    type: 'init',
    modelUrl,
    ...(force ? { forceBackend: force } : {}),
    tokenizerBase: absolute('model/tokenizer/'),
    metaUrl: absolute('model/model-meta.json'),
    ortBase: absolute('ort/'),
    cacheKey: CACHE_KEY,
  };
}
