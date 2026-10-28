/**
 * Everything that touches the model. Lives in the worker only, so the main thread never
 * blocks on inference or on an 89 MB download.
 *
 * Two deliberate choices here:
 *
 * - **onnxruntime-web is loaded at runtime, not bundled.** ORT ships ~55 MB of prebuilt
 *   WASM variants; letting a bundler follow its `new URL(..., import.meta.url)` calls emits
 *   every one of them into `dist/`. Importing the prebuilt ESM bundle from our own
 *   `public/ort/` directory keeps the build small and the deployed asset set explicit.
 * - **`@huggingface/tokenizers` rather than the full transformers.js.** The exported graph
 *   has six outputs and is not an `AutoModelForCausalLM`, so transformers.js could never
 *   drive it; pulling it in just for tokenization would add a second copy of ORT. This is
 *   the same tokenizer implementation, as a standalone package, and it gives back the raw
 *   BPE token strings the token strip wants to show.
 */
import { Tokenizer } from '@huggingface/tokenizers';
import type * as Ort from 'onnxruntime-web';
import type { Backend, ModelMeta } from '../lib/protocol';

export interface ProgressEvent {
  loaded: number;
  total: number;
  fromCache: boolean;
}

/** Raw tensors for one forward pass, in the shapes the exported graph produces. */
export interface ForwardResult {
  /** `(V)` next-token logits at the last position. */
  logits: Float32Array;
  /** `(layers * heads * T * T)` post-softmax attention weights. */
  attention: Float32Array;
  /** `((layers + 1) * T)` residual-stream L2 norms. */
  hiddenNorms: Float32Array;
  /** `(T)` log p(token_t | token_<t); index 0 is meaningless. */
  tokenLogprobs: Float32Array;
  /** `((layers + 1) * K)` logit-lens log-probs. */
  lensValues: Float32Array;
  /** `((layers + 1) * K)` logit-lens token ids. */
  lensIndices: Int32Array;
  tokens: number;
  ms: number;
}

type OrtModule = typeof Ort;

export class Engine {
  private ort: OrtModule | null = null;
  private session: Ort.InferenceSession | null = null;
  private tokenizer: Tokenizer | null = null;
  private decodeCache = new Map<number, string>();

  meta: ModelMeta | null = null;
  backend: Backend = 'wasm';

  async loadMeta(url: string): Promise<ModelMeta> {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`could not read model metadata (${response.status})`);
    const meta = (await response.json()) as ModelMeta;
    this.meta = meta;
    return meta;
  }

  /** `tokenizer.json` and `tokenizer_config.json` are served from this app's own origin. */
  async loadTokenizer(base: string): Promise<void> {
    const prefix = base.endsWith('/') ? base : `${base}/`;
    const [tokenizerJson, configJson] = await Promise.all([
      fetchJson(`${prefix}tokenizer.json`),
      fetchJson(`${prefix}tokenizer_config.json`),
    ]);
    this.tokenizer = new Tokenizer(tokenizerJson, configJson);
  }

  /**
   * Downloads the weights with byte-level progress and keeps them in the Cache API, so a
   * second visit starts without touching the network.
   */
  async loadWeights(
    url: string,
    cacheKey: string,
    onProgress: (event: ProgressEvent) => void,
  ): Promise<ArrayBuffer> {
    const cache = await openCache(cacheKey);

    if (cache) {
      const hit = await cache.match(url);
      if (hit) {
        const buffer = await hit.arrayBuffer();
        onProgress({ loaded: buffer.byteLength, total: buffer.byteLength, fromCache: true });
        return buffer;
      }
    }

    const response = await fetch(url);
    if (!response.ok || !response.body) {
      throw new Error(`model download failed (${response.status} ${response.statusText})`);
    }

    const total = Number(response.headers.get('content-length') ?? 0);
    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    let loaded = 0;

    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      onProgress({ loaded, total, fromCache: false });
    }

    const bytes = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.byteLength;
    }

    if (cache) {
      try {
        await cache.put(
          url,
          new Response(bytes.slice(), {
            headers: {
              'content-type': 'application/octet-stream',
              'content-length': String(bytes.byteLength),
            },
          }),
        );
      } catch {
        // A full or unavailable cache is not a reason to fail the run.
      }
    }

    return bytes.buffer as ArrayBuffer;
  }

  /** Creates the session on WebGPU when the adapter allows it, otherwise on WASM. */
  async createSession(weights: ArrayBuffer, ortBase: string): Promise<Backend> {
    const prefix = ortBase.endsWith('/') ? ortBase : `${ortBase}/`;
    const useWebgpu = await webgpuAvailable();

    // Loaded as a plain URL so the bundler never sees - and never copies - ORT's WASM set.
    // The WebGPU runtime carries the JSEP kernels and is twice the size of the plain WASM
    // one, so a machine without a GPU adapter is not made to download it.
    const entry = useWebgpu ? 'ort.webgpu.bundle.min.mjs' : 'ort.wasm.bundle.min.mjs';
    const ort = (await import(/* @vite-ignore */ `${prefix}${entry}`)) as OrtModule;
    this.ort = ort;

    ort.env.wasm.wasmPaths = prefix;
    // Multi-threaded WASM needs SharedArrayBuffer, which needs COOP/COEP headers that
    // GitHub Pages cannot send. Single thread is the safe default there.
    ort.env.wasm.numThreads = globalThis.crossOriginIsolated
      ? Math.min(4, navigator.hardwareConcurrency || 1)
      : 1;
    ort.env.logLevel = 'error';

    const options: Ort.InferenceSession.SessionOptions = {
      graphOptimizationLevel: 'all',
      executionMode: 'sequential',
    };

    if (useWebgpu) {
      try {
        this.session = await ort.InferenceSession.create(weights, {
          ...options,
          executionProviders: ['webgpu'],
        });
        this.backend = 'webgpu';
        return 'webgpu';
      } catch (error) {
        console.warn('[inside-the-llm] WebGPU session failed, falling back to WASM', error);
      }
    }

    this.session = await ort.InferenceSession.create(weights, {
      ...options,
      executionProviders: ['wasm'],
    });
    this.backend = 'wasm';
    return 'wasm';
  }

  /** Token ids plus the raw BPE strings, which is what the token chips display. */
  encode(text: string): { ids: number[]; tokens: string[] } {
    const encoded = this.requireTokenizer().encode(text, { add_special_tokens: false });
    encoded.ids.forEach((id, index) => {
      const token = encoded.tokens[index];
      if (token !== undefined && !this.decodeCache.has(id)) this.decodeCache.set(id, token);
    });
    return { ids: encoded.ids, tokens: encoded.tokens };
  }

  /** The raw vocabulary entry for one id, cached: the views ask for these constantly. */
  decodeToken(id: number): string {
    const cached = this.decodeCache.get(id);
    if (cached !== undefined) return cached;
    const text = this.requireTokenizer().id_to_token(id) ?? `<${id}>`;
    this.decodeCache.set(id, text);
    return text;
  }

  decode(ids: number[]): string {
    return this.requireTokenizer().decode(ids, { skip_special_tokens: true });
  }

  async forward(ids: number[]): Promise<ForwardResult> {
    const session = this.session;
    const ort = this.ort;
    if (!session || !ort) throw new Error('session is not ready');

    const started = performance.now();
    const input = new ort.Tensor('int32', Int32Array.from(ids), [1, ids.length]);
    const output = await session.run({ input_ids: input });
    const ms = performance.now() - started;

    return {
      // Sliced, because the step records keep these arrays alive and ORT may reuse buffers.
      logits: readFloat(output, 'logits').slice(),
      attention: readFloat(output, 'attentions').slice(),
      hiddenNorms: readFloat(output, 'hidden_norms').slice(),
      tokenLogprobs: readFloat(output, 'token_logprobs').slice(),
      lensValues: readFloat(output, 'lens_values').slice(),
      lensIndices: readInt(output, 'lens_indices').slice(),
      tokens: ids.length,
      ms,
    };
  }

  private requireTokenizer(): Tokenizer {
    if (!this.tokenizer) throw new Error('tokenizer is not ready');
    return this.tokenizer;
  }
}

async function fetchJson(url: string): Promise<Record<string, unknown>> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`could not load ${url} (${response.status})`);
  return (await response.json()) as Record<string, unknown>;
}

function readFloat(output: Ort.InferenceSession.OnnxValueMapType, name: string): Float32Array {
  const tensor = output[name];
  if (!tensor) throw new Error(`model output "${name}" is missing`);
  return tensor.data as Float32Array;
}

function readInt(output: Ort.InferenceSession.OnnxValueMapType, name: string): Int32Array {
  const tensor = output[name];
  if (!tensor) throw new Error(`model output "${name}" is missing`);
  const data = tensor.data;
  if (data instanceof BigInt64Array) return Int32Array.from(data, (value) => Number(value));
  return data as Int32Array;
}

async function openCache(key: string): Promise<Cache | null> {
  if (!('caches' in globalThis)) return null;
  try {
    return await caches.open(key);
  } catch {
    // Private windows and some embedded webviews refuse the Cache API.
    return null;
  }
}

async function webgpuAvailable(): Promise<boolean> {
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  if (!gpu) return false;
  try {
    const adapter = await gpu.requestAdapter();
    return adapter !== null;
  } catch {
    return false;
  }
}
