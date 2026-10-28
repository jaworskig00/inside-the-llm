/// <reference lib="webworker" />
/**
 * The inference worker: owns the model, runs generation, and answers the UI's questions
 * about alternative branches and re-shaped distributions.
 *
 * Nothing here touches the DOM, and every reply is one of the variants declared in
 * `src/lib/protocol.ts`.
 */
import {
  DEFAULT_SAMPLING,
  LIMITS,
  type Candidate,
  type LensLayer,
  type MainToWorker,
  type SamplingConfig,
  type StepRecord,
  type TokenInfo,
  type WorkerToMain,
} from '../lib/protocol';
import { makeRng, sampleFrom, shapeDistribution, topIndices } from '../lib/sampling';
import { Engine, type ForwardResult } from './engine';

const ctx = globalThis as unknown as {
  postMessage: (message: WorkerToMain, transfer?: Transferable[]) => void;
  onmessage: ((event: MessageEvent<MainToWorker>) => void) | null;
};

const engine = new Engine();

/** Token ids of the prompt for the run in flight — the tree expands relative to these. */
let promptIds: number[] = [];
/** Full logits per step, kept so the sampling sliders can be re-applied exactly. */
const stepLogits = new Map<number, Float32Array>();
let cancelRequested = false;
let busy = false;

function send(message: WorkerToMain, transfer?: Transferable[]): void {
  ctx.postMessage(message, transfer);
}

function fail(error: unknown, fatal: boolean): void {
  const message = error instanceof Error ? error.message : String(error);
  send({ type: 'error', message, fatal });
}

async function init(message: Extract<MainToWorker, { type: 'init' }>): Promise<void> {
  const started = performance.now();

  send({
    type: 'progress',
    phase: 'tokenizer',
    loaded: 0,
    total: 0,
    label: 'Loading tokenizer',
    fromCache: false,
  });
  const meta = await engine.loadMeta(message.metaUrl);
  await engine.loadTokenizer(message.tokenizerBase);

  let cached = false;
  const weights = await engine.loadWeights(message.modelUrl, message.cacheKey, (event) => {
    cached = event.fromCache;
    send({
      type: 'progress',
      phase: 'model',
      loaded: event.loaded,
      total: event.total,
      label: event.fromCache ? 'Reading cached weights' : 'Downloading DistilGPT-2',
      fromCache: event.fromCache,
    });
  });

  send({
    type: 'progress',
    phase: 'session',
    loaded: 1,
    total: 1,
    label: 'Starting runtime',
    fromCache: cached,
  });
  const backend = await engine.createSession(weights, message.ortBase, message.forceBackend);

  // One throwaway pass: it compiles the WebGPU shaders so the first real token is not
  // ten times slower than every token after it.
  send({
    type: 'progress',
    phase: 'warmup',
    loaded: 1,
    total: 1,
    label: 'Warming up kernels',
    fromCache: cached,
  });
  await engine.forward(engine.encode('Hello world').ids.slice(0, 4));

  send({ type: 'ready', backend, meta, loadMs: performance.now() - started, fromCache: cached });
}

/** exp() of the log-probs the graph produced, as `Candidate` records. */
function lensFromForward(result: ForwardResult, layers: number, topK: number): LensLayer[] {
  const out: LensLayer[] = [];
  for (let layer = 0; layer <= layers; layer += 1) {
    const candidates: Candidate[] = [];
    for (let k = 0; k < topK; k += 1) {
      const offset = layer * topK + k;
      const id = result.lensIndices[offset] as number;
      candidates.push({
        id,
        text: engine.decodeToken(id),
        prob: Math.exp(result.lensValues[offset] as number),
      });
    }
    out.push({ layer, candidates });
  }
  return out;
}

function hiddenNormsFromForward(result: ForwardResult, layers: number): number[][] {
  const rows: number[][] = [];
  for (let layer = 0; layer <= layers; layer += 1) {
    const row: number[] = [];
    for (let token = 0; token < result.tokens; token += 1) {
      row.push(result.hiddenNorms[layer * result.tokens + token] as number);
    }
    rows.push(row);
  }
  return rows;
}

function toCandidates(logits: Float32Array, ids: number[]): Candidate[] {
  // Softmax over the full vocabulary, so the reported probabilities are the real ones.
  let max = -Infinity;
  for (let i = 0; i < logits.length; i += 1) {
    const v = logits[i] as number;
    if (v > max) max = v;
  }
  let sum = 0;
  for (let i = 0; i < logits.length; i += 1) sum += Math.exp((logits[i] as number) - max);

  return ids.map((id) => ({
    id,
    text: engine.decodeToken(id),
    prob: Math.exp((logits[id] as number) - max) / sum,
  }));
}

function shapedCandidates(logits: Float32Array, sampling: SamplingConfig) {
  const shaped = shapeDistribution(logits, {
    temperature: sampling.temperature,
    topK: sampling.topK,
    topP: sampling.topP,
    limit: LIMITS.rawTopCandidates,
  });
  return {
    candidates: shaped.candidates.map((candidate) => ({
      id: candidate.id,
      text: engine.decodeToken(candidate.id),
      prob: candidate.prob,
    })),
    kept: shaped.kept,
  };
}

async function generate(message: Extract<MainToWorker, { type: 'generate' }>): Promise<void> {
  const { runId, maxNewTokens } = message;
  const sampling = message.sampling ?? DEFAULT_SAMPLING;
  const meta = engine.meta;
  if (!meta) throw new Error('model metadata is not loaded');

  cancelRequested = false;
  stepLogits.clear();

  const encoded = message.promptTokenIds ?? engine.encode(message.prompt).ids;
  if (encoded.length === 0) throw new Error('the prompt tokenizes to nothing — type some text');
  promptIds = encoded.slice(0, LIMITS.maxPromptTokens);

  const ids = [...promptIds];
  const random = makeRng(sampling.seed);
  const runStarted = performance.now();

  // First pass over the prompt alone: this is what gives every prompt token a probability.
  let result = await engine.forward(ids);

  const promptTokens: TokenInfo[] = promptIds.map((id, index) => ({
    index,
    id,
    text: engine.decodeToken(id),
    source: 'prompt' as const,
    prob: index === 0 ? null : Math.exp(result.tokenLogprobs[index] as number),
  }));
  send({ type: 'prompt', runId, tokens: promptTokens });

  const cap = Math.min(maxNewTokens, LIMITS.maxNewTokens);
  let generated = 0;

  for (let step = 0; step < cap; step += 1) {
    if (cancelRequested) {
      send({ type: 'cancelled', runId });
      return;
    }

    const logits = result.logits;
    stepLogits.set(step, logits);

    const rawIds = topIndices(logits, LIMITS.rawTopCandidates);
    const rawTop = toCandidates(logits, rawIds);
    const shaped = shapedCandidates(logits, sampling);

    const chosenId = sampleFrom(
      shaped.candidates.map((c) => ({ id: c.id, prob: c.prob })),
      random,
    );
    const chosen: Candidate = {
      id: chosenId,
      text: engine.decodeToken(chosenId),
      prob: rawTop.find((c) => c.id === chosenId)?.prob ?? probOf(logits, chosenId),
    };

    const record: StepRecord = {
      step,
      position: ids.length,
      chosen,
      rawTop,
      shapedTop: shaped.candidates,
      keptCandidates: shaped.kept,
      lens: lensFromForward(result, meta.numLayers, meta.lensTopK),
      attention: {
        layers: meta.numLayers,
        heads: meta.numHeads,
        tokens: result.tokens,
        data: result.attention,
      },
      hiddenNorms: hiddenNormsFromForward(result, meta.numLayers),
      contextLength: result.tokens,
      ms: result.ms,
    };

    const token: TokenInfo = {
      index: ids.length,
      id: chosenId,
      text: chosen.text,
      source: 'generated',
      prob: chosen.prob,
    };

    // The attention buffer is transferred rather than copied: 1.5 MB per step at the
    // longest context this app allows, and structured cloning it would be wasteful.
    send({ type: 'step', runId, record, token }, [record.attention.data.buffer]);

    ids.push(chosenId);
    generated += 1;

    if (ids.length >= meta.maxPositions) break;
    if (step + 1 >= cap) break;

    result = await engine.forward(ids);
  }

  const totalMs = performance.now() - runStarted;
  send({
    type: 'done',
    runId,
    totalMs,
    tokensPerSecond: generated > 0 ? generated / (totalMs / 1000) : 0,
  });
}

function probOf(logits: Float32Array, id: number): number {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i += 1) {
    const v = logits[i] as number;
    if (v > max) max = v;
  }
  let sum = 0;
  for (let i = 0; i < logits.length; i += 1) sum += Math.exp((logits[i] as number) - max);
  return Math.exp((logits[id] as number) - max) / sum;
}

/** One forward pass on `prompt + path`, returning the branches available at that point. */
async function expand(message: Extract<MainToWorker, { type: 'expand' }>): Promise<void> {
  const ids = [...promptIds, ...message.path];
  const result = await engine.forward(ids);
  const branchIds = topIndices(result.logits, message.branches);
  send({
    type: 'expanded',
    requestId: message.requestId,
    expansion: {
      path: message.path,
      children: toCandidates(result.logits, branchIds),
    },
  });
}

/** Re-applies the sampling sliders to a step that already ran. Exact, not approximate. */
function rescore(message: Extract<MainToWorker, { type: 'rescore' }>): void {
  const logits = stepLogits.get(message.step);
  if (!logits) return;
  const shaped = shapedCandidates(logits, message.sampling);
  send({
    type: 'rescored',
    runId: message.runId,
    step: message.step,
    shapedTop: shaped.candidates,
    keptCandidates: shaped.kept,
  });
}

ctx.onmessage = (event: MessageEvent<MainToWorker>) => {
  const message = event.data;

  if (message.type === 'cancel') {
    cancelRequested = true;
    return;
  }

  if (message.type === 'rescore') {
    try {
      rescore(message);
    } catch (error) {
      fail(error, false);
    }
    return;
  }

  void (async () => {
    if (busy && message.type === 'generate') cancelRequested = true;
    busy = true;
    try {
      if (message.type === 'init') await init(message);
      else if (message.type === 'generate') await generate(message);
      else if (message.type === 'expand') await expand(message);
    } catch (error) {
      fail(error, message.type === 'init');
    } finally {
      busy = false;
    }
  })();
};
