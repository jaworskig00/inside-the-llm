/**
 * The contract between the UI and the inference worker.
 *
 * Everything that crosses the worker boundary is declared here, once, so that a change to
 * the payload shape is a type error in both directions instead of a runtime surprise.
 */

/** Static facts about the exported graph, read from `model/model-meta.json`. */
export interface ModelMeta {
  modelId: string;
  numLayers: number;
  numHeads: number;
  hiddenSize: number;
  vocabSize: number;
  maxPositions: number;
  lensTopK: number;
  opset: number;
}

export type Backend = 'webgpu' | 'wasm';

export interface SamplingConfig {
  /** 0 means greedy; the UI never sends a negative value. */
  temperature: number;
  /** 0 disables top-k filtering. */
  topK: number;
  /** 1 disables nucleus filtering. */
  topP: number;
  /** Fixed so that the same prompt and settings replay identically. */
  seed: number;
}

export interface Candidate {
  id: number;
  text: string;
  prob: number;
}

export type TokenSource = 'prompt' | 'generated';

export interface TokenInfo {
  index: number;
  id: number;
  text: string;
  source: TokenSource;
  /**
   * Probability the model assigned to this token given everything before it.
   * `null` for the very first token, which has no context to be predicted from.
   */
  prob: number | null;
}

/** Logit-lens read-out for one layer: what the model would say if it stopped here. */
export interface LensLayer {
  layer: number;
  candidates: Candidate[];
}

/**
 * Attention weights for one generation step, flattened for cheap transfer.
 * Index of (layer, head, query, key) is `((layer * heads + head) * tokens + query) * tokens + key`.
 */
export interface AttentionPack {
  layers: number;
  heads: number;
  tokens: number;
  data: Float32Array;
}

/** Everything the views need about a single generated token. */
export interface StepRecord {
  step: number;
  /** Index in the token list that this step produced. */
  position: number;
  chosen: Candidate;
  /** Top of the raw (unmodified) next-token distribution. */
  rawTop: Candidate[];
  /** Top of the distribution after temperature / top-k / top-p were applied. */
  shapedTop: Candidate[];
  /** How many tokens survived the top-k / top-p filter. */
  keptCandidates: number;
  lens: LensLayer[];
  attention: AttentionPack;
  /** `[layer][token]` L2 norm of the residual stream. */
  hiddenNorms: number[][];
  contextLength: number;
  ms: number;
}

export interface TreeChild {
  id: number;
  text: string;
  prob: number;
}

/** One lazily expanded node of the alternatives tree. */
export interface TreeExpansion {
  /** Token ids from the start of the sequence up to and including this node. */
  path: number[];
  children: TreeChild[];
}

export type LoadPhase = 'idle' | 'tokenizer' | 'model' | 'session' | 'warmup' | 'ready';

export type MainToWorker =
  | {
      type: 'init';
      modelUrl: string;
      tokenizerBase: string;
      metaUrl: string;
      ortBase: string;
      cacheKey: string;
    }
  | {
      type: 'generate';
      runId: number;
      prompt: string;
      promptTokenIds?: number[];
      maxNewTokens: number;
      sampling: SamplingConfig;
    }
  | { type: 'expand'; requestId: number; path: number[]; branches: number }
  | { type: 'rescore'; runId: number; step: number; sampling: SamplingConfig }
  | { type: 'cancel' };

export type WorkerToMain =
  | {
      type: 'progress';
      phase: LoadPhase;
      loaded: number;
      total: number;
      label: string;
      fromCache: boolean;
    }
  | { type: 'ready'; backend: Backend; meta: ModelMeta; loadMs: number; fromCache: boolean }
  | { type: 'prompt'; runId: number; tokens: TokenInfo[] }
  | { type: 'step'; runId: number; record: StepRecord; token: TokenInfo }
  | { type: 'done'; runId: number; totalMs: number; tokensPerSecond: number }
  | { type: 'expanded'; requestId: number; expansion: TreeExpansion }
  | {
      type: 'rescored';
      runId: number;
      step: number;
      shapedTop: Candidate[];
      keptCandidates: number;
    }
  | { type: 'cancelled'; runId: number }
  | { type: 'error'; message: string; fatal: boolean };

/** Hard limits, kept here so the worker and the UI cannot disagree about them. */
export const LIMITS = {
  maxPromptTokens: 64,
  maxNewTokens: 32,
  defaultNewTokens: 16,
  /** How much of the next-token distribution the worker ships per step. */
  rawTopCandidates: 24,
  /** Branching factor of the alternatives tree. */
  treeBranches: 5,
  /** Depth the tree is allowed to reach. */
  treeMaxDepth: 4,
} as const;

export const DEFAULT_SAMPLING: SamplingConfig = {
  temperature: 0.8,
  topK: 40,
  topP: 0.95,
  seed: 7,
};
