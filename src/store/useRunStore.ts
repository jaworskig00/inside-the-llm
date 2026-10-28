/**
 * One store for the whole app.
 *
 * Every view reads from here and nothing else, which is what keeps them linked: selecting
 * a step in the token strip moves the attention grid, the Sankey and the sampling chart,
 * because all three are derived from `selectedStep`.
 */
import { create } from 'zustand';
import {
  DEFAULT_SAMPLING,
  LIMITS,
  type Backend,
  type Candidate,
  type LoadPhase,
  type ModelMeta,
  type SamplingConfig,
  type StepRecord,
  type TokenInfo,
  type WorkerToMain,
} from '../lib/protocol';
import {
  applyExpansion,
  collapse,
  createRoot,
  findNode,
  markLoading,
  type TreeNode,
} from '../lib/tree';

export interface LoadProgress {
  loaded: number;
  total: number;
  label: string;
  fromCache: boolean;
}

export interface RunStats {
  totalMs: number;
  tokensPerSecond: number;
  loadMs: number;
}

export interface RunState {
  phase: LoadPhase;
  progress: LoadProgress;
  backend: Backend | null;
  meta: ModelMeta | null;
  error: string | null;
  fatal: boolean;
  demoMode: boolean;
  /** Whether this browser exposes a WebGPU adapter at all. Decides what the gate warns about. */
  gpuAvailable: boolean;

  prompt: string;
  maxNewTokens: number;
  sampling: SamplingConfig;

  runId: number;
  running: boolean;
  tokens: TokenInfo[];
  steps: StepRecord[];
  stats: RunStats;

  /** Index into `steps`; -1 before anything has been generated. */
  selectedStep: number;
  /** Index into `tokens` — the token whose attention the views show. */
  selectedToken: number;
  selectedLayer: number;
  /** `null` means "averaged over all heads". */
  selectedHead: number | null;
  playing: boolean;
  explain: boolean;
  /** The first-run walkthrough. Modal while open, so the keyboard shortcuts stand down. */
  tourOpen: boolean;
  tourStep: number;

  tree: TreeNode;
  /** requestId -> the path that request is expanding. */
  pendingExpansions: Record<number, number[]>;
  nextRequestId: number;

  setPrompt: (prompt: string) => void;
  setMaxNewTokens: (value: number) => void;
  setSampling: (patch: Partial<SamplingConfig>) => void;
  setSelectedStep: (step: number) => void;
  setSelectedToken: (index: number) => void;
  setSelectedLayer: (layer: number) => void;
  setSelectedHead: (head: number | null) => void;
  setPlaying: (playing: boolean) => void;
  toggleExplain: () => void;
  startTour: () => void;
  endTour: () => void;
  goToTourStep: (step: number) => void;
  stepBy: (delta: number) => void;
  beginRun: () => number;
  requestExpansion: (path: number[]) => number | null;
  collapseNode: (path: number[]) => void;
  dismissError: () => void;
  applyWorkerMessage: (message: WorkerToMain) => void;
}

const initialProgress: LoadProgress = { loaded: 0, total: 0, label: 'Starting', fromCache: false };

export const useRunStore = create<RunState>((set, get) => ({
  phase: 'idle',
  progress: initialProgress,
  backend: null,
  meta: null,
  error: null,
  fatal: false,
  demoMode: false,
  gpuAvailable: typeof navigator !== 'undefined' && 'gpu' in navigator,

  prompt: 'The lighthouse keeper had not spoken to anyone in',
  maxNewTokens: LIMITS.defaultNewTokens,
  sampling: { ...DEFAULT_SAMPLING },

  runId: 0,
  running: false,
  tokens: [],
  steps: [],
  stats: { totalMs: 0, tokensPerSecond: 0, loadMs: 0 },

  selectedStep: -1,
  selectedToken: 0,
  selectedLayer: 0,
  selectedHead: null,
  playing: false,
  explain: true,
  tourOpen: false,
  tourStep: 0,

  tree: createRoot(0),
  pendingExpansions: {},
  nextRequestId: 1,

  setPrompt: (prompt) => set({ prompt }),
  setMaxNewTokens: (value) =>
    set({ maxNewTokens: Math.max(1, Math.min(LIMITS.maxNewTokens, Math.round(value))) }),
  setSampling: (patch) => set((state) => ({ sampling: { ...state.sampling, ...patch } })),

  setSelectedStep: (step) => {
    const { steps } = get();
    if (steps.length === 0) return;
    const clamped = Math.max(0, Math.min(steps.length - 1, step));
    const record = steps[clamped];
    set({
      selectedStep: clamped,
      // Following the newly produced token is almost always what you want.
      selectedToken: record ? record.position : 0,
    });
  },

  setSelectedToken: (index) => {
    const { tokens } = get();
    if (tokens.length === 0) return;
    set({ selectedToken: Math.max(0, Math.min(tokens.length - 1, index)) });
  },

  setSelectedLayer: (layer) => set({ selectedLayer: Math.max(0, layer) }),
  setSelectedHead: (head) => set({ selectedHead: head }),
  setPlaying: (playing) => set({ playing }),
  toggleExplain: () => set((state) => ({ explain: !state.explain })),
  // Opening the tour stops playback: a moving target is a bad thing to point at.
  startTour: () => set({ tourOpen: true, tourStep: 0, playing: false }),
  endTour: () => set({ tourOpen: false }),
  goToTourStep: (step) => set({ tourStep: Math.max(0, step) }),

  stepBy: (delta) => {
    const { selectedStep, steps } = get();
    if (steps.length === 0) return;
    const next = Math.max(0, Math.min(steps.length - 1, selectedStep + delta));
    get().setSelectedStep(next);
  },

  beginRun: () => {
    const runId = get().runId + 1;
    set({
      runId,
      running: true,
      tokens: [],
      steps: [],
      selectedStep: -1,
      selectedToken: 0,
      playing: false,
      error: null,
      stats: { ...get().stats, totalMs: 0, tokensPerSecond: 0 },
      tree: createRoot(0),
      pendingExpansions: {},
    });
    return runId;
  },

  requestExpansion: (path) => {
    const state = get();
    if (path.length >= LIMITS.treeMaxDepth) return null;
    const node = findNode(state.tree, path);
    if (!node || node.status === 'loading') return null;
    const requestId = state.nextRequestId;
    set({
      nextRequestId: requestId + 1,
      pendingExpansions: { ...state.pendingExpansions, [requestId]: path },
      tree: markLoading(state.tree, path),
    });
    return requestId;
  },

  collapseNode: (path) => set((state) => ({ tree: collapse(state.tree, path) })),
  dismissError: () => set({ error: null }),

  applyWorkerMessage: (message) => {
    switch (message.type) {
      case 'progress':
        set({
          phase: message.phase,
          progress: {
            loaded: message.loaded,
            total: message.total,
            label: message.label,
            fromCache: message.fromCache,
          },
        });
        break;

      case 'ready':
        set((state) => ({
          phase: 'ready',
          backend: message.backend,
          meta: message.meta,
          stats: { ...state.stats, loadMs: message.loadMs },
          selectedLayer: Math.max(0, message.meta.numLayers - 1),
          progress: { ...state.progress, fromCache: message.fromCache },
        }));
        break;

      case 'prompt':
        set({
          tokens: message.tokens,
          selectedToken: Math.max(0, message.tokens.length - 1),
          tree: createRoot(message.tokens.length),
        });
        break;

      case 'step': {
        const state = get();
        if (message.runId !== state.runId) return;
        const steps = [...state.steps, message.record];
        const tokens = [...state.tokens, message.token];
        const follow = state.selectedStep === steps.length - 2 || state.selectedStep === -1;
        set({
          steps,
          tokens,
          selectedStep: follow ? steps.length - 1 : state.selectedStep,
          selectedToken: follow ? message.record.position : state.selectedToken,
        });
        break;
      }

      case 'done': {
        const state = get();
        if (message.runId !== state.runId) return;
        set({
          running: false,
          stats: {
            ...state.stats,
            totalMs: message.totalMs,
            tokensPerSecond: message.tokensPerSecond,
          },
        });
        break;
      }

      case 'cancelled':
        set({ running: false });
        break;

      case 'expanded': {
        const state = get();
        const chosen = state.tokens
          .filter((token) => token.source === 'generated')
          .map((token) => token.id);
        const rest = { ...state.pendingExpansions };
        delete rest[message.requestId];
        set({
          tree: applyExpansion(state.tree, message.expansion, chosen),
          pendingExpansions: rest,
        });
        break;
      }

      case 'rescored': {
        const state = get();
        if (message.runId !== state.runId) return;
        set({
          steps: state.steps.map((step) =>
            step.step === message.step
              ? { ...step, shapedTop: message.shapedTop, keptCandidates: message.keptCandidates }
              : step,
          ),
        });
        break;
      }

      case 'error':
        set({ error: message.message, fatal: message.fatal, running: false });
        break;
    }
  },
}));

/** The step the views are currently showing, or `null` before the first token. */
export function selectCurrentStep(state: RunState): StepRecord | null {
  if (state.selectedStep < 0) return null;
  return state.steps[state.selectedStep] ?? null;
}

export function selectGeneratedText(state: RunState): string {
  return state.tokens
    .filter((token) => token.source === 'generated')
    .map((token) => token.text)
    .join('');
}

export function selectChosenContinuation(state: RunState): Candidate[] {
  return state.steps.map((step) => step.chosen);
}
