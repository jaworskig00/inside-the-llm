/**
 * A stand-in for the inference worker that replays a recorded run.
 *
 * Used by `?demo=1` and by the Playwright test. It speaks exactly the same protocol as the
 * real worker, so nothing in the UI knows the difference — which is the point: if the UI
 * works against this, the protocol boundary is honest.
 */
import type { MainToWorker, WorkerToMain } from '../lib/protocol';
import { shapeDistribution } from '../lib/sampling';
import { decodeAttention, type Fixture } from './fixture';
import type { WorkerLike } from './client';

const STEP_DELAY_MS = 90;

export function createMockWorker(fixtureUrl: string): WorkerLike {
  const listeners = new Set<(event: MessageEvent<WorkerToMain>) => void>();
  let fixture: Fixture | null = null;
  let cancelled = false;

  const emit = (message: WorkerToMain): void => {
    const event = { data: message } as MessageEvent<WorkerToMain>;
    for (const listener of listeners) listener(event);
  };

  const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

  async function load(): Promise<Fixture> {
    if (fixture) return fixture;
    const response = await fetch(fixtureUrl);
    if (!response.ok) throw new Error(`could not load the demo recording (${response.status})`);
    fixture = (await response.json()) as Fixture;
    return fixture;
  }

  async function init(): Promise<void> {
    emit({
      type: 'progress',
      phase: 'model',
      loaded: 0,
      total: 1,
      label: 'Loading recorded run',
      fromCache: true,
    });
    const data = await load();
    emit({
      type: 'ready',
      backend: 'wasm',
      meta: data.meta,
      loadMs: data.stats.loadMs,
      fromCache: true,
    });
  }

  async function generate(message: Extract<MainToWorker, { type: 'generate' }>): Promise<void> {
    const data = await load();
    cancelled = false;

    emit({ type: 'prompt', runId: message.runId, tokens: data.promptTokens });

    const cap = Math.min(message.maxNewTokens, data.steps.length);
    for (let i = 0; i < cap; i += 1) {
      if (cancelled) {
        emit({ type: 'cancelled', runId: message.runId });
        return;
      }
      await sleep(STEP_DELAY_MS);
      const step = data.steps[i];
      if (!step) break;
      emit({
        type: 'step',
        runId: message.runId,
        record: {
          step: step.step,
          position: step.position,
          chosen: step.chosen,
          rawTop: step.rawTop,
          shapedTop: step.shapedTop,
          keptCandidates: step.keptCandidates,
          lens: step.lens,
          attention: {
            layers: step.attention.layers,
            heads: step.attention.heads,
            tokens: step.attention.tokens,
            data: decodeAttention(step.attention),
          },
          hiddenNorms: step.hiddenNorms,
          contextLength: step.contextLength,
          ms: step.ms,
        },
        token: step.token,
      });
    }

    emit({
      type: 'done',
      runId: message.runId,
      totalMs: data.stats.totalMs,
      tokensPerSecond: data.stats.tokensPerSecond,
    });
  }

  async function expand(message: Extract<MainToWorker, { type: 'expand' }>): Promise<void> {
    const data = await load();
    const key = message.path.join('.');
    const match = data.expansions.find((expansion) => expansion.path.join('.') === key);
    await sleep(60);
    emit({
      type: 'expanded',
      requestId: message.requestId,
      expansion: match ?? { path: message.path, children: [] },
    });
  }

  /**
   * The recording keeps the top of each step's raw distribution rather than all 50257
   * logits, so re-shaping in demo mode works on that head of the distribution. Exact for
   * any top-k below 24, which is every setting the sliders can reach in demo mode.
   */
  async function rescore(message: Extract<MainToWorker, { type: 'rescore' }>): Promise<void> {
    const data = await load();
    const step = data.steps.find((candidate) => candidate.step === message.step);
    if (!step) return;

    const ids = step.rawTop.map((candidate) => candidate.id);
    const logits = step.rawTop.map((candidate) => Math.log(Math.max(candidate.prob, 1e-12)));
    const shaped = shapeDistribution(logits, {
      temperature: message.sampling.temperature,
      topK: message.sampling.topK,
      topP: message.sampling.topP,
      limit: ids.length,
    });

    emit({
      type: 'rescored',
      runId: message.runId,
      step: message.step,
      shapedTop: shaped.candidates.map((candidate) => ({
        id: ids[candidate.id] as number,
        text: (step.rawTop[candidate.id]?.text ?? '') as string,
        prob: candidate.prob,
      })),
      keptCandidates: shaped.kept,
    });
  }

  return {
    postMessage(message: MainToWorker) {
      void (async () => {
        try {
          if (message.type === 'init') await init();
          else if (message.type === 'generate') await generate(message);
          else if (message.type === 'expand') await expand(message);
          else if (message.type === 'rescore') await rescore(message);
          else if (message.type === 'cancel') cancelled = true;
        } catch (error) {
          emit({
            type: 'error',
            message: error instanceof Error ? error.message : String(error),
            fatal: message.type === 'init',
          });
        }
      })();
    },
    addEventListener(listener) {
      listeners.add(listener);
    },
    terminate() {
      listeners.clear();
    },
  };
}
