/**
 * Wires the worker to the store: one effect, created once, that forwards every worker
 * message into `applyWorkerMessage` and hands back the few commands the UI can issue.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { LIMITS, type SamplingConfig } from '../lib/protocol';
import { createWorker, initMessage, type WorkerLike } from '../worker/client';
import { useRunStore } from '../store/useRunStore';

export interface InferenceApi {
  generate: () => void;
  cancel: () => void;
  expand: (path: number[]) => void;
  rescore: (step: number, sampling: SamplingConfig) => void;
  forkFrom: (path: number[]) => void;
}

export function useInference(): InferenceApi {
  const workerRef = useRef<WorkerLike | null>(null);
  const [, setMounted] = useState(false);

  useEffect(() => {
    let disposed = false;
    let created: WorkerLike | null = null;

    void (async () => {
      try {
        const { worker, demoMode } = await createWorker();
        if (disposed) {
          worker.terminate();
          return;
        }
        created = worker;
        workerRef.current = worker;
        useRunStore.setState({ demoMode });
        worker.addEventListener((event) => useRunStore.getState().applyWorkerMessage(event.data));
        worker.postMessage(initMessage());
        setMounted(true);
      } catch (error) {
        useRunStore.setState({
          error: error instanceof Error ? error.message : String(error),
          fatal: true,
        });
      }
    })();

    return () => {
      disposed = true;
      created?.terminate();
      workerRef.current = null;
    };
  }, []);

  const generate = useCallback(() => {
    const worker = workerRef.current;
    if (!worker) return;
    const state = useRunStore.getState();
    const runId = state.beginRun();
    worker.postMessage({
      type: 'generate',
      runId,
      prompt: state.prompt,
      maxNewTokens: state.maxNewTokens,
      sampling: state.sampling,
    });
  }, []);

  const cancel = useCallback(() => {
    workerRef.current?.postMessage({ type: 'cancel' });
  }, []);

  const expand = useCallback((path: number[]) => {
    const worker = workerRef.current;
    if (!worker) return;
    const requestId = useRunStore.getState().requestExpansion(path);
    if (requestId === null) return;
    worker.postMessage({ type: 'expand', requestId, path, branches: LIMITS.treeBranches });
  }, []);

  const rescore = useCallback((step: number, sampling: SamplingConfig) => {
    const worker = workerRef.current;
    if (!worker) return;
    worker.postMessage({ type: 'rescore', runId: useRunStore.getState().runId, step, sampling });
  }, []);

  /** "What if the model had picked this branch instead?" — re-runs from a forked prefix. */
  const forkFrom = useCallback((path: number[]) => {
    const worker = workerRef.current;
    if (!worker) return;
    const state = useRunStore.getState();
    const promptIds = state.tokens.filter((token) => token.source === 'prompt').map((t) => t.id);
    const runId = state.beginRun();
    worker.postMessage({
      type: 'generate',
      runId,
      prompt: state.prompt,
      promptTokenIds: [...promptIds, ...path],
      maxNewTokens: state.maxNewTokens,
      sampling: state.sampling,
    });
  }, []);

  return { generate, cancel, expand, rescore, forkFrom };
}
