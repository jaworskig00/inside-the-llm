import { useRunStore } from '../store/useRunStore';
import { bytes } from '../lib/format';

const PHASES = [
  { key: 'tokenizer', label: 'Tokenizer' },
  { key: 'model', label: 'Weights' },
  { key: 'session', label: 'Runtime' },
  { key: 'warmup', label: 'Warm-up' },
] as const;

/**
 * The first screen anyone sees. An 89 MB download needs to explain itself, so this says
 * what is being fetched, how far along it is, and that it only happens once.
 */
export function LoadingGate() {
  const phase = useRunStore((state) => state.phase);
  const progress = useRunStore((state) => state.progress);
  const error = useRunStore((state) => state.error);
  const fatal = useRunStore((state) => state.fatal);
  const gpuAvailable = useRunStore((state) => state.gpuAvailable);
  const demoMode = useRunStore((state) => state.demoMode);

  if (phase === 'ready' && !fatal) return null;

  const fraction = progress.total > 0 ? progress.loaded / progress.total : 0;
  const activeIndex = PHASES.findIndex((entry) => entry.key === phase);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-void/95 px-6 backdrop-blur-sm">
      <div className="panel w-full max-w-lg p-6" role="status" aria-live="polite">
        <h2 className="text-[15px] font-semibold">Inside the LLM</h2>

        {fatal && error ? (
          <>
            <p className="mt-3 text-[13px] leading-relaxed text-ink-dim">
              The model could not be loaded.
            </p>
            <pre className="mt-3 overflow-x-auto rounded-lg border border-edge bg-panel-2 p-3 font-mono text-[11.5px] text-amber">
              {error}
            </pre>
            <p className="mt-3 text-[12px] leading-relaxed text-ink-faint">
              The weights ship as a GitHub Release asset. If the download is blocked, you can still
              see the whole interface against a recorded run:{' '}
              <a className="text-accent underline" href="?demo=1">
                open the recorded run
              </a>
              .
            </p>
          </>
        ) : (
          <>
            <p className="mt-2 text-[13px] leading-relaxed text-ink-dim">
              Downloading DistilGPT-2 and starting it in a Web Worker. This happens once — the
              weights are kept in the browser cache afterwards.
            </p>

            <div className="mt-5">
              <div className="mb-1.5 flex items-baseline justify-between text-[12px]">
                <span className="text-ink">{progress.label}</span>
                <span className="tabular text-ink-faint">
                  {progress.total > 0 && phase === 'model'
                    ? `${bytes(progress.loaded)} / ${bytes(progress.total)}`
                    : `${Math.round(fraction * 100)}%`}
                </span>
              </div>
              <div
                className="h-2 w-full overflow-hidden rounded-full bg-edge"
                role="progressbar"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(fraction * 100)}
              >
                <div
                  className="h-full rounded-full bg-accent transition-[width] duration-200"
                  style={{ width: `${Math.max(2, fraction * 100)}%` }}
                />
              </div>
            </div>

            <ol className="mt-5 grid grid-cols-4 gap-2">
              {PHASES.map((entry, index) => {
                const done = activeIndex > index || phase === 'ready';
                const active = activeIndex === index;
                return (
                  <li
                    key={entry.key}
                    className={`rounded-md border px-2 py-1.5 text-center text-[11px] ${
                      active
                        ? 'border-accent text-accent'
                        : done
                          ? 'border-edge-bright text-ink-dim'
                          : 'border-edge text-ink-faint'
                    }`}
                  >
                    {done && !active ? '✓ ' : ''}
                    {entry.label}
                  </li>
                );
              })}
            </ol>

            {!gpuAvailable && !demoMode && (
              <p className="mt-4 text-[12px] leading-relaxed text-amber">
                This browser exposes no WebGPU adapter, so inference falls back to WebAssembly on
                the CPU. GitHub Pages cannot send the headers that would let it use more than one
                thread, and building the session single-threaded takes several minutes.{' '}
                <a className="underline" href="?demo=1">
                  Open the recorded run instead
                </a>{' '}
                — same interface, same model output, nothing to download.
              </p>
            )}

            {error && !fatal && <p className="mt-4 text-[12px] text-amber">{error}</p>}
          </>
        )}
      </div>
    </div>
  );
}
