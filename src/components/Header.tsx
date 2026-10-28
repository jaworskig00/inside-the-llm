import { useRunStore } from '../store/useRunStore';
import { ms } from '../lib/format';

const REPO_URL = 'https://github.com/jaworskig00/inside-the-llm';

/**
 * Title, the model's provenance, and badges that answer the two questions anyone technical
 * asks immediately: what is actually running, and on what.
 */
export function Header() {
  const backend = useRunStore((state) => state.backend);
  const meta = useRunStore((state) => state.meta);
  const stats = useRunStore((state) => state.stats);
  const fromCache = useRunStore((state) => state.progress.fromCache);
  const demoMode = useRunStore((state) => state.demoMode);
  const explain = useRunStore((state) => state.explain);
  const toggleExplain = useRunStore((state) => state.toggleExplain);
  const startTour = useRunStore((state) => state.startTour);

  return (
    <header className="flex flex-wrap items-end justify-between gap-4 border-b border-edge px-5 py-4">
      <div>
        <h1 className="text-[17px] font-semibold tracking-tight">
          Inside the LLM
          <span className="ml-2 align-middle text-[12px] font-normal text-ink-dim">
            a language model, running in this tab, with the lid off
          </span>
        </h1>
        <p className="mt-1 text-[12px] text-ink-dim">
          {meta
            ? `${meta.modelId} · ${meta.numLayers} layers · ${meta.numHeads} heads · ${meta.hiddenSize} dims · ${meta.vocabSize.toLocaleString('en-US')} tokens of vocabulary`
            : 'loading model metadata…'}
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {demoMode && (
          <Badge
            tone="violet"
            title="Replaying a run that was recorded from the real model, so nothing is downloaded and nothing is computed here"
          >
            recorded run
          </Badge>
        )}
        {backend && !demoMode && (
          <Badge
            tone={backend === 'webgpu' ? 'accent' : 'neutral'}
            title={
              backend === 'webgpu'
                ? 'Inference is running on the GPU through WebGPU'
                : 'WebGPU was unavailable, so inference fell back to WebAssembly on the CPU'
            }
          >
            {backend === 'webgpu' ? 'WebGPU' : 'WASM'}
          </Badge>
        )}
        {fromCache && !demoMode && (
          <Badge tone="neutral" title="Weights came from the Cache API, not the network">
            cached
          </Badge>
        )}
        {stats.loadMs > 0 && !demoMode && (
          <Badge tone="neutral" title="Time from page load to a warmed-up session">
            ready in {ms(stats.loadMs)}
          </Badge>
        )}

        <button
          type="button"
          className={`btn ${explain ? 'border-accent text-accent' : ''}`}
          onClick={toggleExplain}
          aria-pressed={explain}
          title="Show or hide the plain-language caption on every panel"
        >
          {explain ? 'Explain: on' : 'Explain: off'}
        </button>

        <button type="button" className="btn" onClick={startTour} title="Replay the walkthrough">
          Walkthrough
        </button>

        <a className="btn" href={REPO_URL} target="_blank" rel="noreferrer noopener">
          Source ↗
        </a>
      </div>
    </header>
  );
}

function Badge({
  children,
  tone,
  title,
}: {
  children: React.ReactNode;
  tone: 'accent' | 'neutral' | 'violet';
  title: string;
}) {
  const tones = {
    accent: 'border-accent/50 text-accent bg-accent/10',
    violet: 'border-violet/50 text-violet bg-violet/10',
    neutral: 'border-edge-bright text-ink-dim',
  } as const;
  return (
    <span
      className={`rounded-md border px-2 py-1 text-[11px] font-medium ${tones[tone]}`}
      title={title}
    >
      {children}
    </span>
  );
}
