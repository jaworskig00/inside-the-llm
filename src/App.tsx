import { Header } from './components/Header';
import { PromptBar } from './components/PromptBar';
import { PlaybackControls } from './components/PlaybackControls';
import { TokenStrip } from './components/TokenStrip';
import { SamplingPanel } from './components/SamplingPanel';
import { AttentionView } from './components/AttentionView';
import { AlternativesTree } from './components/AlternativesTree';
import { LayerFlowSankey } from './components/LayerFlowSankey';
import { ResidualStrip } from './components/ResidualStrip';
import { LoadingGate } from './components/LoadingGate';
import { Tour } from './components/Tour';
import { MobileNotice } from './components/MobileNotice';
import { useInference } from './hooks/useInference';
import { useRunStore } from './store/useRunStore';

export default function App() {
  const { generate, cancel, expand, rescore, forkFrom } = useInference();
  const error = useRunStore((state) => state.error);
  const fatal = useRunStore((state) => state.fatal);
  const dismissError = useRunStore((state) => state.dismissError);

  return (
    <div className="min-h-full">
      <LoadingGate />
      <Tour onGenerate={generate} />

      <a
        href="#tokens-title"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded focus:bg-accent focus:px-3 focus:py-2 focus:text-[13px] focus:text-void"
      >
        Skip to the visualisations
      </a>

      <Header />
      <MobileNotice />

      <main className="mx-auto grid max-w-[1700px] gap-4 p-4" data-testid="app-main">
        <PromptBar onGenerate={generate} onCancel={cancel} />

        {error && !fatal && (
          <div className="panel flex items-start justify-between gap-4 border-amber/40 bg-amber/5 p-3">
            <p className="font-mono text-[12px] text-amber">{error}</p>
            <button type="button" className="btn px-2 py-1 text-[11px]" onClick={dismissError}>
              dismiss
            </button>
          </div>
        )}

        <TokenStrip />
        <PlaybackControls />

        <div className="grid gap-4 xl:grid-cols-[minmax(0,5fr)_minmax(0,4fr)]">
          <SamplingPanel onRescore={rescore} />
          <LayerFlowSankey />
        </div>

        <AttentionView />

        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <AlternativesTree onExpand={expand} onFork={forkFrom} />
          <ResidualStrip />
        </div>

        <footer className="flex flex-wrap items-center justify-between gap-3 px-1 py-6 text-[11.5px] text-ink-faint">
          <p>
            DistilGPT-2 exported to ONNX with attention, hidden-state and logit-lens outputs, run
            client-side with onnxruntime-web. No server, no API key, nothing leaves this tab.
          </p>
          <p className="flex gap-3">
            <a
              className="hover:text-ink"
              href="https://gj-dev.com"
              target="_blank"
              rel="noreferrer noopener"
            >
              Grzegorz Jaworski
            </a>
            <a
              className="hover:text-ink"
              href="https://github.com/jaworskig00/inside-the-llm"
              target="_blank"
              rel="noreferrer noopener"
            >
              GitHub
            </a>
            <a
              className="hover:text-ink"
              href="https://linkedin.com/in/jaworskig00"
              target="_blank"
              rel="noreferrer noopener"
            >
              LinkedIn
            </a>
          </p>
        </footer>
      </main>
    </div>
  );
}
