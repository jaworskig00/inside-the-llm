import { useRunStore } from '../store/useRunStore';
import { LIMITS } from '../lib/protocol';
import { PRESET_PROMPTS } from '../lib/format';

interface PromptBarProps {
  onGenerate: () => void;
  onCancel: () => void;
}

export function PromptBar({ onGenerate, onCancel }: PromptBarProps) {
  const prompt = useRunStore((state) => state.prompt);
  const setPrompt = useRunStore((state) => state.setPrompt);
  const maxNewTokens = useRunStore((state) => state.maxNewTokens);
  const setMaxNewTokens = useRunStore((state) => state.setMaxNewTokens);
  const running = useRunStore((state) => state.running);
  const phase = useRunStore((state) => state.phase);
  const demoMode = useRunStore((state) => state.demoMode);

  const ready = phase === 'ready';

  return (
    <div className="panel grid gap-3 p-4" data-tour="prompt">
      <div className="flex flex-wrap items-center gap-2">
        <span className="label">presets</span>
        {PRESET_PROMPTS.map((preset) => (
          <button
            key={preset.label}
            type="button"
            className={`btn px-2.5 py-1 text-[12px] ${prompt === preset.text ? 'border-accent text-accent' : ''}`}
            onClick={() => setPrompt(preset.text)}
            title={preset.note}
            disabled={demoMode}
          >
            {preset.label}
          </button>
        ))}
        {demoMode && (
          <span className="text-[11px] text-ink-faint">
            the recorded run has a fixed prompt — add <code className="font-mono">?demo=0</code> to
            run the real model
          </span>
        )}
      </div>

      <div className="flex flex-col gap-3 lg:flex-row lg:items-end">
        <div className="min-w-0 flex-1">
          <label htmlFor="prompt" className="label mb-1.5 block">
            prompt
          </label>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            onKeyDown={(event) => {
              if ((event.metaKey || event.ctrlKey) && event.key === 'Enter' && ready && !running) {
                event.preventDefault();
                onGenerate();
              }
            }}
            rows={2}
            spellCheck={false}
            disabled={demoMode}
            className="w-full resize-y rounded-lg border border-edge-bright bg-panel-2 p-3 font-mono text-[13px] leading-relaxed text-ink outline-none disabled:opacity-60"
            placeholder="The capital of France is"
            aria-describedby="prompt-hint"
          />
          <p id="prompt-hint" className="mt-1 text-[11px] text-ink-faint">
            ⌘/Ctrl + Enter to generate · the first {LIMITS.maxPromptTokens} tokens are used
          </p>
        </div>

        <div className="w-full lg:w-44">
          <div className="mb-1.5 flex items-baseline justify-between">
            <label htmlFor="new-tokens" className="label">
              new tokens
            </label>
            <span className="tabular text-[12px] font-medium text-accent">{maxNewTokens}</span>
          </div>
          <input
            id="new-tokens"
            type="range"
            className="slider"
            min={1}
            max={LIMITS.maxNewTokens}
            step={1}
            value={maxNewTokens}
            onChange={(event) => setMaxNewTokens(Number(event.target.value))}
          />
        </div>

        {running ? (
          <button type="button" className="btn w-full lg:w-auto" onClick={onCancel}>
            Stop
          </button>
        ) : (
          <button
            type="button"
            className="btn btn-primary w-full lg:w-auto"
            onClick={onGenerate}
            disabled={!ready || prompt.trim().length === 0}
          >
            {ready ? 'Generate' : 'Loading model…'}
          </button>
        )}
      </div>
    </div>
  );
}
