import { useEffect } from 'react';
import { Panel, Empty, Stat } from './Panel';
import { ProbabilityBars } from './ProbabilityBars';
import { useRunStore, selectCurrentStep } from '../store/useRunStore';
import { entropyBits } from '../lib/sampling';
import { fixed, percent } from '../lib/format';
import type { SamplingConfig } from '../lib/protocol';

interface SamplingPanelProps {
  onRescore: (step: number, sampling: SamplingConfig) => void;
}

/**
 * Temperature, top-k and top-p, wired straight to the chart.
 *
 * Moving a slider does not re-run the model: the step's logits are still in the worker, so
 * it re-applies the filters to the real distribution and sends back the result. Exact, and
 * fast enough to feel live.
 */
export function SamplingPanel({ onRescore }: SamplingPanelProps) {
  const step = useRunStore(selectCurrentStep);
  const sampling = useRunStore((state) => state.sampling);
  const setSampling = useRunStore((state) => state.setSampling);
  const selectedStep = useRunStore((state) => state.selectedStep);
  const running = useRunStore((state) => state.running);

  // Debounced so dragging a slider does not queue dozens of worker round-trips.
  useEffect(() => {
    if (selectedStep < 0 || running) return;
    const timer = window.setTimeout(() => onRescore(selectedStep, sampling), 90);
    return () => window.clearTimeout(timer);
  }, [sampling, selectedStep, running, onRescore]);

  const entropy = step ? entropyBits(step.rawTop.map((c) => c.prob)) : 0;

  return (
    <Panel
      id="sampling"
      title="Sampling"
      explain="The model outputs a probability for every one of 50 257 tokens. These three dials decide how much of that list it is allowed to pick from — the faint bars are the model's real opinion, the solid bars are what is left after filtering."
      aside={
        step && (
          <div className="flex items-center gap-4">
            <Stat label="kept" value={String(step.keptCandidates)} />
            <Stat label="entropy" value={`${fixed(entropy, 2)} bits`} />
          </div>
        )
      }
    >
      <div className="grid gap-4">
        <div className="grid gap-3 sm:grid-cols-3">
          <Slider
            label="temperature"
            hint="0 = always the top token, >1 = wilder"
            value={sampling.temperature}
            min={0}
            max={2}
            step={0.05}
            format={(v) => (v === 0 ? 'greedy' : v.toFixed(2))}
            onChange={(temperature) => setSampling({ temperature })}
          />
          <Slider
            label="top-k"
            hint="only the k most likely tokens survive"
            value={sampling.topK}
            min={0}
            max={100}
            step={1}
            format={(v) => (v === 0 ? 'off' : String(v))}
            onChange={(topK) => setSampling({ topK })}
          />
          <Slider
            label="top-p"
            hint="keep tokens until their mass reaches p"
            value={sampling.topP}
            min={0.05}
            max={1}
            step={0.01}
            format={(v) => (v >= 1 ? 'off' : v.toFixed(2))}
            onChange={(topP) => setSampling({ topP })}
          />
        </div>

        {step ? (
          <>
            <ProbabilityBars raw={step.rawTop} shaped={step.shapedTop} chosenId={step.chosen.id} />
            <p className="text-[11px] text-ink-faint">
              The model picked{' '}
              <span className="font-mono text-amber">{step.chosen.text || '␣'}</span> at{' '}
              {percent(step.chosen.prob, 2)}. Changing the dials here reshapes the chart but does
              not rewrite history — press Generate to sample a new run.
            </p>
          </>
        ) : (
          <Empty>Generate something to see the distribution.</Empty>
        )}
      </div>
    </Panel>
  );
}

interface SliderProps {
  label: string;
  hint: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format: (value: number) => string;
  onChange: (value: number) => void;
}

function Slider({ label, hint, value, min, max, step, format, onChange }: SliderProps) {
  const id = `slider-${label}`;
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between gap-2">
        <label htmlFor={id} className="label">
          {label}
        </label>
        <span className="tabular text-[12px] font-medium text-accent">{format(value)}</span>
      </div>
      <input
        id={id}
        type="range"
        className="slider"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        aria-describedby={`${id}-hint`}
      />
      <p id={`${id}-hint`} className="mt-1 text-[10.5px] leading-snug text-ink-faint">
        {hint}
      </p>
    </div>
  );
}
