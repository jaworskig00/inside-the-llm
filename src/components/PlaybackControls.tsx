import { useEffect } from 'react';
import { useRunStore } from '../store/useRunStore';
import { ms } from '../lib/format';

const PLAY_INTERVAL_MS = 650;

/**
 * Play / pause / step through the run. Keyboard shortcuts are registered here rather than
 * on the buttons so they work no matter what has focus: left and right arrows step,
 * space toggles playback.
 */
export function PlaybackControls() {
  const steps = useRunStore((state) => state.steps);
  const selectedStep = useRunStore((state) => state.selectedStep);
  const playing = useRunStore((state) => state.playing);
  const running = useRunStore((state) => state.running);
  const stats = useRunStore((state) => state.stats);
  const setPlaying = useRunStore((state) => state.setPlaying);
  const stepBy = useRunStore((state) => state.stepBy);
  const setSelectedStep = useRunStore((state) => state.setSelectedStep);

  const atEnd = selectedStep >= steps.length - 1;

  useEffect(() => {
    if (!playing || steps.length === 0) return;
    const timer = window.setInterval(() => {
      const state = useRunStore.getState();
      if (state.selectedStep >= state.steps.length - 1) {
        state.setPlaying(false);
        return;
      }
      state.stepBy(1);
    }, PLAY_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [playing, steps.length]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;

      if (event.key === 'ArrowLeft') {
        event.preventDefault();
        stepBy(-1);
      } else if (event.key === 'ArrowRight') {
        event.preventDefault();
        stepBy(1);
      } else if (event.key === ' ') {
        event.preventDefault();
        const state = useRunStore.getState();
        if (state.steps.length > 0) state.setPlaying(!state.playing);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [stepBy]);

  const disabled = steps.length === 0;

  return (
    <div
      className="panel flex flex-wrap items-center gap-3 px-4 py-3"
      role="group"
      aria-label="Playback"
    >
      <button
        type="button"
        className="btn"
        onClick={() => stepBy(-1)}
        disabled={disabled || selectedStep <= 0}
        aria-label="Previous step (left arrow)"
        title="Previous step — ←"
      >
        ◀
      </button>
      <button
        type="button"
        className="btn btn-primary min-w-[88px]"
        onClick={() => {
          if (atEnd && !running) setSelectedStep(0);
          setPlaying(!playing);
        }}
        disabled={disabled}
        aria-label={playing ? 'Pause playback (space)' : 'Play the run (space)'}
        title="Play / pause — space"
      >
        {playing ? '❙❙ Pause' : atEnd && !running ? '↺ Replay' : '▶ Play'}
      </button>
      <button
        type="button"
        className="btn"
        onClick={() => stepBy(1)}
        disabled={disabled || atEnd}
        aria-label="Next step (right arrow)"
        title="Next step — →"
      >
        ▶
      </button>

      <input
        type="range"
        className="slider mx-2 min-w-[140px] flex-1"
        min={0}
        max={Math.max(0, steps.length - 1)}
        value={Math.max(0, selectedStep)}
        onChange={(event) => setSelectedStep(Number(event.target.value))}
        disabled={disabled}
        aria-label="Generation step"
        aria-valuetext={`Step ${selectedStep + 1} of ${steps.length}`}
      />

      <div className="flex items-center gap-4 text-[11px] text-ink-faint">
        <span className="tabular">
          {steps.length > 0 ? `step ${selectedStep + 1} / ${steps.length}` : 'no run yet'}
        </span>
        {stats.tokensPerSecond > 0 && (
          <span
            className="tabular"
            title="Measured end to end, including the forward pass per token"
          >
            {stats.tokensPerSecond.toFixed(1)} tok/s
          </span>
        )}
        {steps.length > 0 && (
          <span className="tabular" title="Forward pass time for the selected step">
            {ms(steps[Math.max(0, selectedStep)]?.ms ?? 0)} / pass
          </span>
        )}
      </div>
      <kbd className="hidden rounded border border-edge-bright px-1.5 py-0.5 text-[10px] text-ink-faint lg:inline">
        ← → step · space play
      </kbd>
    </div>
  );
}
