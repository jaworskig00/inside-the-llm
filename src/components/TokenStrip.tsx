import { useMemo } from 'react';
import { Panel, Empty, Stat } from './Panel';
import { useRunStore } from '../store/useRunStore';
import { displayToken, percent, plainToken } from '../lib/format';
import { contrastInk, probabilityColor, clamp01 } from '../lib/color';

/**
 * The spine of the app: the prompt and the generated text as individual tokens.
 *
 * Colour is the probability the model assigned to that token given everything before it —
 * for the prompt that is teacher forcing (how surprising the text was), for generated
 * tokens it is the probability of the token it actually picked. Clicking a chip selects it,
 * which is what every other view keys off.
 */
export function TokenStrip() {
  const tokens = useRunStore((state) => state.tokens);
  const steps = useRunStore((state) => state.steps);
  const selectedToken = useRunStore((state) => state.selectedToken);
  const selectedStep = useRunStore((state) => state.selectedStep);
  const setSelectedToken = useRunStore((state) => state.setSelectedToken);
  const setSelectedStep = useRunStore((state) => state.setSelectedStep);

  /** Token index -> step index, so clicking a generated chip also moves the timeline. */
  const stepByPosition = useMemo(() => {
    const map = new Map<number, number>();
    steps.forEach((step, index) => map.set(step.position, index));
    return map;
  }, [steps]);

  const generated = useMemo(
    () =>
      tokens
        .filter((token) => token.source === 'generated')
        .map((t) => plainToken(t.text))
        .join(''),
    [tokens],
  );

  const onSelect = (index: number) => {
    setSelectedToken(index);
    const step = stepByPosition.get(index);
    if (step !== undefined) setSelectedStep(step);
  };

  return (
    <Panel
      id="tokens"
      title="Tokens"
      explain="The model does not see letters — it sees these chunks. Brighter chips are tokens the model found more likely; a dim chip means it was surprised. Click any chip to inspect that moment."
      aside={
        <div className="flex items-center gap-4">
          <Stat label="tokens" value={String(tokens.length)} />
          <Stat
            label="step"
            value={selectedStep >= 0 ? `${selectedStep + 1}/${steps.length}` : '—'}
          />
        </div>
      }
    >
      {tokens.length === 0 ? (
        <Empty>Type a prompt and press Generate.</Empty>
      ) : (
        <>
          <div className="flex flex-wrap gap-1" role="list" aria-label="Token sequence">
            {tokens.map((token) => {
              const isSelected = token.index === selectedToken;
              const t = token.prob === null ? 0.35 : clamp01(token.prob);
              const isGenerated = token.source === 'generated';
              return (
                <button
                  key={`${token.index}-${token.id}`}
                  role="listitem"
                  type="button"
                  onClick={() => onSelect(token.index)}
                  className={`chip ${isSelected ? 'chip-selected' : ''} ${
                    isGenerated ? '' : 'opacity-90'
                  }`}
                  style={{
                    background: probabilityColor(t),
                    color: contrastInk(t),
                    borderStyle: isGenerated ? 'solid' : 'dashed',
                    borderColor: isSelected ? undefined : 'rgba(255,255,255,0.12)',
                  }}
                  title={`#${token.index} · id ${token.id} · ${
                    token.prob === null
                      ? 'first token (no context)'
                      : `p = ${percent(token.prob, 2)}`
                  } · ${isGenerated ? 'generated' : 'prompt'}`}
                  aria-label={`Token ${token.index + 1}: ${displayToken(token.text)}, ${
                    token.prob === null ? 'no probability' : `probability ${percent(token.prob)}`
                  }`}
                  aria-pressed={isSelected}
                >
                  {displayToken(token.text) || '␣'}
                </button>
              );
            })}
          </div>

          <div className="mt-4 flex items-start gap-3 border-t border-edge pt-3">
            <span className="label mt-[2px] shrink-0">output</span>
            <p className="min-w-0 whitespace-pre-wrap break-words font-mono text-[12.5px] leading-relaxed text-ink-dim">
              {generated ? generated : <span className="text-ink-faint">nothing yet</span>}
            </p>
          </div>

          <p className="mt-3 text-[11px] text-ink-faint">
            <span className="font-mono">·</span> marks a leading space,{' '}
            <span className="font-mono">↵</span> a newline. Dashed borders are prompt tokens.
          </p>
        </>
      )}
    </Panel>
  );
}
