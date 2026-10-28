import { useMemo } from 'react';
import { interpolateMagma, scaleSequential } from 'd3';
import { Panel, Empty } from './Panel';
import { useRunStore, selectCurrentStep } from '../store/useRunStore';
import { displayToken, fixed } from '../lib/format';

const CELL_H = 13;
const LABEL_W = 34;

/**
 * The residual stream, as magnitude only.
 *
 * `hidden_norms` is the L2 norm of each token's hidden state at each layer — the one piece
 * of the 768-dimensional state that can be shown honestly without projecting it. It is
 * here because the growth pattern is a real, visible property of transformers: norms climb
 * steadily with depth, and the final-layer norm spikes on the tokens the model is most
 * committed about.
 */
export function ResidualStrip() {
  const step = useRunStore(selectCurrentStep);
  const tokens = useRunStore((state) => state.tokens);
  const selectedToken = useRunStore((state) => state.selectedToken);
  const setSelectedToken = useRunStore((state) => state.setSelectedToken);

  const grid = step?.hiddenNorms ?? null;

  const { color, max } = useMemo(() => {
    if (!grid) return { color: null, max: 0 };
    let hi = 0;
    for (const row of grid) for (const value of row) if (value > hi) hi = value;
    return { color: scaleSequential(interpolateMagma).domain([0, hi]), max: hi };
  }, [grid]);

  if (!grid || !color) {
    return (
      <Panel
        id="residual"
        title="Residual stream"
        explain="Each token carries a 768-number state that every layer adds to. This strip shows how large that state has grown — the only part of it that can be drawn without distortion."
      >
        <Empty>Nothing to measure yet.</Empty>
      </Panel>
    );
  }

  const count = grid[0]?.length ?? 0;
  const cellW = Math.max(6, Math.min(18, Math.floor(620 / Math.max(count, 1))));
  const width = LABEL_W + count * cellW;
  const height = grid.length * CELL_H;

  return (
    <Panel
      id="residual"
      title="Residual stream"
      explain="Each token carries a 768-number state that every layer adds to. Brighter means a larger state — notice how it grows with depth, and spikes on the tokens the model cares about."
    >
      <div className="overflow-x-auto">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          style={{ width, height }}
          className="max-w-full"
          role="img"
          aria-label={`Residual stream magnitude, ${grid.length} layers by ${count} tokens`}
        >
          {grid.map((row, layer) => (
            <g key={layer} transform={`translate(0,${layer * CELL_H})`}>
              <text
                x={LABEL_W - 6}
                y={CELL_H / 2 + 3}
                textAnchor="end"
                fontSize={8.5}
                fill="var(--color-ink-faint)"
                fontFamily="var(--font-mono)"
              >
                {layer === 0 ? 'emb' : `L${layer - 1}`}
              </text>
              {row.map((value, index) => (
                <rect
                  key={index}
                  x={LABEL_W + index * cellW}
                  y={1}
                  width={cellW - 1}
                  height={CELL_H - 2}
                  fill={color(value)}
                  stroke={index === selectedToken ? 'var(--color-accent)' : 'none'}
                  strokeWidth={index === selectedToken ? 1 : 0}
                  onClick={() => setSelectedToken(index)}
                  style={{ cursor: 'pointer' }}
                >
                  <title>{`${displayToken(tokens[index]?.text ?? '')} · ${
                    layer === 0 ? 'embedding' : `layer ${layer - 1}`
                  } · ‖h‖ = ${fixed(value, 1)}`}</title>
                </rect>
              ))}
            </g>
          ))}
        </svg>
      </div>
      <p className="mt-2 text-[11px] text-ink-faint tabular">
        peak ‖h‖ = {fixed(max, 1)} · bottom row is the output layer
      </p>
    </Panel>
  );
}
