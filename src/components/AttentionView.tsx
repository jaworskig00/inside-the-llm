import { Panel, Empty, Stat } from './Panel';
import { HeadGrid } from './HeadGrid';
import { ArcDiagram } from './ArcDiagram';
import { useRunStore, selectCurrentStep } from '../store/useRunStore';
import { displayToken, percent } from '../lib/format';
import { attentionRow, averageHeadsRow, maxOf } from '../lib/attention';

/**
 * Attention, at two zoom levels: the whole model at once (head grid) and one head in
 * detail (arc diagram). Both follow the selected step and the selected token, so stepping
 * through the run animates them together.
 */
export function AttentionView() {
  const step = useRunStore(selectCurrentStep);
  const tokens = useRunStore((state) => state.tokens);
  const selectedToken = useRunStore((state) => state.selectedToken);
  const selectedLayer = useRunStore((state) => state.selectedLayer);
  const selectedHead = useRunStore((state) => state.selectedHead);
  const setSelectedLayer = useRunStore((state) => state.setSelectedLayer);
  const setSelectedHead = useRunStore((state) => state.setSelectedHead);
  const setSelectedToken = useRunStore((state) => state.setSelectedToken);
  const meta = useRunStore((state) => state.meta);

  if (!step || !meta) {
    return (
      <Panel
        id="attention"
        title="Attention"
        explain="Each layer has several attention heads. A head decides, for every token, which earlier tokens matter to it — that is the only way information moves sideways in a transformer."
      >
        <Empty>Attention appears once a run has started.</Empty>
      </Panel>
    );
  }

  const pack = step.attention;
  const query = Math.min(selectedToken, pack.tokens - 1);
  const layer = Math.min(selectedLayer, pack.layers - 1);
  const row =
    selectedHead === null
      ? averageHeadsRow(pack, layer, query)
      : attentionRow(pack, layer, selectedHead, query);
  const peak = maxOf(row);
  const peakIndex = Array.from(row).indexOf(peak);
  const peakToken = tokens[peakIndex];

  return (
    <Panel
      id="attention"
      title="Attention"
      explain="Each layer has several attention heads. A head decides, for every token, which earlier tokens matter to it — that is the only way information moves sideways in a transformer. Bright squares in the grid are strong links; the arcs below show where the selected token is looking."
      aside={
        <div className="flex items-center gap-4">
          <Stat label="looking at" value={displayToken(peakToken?.text ?? '') || '—'} />
          <Stat label="strength" value={percent(peak)} />
        </div>
      }
    >
      <div className="grid gap-5">
        <HeadGrid
          pack={pack}
          selectedLayer={layer}
          selectedHead={selectedHead}
          onSelect={(nextLayer, nextHead) => {
            setSelectedLayer(nextLayer);
            setSelectedHead(nextHead);
          }}
        />

        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-edge pt-4">
          <div className="flex items-center gap-2">
            <span className="label">layer</span>
            <div className="flex gap-1" role="group" aria-label="Layer">
              {Array.from({ length: pack.layers }, (_, index) => (
                <button
                  key={index}
                  type="button"
                  onClick={() => setSelectedLayer(index)}
                  className={`btn px-2.5 py-1 text-[12px] ${
                    index === layer ? 'border-accent text-accent' : ''
                  }`}
                  aria-pressed={index === layer}
                >
                  {index}
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className="label">head</span>
            <button
              type="button"
              onClick={() => setSelectedHead(null)}
              className={`btn px-2.5 py-1 text-[12px] ${
                selectedHead === null ? 'border-accent text-accent' : ''
              }`}
              aria-pressed={selectedHead === null}
              title="Mean over all heads in this layer"
            >
              mean
            </button>
            <div className="flex flex-wrap gap-1" role="group" aria-label="Head">
              {Array.from({ length: pack.heads }, (_, index) => (
                <button
                  key={index}
                  type="button"
                  onClick={() => setSelectedHead(index)}
                  className={`btn px-2 py-1 text-[12px] ${
                    selectedHead === index ? 'border-accent text-accent' : ''
                  }`}
                  aria-pressed={selectedHead === index}
                >
                  {index}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div>
          <div className="mb-1 flex items-baseline justify-between">
            <span className="label">
              from{' '}
              <span className="font-mono text-amber">
                {displayToken(tokens[query]?.text ?? '')}
              </span>{' '}
              · layer {layer} · {selectedHead === null ? 'all heads' : `head ${selectedHead}`}
            </span>
            <span className="text-[11px] text-ink-faint">click a dot to move the query token</span>
          </div>
          <ArcDiagram
            pack={pack}
            tokens={tokens}
            layer={layer}
            head={selectedHead}
            query={query}
            onSelectToken={setSelectedToken}
          />
        </div>
      </div>
    </Panel>
  );
}
