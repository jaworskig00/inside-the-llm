import { useEffect, useRef } from 'react';
import { scalePoint, select } from 'd3';
import type { AttentionPack, TokenInfo } from '../lib/protocol';
import { arcLinks, attentionRow, averageHeadsRow, maxOf } from '../lib/attention';
import { displayToken, percent } from '../lib/format';

interface ArcDiagramProps {
  pack: AttentionPack;
  tokens: TokenInfo[];
  layer: number;
  /** `null` renders the mean over all heads for this layer. */
  head: number | null;
  /** The token doing the looking. */
  query: number;
  onSelectToken: (index: number) => void;
}

const HEIGHT = 210;
// bottom is generous because the token labels are rotated 55 degrees under the baseline.
const MARGIN = { top: 16, right: 14, bottom: 68, left: 14 };
const DURATION = 320;

/**
 * Where one token looks. Tokens sit on a line; an arc from the selected token to each
 * earlier token is drawn with thickness and opacity proportional to the attention weight.
 *
 * SVG here, not canvas: there are at most a couple of dozen arcs after thresholding, they
 * need hit testing and tooltips, and d3 transitions on `d` make the step-to-step change
 * legible instead of a jump cut.
 */
export function ArcDiagram({ pack, tokens, layer, head, query, onSelectToken }: ArcDiagramProps) {
  const ref = useRef<SVGSVGElement | null>(null);
  // Held in a ref so a new inline callback from the parent does not re-run the whole draw.
  const onSelectRef = useRef(onSelectToken);
  useEffect(() => {
    onSelectRef.current = onSelectToken;
  }, [onSelectToken]);

  useEffect(() => {
    const svg = ref.current;
    if (!svg) return;

    const count = Math.min(pack.tokens, tokens.length);
    if (count === 0) return;

    const width = Math.max(320, svg.clientWidth || 640);
    const innerWidth = width - MARGIN.left - MARGIN.right;
    const baseline = HEIGHT - MARGIN.bottom;

    const row =
      head === null ? averageHeadsRow(pack, layer, query) : attentionRow(pack, layer, head, query);
    const peak = Math.max(maxOf(row), 1e-6);
    const links = arcLinks(row, query, { minWeight: Math.max(0.015, peak * 0.06), maxLinks: 28 });

    const x = scalePoint<number>()
      .domain(Array.from({ length: count }, (_, i) => i))
      .range([MARGIN.left, MARGIN.left + innerWidth])
      .padding(0.5);

    const root = select(svg).attr('viewBox', `0 0 ${width} ${HEIGHT}`).attr('height', HEIGHT);

    let layerG = root.select<SVGGElement>('g.arcs');
    if (layerG.empty()) {
      root.append('line').attr('class', 'baseline');
      layerG = root.append('g').attr('class', 'arcs');
      root.append('g').attr('class', 'ticks');
    }

    root
      .select<SVGLineElement>('line.baseline')
      .attr('x1', MARGIN.left)
      .attr('x2', MARGIN.left + innerWidth)
      .attr('y1', baseline)
      .attr('y2', baseline)
      .attr('stroke', 'var(--color-edge-bright)')
      .attr('stroke-width', 1);

    const arcPath = (source: number, target: number): string => {
      const x1 = x(source) ?? 0;
      const x2 = x(target) ?? 0;
      const span = Math.abs(x1 - x2);
      const lift = Math.min(baseline - MARGIN.top, 18 + span * 0.5);
      return `M${x1},${baseline} C${x1},${baseline - lift} ${x2},${baseline - lift} ${x2},${baseline}`;
    };

    layerG
      .selectAll<SVGPathElement, (typeof links)[number]>('path')
      .data(links, (d) => `${d.source}-${d.target}`)
      .join(
        (enter) =>
          enter
            .append('path')
            .attr('fill', 'none')
            .attr('stroke', 'var(--color-accent)')
            .attr('stroke-linecap', 'round')
            .attr('d', (d) => arcPath(d.source, d.target))
            .attr('stroke-width', 0)
            .attr('opacity', 0),
        (update) => update,
        (exit) => exit.transition().duration(DURATION).attr('opacity', 0).remove(),
      )
      .transition()
      .duration(DURATION)
      .attr('d', (d) => arcPath(d.source, d.target))
      .attr('stroke-width', (d) => 1 + 7 * (d.weight / peak))
      .attr('opacity', (d) => 0.22 + 0.62 * (d.weight / peak));

    // Token labels. Past ~34 tokens there is no room for text, so they become ticks and
    // the detail moves into the tooltip.
    const dense = count > 34;
    const ticks = root.select<SVGGElement>('g.ticks');

    ticks
      .selectAll<SVGGElement, number>('g.tick')
      .data(
        Array.from({ length: count }, (_, i) => i),
        (d) => d,
      )
      .join(
        (enter) => {
          const g = enter.append('g').attr('class', 'tick').style('cursor', 'pointer');
          g.append('circle').attr('r', 3).attr('cy', 0);
          g.append('text')
            .attr('y', 14)
            .attr('font-family', 'var(--font-mono)')
            .attr('font-size', 9.5)
            .attr('text-anchor', 'end')
            .attr('transform', 'rotate(-55)');
          g.append('title');
          return g;
        },
        (update) => update,
        (exit) => exit.remove(),
      )
      .attr('transform', (d) => `translate(${x(d) ?? 0},${baseline})`)
      .each(function (d) {
        const token = tokens[d];
        const weight = row[d] ?? 0;
        const isQuery = d === query;
        const g = select(this);
        g.select('circle')
          .attr(
            'fill',
            isQuery
              ? 'var(--color-amber)'
              : weight > 0.02
                ? 'var(--color-accent)'
                : 'var(--color-edge-bright)',
          )
          .attr('r', isQuery ? 4.5 : 3);
        g.select('text')
          .text(dense ? '' : displayToken(token?.text ?? '') || '␣')
          .attr(
            'fill',
            isQuery
              ? 'var(--color-amber)'
              : weight > 0.02
                ? 'var(--color-ink)'
                : 'var(--color-ink-faint)',
          );
        g.select('title').text(
          `#${d} ${displayToken(token?.text ?? '')} — attention ${percent(weight, 2)}`,
        );
      })
      .on('click', (_event, d) => onSelectRef.current(d));
  }, [pack, tokens, layer, head, query]);

  return (
    <svg
      ref={ref}
      className="w-full"
      role="img"
      aria-label={`Attention arcs from token ${query} in layer ${layer}, ${
        head === null ? 'averaged over heads' : `head ${head}`
      }`}
    />
  );
}
