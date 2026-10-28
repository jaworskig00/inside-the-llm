import { useEffect, useRef } from 'react';
import { scaleLinear, select } from 'd3';
import type { Candidate } from '../lib/protocol';
import { displayToken, percent } from '../lib/format';
import { probabilityColor } from '../lib/color';

interface ProbabilityBarsProps {
  /** The real next-token distribution, before temperature / top-k / top-p. */
  raw: Candidate[];
  /** What the sliders turned it into — this is what the model actually samples from. */
  shaped: Candidate[];
  chosenId: number | null;
  rows?: number;
  onPick?: (candidate: Candidate) => void;
}

interface Row {
  id: number;
  text: string;
  raw: number;
  shaped: number;
  chosen: boolean;
}

const ROW_HEIGHT = 22;
const LABEL_WIDTH = 92;
const VALUE_WIDTH = 54;
const DURATION = 260;

/**
 * The before/after bar chart. The faint bar is the model's own distribution; the solid bar
 * is what the sampling settings leave of it. Seeing one shrink as the other is reshaped is
 * the clearest way to explain what temperature and nucleus sampling actually do.
 *
 * D3 owns this SVG end to end — a keyed join plus transitions, so bars slide and recolour
 * instead of popping when the step or the sliders change.
 */
export function ProbabilityBars({
  raw,
  shaped,
  chosenId,
  rows = 12,
  onPick,
}: ProbabilityBarsProps) {
  const ref = useRef<SVGSVGElement | null>(null);
  // Held in a ref so a new inline callback from the parent does not re-run the whole draw.
  const onPickRef = useRef(onPick);
  useEffect(() => {
    onPickRef.current = onPick;
  }, [onPick]);

  useEffect(() => {
    const svg = ref.current;
    if (!svg) return;

    const shapedById = new Map(shaped.map((c) => [c.id, c.prob]));
    const data: Row[] = raw.slice(0, rows).map((candidate) => ({
      id: candidate.id,
      text: displayToken(candidate.text) || '␣',
      raw: candidate.prob,
      shaped: shapedById.get(candidate.id) ?? 0,
      chosen: candidate.id === chosenId,
    }));

    const height = Math.max(1, data.length) * ROW_HEIGHT;
    const width = svg.clientWidth || 420;
    const barWidth = Math.max(40, width - LABEL_WIDTH - VALUE_WIDTH);
    const maxValue = Math.max(0.0001, ...data.map((d) => Math.max(d.raw, d.shaped)));

    const x = scaleLinear().domain([0, maxValue]).range([0, barWidth]);

    const root = select(svg).attr('viewBox', `0 0 ${width} ${height}`).attr('height', height);

    const groups = root
      .selectAll<SVGGElement, Row>('g.row')
      .data(data, (d) => d.id)
      .join(
        (enter) => {
          const g = enter.append('g').attr('class', 'row').style('opacity', 0);

          g.append('text')
            .attr('class', 'token')
            .attr('x', LABEL_WIDTH - 8)
            .attr('y', ROW_HEIGHT / 2)
            .attr('text-anchor', 'end')
            .attr('dominant-baseline', 'middle')
            .attr('font-family', 'var(--font-mono)')
            .attr('font-size', 11.5);

          g.append('rect')
            .attr('class', 'raw')
            .attr('x', LABEL_WIDTH)
            .attr('y', 5)
            .attr('height', ROW_HEIGHT - 10)
            .attr('rx', 2)
            .attr('fill', 'var(--color-edge-bright)')
            .attr('width', 0);

          g.append('rect')
            .attr('class', 'shaped')
            .attr('x', LABEL_WIDTH)
            .attr('y', 5)
            .attr('height', ROW_HEIGHT - 10)
            .attr('rx', 2)
            .attr('width', 0);

          g.append('text')
            .attr('class', 'value')
            .attr('y', ROW_HEIGHT / 2)
            .attr('dominant-baseline', 'middle')
            .attr('font-size', 10.5)
            .attr('font-variant-numeric', 'tabular-nums')
            .attr('fill', 'var(--color-ink-dim)');

          g.append('rect')
            .attr('class', 'hit')
            .attr('x', 0)
            .attr('y', 0)
            .attr('height', ROW_HEIGHT)
            .attr('fill', 'transparent')
            .style('cursor', 'pointer');

          return g;
        },
        (update) => update,
        (exit) => exit.transition().duration(DURATION).style('opacity', 0).remove(),
      );

    groups
      .transition()
      .duration(DURATION)
      .style('opacity', 1)
      .attr('transform', (_d, i) => `translate(0,${i * ROW_HEIGHT})`);

    groups
      .select<SVGTextElement>('text.token')
      .text((d) => d.text)
      .attr('fill', (d) => (d.chosen ? 'var(--color-amber)' : 'var(--color-ink)'))
      .attr('font-weight', (d) => (d.chosen ? 600 : 400));

    groups
      .select<SVGRectElement>('rect.raw')
      .transition()
      .duration(DURATION)
      .attr('width', (d) => Math.max(1, x(d.raw)));

    groups
      .select<SVGRectElement>('rect.shaped')
      .transition()
      .duration(DURATION)
      .attr('width', (d) => x(d.shaped))
      .attr('fill', (d) =>
        d.chosen ? 'var(--color-amber)' : probabilityColor(d.shaped / maxValue),
      );

    groups
      .select<SVGTextElement>('text.value')
      .attr('x', LABEL_WIDTH + barWidth + 8)
      .text((d) => (d.shaped > 0 ? percent(d.shaped) : 'cut'))
      .attr('fill', (d) => (d.shaped > 0 ? 'var(--color-ink-dim)' : 'var(--color-ink-faint)'));

    groups
      .select<SVGRectElement>('rect.hit')
      .attr('width', width)
      .on('click', (_event, d) => {
        onPickRef.current?.({ id: d.id, text: d.text, prob: d.raw });
      })
      .append('title')
      .text(
        (d) => `id ${d.id} · model ${percent(d.raw, 2)} · after sampling ${percent(d.shaped, 2)}`,
      );
  }, [raw, shaped, chosenId, rows]);

  return <svg ref={ref} className="w-full" role="img" aria-label="Next-token probabilities" />;
}
