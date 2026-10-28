import { useEffect, useMemo, useRef } from 'react';
import { select } from 'd3';
import { sankey, sankeyLeft, sankeyLinkHorizontal } from 'd3-sankey';
import type { SankeyGraph, SankeyNode, SankeyLink } from 'd3-sankey';
import { Panel, Empty, Stat } from './Panel';
import { useRunStore, selectCurrentStep } from '../store/useRunStore';
import { buildLensSankey, type SankeyLinkInput, type SankeyNodeInput } from '../lib/lens';
import { displayToken, percent } from '../lib/format';
import { tokenColor } from '../lib/color';

type Node = SankeyNode<SankeyNodeInput, SankeyLinkInput>;
type Link = SankeyLink<SankeyNodeInput, SankeyLinkInput>;

const HEIGHT = 300;
const MARGIN = { top: 14, right: 70, bottom: 26, left: 24 };
const DURATION = 420;

/**
 * The logit lens as a flow.
 *
 * Reading the residual stream at every layer gives a running guess at the next token. This
 * chart shows the top candidates as columns, one per layer, and the ribbons between them
 * are probability mass moving: a ribbon that stays flat is a candidate holding its ground,
 * a crossing ribbon is the model changing its mind on the way up the stack.
 */
export function LayerFlowSankey() {
  const step = useRunStore(selectCurrentStep);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const data = useMemo(
    () => (step ? buildLensSankey(step.lens, { maxCandidates: 5 }) : null),
    [step],
  );

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg || !data || data.nodes.length === 0) return;

    const width = Math.max(360, svg.clientWidth || 560);
    const innerWidth = width - MARGIN.left - MARGIN.right;
    const innerHeight = HEIGHT - MARGIN.top - MARGIN.bottom;

    const layout = sankey<SankeyNodeInput, SankeyLinkInput>()
      .nodeId((node) => node.id)
      .nodeAlign(sankeyLeft)
      .nodeWidth(11)
      .nodePadding(9)
      .extent([
        [MARGIN.left, MARGIN.top],
        [MARGIN.left + innerWidth, MARGIN.top + innerHeight],
      ]);

    // d3-sankey mutates what it is given, so it gets copies.
    const graph: SankeyGraph<SankeyNodeInput, SankeyLinkInput> = layout({
      nodes: data.nodes.map((node) => ({ ...node })),
      links: data.links.map((link) => ({ ...link })),
    });

    const root = select(svg).attr('viewBox', `0 0 ${width} ${HEIGHT}`).attr('height', HEIGHT);

    let linkG = root.select<SVGGElement>('g.links');
    if (linkG.empty()) {
      linkG = root.append('g').attr('class', 'links').attr('fill', 'none');
      root.append('g').attr('class', 'nodes');
      root.append('g').attr('class', 'axis');
    }

    const path = sankeyLinkHorizontal<SankeyNodeInput, SankeyLinkInput>();

    linkG
      .selectAll<SVGPathElement, Link>('path')
      .data(graph.links, (d) => `${(d.source as Node).id}->${(d.target as Node).id}`)
      .join(
        (enter) =>
          enter
            .append('path')
            .attr('d', path)
            .attr('stroke-width', 0)
            .attr('stroke-opacity', 0)
            .call((selection) => selection.append('title')),
        (update) => update,
        (exit) => exit.transition().duration(DURATION).attr('stroke-opacity', 0).remove(),
      )
      .attr('stroke', (d) => tokenColor(d.tokenId, data.tokenOrder))
      .call((selection) =>
        selection.select('title').text((d) => {
          const source = d.source as Node;
          const target = d.target as Node;
          return d.kept
            ? `${displayToken(source.text)} keeps ${percent(d.value ?? 0)} from layer ${source.layer} to ${target.layer}`
            : `${percent(d.value ?? 0)} moves from ${displayToken(source.text)} to ${displayToken(target.text)}`;
        }),
      )
      .transition()
      .duration(DURATION)
      .attr('d', path)
      .attr('stroke-width', (d) => Math.max(1, d.width ?? 0))
      // Traded ribbons are the ones that slope from one candidate to another, which the
      // geometry already shows; a dash pattern on a thick near-horizontal ribbon just reads
      // as a barcode, so they are separated by weight instead.
      .attr('stroke-opacity', (d) => (d.kept ? 0.5 : 0.22));

    root
      .select<SVGGElement>('g.nodes')
      .selectAll<SVGGElement, Node>('g.node')
      .data(graph.nodes, (d) => d.id)
      .join(
        (enter) => {
          const g = enter.append('g').attr('class', 'node');
          g.append('rect').attr('rx', 2);
          g.append('text')
            .attr('font-family', 'var(--font-mono)')
            .attr('font-size', 10.5)
            .attr('dominant-baseline', 'middle');
          g.append('title');
          return g;
        },
        (update) => update,
        (exit) => exit.remove(),
      )
      .call((selection) => {
        selection
          .select('rect')
          .attr('fill', (d) => tokenColor(d.tokenId, data.tokenOrder))
          .transition()
          .duration(DURATION)
          .attr('x', (d) => d.x0 ?? 0)
          .attr('y', (d) => d.y0 ?? 0)
          .attr('width', (d) => (d.x1 ?? 0) - (d.x0 ?? 0))
          .attr('height', (d) => Math.max(1, (d.y1 ?? 0) - (d.y0 ?? 0)));

        selection
          .select('text')
          // Only the last column is labelled: repeating the same five words six times is
          // noise, and the ribbons already carry identity through colour.
          .text((d) => (d.layer === data.coverage.length - 1 ? displayToken(d.text) : ''))
          .attr('fill', 'var(--color-ink)')
          .transition()
          .duration(DURATION)
          .attr('x', (d) => (d.x1 ?? 0) + 6)
          .attr('y', (d) => ((d.y0 ?? 0) + (d.y1 ?? 0)) / 2);

        selection
          .select('title')
          .text(
            (d) =>
              `layer ${d.layer} · ${displayToken(d.text)} · ${percent(d.prob)} of the top-5 mass`,
          );
      });

    root
      .select<SVGGElement>('g.axis')
      .selectAll<SVGTextElement, number>('text')
      .data(
        data.coverage.map((_, index) => index),
        (d) => d,
      )
      .join('text')
      .attr('y', HEIGHT - 8)
      .attr('text-anchor', 'middle')
      .attr('font-size', 10)
      .attr('fill', 'var(--color-ink-faint)')
      .attr('x', (d) => {
        const node = graph.nodes.find((candidate) => candidate.layer === d);
        return ((node?.x0 ?? 0) + (node?.x1 ?? 0)) / 2 || 0;
      })
      .text((d) => (d === 0 ? 'embed' : `L${d}`));
  }, [data]);

  if (!step || !data || data.nodes.length === 0) {
    return (
      <Panel
        id="sankey"
        title="Layer flow (logit lens)"
        explain="If the model were forced to answer after each layer, what would it say? This shows that guess changing as the text travels up through the network."
      >
        <Empty>Run a generation to read the layers.</Empty>
      </Panel>
    );
  }

  const finalCoverage = data.coverage[data.coverage.length - 1] ?? 0;

  return (
    <Panel
      id="sankey"
      title="Layer flow (logit lens)"
      explain="If the model were forced to answer after each layer, what would it say? Each column is one layer; ribbon thickness is probability. A ribbon crossing to another word is the model changing its mind on the way up."
      aside={<Stat label="top-5 covers" value={percent(finalCoverage)} />}
    >
      <svg ref={svgRef} className="w-full" role="img" aria-label="Logit lens flow across layers" />
      <p className="mt-2 text-[11px] leading-relaxed text-ink-faint">
        Widths are shares of the five strongest candidates (together {percent(finalCoverage)} of
        the real distribution at the output layer), renormalised per layer so every column sums
        to one. A ribbon that runs flat is a candidate holding its mass; a ribbon that slopes
        across is mass moving to a different word.
      </p>
    </Panel>
  );
}
