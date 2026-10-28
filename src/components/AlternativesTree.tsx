import { useEffect, useMemo, useRef } from 'react';
import { hierarchy, linkHorizontal, select, tree as d3tree } from 'd3';
import type { HierarchyPointLink, HierarchyPointNode } from 'd3';
import { Panel, Empty, Stat } from './Panel';
import { useRunStore } from '../store/useRunStore';
import { LIMITS } from '../lib/protocol';
import { countNodes, type TreeNode } from '../lib/tree';
import { displayToken, percent } from '../lib/format';

interface AlternativesTreeProps {
  onExpand: (path: number[]) => void;
  onFork: (path: number[]) => void;
}

const MARGIN = { top: 16, right: 110, bottom: 16, left: 86 };
const ROW = 34;
/** Fixed horizontal distance between depths, so one open branch does not span the panel. */
const COLUMN = 132;
const DURATION = 360;

/**
 * The roads not taken.
 *
 * Every node is a token the model could have emitted; its children are the top-5
 * continuations *from that point*, which means each expansion is a real forward pass on a
 * different prefix. Nothing is computed until a branch is opened — a fully expanded
 * 4-deep tree would be 780 forward passes.
 *
 * The branch the model actually walked is highlighted, and "fork" re-runs generation from
 * any node, which is the most direct way to feel how much one token choice changes.
 */
export function AlternativesTree({ onExpand, onFork }: AlternativesTreeProps) {
  const treeData = useRunStore((state) => state.tree);
  const tokens = useRunStore((state) => state.tokens);
  const running = useRunStore((state) => state.running);
  const collapseNode = useRunStore((state) => state.collapseNode);
  const svgRef = useRef<SVGSVGElement | null>(null);

  // Held in a ref so a new inline callback from the parent does not re-run the whole draw.
  const handlers = useRef({ onExpand, onFork, collapseNode });
  useEffect(() => {
    handlers.current = { onExpand, onFork, collapseNode };
  }, [onExpand, onFork, collapseNode]);

  const root = useMemo(() => hierarchy<TreeNode>(treeData, (node) => node.children), [treeData]);
  const nodeCount = useMemo(() => countNodes(treeData), [treeData]);
  const hasRun = tokens.length > 0;

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg || !hasRun) return;

    // nodeSize rather than size: the tree grows to fit its content instead of being
    // stretched across the panel, so a single open branch does not draw 400 px of empty link.
    const positioned = d3tree<TreeNode>().nodeSize([ROW, COLUMN])(root);

    let minX = Infinity;
    let maxX = -Infinity;
    let maxY = 0;
    positioned.each((node) => {
      if (node.x < minX) minX = node.x;
      if (node.x > maxX) maxX = node.x;
      if (node.y > maxY) maxY = node.y;
    });

    const height = Math.max(150, maxX - minX + MARGIN.top + MARGIN.bottom + ROW);
    const width = Math.max(380, maxY + MARGIN.left + MARGIN.right);

    const container = select(svg)
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('height', height)
      // Left-align: the default centres the viewBox inside the full-width element, which
      // would float a shallow tree in the middle of the panel.
      .attr('preserveAspectRatio', 'xMinYMid meet');

    let group = container.select<SVGGElement>('g.plot');
    if (group.empty()) {
      group = container.append('g').attr('class', 'plot');
      group.append('g').attr('class', 'links').attr('fill', 'none');
      group.append('g').attr('class', 'nodes');
    }
    group.attr('transform', `translate(${MARGIN.left},${MARGIN.top - minX + ROW / 2})`);

    const linkPath = linkHorizontal<HierarchyPointLink<TreeNode>, HierarchyPointNode<TreeNode>>()
      .x((d) => d.y)
      .y((d) => d.x);

    group
      .select<SVGGElement>('g.links')
      .selectAll<SVGPathElement, HierarchyPointLink<TreeNode>>('path')
      .data(positioned.links(), (d) => d.target.data.key)
      .join(
        (enter) => enter.append('path').attr('stroke-opacity', 0).attr('d', linkPath),
        (update) => update,
        (exit) => exit.transition().duration(DURATION).attr('stroke-opacity', 0).remove(),
      )
      .attr('stroke', (d) =>
        d.target.data.onChosenPath ? 'var(--color-amber)' : 'var(--color-edge-bright)',
      )
      .transition()
      .duration(DURATION)
      .attr('d', linkPath)
      // Thickness is the probability of taking this edge, so the model's own path is thick.
      .attr('stroke-width', (d) => 1 + 5 * d.target.data.prob)
      .attr('stroke-opacity', (d) => (d.target.data.onChosenPath ? 0.95 : 0.5));

    group
      .select<SVGGElement>('g.nodes')
      .selectAll<SVGGElement, HierarchyPointNode<TreeNode>>('g.node')
      .data(positioned.descendants(), (d) => d.data.key)
      .join(
        (enter) => {
          const g = enter.append('g').attr('class', 'node').style('opacity', 0);
          g.append('circle').attr('r', 4);
          g.append('text')
            .attr('class', 'token')
            .attr('dy', '0.32em')
            .attr('font-family', 'var(--font-mono)')
            .attr('font-size', 11.5);
          g.append('text')
            .attr('class', 'prob')
            .attr('dy', '0.32em')
            .attr('font-size', 10)
            .attr('font-variant-numeric', 'tabular-nums')
            .attr('fill', 'var(--color-ink-faint)');
          g.append('title');
          return g;
        },
        (update) => update,
        (exit) => exit.transition().duration(DURATION).style('opacity', 0).remove(),
      )
      .call((selection) => {
        selection
          .transition()
          .duration(DURATION)
          .style('opacity', 1)
          .attr('transform', (d) => `translate(${d.y},${d.x})`);

        selection
          .select('circle')
          .attr('fill', (d) =>
            d.data.status === 'loading'
              ? 'var(--color-violet)'
              : d.data.onChosenPath
                ? 'var(--color-amber)'
                : 'var(--color-accent)',
          )
          .attr('r', (d) => (d.data.depth === 0 ? 5 : 4))
          .style('cursor', 'pointer');

        selection
          .select('text.token')
          .attr('x', (d) => (d.data.depth === 0 ? -10 : 9))
          .attr('text-anchor', (d) => (d.data.depth === 0 ? 'end' : 'start'))
          .attr('fill', (d) => (d.data.onChosenPath ? 'var(--color-amber)' : 'var(--color-ink)'))
          .text((d) =>
            d.data.depth === 0
              ? 'prompt'
              : `${displayToken(d.data.text) || '␣'}${d.data.status === 'loading' ? ' …' : ''}`,
          );

        selection
          .select('text.prob')
          .attr('x', (d) => (d.data.depth === 0 ? -10 : 9))
          .attr('dy', '1.5em')
          .attr('text-anchor', (d) => (d.data.depth === 0 ? 'end' : 'start'))
          .text((d) => (d.data.depth === 0 ? '' : percent(d.data.prob)));

        selection
          .select('title')
          .text((d) =>
            d.data.depth === 0
              ? 'Click to see the first alternatives'
              : `${displayToken(d.data.text)} — ${percent(d.data.prob)} at this point, ${percent(
                  d.data.cumulative,
                )} for the whole path. Click to expand, shift-click to fork the run from here.`,
          );

        selection.on('click', (event: MouseEvent, d) => {
          if (event.shiftKey && d.data.depth > 0) {
            handlers.current.onFork(d.data.path);
            return;
          }
          if (d.data.status === 'expanded') handlers.current.collapseNode(d.data.path);
          else if (d.data.status === 'collapsed') handlers.current.onExpand(d.data.path);
        });
      });
  }, [root, hasRun]);

  return (
    <Panel
      id="tree"
      title="Alternatives"
      explain="At every step the model had other options. Click a node to see the five most likely continuations from that exact point — each click is a real forward pass. Shift-click to re-run the whole generation from that branch."
      aside={<Stat label="nodes" value={String(nodeCount)} />}
    >
      {hasRun ? (
        <>
          <div className="max-h-[460px] overflow-y-auto">
            <svg
              ref={svgRef}
              className="w-full"
              role="img"
              aria-label="Tree of alternative continuations"
            />
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-edge pt-3 text-[11px] text-ink-faint">
            <span>
              <span className="text-amber">■</span> the path taken
            </span>
            <span>click = expand · shift-click = fork</span>
            <span>max depth {LIMITS.treeMaxDepth}</span>
            {running && <span className="text-violet">model is busy — expansions queue up</span>}
          </div>
        </>
      ) : (
        <Empty>Generate first, then open a branch.</Empty>
      )}
    </Panel>
  );
}
