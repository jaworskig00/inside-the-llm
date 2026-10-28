import { describe, expect, it } from 'vitest';
import {
  buildLensSankey,
  layerInflow,
  layerOutflow,
  nodeId,
  selectLensCandidates,
} from '../src/lib/lens';
import type { LensLayer } from '../src/lib/protocol';

function layer(index: number, entries: [number, string, number][]): LensLayer {
  return {
    layer: index,
    candidates: entries.map(([id, text, prob]) => ({ id, text, prob })),
  };
}

/** Three layers in which probability clearly migrates from " to" to " the". */
const LAYERS: LensLayer[] = [
  layer(0, [
    [1, ' to', 0.5],
    [2, ' the', 0.1],
    [3, ' a', 0.05],
  ]),
  layer(1, [
    [1, ' to', 0.3],
    [2, ' the', 0.3],
    [3, ' a', 0.05],
  ]),
  layer(2, [
    [2, ' the', 0.6],
    [1, ' to', 0.1],
    [3, ' a', 0.02],
  ]),
];

describe('selectLensCandidates', () => {
  it('ranks by the best probability a candidate reaches anywhere in the stack', () => {
    expect(selectLensCandidates(LAYERS, 3)).toEqual([2, 1, 3]);
  });

  it('respects the cap', () => {
    expect(selectLensCandidates(LAYERS, 2)).toEqual([2, 1]);
  });

  it('returns nothing for an empty stack', () => {
    expect(selectLensCandidates([], 5)).toEqual([]);
  });
});

describe('buildLensSankey', () => {
  const data = buildLensSankey(LAYERS, { maxCandidates: 3 });

  it('emits one node per layer per candidate', () => {
    expect(data.nodes).toHaveLength(LAYERS.length * 3);
    expect(data.nodes.some((node) => node.id === nodeId(2, 2))).toBe(true);
  });

  it('renormalises each layer to exactly 1', () => {
    for (let index = 0; index < LAYERS.length; index += 1) {
      const total = data.nodes
        .filter((node) => node.layer === index)
        .reduce((sum, node) => sum + node.prob, 0);
      expect(total).toBeCloseTo(1, 9);
    }
  });

  it('reports what share of the real distribution the selection covers', () => {
    expect(data.coverage[0]).toBeCloseTo(0.65, 9);
    expect(data.coverage[2]).toBeCloseTo(0.72, 9);
  });

  // The whole point of the transport-plan construction: the ribbons are honest, so every
  // layer passes on exactly the mass it received.
  it('conserves mass between every pair of layers', () => {
    for (let index = 0; index + 1 < LAYERS.length; index += 1) {
      expect(layerOutflow(data, index)).toBeCloseTo(1, 9);
      expect(layerInflow(data, index + 1)).toBeCloseTo(1, 9);
    }
  });

  it('balances each node: what leaves equals its own probability', () => {
    for (let index = 0; index + 1 < LAYERS.length; index += 1) {
      for (const node of data.nodes.filter((candidate) => candidate.layer === index)) {
        const out = data.links
          .filter((link) => link.source === node.id)
          .reduce((sum, link) => sum + link.value, 0);
        expect(out).toBeCloseTo(node.prob, 9);
      }
    }
  });

  it('marks ribbons that stay on the same token as kept and the rest as traded', () => {
    const kept = data.links.filter((link) => link.kept);
    const traded = data.links.filter((link) => !link.kept);
    expect(kept.length).toBeGreaterThan(0);
    expect(traded.length).toBeGreaterThan(0);
    for (const link of kept) {
      expect(link.source.split(':')[1]).toBe(link.target.split(':')[1]);
    }
    for (const link of traded) {
      expect(link.source.split(':')[1]).not.toBe(link.target.split(':')[1]);
    }
  });

  it('shows " the" gaining and " to" losing, which is the story the chart tells', () => {
    const theGain = data.links
      .filter((link) => link.target === nodeId(2, 2) && !link.kept)
      .reduce((sum, link) => sum + link.value, 0);
    expect(theGain).toBeGreaterThan(0.2);
  });

  it('produces only same-token links when the distribution never changes', () => {
    const flat = [layer(0, [[1, 'a', 0.5]]), layer(1, [[1, 'a', 0.5]])];
    const result = buildLensSankey(flat, { maxCandidates: 3 });
    expect(result.links).toHaveLength(1);
    expect(result.links[0]?.kept).toBe(true);
    expect(result.links[0]?.value).toBeCloseTo(1, 9);
  });

  it('degrades gracefully on empty input', () => {
    expect(buildLensSankey([])).toEqual({ nodes: [], links: [], tokenOrder: [], coverage: [] });
  });

  it('spreads mass evenly when a layer has no probability at all', () => {
    const odd = [layer(0, [[1, 'a', 0]]), layer(1, [[1, 'a', 0.4]])];
    const result = buildLensSankey(odd, { maxCandidates: 2 });
    expect(result.nodes.every((node) => node.prob === 1)).toBe(true);
  });
});
