/**
 * Turning the logit lens into a Sankey diagram.
 *
 * The logit lens asks: if the model stopped at layer N and decoded right now, what would
 * it say? Running that for every layer gives a sequence of distributions, and the
 * interesting thing is how probability *moves between candidates* as depth increases.
 *
 * d3-sankey wants a conserved flow, so the construction below is a transport plan:
 *
 *   1. Pick the candidate set U - the tokens that matter anywhere in the stack.
 *   2. At each layer, renormalise over U so every layer carries exactly 1.0 of mass.
 *   3. Between consecutive layers, a candidate keeps `min(p_l, p_next)` of its own mass and
 *      sheds any surplus; the shed mass is handed to the candidates that gained, matched up
 *      by the north-west corner rule so the diagram stays sparse.
 *
 * Mass is conserved by construction (the surplus and the deficit both equal
 * `0.5 * L1(p_l, p_next)`), so the ribbon widths are honest and the crossings are the
 * story: probability physically moving from one word to another as the model thinks.
 */
import type { LensLayer } from './protocol';

export interface SankeyNodeInput {
  /** Stable id of the form `L<layer>:<tokenId>`. */
  id: string;
  layer: number;
  tokenId: number;
  text: string;
  /** Renormalised probability of this candidate at this layer. */
  prob: number;
}

export interface SankeyLinkInput {
  source: string;
  target: string;
  value: number;
  /** True when the ribbon stays on the same token (mass kept rather than traded). */
  kept: boolean;
  tokenId: number;
}

export interface LensSankeyData {
  nodes: SankeyNodeInput[];
  links: SankeyLinkInput[];
  /** Candidate ids in a stable order, so colours do not jump between steps. */
  tokenOrder: number[];
  /**
   * Share of the real distribution that the chosen candidate set covers, per layer.
   * Shown as a caption so nobody mistakes the renormalised view for the full picture.
   */
  coverage: number[];
}

const EPSILON = 1e-9;

/** Picks the candidates worth drawing: highest probability reached anywhere in the stack. */
export function selectLensCandidates(layers: LensLayer[], maxCandidates: number): number[] {
  const best = new Map<number, number>();
  for (const layer of layers) {
    for (const candidate of layer.candidates) {
      const current = best.get(candidate.id) ?? 0;
      if (candidate.prob > current) best.set(candidate.id, candidate.prob);
    }
  }
  return [...best.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, maxCandidates)
    .map(([id]) => id);
}

export function buildLensSankey(
  layers: LensLayer[],
  options: { maxCandidates?: number } = {},
): LensSankeyData {
  const maxCandidates = options.maxCandidates ?? 5;
  const empty: LensSankeyData = { nodes: [], links: [], tokenOrder: [], coverage: [] };
  if (layers.length === 0) return empty;

  const tokenOrder = selectLensCandidates(layers, maxCandidates);
  if (tokenOrder.length === 0) return empty;

  const labels = new Map<number, string>();
  for (const layer of layers) {
    for (const candidate of layer.candidates) {
      if (!labels.has(candidate.id)) labels.set(candidate.id, candidate.text);
    }
  }

  // Renormalised probability of each selected candidate, per layer.
  const matrix: number[][] = [];
  const coverage: number[] = [];
  for (const layer of layers) {
    const byId = new Map(layer.candidates.map((c) => [c.id, c.prob]));
    const raw = tokenOrder.map((id) => byId.get(id) ?? 0);
    const total = raw.reduce((sum, v) => sum + v, 0);
    coverage.push(total);
    matrix.push(total > EPSILON ? raw.map((v) => v / total) : raw.map(() => 1 / raw.length));
  }

  const nodes: SankeyNodeInput[] = [];
  for (let layer = 0; layer < layers.length; layer += 1) {
    tokenOrder.forEach((tokenId, slot) => {
      const prob = (matrix[layer] as number[])[slot] as number;
      // A candidate that carries no probability at this layer gets no node. Keeping it
      // would hand d3-sankey an isolated zero-value node, which it parks in the first
      // column as a sliver and which squashes the real ribbons.
      if (prob <= EPSILON) return;
      nodes.push({
        id: nodeId(layer, tokenId),
        layer,
        tokenId,
        text: labels.get(tokenId) ?? String(tokenId),
        prob,
      });
    });
  }

  const links: SankeyLinkInput[] = [];
  for (let layer = 0; layer + 1 < layers.length; layer += 1) {
    const here = matrix[layer] as number[];
    const next = matrix[layer + 1] as number[];

    const surplus: { slot: number; amount: number }[] = [];
    const deficit: { slot: number; amount: number }[] = [];

    tokenOrder.forEach((tokenId, slot) => {
      const a = here[slot] as number;
      const b = next[slot] as number;
      const keptMass = Math.min(a, b);
      if (keptMass > EPSILON) {
        links.push({
          source: nodeId(layer, tokenId),
          target: nodeId(layer + 1, tokenId),
          value: keptMass,
          kept: true,
          tokenId,
        });
      }
      if (a - b > EPSILON) surplus.push({ slot, amount: a - b });
      if (b - a > EPSILON) deficit.push({ slot, amount: b - a });
    });

    // North-west corner rule: walk the shed and the gained side in step, settling whichever
    // runs out first. It produces the same marginals as a proportional split with at most
    // |surplus| + |deficit| - 1 ribbons instead of |surplus| * |deficit| - the difference
    // between a readable diagram and a barcode.
    let from = 0;
    let to = 0;
    while (from < surplus.length && to < deficit.length) {
      const source = surplus[from] as { slot: number; amount: number };
      const target = deficit[to] as { slot: number; amount: number };
      const value = Math.min(source.amount, target.amount);

      if (value > EPSILON) {
        const fromToken = tokenOrder[source.slot] as number;
        links.push({
          source: nodeId(layer, fromToken),
          target: nodeId(layer + 1, tokenOrder[target.slot] as number),
          value,
          kept: false,
          tokenId: fromToken,
        });
      }

      source.amount -= value;
      target.amount -= value;
      if (source.amount <= EPSILON) from += 1;
      if (target.amount <= EPSILON) to += 1;
    }
  }

  return { nodes, links, tokenOrder, coverage };
}

export function nodeId(layer: number, tokenId: number): string {
  return `L${layer}:${tokenId}`;
}

/** Total mass leaving a layer — 1 for every layer but the last. Used by the unit tests. */
export function layerOutflow(data: LensSankeyData, layer: number): number {
  return data.links
    .filter((link) => link.source.startsWith(`L${layer}:`))
    .reduce((sum, link) => sum + link.value, 0);
}

/** Total mass arriving at a layer — 1 for every layer but the first. */
export function layerInflow(data: LensSankeyData, layer: number): number {
  return data.links
    .filter((link) => link.target.startsWith(`L${layer}:`))
    .reduce((sum, link) => sum + link.value, 0);
}
