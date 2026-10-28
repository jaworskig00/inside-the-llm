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
 *   3. Between consecutive layers, a candidate keeps `min(p_l, p_next)` of its own mass,
 *      sheds any surplus, and the shed mass is distributed over the candidates that
 *      gained, in proportion to how much each gained.
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
      nodes.push({
        id: nodeId(layer, tokenId),
        layer,
        tokenId,
        text: labels.get(tokenId) ?? String(tokenId),
        prob: (matrix[layer] as number[])[slot] as number,
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

    const deficitTotal = deficit.reduce((sum, d) => sum + d.amount, 0);
    if (deficitTotal <= EPSILON) continue;

    for (const from of surplus) {
      for (const to of deficit) {
        const value = (from.amount * to.amount) / deficitTotal;
        if (value <= EPSILON) continue;
        const fromToken = tokenOrder[from.slot] as number;
        links.push({
          source: nodeId(layer, fromToken),
          target: nodeId(layer + 1, tokenOrder[to.slot] as number),
          value,
          kept: false,
          tokenId: fromToken,
        });
      }
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
