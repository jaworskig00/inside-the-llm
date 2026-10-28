/**
 * Reading and reshaping attention tensors.
 *
 * The worker ships one flat `Float32Array` per step holding `layers x heads x T x T`
 * weights. These helpers are the only place that knows that layout.
 */
import type { AttentionPack } from './protocol';

export function headOffset(pack: AttentionPack, layer: number, head: number): number {
  return (layer * pack.heads + head) * pack.tokens * pack.tokens;
}

/** One head's full `T x T` matrix, as a view onto the original buffer (no copy). */
export function headMatrix(pack: AttentionPack, layer: number, head: number): Float32Array {
  const start = headOffset(pack, layer, head);
  return pack.data.subarray(start, start + pack.tokens * pack.tokens);
}

/** The attention a single query token pays to every key token, for one head. */
export function attentionRow(
  pack: AttentionPack,
  layer: number,
  head: number,
  query: number,
): Float32Array {
  const start = headOffset(pack, layer, head) + query * pack.tokens;
  return pack.data.subarray(start, start + pack.tokens);
}

/**
 * Mean over heads for one layer. Averaging post-softmax rows keeps the result a valid
 * distribution (each row still sums to 1), which is why it is a defensible "overview".
 */
export function averageHeads(pack: AttentionPack, layer: number): Float32Array {
  const size = pack.tokens * pack.tokens;
  const out = new Float32Array(size);
  for (let head = 0; head < pack.heads; head += 1) {
    const start = headOffset(pack, layer, head);
    for (let i = 0; i < size; i += 1)
      out[i] = (out[i] as number) + (pack.data[start + i] as number);
  }
  for (let i = 0; i < size; i += 1) out[i] = (out[i] as number) / pack.heads;
  return out;
}

/** Mean over heads of a single query row. */
export function averageHeadsRow(pack: AttentionPack, layer: number, query: number): Float32Array {
  const out = new Float32Array(pack.tokens);
  for (let head = 0; head < pack.heads; head += 1) {
    const start = headOffset(pack, layer, head) + query * pack.tokens;
    for (let i = 0; i < pack.tokens; i += 1) {
      out[i] = (out[i] as number) + (pack.data[start + i] as number);
    }
  }
  for (let i = 0; i < pack.tokens; i += 1) out[i] = (out[i] as number) / pack.heads;
  return out;
}

export function maxOf(values: ArrayLike<number>): number {
  let max = 0;
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i] as number;
    if (v > max) max = v;
  }
  return max;
}

/**
 * Scales a causal attention matrix so each heatmap uses its full colour range.
 *
 * Attention is causal, so row `i` spreads its mass over `i + 1` keys: early rows are
 * near 1.0 and late rows are near `1/T`. Colouring raw values makes every head look like
 * the same bright triangle, so each row is divided by its own maximum. The upper
 * triangle is left as NaN, which the renderer draws as empty.
 */
export function normalizeRows(matrix: ArrayLike<number>, tokens: number): Float32Array {
  const out = new Float32Array(tokens * tokens).fill(NaN);
  for (let row = 0; row < tokens; row += 1) {
    const base = row * tokens;
    let max = 0;
    for (let col = 0; col <= row; col += 1) {
      const v = matrix[base + col] as number;
      if (v > max) max = v;
    }
    const scale = max > 0 ? 1 / max : 0;
    for (let col = 0; col <= row; col += 1) {
      out[base + col] = (matrix[base + col] as number) * scale;
    }
  }
  return out;
}

export interface AttentionLink {
  source: number;
  target: number;
  weight: number;
}

/**
 * Edges for the arc diagram: where the selected token looks.
 *
 * Only the strongest edges are kept — a 64-token context has 2080 causal pairs and drawing
 * all of them is noise, not information.
 */
export function arcLinks(
  row: ArrayLike<number>,
  query: number,
  options: { minWeight?: number; maxLinks?: number } = {},
): AttentionLink[] {
  const minWeight = options.minWeight ?? 0.02;
  const maxLinks = options.maxLinks ?? 24;

  const links: AttentionLink[] = [];
  for (let key = 0; key <= query && key < row.length; key += 1) {
    const weight = row[key] as number;
    if (weight >= minWeight) links.push({ source: query, target: key, weight });
  }
  links.sort((a, b) => b.weight - a.weight);
  return links.slice(0, maxLinks);
}

/**
 * How concentrated a head is: the share of attention that the single strongest key takes.
 * Used to sort / label heads in the grid ("this one is a previous-token head").
 */
export function headFocus(pack: AttentionPack, layer: number, head: number): number {
  const row = attentionRow(pack, layer, head, pack.tokens - 1);
  return maxOf(row);
}
