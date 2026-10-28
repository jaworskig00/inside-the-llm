/**
 * The sampling maths, kept free of DOM and worker concerns so it can be unit tested and
 * reused by the live probability chart (which has to show what the sliders *would* do
 * before the next token is drawn).
 */

/** Numerically stable softmax with an optional temperature. */
export function softmax(logits: ArrayLike<number>, temperature = 1): Float32Array {
  const n = logits.length;
  const out = new Float32Array(n);
  const t = temperature > 0 ? temperature : 1;

  let max = -Infinity;
  for (let i = 0; i < n; i += 1) {
    const v = logits[i] as number;
    if (v > max) max = v;
  }

  let sum = 0;
  for (let i = 0; i < n; i += 1) {
    const e = Math.exp(((logits[i] as number) - max) / t);
    out[i] = e;
    sum += e;
  }

  if (sum === 0) return out;
  for (let i = 0; i < n; i += 1) out[i] = (out[i] as number) / sum;
  return out;
}

export interface IndexedProb {
  id: number;
  prob: number;
}

/**
 * Indices of the `k` largest values, largest first.
 *
 * Uses a bounded insertion list rather than sorting 50257 entries: at k=24 this is the
 * difference between ~1 ms and ~15 ms per generated token.
 */
export function topIndices(values: ArrayLike<number>, k: number): number[] {
  const limit = Math.max(0, Math.min(k, values.length));
  if (limit === 0) return [];

  const ids: number[] = [];
  const scores: number[] = [];

  for (let i = 0; i < values.length; i += 1) {
    const v = values[i] as number;
    if (ids.length === limit && v <= (scores[limit - 1] as number)) continue;

    let pos = ids.length === limit ? limit - 1 : ids.length;
    if (ids.length < limit) {
      ids.push(i);
      scores.push(v);
    } else {
      ids[pos] = i;
      scores[pos] = v;
    }
    while (pos > 0 && (scores[pos - 1] as number) < v) {
      ids[pos] = ids[pos - 1] as number;
      scores[pos] = scores[pos - 1] as number;
      pos -= 1;
      ids[pos] = i;
      scores[pos] = v;
    }
  }

  return ids;
}

/**
 * Applies temperature, then top-k, then top-p, in that order — the order matters and this
 * is the order llama.cpp and HF `generate` use.
 *
 * Returns a sparse view: only the candidates that survived, with probabilities
 * renormalised to sum to 1. A temperature of 0 collapses to greedy decoding.
 */
export function shapeDistribution(
  logits: ArrayLike<number>,
  options: { temperature: number; topK: number; topP: number; limit?: number },
): { candidates: IndexedProb[]; kept: number } {
  const { temperature, topK, topP } = options;
  const limit = options.limit ?? 64;

  if (temperature <= 0) {
    const [best] = topIndices(logits, 1);
    if (best === undefined) return { candidates: [], kept: 0 };
    return { candidates: [{ id: best, prob: 1 }], kept: 1 };
  }

  const probs = softmax(logits, temperature);

  // top-k, taken over the temperature-scaled distribution.
  const kCap = topK > 0 ? Math.min(topK, probs.length) : probs.length;
  const pool = topIndices(probs, Math.max(kCap, 1));

  // top-p (nucleus): keep the shortest prefix whose mass reaches p.
  let kept = pool.length;
  if (topP < 1) {
    let cumulative = 0;
    kept = 0;
    for (const id of pool) {
      cumulative += probs[id] as number;
      kept += 1;
      if (cumulative >= topP) break;
    }
  }

  const survivors = pool.slice(0, Math.max(kept, 1));
  let total = 0;
  for (const id of survivors) total += probs[id] as number;

  const candidates: IndexedProb[] = survivors
    .slice(0, limit)
    .map((id) => ({ id, prob: total > 0 ? (probs[id] as number) / total : 0 }));

  return { candidates, kept: survivors.length };
}

/**
 * mulberry32 — a small, fast, well-distributed PRNG.
 * Seeded so that the same prompt and settings always produce the same text, which matters
 * for a demo people will screenshot and for the Playwright run.
 */
export function makeRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Draws one candidate. `candidates` must carry probabilities that sum to ~1. */
export function sampleFrom(candidates: IndexedProb[], random: () => number): number {
  if (candidates.length === 0) return -1;
  if (candidates.length === 1) return (candidates[0] as IndexedProb).id;

  const roll = random();
  let cumulative = 0;
  for (const candidate of candidates) {
    cumulative += candidate.prob;
    if (roll < cumulative) return candidate.id;
  }
  return (candidates[candidates.length - 1] as IndexedProb).id;
}

/** Shannon entropy in bits — a one-number summary of "how undecided is the model". */
export function entropyBits(probs: ArrayLike<number>): number {
  let h = 0;
  for (let i = 0; i < probs.length; i += 1) {
    const p = probs[i] as number;
    if (p > 0) h -= p * Math.log2(p);
  }
  return h;
}
