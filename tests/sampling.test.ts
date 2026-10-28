import { describe, expect, it } from 'vitest';
import {
  entropyBits,
  makeRng,
  sampleFrom,
  shapeDistribution,
  softmax,
  topIndices,
} from '../src/lib/sampling';

// softmax returns a Float32Array, so sums are compared to six decimals rather than ten -
// float32 carries about seven decimal digits and the accumulation eats the rest.
describe('softmax', () => {
  it('produces a distribution', () => {
    const probs = softmax([1, 2, 3]);
    expect(sum(probs)).toBeCloseTo(1, 6);
    expect(probs[2]).toBeGreaterThan(probs[1] as number);
  });

  it('is stable for large logits', () => {
    const probs = softmax([1000, 1001, 999]);
    expect(Number.isNaN(probs[0] as number)).toBe(false);
    expect(sum(probs)).toBeCloseTo(1, 6);
  });

  it('flattens as temperature rises and sharpens as it falls', () => {
    const hot = softmax([0, 1, 2], 5);
    const cold = softmax([0, 1, 2], 0.2);
    expect(entropyBits(hot)).toBeGreaterThan(entropyBits(cold));
    expect(cold[2]).toBeGreaterThan(0.99);
  });

  it('treats a non-positive temperature as 1 rather than dividing by zero', () => {
    expect(sum(softmax([0, 1, 2], 0))).toBeCloseTo(1, 6);
  });
});

describe('topIndices', () => {
  it('returns the k largest, largest first', () => {
    expect(topIndices([0.1, 0.9, 0.4, 0.95, 0.2], 3)).toEqual([3, 1, 2]);
  });

  it('agrees with a full sort on random data', () => {
    const values = Array.from({ length: 5000 }, (_, i) => (Math.sin(i * 12.9898) * 43758.5453) % 1);
    const expected = values
      .map((value, index) => ({ value, index }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 24)
      .map((entry) => entry.index);
    expect(topIndices(values, 24)).toEqual(expected);
  });

  it('handles k larger than the input and k of zero', () => {
    expect(topIndices([3, 1], 9)).toEqual([0, 1]);
    expect(topIndices([3, 1], 0)).toEqual([]);
  });
});

describe('shapeDistribution', () => {
  const logits = [3, 2.5, 2, 1, 0, -1, -4];

  it('renormalises whatever survives', () => {
    const { candidates } = shapeDistribution(logits, { temperature: 1, topK: 3, topP: 1 });
    expect(candidates).toHaveLength(3);
    expect(candidates.reduce((total, c) => total + c.prob, 0)).toBeCloseTo(1, 10);
  });

  it('collapses to the single best token at temperature 0', () => {
    const { candidates, kept } = shapeDistribution(logits, { temperature: 0, topK: 50, topP: 1 });
    expect(kept).toBe(1);
    expect(candidates).toEqual([{ id: 0, prob: 1 }]);
  });

  it('keeps the shortest prefix that reaches p', () => {
    const relaxed = shapeDistribution(logits, { temperature: 1, topK: 0, topP: 0.99 });
    const strict = shapeDistribution(logits, { temperature: 1, topK: 0, topP: 0.5 });
    expect(strict.kept).toBeLessThan(relaxed.kept);
    // top-p never returns an empty set, even when p is smaller than the top probability.
    expect(shapeDistribution(logits, { temperature: 1, topK: 0, topP: 0.05 }).kept).toBe(1);
  });

  it('applies top-k before top-p', () => {
    // top-k=2 leaves ~0.73 of the mass; a p of 0.9 then cannot pull in a third token.
    const { kept } = shapeDistribution(logits, { temperature: 1, topK: 2, topP: 0.9 });
    expect(kept).toBe(2);
  });

  it('never lets a filtered-out token keep probability', () => {
    const { candidates } = shapeDistribution(logits, { temperature: 1, topK: 2, topP: 1 });
    expect(candidates.map((c) => c.id)).toEqual([0, 1]);
  });
});

describe('makeRng / sampleFrom', () => {
  it('is deterministic for a seed', () => {
    const a = makeRng(42);
    const b = makeRng(42);
    expect([a(), a(), a()]).toEqual([b(), b(), b()]);
  });

  it('stays inside [0, 1)', () => {
    const rng = makeRng(1);
    for (let i = 0; i < 2000; i += 1) {
      const value = rng();
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
  });

  it('picks candidates in proportion to their probability', () => {
    const candidates = [
      { id: 10, prob: 0.7 },
      { id: 20, prob: 0.2 },
      { id: 30, prob: 0.1 },
    ];
    const rng = makeRng(2026);
    const counts = new Map<number, number>();
    for (let i = 0; i < 20000; i += 1) {
      const id = sampleFrom(candidates, rng);
      counts.set(id, (counts.get(id) ?? 0) + 1);
    }
    expect((counts.get(10) ?? 0) / 20000).toBeCloseTo(0.7, 1);
    expect((counts.get(20) ?? 0) / 20000).toBeCloseTo(0.2, 1);
    expect((counts.get(30) ?? 0) / 20000).toBeCloseTo(0.1, 1);
  });

  it('returns the last candidate rather than -1 when rounding leaves a gap', () => {
    expect(
      sampleFrom(
        [
          { id: 5, prob: 0.4 },
          { id: 6, prob: 0.4 },
        ],
        () => 0.999,
      ),
    ).toBe(6);
    expect(sampleFrom([], () => 0.5)).toBe(-1);
  });
});

describe('entropyBits', () => {
  it('is 0 for a certain distribution and log2(n) for a uniform one', () => {
    expect(entropyBits([1, 0, 0, 0])).toBeCloseTo(0, 10);
    expect(entropyBits([0.25, 0.25, 0.25, 0.25])).toBeCloseTo(2, 10);
  });
});

function sum(values: ArrayLike<number>): number {
  let total = 0;
  for (let i = 0; i < values.length; i += 1) total += values[i] as number;
  return total;
}
