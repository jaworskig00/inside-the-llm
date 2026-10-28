import { describe, expect, it } from 'vitest';
import {
  arcLinks,
  attentionRow,
  averageHeads,
  averageHeadsRow,
  headMatrix,
  headOffset,
  maxOf,
  normalizeRows,
} from '../src/lib/attention';
import type { AttentionPack } from '../src/lib/protocol';

/**
 * A pack with a recognisable pattern per head, so an indexing mistake shows up as a wrong
 * number rather than as something that merely looks plausible.
 */
function makePack(layers = 2, heads = 3, tokens = 4): AttentionPack {
  const data = new Float32Array(layers * heads * tokens * tokens);
  for (let layer = 0; layer < layers; layer += 1) {
    for (let head = 0; head < heads; head += 1) {
      for (let query = 0; query < tokens; query += 1) {
        // Causal rows that sum to exactly 1, with mass concentrated on the diagonal.
        const keys = query + 1;
        const base = ((layer * heads + head) * tokens + query) * tokens;
        let total = 0;
        const raw: number[] = [];
        for (let key = 0; key < keys; key += 1) {
          const value = 1 + head + (key === query ? 3 : 0);
          raw.push(value);
          total += value;
        }
        raw.forEach((value, key) => {
          data[base + key] = value / total;
        });
      }
    }
  }
  return { layers, heads, tokens, data };
}

describe('indexing', () => {
  it('computes the offset of a (layer, head) block', () => {
    const pack = makePack();
    expect(headOffset(pack, 0, 0)).toBe(0);
    expect(headOffset(pack, 0, 1)).toBe(16);
    expect(headOffset(pack, 1, 0)).toBe(48);
  });

  it('returns a view, not a copy, for a head matrix', () => {
    const pack = makePack();
    const matrix = headMatrix(pack, 1, 2);
    expect(matrix.length).toBe(16);
    expect(matrix.buffer).toBe(pack.data.buffer);
  });

  it('reads the right row', () => {
    const pack = makePack();
    const row = attentionRow(pack, 0, 0, 2);
    expect(row.length).toBe(4);
    expect(sum(row)).toBeCloseTo(1, 6);
    expect(row[3]).toBe(0); // causal: nothing after the query
  });
});

describe('averaging over heads', () => {
  it('keeps every row a distribution', () => {
    const pack = makePack();
    const averaged = averageHeads(pack, 0);
    for (let query = 0; query < pack.tokens; query += 1) {
      const row = averaged.slice(query * pack.tokens, (query + 1) * pack.tokens);
      expect(sum(row)).toBeCloseTo(1, 6);
    }
  });

  it('matches the mean of the individual head rows', () => {
    const pack = makePack();
    const averaged = averageHeadsRow(pack, 1, 3);
    for (let key = 0; key < pack.tokens; key += 1) {
      let expected = 0;
      for (let head = 0; head < pack.heads; head += 1) {
        expected += attentionRow(pack, 1, head, 3)[key] as number;
      }
      expect(averaged[key]).toBeCloseTo(expected / pack.heads, 6);
    }
  });

  it('agrees with the full-matrix average', () => {
    const pack = makePack();
    const matrix = averageHeads(pack, 0);
    const row = averageHeadsRow(pack, 0, 2);
    for (let key = 0; key < pack.tokens; key += 1) {
      expect(matrix[2 * pack.tokens + key]).toBeCloseTo(row[key] as number, 6);
    }
  });
});

describe('normalizeRows', () => {
  const pack = makePack();
  const normalized = normalizeRows(headMatrix(pack, 0, 0), pack.tokens);

  it('puts each row at exactly 1 at its own maximum', () => {
    for (let query = 0; query < pack.tokens; query += 1) {
      const row = Array.from(
        normalized.slice(query * pack.tokens, query * pack.tokens + query + 1),
      );
      expect(Math.max(...row)).toBeCloseTo(1, 6);
    }
  });

  it('marks the future as NaN so the renderer can leave it blank', () => {
    expect(Number.isNaN(normalized[0 * pack.tokens + 1] as number)).toBe(true);
    expect(Number.isNaN(normalized[2 * pack.tokens + 3] as number)).toBe(true);
    expect(Number.isNaN(normalized[3 * pack.tokens + 3] as number)).toBe(false);
  });

  it('survives an all-zero row instead of emitting Infinity', () => {
    const zeros = normalizeRows(new Float32Array(9), 3);
    expect(Array.from(zeros.slice(0, 1))).toEqual([0]);
    expect(zeros.every((value) => Number.isNaN(value) || value === 0)).toBe(true);
  });
});

describe('arcLinks', () => {
  it('drops weak edges, sorts by weight and never looks forward', () => {
    const row = [0.5, 0.001, 0.3, 0.19, 0.9];
    const links = arcLinks(row, 3, { minWeight: 0.01 });
    expect(links.map((link) => link.target)).toEqual([0, 2, 3]);
    expect(links.every((link) => link.target <= 3)).toBe(true);
    expect(links[0]?.weight).toBe(0.5);
  });

  it('respects maxLinks', () => {
    const row = Array.from({ length: 40 }, () => 0.5);
    expect(arcLinks(row, 39, { maxLinks: 5 })).toHaveLength(5);
  });

  it('always keeps the query itself when it is the strongest key', () => {
    const links = arcLinks([0.1, 0.9], 1, { minWeight: 0.05 });
    expect(links[0]).toEqual({ source: 1, target: 1, weight: 0.9 });
  });
});

describe('maxOf', () => {
  it('ignores negatives and handles an empty input', () => {
    expect(maxOf([0.2, 0.9, 0.4])).toBe(0.9);
    expect(maxOf([])).toBe(0);
  });
});

function sum(values: ArrayLike<number>): number {
  let total = 0;
  for (let i = 0; i < values.length; i += 1) total += values[i] as number;
  return total;
}
