import { describe, expect, it } from 'vitest';
import { bytes, displayToken, fixed, ms, percent, plainToken } from '../src/lib/format';
import { decodeAttention, encodeAttention } from '../src/worker/fixture';
import { clamp01, contrastInk, tokenColor } from '../src/lib/color';

describe('token display', () => {
  it('makes GPT-2 byte-level BPE markers visible', () => {
    // U+0120 is how the tokenizer writes a leading space; showing it is the point.
    expect(displayToken('Ġthe')).toBe('·the');
    expect(displayToken('Ċ')).toBe('↵');
    expect(displayToken(' the')).toBe('·the');
    expect(displayToken('the')).toBe('the');
  });

  it('turns the same markers back into real whitespace for the output readout', () => {
    expect(plainToken('Ġthe')).toBe(' the');
    expect(plainToken('Ċ')).toBe('\n');
  });
});

describe('number formatting', () => {
  it('formats percentages, with a floor so tiny values do not read as zero', () => {
    expect(percent(0.5)).toBe('50.0%');
    expect(percent(0.12345, 2)).toBe('12.35%');
    expect(percent(0.00004)).toBe('<0.1%');
    expect(percent(0)).toBe('0.0%');
    expect(percent(Number.NaN)).toBe('—');
  });

  it('formats fixed-point numbers and bails on non-finite input', () => {
    expect(fixed(1.239, 2)).toBe('1.24');
    expect(fixed(Number.POSITIVE_INFINITY)).toBe('—');
  });

  it('formats byte counts and durations', () => {
    expect(bytes(512)).toBe('512 B');
    expect(bytes(2048)).toBe('2 KB');
    expect(bytes(89_000_000)).toBe('84.9 MB');
    expect(ms(450)).toBe('450 ms');
    expect(ms(2500)).toBe('2.50 s');
  });
});

describe('colour helpers', () => {
  it('clamps and treats NaN as the bottom of the scale', () => {
    expect(clamp01(-1)).toBe(0);
    expect(clamp01(2)).toBe(1);
    expect(clamp01(Number.NaN)).toBe(0);
  });

  it('switches the ink colour on bright backgrounds', () => {
    expect(contrastInk(0.9)).not.toBe(contrastInk(0.1));
  });

  it('keeps a token colour stable and falls back to grey for unknown ids', () => {
    const order = [5, 9, 11];
    expect(tokenColor(9, order)).toBe(tokenColor(9, order));
    expect(tokenColor(5, order)).not.toBe(tokenColor(9, order));
    expect(tokenColor(404, order)).toBe('#6b7280');
  });
});

describe('fixture attention codec', () => {
  it('round-trips within the 8-bit quantisation error', () => {
    const original = new Float32Array([0, 0.25, 0.5, 0.75, 1]);
    const restored = decodeAttention({
      layers: 1,
      heads: 1,
      tokens: 1,
      u8: encodeAttention(original),
    });
    expect(restored).toHaveLength(original.length);
    original.forEach((value, index) => {
      expect(restored[index]).toBeCloseTo(value, 2);
    });
  });

  it('clamps out-of-range input instead of wrapping around', () => {
    const restored = decodeAttention({
      layers: 1,
      heads: 1,
      tokens: 1,
      u8: encodeAttention(new Float32Array([-0.5, 1.5])),
    });
    expect(restored[0]).toBe(0);
    expect(restored[1]).toBe(1);
  });
});
