/** Small formatting helpers shared by every view. */

/**
 * GPT-2's byte-level BPE writes a leading space as U+0120 and a newline as U+010A.
 * Tokens have to be shown as they really are - a chip reading "·the" tells the truth about
 * why " the" and "the" are different tokens, which is half the point of the token strip.
 */
export function displayToken(text: string): string {
  return text.replace(/Ġ/g, '·').replace(/Ċ/g, '↵').replace(/^ /, '·').replace(/\n/g, '↵');
}

/**
 * GPT-2's byte-to-unicode alphabet, inverted.
 *
 * Byte-level BPE never stores bytes directly - every byte is mapped to a printable
 * codepoint so that the vocabulary is text. "o" with an acute accent is two UTF-8 bytes and
 * appears in the vocabulary as two odd-looking characters; a space is U+0120. Concatenating
 * those strings and calling the result output produces mojibake for anything outside ASCII,
 * so the mapping has to be undone to bytes and the bytes decoded as UTF-8.
 */
const BYTE_DECODER = buildByteDecoder();

function buildByteDecoder(): Map<string, number> {
  const printable: number[] = [];
  for (let b = 0x21; b <= 0x7e; b += 1) printable.push(b);
  for (let b = 0xa1; b <= 0xac; b += 1) printable.push(b);
  for (let b = 0xae; b <= 0xff; b += 1) printable.push(b);

  const bytes = [...printable];
  const codes = [...printable];
  let extra = 0;
  for (let b = 0; b < 256; b += 1) {
    if (printable.includes(b)) continue;
    bytes.push(b);
    codes.push(256 + extra);
    extra += 1;
  }

  const decoder = new Map<string, number>();
  bytes.forEach((byte, index) => decoder.set(String.fromCharCode(codes[index] as number), byte));
  return decoder;
}

const UTF8 = new TextDecoder('utf-8');
const UTF8_ENCODER = new TextEncoder();

/**
 * Turns a run of raw vocabulary entries back into the text they encode.
 *
 * Decoding the whole run at once rather than token by token matters: one token is often
 * half of a multi-byte character, and decoding it alone yields a replacement character
 * where joining it with its neighbour gives the real one.
 */
export function decodeTokens(tokens: readonly string[]): string {
  const bytes: number[] = [];
  for (const token of tokens) {
    for (const character of token) {
      const byte = BYTE_DECODER.get(character);
      if (byte === undefined) {
        // Not part of the byte alphabet - an added or special token. Pass it through.
        for (const unit of UTF8_ENCODER.encode(character)) bytes.push(unit);
      } else {
        bytes.push(byte);
      }
    }
  }
  return UTF8.decode(new Uint8Array(bytes));
}

/** One token on its own, for tooltips. Mid-sequence bytes decode to a replacement char. */
export function decodeToken(token: string): string {
  return decodeTokens([token]);
}

export function percent(value: number, digits = 1): string {
  if (!Number.isFinite(value)) return '—';
  if (value > 0 && value < 0.001) return '<0.1%';
  return `${(value * 100).toFixed(digits)}%`;
}

export function fixed(value: number, digits = 2): string {
  return Number.isFinite(value) ? value.toFixed(digits) : '—';
}

export function bytes(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(0)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

export function ms(value: number): string {
  return value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(2)} s`;
}

export const PRESET_PROMPTS = [
  {
    label: 'Capital city',
    text: 'The capital of France is',
    note: 'A fact small models usually get right — watch the lens lock on early.',
  },
  {
    label: 'Recipe',
    text: 'To make a good omelette you need eggs, butter and',
    note: 'A list the model wants to continue — attention leans on the previous items.',
  },
  {
    label: 'Story opening',
    text: 'The lighthouse keeper had not spoken to anyone in',
    note: 'Open-ended narrative: high entropy, the alternatives tree branches wide.',
  },
  {
    label: 'Code comment',
    text: '// returns the sum of all numbers in the list\nfunction sum(numbers) {',
    note: 'Code puts a lot of weight on punctuation tokens — a good head-grid example.',
  },
  {
    label: 'Repetition trap',
    text: 'I really really really',
    note: 'Induction heads at work: one head copies the previous occurrence.',
  },
] as const;
