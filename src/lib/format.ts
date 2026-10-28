/** Small formatting helpers shared by every view. */

/**
 * GPT-2's byte-level BPE writes a leading space as U+0120 and a newline as U+010A.
 * Tokens have to be shown as they really are - a chip reading "·the" tells the truth about
 * why " the" and "the" are different tokens, which is half the point of the token strip.
 */
export function displayToken(text: string): string {
  return text.replace(/Ġ/g, '·').replace(/Ċ/g, '↵').replace(/^ /, '·').replace(/\n/g, '↵');
}

/** The same token, decoded for the "generated text" readout. */
export function plainToken(text: string): string {
  return text.replace(/Ġ/g, ' ').replace(/Ċ/g, '\n');
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
