/**
 * Colour decisions, in one place.
 *
 * Everything here is colour-vision-deficiency safe: the sequential scales are Cividis and
 * Viridis (both monotonic in lightness and designed for deuteranopia), and the categorical
 * scale is Okabe-Ito, which stays distinguishable under all three common CVD types.
 * No red/green pairs carry meaning anywhere in the app.
 */
import { interpolateCividis, interpolateViridis } from 'd3';

/** Attention weight -> colour. Viridis reads well on a dark background. */
export function attentionColor(t: number): string {
  return interpolateViridis(clamp01(t));
}

/** Probability -> colour, used by the token strip and the bar chart. */
export function probabilityColor(p: number): string {
  return interpolateCividis(0.15 + 0.85 * clamp01(p));
}

/**
 * Okabe-Ito, minus the pure black entry. Used for the handful of named token candidates
 * in the Sankey and the alternatives tree, where identity matters more than magnitude.
 */
export const CATEGORICAL = [
  '#56b4e9', // sky blue
  '#e69f00', // orange
  '#009e73', // bluish green
  '#cc79a7', // reddish purple
  '#f0e442', // yellow
  '#0072b2', // blue
  '#d55e00', // vermillion
] as const;

export function categorical(index: number): string {
  return CATEGORICAL[index % CATEGORICAL.length] as string;
}

/** Keeps a token's colour stable across steps and views. */
export function tokenColor(tokenId: number, order: number[]): string {
  const index = order.indexOf(tokenId);
  return index === -1 ? '#6b7280' : categorical(index);
}

export function clamp01(value: number): number {
  if (Number.isNaN(value)) return 0;
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}

/**
 * Text colour that stays readable on a given Cividis/Viridis background.
 * Both scales are dark at the low end and bright at the high end, so a single threshold
 * is enough.
 */
export function contrastInk(t: number): string {
  return t > 0.6 ? '#0b0c10' : '#f2f4f8';
}
