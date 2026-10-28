/**
 * Shape of a recorded run.
 *
 * `scripts/record_fixture.mjs` runs the real ONNX graph under Node and writes one of these.
 * It powers two things: the Playwright e2e test (CI must not download 89 MB of weights) and
 * the app's own `?demo=1` mode, which lets someone on a phone or a slow connection see the
 * whole UI working against real model output.
 *
 * Attention is stored 8-bit quantised and base64 encoded, which keeps a 12-step recording
 * near 200 KB instead of several megabytes. The live app always uses full float32.
 */
import type {
  Candidate,
  LensLayer,
  ModelMeta,
  SamplingConfig,
  TokenInfo,
  TreeExpansion,
} from '../lib/protocol';

export interface FixtureAttention {
  layers: number;
  heads: number;
  tokens: number;
  /** base64 of a Uint8Array holding `round(weight * 255)`. */
  u8: string;
}

export interface FixtureStep {
  step: number;
  position: number;
  chosen: Candidate;
  rawTop: Candidate[];
  shapedTop: Candidate[];
  keptCandidates: number;
  lens: LensLayer[];
  hiddenNorms: number[][];
  contextLength: number;
  ms: number;
  token: TokenInfo;
  attention: FixtureAttention;
}

export interface Fixture {
  version: 1;
  recordedAt: string;
  meta: ModelMeta;
  prompt: string;
  sampling: SamplingConfig;
  promptTokens: TokenInfo[];
  steps: FixtureStep[];
  expansions: TreeExpansion[];
  stats: { loadMs: number; totalMs: number; tokensPerSecond: number };
}

export function decodeAttention(attention: FixtureAttention): Float32Array {
  const binary = atob(attention.u8);
  const out = new Float32Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) out[i] = binary.charCodeAt(i) / 255;
  return out;
}

export function encodeAttention(data: Float32Array): string {
  const bytes = new Uint8Array(data.length);
  for (let i = 0; i < data.length; i += 1) {
    bytes[i] = Math.max(0, Math.min(255, Math.round((data[i] as number) * 255)));
  }
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}
