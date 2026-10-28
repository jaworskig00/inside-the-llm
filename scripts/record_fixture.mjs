#!/usr/bin/env node
/**
 * Records a real run of the exported graph into `public/demo-run.json`.
 *
 * The recording does two jobs:
 *   - CI never downloads 89 MB of weights; the Playwright test drives the app against this
 *     file through the same worker protocol the real worker speaks.
 *   - The app's own `?demo=1` mode replays it, so someone on a phone or a slow connection
 *     still gets the whole interface working on genuine model output.
 *
 * Everything here comes from the actual ONNX graph under onnxruntime-node - no synthetic
 * numbers. Attention is quantised to 8 bits before being base64'd, which is the difference
 * between a ~250 KB fixture and a multi-megabyte one, and is well under what a heatmap or
 * an arc can show.
 *
 * Usage: node scripts/record_fixture.mjs [--model .devmodel/model.onnx] [--tokens 12]
 */
import { readFile, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import ort from 'onnxruntime-node';
import { Tokenizer } from '@huggingface/tokenizers';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');

const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) {
  args.set(process.argv[i]?.replace(/^--/, ''), process.argv[i + 1]);
}

const MODEL_PATH = resolve(root, args.get('model') ?? '.devmodel/model.onnx');
const OUT_PATH = resolve(root, args.get('out') ?? 'public/demo-run.json');
const PROMPT = args.get('prompt') ?? 'The capital of France is';
const MAX_NEW_TOKENS = Number(args.get('tokens') ?? 12);
const SAMPLING = { temperature: 0.8, topK: 40, topP: 0.95, seed: 7 };
const RAW_TOP = 24;
const TREE_BRANCHES = 5;
const TREE_DEPTH = 2;

const meta = JSON.parse(await readFile(join(root, 'public/model/model-meta.json'), 'utf8'));
const tokenizer = new Tokenizer(
  JSON.parse(await readFile(join(root, 'public/model/tokenizer/tokenizer.json'), 'utf8')),
  JSON.parse(await readFile(join(root, 'public/model/tokenizer/tokenizer_config.json'), 'utf8')),
);

const loadStarted = performance.now();
const session = await ort.InferenceSession.create(MODEL_PATH);
const loadMs = performance.now() - loadStarted;
console.log(`[record] session ready in ${loadMs.toFixed(0)} ms`);

const decodeCache = new Map();
const tokenText = (id) => {
  if (!decodeCache.has(id)) decodeCache.set(id, tokenizer.id_to_token(id) ?? `<${id}>`);
  return decodeCache.get(id);
};

async function forward(ids) {
  const started = performance.now();
  const input = new ort.Tensor('int32', Int32Array.from(ids), [1, ids.length]);
  const output = await session.run({ input_ids: input });
  return {
    logits: output.logits.data,
    attention: output.attentions.data,
    hiddenNorms: output.hidden_norms.data,
    tokenLogprobs: output.token_logprobs.data,
    lensValues: output.lens_values.data,
    lensIndices: output.lens_indices.data,
    tokens: ids.length,
    ms: performance.now() - started,
  };
}

function softmaxTop(logits, k) {
  let max = -Infinity;
  for (const value of logits) if (value > max) max = value;
  let total = 0;
  for (const value of logits) total += Math.exp(value - max);

  const ids = [...logits.keys()].sort((a, b) => logits[b] - logits[a]).slice(0, k);
  return ids.map((id) => ({
    id,
    text: tokenText(id),
    prob: Math.exp(logits[id] - max) / total,
  }));
}

/** Mirrors src/lib/sampling.ts; kept local so the recorder has no bundler dependency. */
function shape(candidates, { temperature, topK, topP }) {
  if (temperature <= 0)
    return { candidates: candidates.slice(0, 1).map((c) => ({ ...c, prob: 1 })), kept: 1 };

  const scaled = candidates.map((c) => ({ ...c, raw: Math.log(c.prob) / temperature }));
  let max = -Infinity;
  for (const entry of scaled) if (entry.raw > max) max = entry.raw;
  let total = 0;
  for (const entry of scaled) total += Math.exp(entry.raw - max);
  const tempered = scaled.map((entry) => ({ ...entry, prob: Math.exp(entry.raw - max) / total }));

  const pool = tempered.slice(0, topK > 0 ? topK : tempered.length);
  let kept = pool.length;
  if (topP < 1) {
    let cumulative = 0;
    kept = 0;
    for (const entry of pool) {
      cumulative += entry.prob;
      kept += 1;
      if (cumulative >= topP) break;
    }
  }

  const survivors = pool.slice(0, Math.max(kept, 1));
  const mass = survivors.reduce((sum, entry) => sum + entry.prob, 0);
  return {
    candidates: survivors.map((entry) => ({
      id: entry.id,
      text: entry.text,
      prob: entry.prob / mass,
    })),
    kept: survivors.length,
  };
}

function makeRng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function encodeAttention(data) {
  const bytes = Buffer.alloc(data.length);
  for (let i = 0; i < data.length; i += 1) {
    bytes[i] = Math.max(0, Math.min(255, Math.round(data[i] * 255)));
  }
  return bytes.toString('base64');
}

function lensFrom(result) {
  const out = [];
  for (let layer = 0; layer <= meta.numLayers; layer += 1) {
    const candidates = [];
    for (let k = 0; k < meta.lensTopK; k += 1) {
      const offset = layer * meta.lensTopK + k;
      const id = Number(result.lensIndices[offset]);
      candidates.push({ id, text: tokenText(id), prob: Math.exp(result.lensValues[offset]) });
    }
    out.push({ layer, candidates });
  }
  return out;
}

function hiddenNormsFrom(result) {
  const rows = [];
  for (let layer = 0; layer <= meta.numLayers; layer += 1) {
    const row = [];
    for (let token = 0; token < result.tokens; token += 1) {
      row.push(Number(result.hiddenNorms[layer * result.tokens + token].toFixed(2)));
    }
    rows.push(row);
  }
  return rows;
}

const round = (candidates) =>
  candidates.map((candidate) => ({ ...candidate, prob: Number(candidate.prob.toFixed(6)) }));

// ---- the run ----------------------------------------------------------------------------

const encoded = tokenizer.encode(PROMPT, { add_special_tokens: false });
const promptIds = encoded.ids;
const ids = [...promptIds];
const random = makeRng(SAMPLING.seed);

const runStarted = performance.now();
let result = await forward(ids);

const promptTokens = promptIds.map((id, index) => ({
  index,
  id,
  text: tokenText(id),
  source: 'prompt',
  prob: index === 0 ? null : Number(Math.exp(result.tokenLogprobs[index]).toFixed(6)),
}));

const steps = [];
for (let step = 0; step < MAX_NEW_TOKENS; step += 1) {
  const rawTop = softmaxTop(result.logits, RAW_TOP);
  const shaped = shape(rawTop, SAMPLING);

  const roll = random();
  let cumulative = 0;
  let chosen = shaped.candidates[shaped.candidates.length - 1];
  for (const candidate of shaped.candidates) {
    cumulative += candidate.prob;
    if (roll < cumulative) {
      chosen = candidate;
      break;
    }
  }

  const chosenProb = rawTop.find((candidate) => candidate.id === chosen.id)?.prob ?? chosen.prob;

  steps.push({
    step,
    position: ids.length,
    chosen: { id: chosen.id, text: chosen.text, prob: Number(chosenProb.toFixed(6)) },
    rawTop: round(rawTop),
    shapedTop: round(shaped.candidates),
    keptCandidates: shaped.kept,
    lens: lensFrom(result).map((entry) => ({ ...entry, candidates: round(entry.candidates) })),
    hiddenNorms: hiddenNormsFrom(result),
    contextLength: result.tokens,
    ms: Number(result.ms.toFixed(1)),
    token: {
      index: ids.length,
      id: chosen.id,
      text: chosen.text,
      source: 'generated',
      prob: Number(chosenProb.toFixed(6)),
    },
    attention: {
      layers: meta.numLayers,
      heads: meta.numHeads,
      tokens: result.tokens,
      u8: encodeAttention(result.attention),
    },
  });

  ids.push(chosen.id);
  process.stdout.write(`\r[record] step ${step + 1}/${MAX_NEW_TOKENS}  "${chosen.text}"          `);
  if (step + 1 < MAX_NEW_TOKENS) result = await forward(ids);
}

const totalMs = performance.now() - runStarted;
process.stdout.write('\n');

// ---- the alternatives tree ---------------------------------------------------------------

const expansions = [];
const queue = [[]];
while (queue.length > 0) {
  const path = queue.shift();
  const branchResult = await forward([...promptIds, ...path]);
  const children = round(softmaxTop(branchResult.logits, TREE_BRANCHES));
  expansions.push({ path, children });
  if (path.length < TREE_DEPTH) {
    for (const child of children) queue.push([...path, child.id]);
  }
}
console.log(`[record] ${expansions.length} tree expansions`);

const fixture = {
  version: 1,
  recordedAt: new Date().toISOString().slice(0, 10),
  meta,
  prompt: PROMPT,
  sampling: SAMPLING,
  promptTokens,
  steps,
  expansions,
  stats: {
    loadMs: Number(loadMs.toFixed(0)),
    totalMs: Number(totalMs.toFixed(0)),
    tokensPerSecond: Number((MAX_NEW_TOKENS / (totalMs / 1000)).toFixed(2)),
  },
};

await writeFile(OUT_PATH, JSON.stringify(fixture));
const text = steps
  .map((step) => step.token.text)
  .join('')
  .replace(/Ġ/g, ' ');
console.log(`[record] "${PROMPT}" ->${text}`);
console.log(
  `[record] wrote ${OUT_PATH} (${((await readFile(OUT_PATH)).length / 1024).toFixed(0)} KB)`,
);
