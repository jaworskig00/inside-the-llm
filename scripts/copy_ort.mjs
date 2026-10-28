#!/usr/bin/env node
/**
 * Copies the onnxruntime-web artifacts the app actually loads into `public/ort/`.
 *
 * They have to come from our own origin: a Web Worker cannot reach ORT's default CDN path,
 * and pinning them locally means the deployed app never depends on a third party being up.
 *
 * Six files are copied, not the 26 that ship in the package: the two `.bundle.` entry points
 * and the loader + binary each of them actually imports. The WebGPU runtime pulls the
 * asyncify build (27 MB, ~7 MB over the wire); a machine with no GPU adapter gets the plain
 * WASM build instead (14 MB, ~4 MB), which is half the download for no loss of function.
 *
 * Run by `npm run dev` and `npm run build`, so it can never drift from the installed
 * onnxruntime-web version.
 */
import { cp, mkdir, rm, stat } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const source = join(root, 'node_modules', 'onnxruntime-web', 'dist');
const target = join(root, 'public', 'ort');

const FILES = [
  // WebGPU path
  'ort.webgpu.bundle.min.mjs',
  'ort-wasm-simd-threaded.asyncify.mjs',
  'ort-wasm-simd-threaded.asyncify.wasm',
  // WASM-only path
  'ort.wasm.bundle.min.mjs',
  'ort-wasm-simd-threaded.mjs',
  'ort-wasm-simd-threaded.wasm',
];

try {
  await stat(source);
} catch {
  console.error('[copy_ort] onnxruntime-web is not installed — run npm install first');
  process.exit(1);
}

await rm(target, { recursive: true, force: true });
await mkdir(target, { recursive: true });

let total = 0;
for (const file of FILES) {
  const from = join(source, file);
  const info = await stat(from);
  await cp(from, join(target, file));
  total += info.size;
}

console.log(`[copy_ort] ${FILES.length} files, ${(total / 1e6).toFixed(0)} MB -> public/ort/`);
