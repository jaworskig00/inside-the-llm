#!/usr/bin/env node
/**
 * Captures the README media from the running app.
 *
 * Everything is shot in `?demo=1`, which replays `public/demo-run.json` — a run recorded
 * from the real ONNX graph. So the screenshots are genuine model output, and the script
 * needs neither a GPU nor the 89 MB weights.
 *
 * Produces `docs/media/`:
 *   hero.png        the whole app, generated and stepped to an interesting token
 *   attention.png   the head grid plus the arc diagram
 *   layer-flow.png  the logit-lens Sankey
 *   alternatives.png the expanded tree
 *   hero.gif        a short loop of the generation running (needs ffmpeg; skipped if absent)
 *
 * Usage: node scripts/capture_media.mjs
 */
import { spawn } from 'node:child_process';
import { mkdir, readdir, rm } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from '@playwright/test';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const outDir = join(root, 'docs', 'media');
const framesDir = join(root, 'docs', '.frames');
const PORT = 4188;
const BASE = `http://127.0.0.1:${PORT}/inside-the-llm/`;
const VIEWPORT = { width: 1680, height: 1050 };

await mkdir(outDir, { recursive: true });
await rm(framesDir, { recursive: true, force: true });
await mkdir(framesDir, { recursive: true });

console.log('[media] building');
await run('npm', ['run', 'build']);

console.log('[media] starting preview');
const server = spawn(
  'npx',
  ['vite', 'preview', '--host', '127.0.0.1', '--port', String(PORT), '--strictPort'],
  { cwd: root, stdio: 'ignore' },
);
process.on('exit', () => server.kill());

await waitFor(BASE);

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: VIEWPORT, deviceScaleFactor: 2 });

try {
  await page.goto(`${BASE}?demo=1`);
  await page
    .getByRole('button', { name: 'Generate' })
    .waitFor({ state: 'visible', timeout: 30_000 });

  // --- the animated loop: one frame per generated token ---------------------------------
  console.log('[media] recording the generation loop');
  await page.getByRole('button', { name: 'Generate' }).click();
  for (let frame = 0; frame < 26; frame += 1) {
    await page.screenshot({ path: join(framesDir, `f${String(frame).padStart(3, '0')}.png`) });
    await page.waitForTimeout(110);
  }

  await page.getByText('step 12 / 12').waitFor({ timeout: 30_000 });

  // Step back to a token where the model was genuinely undecided — a 99 %-certain step
  // makes a dull screenshot of a bar chart.
  await page.locator('body').click({ position: { x: 4, y: 4 } });
  for (let i = 0; i < 9; i += 1) await page.keyboard.press('ArrowLeft');
  await page.waitForTimeout(600);

  console.log('[media] hero');
  await page.screenshot({ path: join(outDir, 'hero.png') });

  console.log('[media] panels');
  await shootPanel(page, 'Attention', 'attention.png');
  await shootPanel(page, 'Layer flow (logit lens)', 'layer-flow.png');

  const tree = panelFor(page, 'Alternatives');
  await tree.scrollIntoViewIfNeeded();
  await tree.locator('g.node').first().click();
  await page.waitForTimeout(900);
  await tree.locator('g.node').nth(1).click();
  await page.waitForTimeout(900);
  await tree.screenshot({ path: join(outDir, 'alternatives.png') });
} finally {
  await browser.close();
}

// --- the GIF ----------------------------------------------------------------------------
const frames = (await readdir(framesDir)).filter((name) => name.endsWith('.png'));
if (await hasFfmpeg()) {
  console.log(`[media] encoding ${frames.length} frames into hero.gif`);
  const filters =
    'fps=9,scale=1100:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=160[p];[b][p]paletteuse=dither=bayer:bayer_scale=3';
  await run('ffmpeg', [
    '-y',
    '-loglevel',
    'error',
    '-framerate',
    '9',
    '-pattern_type',
    'glob',
    '-i',
    join(framesDir, '*.png'),
    '-filter_complex',
    filters,
    '-loop',
    '0',
    join(outDir, 'hero.gif'),
  ]);
} else {
  console.warn('[media] ffmpeg not found — skipping hero.gif, the stills are still written');
}

await rm(framesDir, { recursive: true, force: true });
server.kill();

for (const file of (await readdir(outDir)).sort()) console.log(`[media] docs/media/${file}`);

// --- helpers ------------------------------------------------------------------------------

function panelFor(page, title) {
  return page
    .locator('section')
    .filter({ has: page.getByRole('heading', { name: title, exact: true }) });
}

async function shootPanel(page, title, file) {
  const panel = panelFor(page, title);
  await panel.scrollIntoViewIfNeeded();
  await page.waitForTimeout(500);
  await panel.screenshot({ path: join(outDir, file) });
}

function run(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd: root, stdio: 'inherit' });
    child.on('error', reject);
    child.on('exit', (code) =>
      code === 0 ? resolve() : reject(new Error(`${command} exited ${code}`)),
    );
  });
}

async function hasFfmpeg() {
  return new Promise((resolve) => {
    const child = spawn('ffmpeg', ['-version'], { stdio: 'ignore' });
    child.on('error', () => resolve(false));
    child.on('exit', (code) => resolve(code === 0));
  });
}

async function waitFor(url) {
  for (let attempt = 0; attempt < 60; attempt += 1) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {
      // not up yet
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`preview never came up at ${url}`);
}
