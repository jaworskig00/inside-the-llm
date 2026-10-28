import { useEffect, useRef, useState } from 'react';
import type { AttentionPack } from '../lib/protocol';
import { headMatrix, normalizeRows } from '../lib/attention';
import { attentionColor } from '../lib/color';

interface HeadGridProps {
  pack: AttentionPack;
  selectedLayer: number;
  selectedHead: number | null;
  onSelect: (layer: number, head: number | null) => void;
}

const CELL = 46;
const GAP = 3;
const PAD_LEFT = 26;
const PAD_TOP = 18;

/**
 * Every attention head in the model, at once: one row per layer, one column per head.
 *
 * This is deliberately canvas and not SVG. A 6x12 grid of 64x64 matrices is 294 912 cells;
 * as SVG rects that is a third of a million DOM nodes and the tab dies. On canvas it is
 * one draw call per head — 72 `drawImage` calls, around 3 ms — so the grid can be
 * repainted on every step without dropping a frame.
 *
 * Each head is row-normalised (see `normalizeRows`): attention is causal, so without it
 * every head looks like the same bright triangle.
 */
export function HeadGrid({ pack, selectedLayer, selectedHead, onSelect }: HeadGridProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hover, setHover] = useState<{ layer: number; head: number } | null>(null);

  const width = PAD_LEFT + pack.heads * (CELL + GAP);
  const height = PAD_TOP + pack.layers * (CELL + GAP);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    // One reusable offscreen canvas at the matrix's own resolution; the browser scales it
    // up with nearest-neighbour so individual token pairs stay visible.
    const tile = document.createElement('canvas');
    tile.width = pack.tokens;
    tile.height = pack.tokens;
    const tileCtx = tile.getContext('2d');
    if (!tileCtx) return;
    const image = tileCtx.createImageData(pack.tokens, pack.tokens);

    ctx.imageSmoothingEnabled = false;
    ctx.font = '9px ui-monospace, monospace';
    ctx.fillStyle = '#666d7e';

    for (let head = 0; head < pack.heads; head += 1) {
      ctx.textAlign = 'center';
      ctx.fillText(String(head), PAD_LEFT + head * (CELL + GAP) + CELL / 2, 11);
    }

    for (let layer = 0; layer < pack.layers; layer += 1) {
      ctx.textAlign = 'right';
      ctx.fillStyle = '#666d7e';
      ctx.fillText(`L${layer}`, PAD_LEFT - 6, PAD_TOP + layer * (CELL + GAP) + CELL / 2 + 3);

      for (let head = 0; head < pack.heads; head += 1) {
        const matrix = normalizeRows(headMatrix(pack, layer, head), pack.tokens);
        paint(image, matrix, pack.tokens);
        tileCtx.putImageData(image, 0, 0);

        const x = PAD_LEFT + head * (CELL + GAP);
        const y = PAD_TOP + layer * (CELL + GAP);
        ctx.drawImage(tile, 0, 0, pack.tokens, pack.tokens, x, y, CELL, CELL);

        const isSelected =
          layer === selectedLayer && (selectedHead === head || selectedHead === null);
        const isHovered = hover?.layer === layer && hover?.head === head;
        if (isSelected || isHovered) {
          ctx.strokeStyle = isHovered ? '#f0b429' : '#62b0ff';
          ctx.lineWidth = isSelected && selectedHead === head ? 2 : 1;
          ctx.strokeRect(x - 0.5, y - 0.5, CELL + 1, CELL + 1);
        }
      }
    }
  }, [pack, selectedLayer, selectedHead, hover, width, height]);

  const locate = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const scale = rect.width / width;
    const x = (event.clientX - rect.left) / scale - PAD_LEFT;
    const y = (event.clientY - rect.top) / scale - PAD_TOP;
    const head = Math.floor(x / (CELL + GAP));
    const layer = Math.floor(y / (CELL + GAP));
    if (head < 0 || layer < 0 || head >= pack.heads || layer >= pack.layers) return null;
    return { layer, head };
  };

  return (
    <div className="overflow-x-auto">
      <canvas
        ref={canvasRef}
        style={{ width, height }}
        className="max-w-full"
        onMouseMove={(event) => setHover(locate(event))}
        onMouseLeave={() => setHover(null)}
        onClick={(event) => {
          const found = locate(event);
          if (found) onSelect(found.layer, found.head);
        }}
        role="img"
        aria-label={`Attention head grid, ${pack.layers} layers by ${pack.heads} heads. Selected layer ${selectedLayer}, ${
          selectedHead === null ? 'all heads averaged' : `head ${selectedHead}`
        }.`}
      />
      <p className="mt-2 text-[11px] text-ink-faint" aria-live="polite">
        {hover
          ? `layer ${hover.layer}, head ${hover.head} — click to inspect`
          : 'Rows are layers, columns are heads. Click a square to open it below.'}
      </p>
    </div>
  );
}

/**
 * Viridis, baked into a 256-entry byte lookup table.
 *
 * Calling d3's interpolator per pixel means ~300 000 string allocations and parses per
 * repaint. Quantising the scale once costs nothing visually and turns the inner loop into
 * three array reads.
 */
const LUT = buildLut();

function buildLut(): Uint8Array {
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i += 1) {
    const match = /rgb\((\d+),\s*(\d+),\s*(\d+)\)/.exec(attentionColor(i / 255));
    lut[i * 3] = match ? Number(match[1]) : 0;
    lut[i * 3 + 1] = match ? Number(match[2]) : 0;
    lut[i * 3 + 2] = match ? Number(match[3]) : 0;
  }
  return lut;
}

function paint(image: ImageData, matrix: Float32Array, tokens: number): void {
  const pixels = image.data;
  for (let i = 0; i < tokens * tokens; i += 1) {
    const value = matrix[i] as number;
    const offset = i * 4;
    if (Number.isNaN(value)) {
      // The upper triangle: a token cannot attend to the future. Drawn as panel background.
      pixels[offset] = 14;
      pixels[offset + 1] = 16;
      pixels[offset + 2] = 21;
      pixels[offset + 3] = 255;
      continue;
    }
    const slot = (value <= 0 ? 0 : value >= 1 ? 255 : (value * 255) | 0) * 3;
    pixels[offset] = LUT[slot] as number;
    pixels[offset + 1] = LUT[slot + 1] as number;
    pixels[offset + 2] = LUT[slot + 2] as number;
    pixels[offset + 3] = 255;
  }
}
