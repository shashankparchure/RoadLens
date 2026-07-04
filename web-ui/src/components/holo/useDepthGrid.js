import { useEffect, useState } from 'react';

/**
 * Decode the backend's INFERNO-colormapped depth heatmap (base64 image)
 * back into a scalar height grid, client-side.
 *
 * Strategy: nearest-RGB lookup against an inferno LUT — exact, monotonic,
 * and colormap-agnostic (swap the anchors if the backend ever changes maps).
 * api.py:928 applies cv2.COLORMAP_INFERNO to min-max-normalized depth.
 */

// Matplotlib/OpenCV inferno anchor points (t, r, g, b)
const INFERNO_ANCHORS = [
  [0.0, 0, 0, 4], [0.1, 22, 11, 57], [0.2, 66, 10, 104], [0.3, 106, 23, 110],
  [0.4, 147, 38, 103], [0.5, 188, 55, 84], [0.6, 221, 81, 58], [0.7, 243, 120, 25],
  [0.8, 252, 165, 10], [0.9, 246, 215, 70], [1.0, 252, 255, 164],
];

// Interpolate anchors into a 64-entry LUT at module load.
const LUT = (() => {
  const out = [];
  for (let i = 0; i < 64; i++) {
    const t = i / 63;
    let k = 0;
    while (k < INFERNO_ANCHORS.length - 2 && INFERNO_ANCHORS[k + 1][0] < t) k++;
    const [t0, r0, g0, b0] = INFERNO_ANCHORS[k];
    const [t1, r1, g1, b1] = INFERNO_ANCHORS[k + 1];
    const f = t1 === t0 ? 0 : (t - t0) / (t1 - t0);
    out.push([t, r0 + (r1 - r0) * f, g0 + (g1 - g0) * f, b0 + (b1 - b0) * f]);
  }
  return out;
})();

const SAMPLE_W = 80;
const SAMPLE_H = 50;
export const GRID_COLS = 40;
export const GRID_ROWS = 25;
const OFF_LUT_DIST2 = 60 * 60; // beyond this squared RGB distance, pixel is "not inferno"
const MAX_OFF_FRACTION = 0.35;

function decodePixel(r, g, b) {
  let best = 0;
  let bestD = Infinity;
  for (let i = 0; i < LUT.length; i++) {
    const dr = r - LUT[i][1];
    const dg = g - LUT[i][2];
    const db = b - LUT[i][3];
    const d = dr * dr + dg * dg + db * db;
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return [LUT[best][0], bestD];
}

/** Full decode: sample → LUT → downsample → blur → detrend → normalize. */
function decode(img) {
  const canvas = document.createElement('canvas');
  canvas.width = SAMPLE_W;
  canvas.height = SAMPLE_H;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, SAMPLE_W, SAMPLE_H);
  const { data } = ctx.getImageData(0, 0, SAMPLE_W, SAMPLE_H);

  const raw = new Float32Array(SAMPLE_W * SAMPLE_H);
  let offCount = 0;
  for (let i = 0; i < SAMPLE_W * SAMPLE_H; i++) {
    const [t, d] = decodePixel(data[i * 4], data[i * 4 + 1], data[i * 4 + 2]);
    raw[i] = t;
    if (d > OFF_LUT_DIST2) offCount++;
  }
  if (offCount / (SAMPLE_W * SAMPLE_H) > MAX_OFF_FRACTION) return null;

  // 2x2 box downsample → 40x25
  const grid = new Float32Array(GRID_COLS * GRID_ROWS);
  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      const sy = r * 2;
      const sx = c * 2;
      grid[r * GRID_COLS + c] =
        (raw[sy * SAMPLE_W + sx] + raw[sy * SAMPLE_W + sx + 1] +
         raw[(sy + 1) * SAMPLE_W + sx] + raw[(sy + 1) * SAMPLE_W + sx + 1]) / 4;
    }
  }

  // One 3x3 box-blur pass to kill JPEG speckle.
  const blurred = new Float32Array(grid.length);
  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      let sum = 0;
      let n = 0;
      for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          const rr = r + dr;
          const cc = c + dc;
          if (rr >= 0 && rr < GRID_ROWS && cc >= 0 && cc < GRID_COLS) {
            sum += grid[rr * GRID_COLS + cc];
            n++;
          }
        }
      }
      blurred[r * GRID_COLS + c] = sum / n;
    }
  }

  // Detrend: subtract the least-squares plane so the mesh shows local
  // relief relative to the road, not the global scene slope.
  {
    let sz = 0, suu = 0, svv = 0, suz = 0, svz = 0;
    const N = blurred.length;
    for (let r = 0; r < GRID_ROWS; r++) {
      for (let c = 0; c < GRID_COLS; c++) {
        const u = c / (GRID_COLS - 1) - 0.5;
        const v = r / (GRID_ROWS - 1) - 0.5;
        const z = blurred[r * GRID_COLS + c];
        sz += z;
        suu += u * u; svv += v * v;
        suz += u * z; svz += v * z;
      }
    }
    // centered coords → the normal equations decouple
    const b = suz / (suu || 1);
    const cCoef = svz / (svv || 1);
    const a = sz / N;
    for (let r = 0; r < GRID_ROWS; r++) {
      for (let c = 0; c < GRID_COLS; c++) {
        const u = c / (GRID_COLS - 1) - 0.5;
        const v = r / (GRID_ROWS - 1) - 0.5;
        blurred[r * GRID_COLS + c] -= a + b * u + cCoef * v;
      }
    }
  }

  // Normalize to [0,1]; reject a degenerate flat grid.
  let mn = Infinity;
  let mx = -Infinity;
  for (const v of blurred) {
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  if (!(mx - mn > 0.02)) return null;
  for (let i = 0; i < blurred.length; i++) blurred[i] = (blurred[i] - mn) / (mx - mn);
  return blurred;
}

/**
 * @param {string|null} src  data-URL of the depth heatmap
 * @returns {{ grid: Float32Array|null, cols: number, rows: number, ok: boolean, pending: boolean }}
 */
export default function useDepthGrid(src) {
  // Decoded result keyed by the src it was computed from; setState happens
  // only in async image callbacks (never synchronously inside the effect).
  const [decoded, setDecoded] = useState({ src: null, grid: null, ok: false });

  useEffect(() => {
    if (!src) return undefined;
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      if (cancelled) return;
      let grid = null;
      try {
        grid = decode(img);
      } catch (e) {
        console.error('depth grid decode failed', e);
      }
      setDecoded({ src, grid, ok: grid !== null });
    };
    img.onerror = () => {
      if (!cancelled) setDecoded({ src, grid: null, ok: false });
    };
    img.src = src;
    return () => {
      cancelled = true;
    };
  }, [src]);

  const current = src && decoded.src === src ? decoded : null;
  return {
    grid: current?.grid ?? null,
    cols: GRID_COLS,
    rows: GRID_ROWS,
    ok: current?.ok ?? false,
    pending: Boolean(src) && current === null,
  };
}
