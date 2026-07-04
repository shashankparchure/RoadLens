import { useEffect, useMemo, useRef } from 'react';
import InstrumentPanel from '../instrument/InstrumentPanel';
import StatReadout from '../instrument/StatReadout';
import useDepthGrid from './useDepthGrid';
import RidgelineFallback from './RidgelineFallback';

const COL_LINE_STEP = 3;   // draw every 3rd column line
const YAW_AMPLITUDE = (8 * Math.PI) / 180; // ±8°
const YAW_PERIOD_MS = 9000;
const HEIGHT_SCALE = 0.30; // relief exaggeration relative to spread
const INVERT = true;       // Depth-Anything = inverse depth; flip so the pothole reads as a bowl

// Iridescent per-row ramp: violet → teal → magenta
function rowColor(t) {
  const stops = [
    [0.0, 139, 92, 246],
    [0.55, 45, 212, 191],
    [1.0, 232, 121, 249],
  ];
  let k = 0;
  while (k < stops.length - 2 && stops[k + 1][0] < t) k++;
  const [t0, r0, g0, b0] = stops[k];
  const [t1, r1, g1, b1] = stops[k + 1];
  const f = t1 === t0 ? 0 : (t - t0) / (t1 - t0);
  return [r0 + (r1 - r0) * f, g0 + (g1 - g0) * f, b0 + (b1 - b0) * f].map(Math.round);
}

/**
 * The signature panel: the decoded depth field rendered as an animated
 * isometric wireframe hologram. Canvas 2D, single rAF, pauses off-screen.
 */
export default function HologramTerrain({ images, primaryPothole, features }) {
  const { grid, cols, rows, ok, pending } = useDepthGrid(images?.depthHeatmap || null);
  const canvasRef = useRef(null);
  const rafRef = useRef(0);
  const visibleRef = useRef(true);
  const reducedMotion = useMemo(
    () => window.matchMedia('(prefers-reduced-motion: reduce)').matches,
    [],
  );

  useEffect(() => {
    if (!ok || !grid || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    const rowCol = [];
    for (let r = 0; r < rows; r++) rowCol.push(rowColor(r / (rows - 1)));

    function resize() {
      const { clientWidth, clientHeight } = canvas;
      canvas.width = Math.round(clientWidth * dpr);
      canvas.height = Math.round(clientHeight * dpr);
    }
    resize();

    function project(u, v, h, theta, cx, cy, S, Hs) {
      const cos = Math.cos(theta);
      const sin = Math.sin(theta);
      const x = u * cos - v * sin;
      const y = u * sin + v * cos;
      return [cx + (x - y) * S * 0.866, cy + (x + y) * S * 0.5 - h * Hs];
    }

    function draw(tms) {
      const W = canvas.width;
      const H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      const theta = reducedMotion
        ? (6 * Math.PI) / 180
        : YAW_AMPLITUDE * Math.sin((2 * Math.PI * tms) / YAW_PERIOD_MS);
      const S = 0.72 * Math.min(W, H * 1.7);
      const Hs = HEIGHT_SCALE * S;
      const cx = W / 2;
      const cy = H * 0.56;

      // Project all grid points once per frame.
      const px = new Float32Array(cols * rows);
      const py = new Float32Array(cols * rows);
      for (let r = 0; r < rows; r++) {
        const v = r / (rows - 1) - 0.5;
        for (let c = 0; c < cols; c++) {
          const u = c / (cols - 1) - 0.5;
          let h = grid[r * cols + c];
          if (INVERT) h = 1 - h;
          const [X, Y] = project(u, v * 0.72, h, theta, cx, cy, S, Hs);
          px[r * cols + c] = X;
          py[r * cols + c] = Y;
        }
      }

      // Base plate (projection surface) under the mesh.
      const corners = [
        [-0.5, -0.36], [0.5, -0.36], [0.5, 0.36], [-0.5, 0.36],
      ].map(([u, v]) => project(u, v, -0.06, theta, cx, cy, S, Hs));
      ctx.beginPath();
      ctx.moveTo(corners[0][0], corners[0][1]);
      for (let i = 1; i < 4; i++) ctx.lineTo(corners[i][0], corners[i][1]);
      ctx.closePath();
      ctx.fillStyle = 'rgba(45, 212, 191, 0.045)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(45, 212, 191, 0.28)';
      ctx.lineWidth = 1 * dpr;
      ctx.stroke();

      // Row polylines, back to front, two-pass glow.
      for (let r = 0; r < rows; r++) {
        const [cr, cg, cb] = rowCol[r];
        for (const pass of [0, 1]) {
          ctx.beginPath();
          ctx.moveTo(px[r * cols], py[r * cols]);
          for (let c = 1; c < cols; c++) ctx.lineTo(px[r * cols + c], py[r * cols + c]);
          if (pass === 0) {
            ctx.strokeStyle = `rgba(${cr},${cg},${cb},0.16)`;
            ctx.lineWidth = 3.5 * dpr;
          } else {
            ctx.strokeStyle = `rgba(${cr},${cg},${cb},0.9)`;
            ctx.lineWidth = 1.1 * dpr;
          }
          ctx.stroke();
        }
      }

      // Sparse column lines for the wireframe cross-weave.
      ctx.lineWidth = 0.7 * dpr;
      ctx.strokeStyle = 'rgba(167, 139, 250, 0.28)';
      for (let c = 0; c < cols; c += COL_LINE_STEP) {
        ctx.beginPath();
        ctx.moveTo(px[c], py[c]);
        for (let r = 1; r < rows; r++) ctx.lineTo(px[r * cols + c], py[r * cols + c]);
        ctx.stroke();
      }
    }

    if (reducedMotion) {
      draw(0);
      return;
    }

    const start = performance.now();
    function loop(t) {
      if (visibleRef.current && document.visibilityState !== 'hidden') {
        draw(t - start);
      }
      rafRef.current = requestAnimationFrame(loop);
    }
    rafRef.current = requestAnimationFrame(loop);

    const io = new IntersectionObserver(
      ([entry]) => {
        visibleRef.current = entry.intersectionRatio > 0.15;
      },
      { threshold: [0, 0.15, 1] },
    );
    io.observe(canvas);

    const onResize = () => resize();
    window.addEventListener('resize', onResize);

    return () => {
      cancelAnimationFrame(rafRef.current);
      io.disconnect();
      window.removeEventListener('resize', onResize);
    };
  }, [ok, grid, cols, rows, reducedMotion]);

  // Nothing to show at all
  if (!images?.depthHeatmap && !primaryPothole?.depthProfile) return null;
  if (pending) return null;

  const bowl = primaryPothole?.depthProfile?.bowlDepth;
  const predicted = primaryPothole?.depthProfile?.predictedBowlDepth;
  const maxDepth = features?.max_depth;

  return (
    <InstrumentPanel
      title="Depth Field · Holographic Reconstruction"
      accent="holo"
      statusLabel={ok ? 'LIVE MESH' : 'RIDGELINE MODE'}
      flicker={false}
      bodyClassName="relative"
    >
      <div className="relative bg-slate-950/80 instrument-grid">
        {/* Projection rings under the mesh */}
        {!reducedMotion && ok && (
          <div className="absolute left-1/2 top-[62%] -translate-x-1/2 -translate-y-1/2 pointer-events-none" aria-hidden="true">
            <span className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[340px] h-[120px] rounded-[50%] border border-holo-teal/25" style={{ animation: 'radar-ping 3.4s ease-out infinite' }} />
            <span className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[340px] h-[120px] rounded-[50%] border border-holo-violet/25" style={{ animation: 'radar-ping 3.4s ease-out 1.7s infinite' }} />
          </div>
        )}

        <div className="h-[300px] sm:h-[340px]">
          {ok ? (
            <canvas
              ref={canvasRef}
              className="w-full h-full"
              style={{ filter: 'drop-shadow(0 0 18px rgba(139,92,246,0.25))' }}
              role="img"
              aria-label="Isometric wireframe of the road depth field"
            />
          ) : (
            <RidgelineFallback slicePoints={primaryPothole?.depthProfile?.slicePoints} />
          )}
        </div>

        {/* Corner readouts */}
        <div className="absolute top-3 left-4 flex gap-6">
          {Number.isFinite(bowl) && (
            <StatReadout label="Bowl depth" value={`${(bowl * 100).toFixed(1)}%`} color="#a78bfa" size="sm" />
          )}
          {Number.isFinite(predicted) && predicted > 0 && (
            <StatReadout label="Predicted floor" value={`${(predicted * 100).toFixed(1)}%`} color="#e879f9" size="sm" />
          )}
        </div>
        <div className="absolute top-3 right-4">
          {Number.isFinite(maxDepth) && (
            <StatReadout label="Max depth" value={maxDepth.toFixed(3)} color="#5eead4" size="sm" />
          )}
        </div>

        {/* Footer telemetry */}
        <p className="absolute bottom-2.5 left-4 font-mono text-[9px] uppercase tracking-[0.16em] text-slate-600">
          {ok ? 'MESH 40×25 · SRC DEPTH-ANYTHING-V2 · CMAP INFERNO-DECODE' : 'SRC CROSS-SECTION · SYNTHETIC EXTRUSION'}
        </p>
        <p className="absolute bottom-2.5 right-4 font-mono text-[9px] uppercase tracking-[0.16em] holo-text">
          HOLOGRAPHIC PROJECTION
        </p>
      </div>
    </InstrumentPanel>
  );
}
