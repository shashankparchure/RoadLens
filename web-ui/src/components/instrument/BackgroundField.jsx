import { useEffect, useMemo, useRef } from 'react';
import ToonRoadScene from './ToonRoadScene';

/**
 * BackgroundField — the app-wide animated backdrop.
 *
 * A LiDAR ground-sweep scene: a perspective road-grid receding to a horizon,
 * slowly driving toward the viewer, with a slow sine ripple breathing through
 * the lateral lines. A radar azimuth sweeps the ground and lights up grid
 * returns that fade like LiDAR points; every few seconds a hazard blip pings
 * with expanding rings. Faint iridescent contour lines drift near the horizon.
 *
 * Perf: one canvas, one rAF, DPR capped at 1.5, ~80 strokes/frame, pauses on
 * hidden tabs. Reduced motion renders a single static frame.
 */

const CYAN = [103, 232, 249];
const AMBER = [245, 158, 11];
const RED = [239, 68, 68];
const VIOLET = [139, 92, 246];
const TEAL = [45, 212, 191];

const N_LONG = 26;          // longitudinal (fanning) lines
const N_LAT = 16;           // lateral rows visible at once
const DRIVE_PERIOD = 14000; // ms for one lateral-row cycle (forward drift)
const SWEEP_PERIOD = 9000;  // ms for a full radar pass (left→right→left)
const RIPPLE_AMP = 6;       // px, warp-ripple amplitude at the near edge
const BLIP_EVERY = [3500, 6500]; // ms between hazard blips (random range)

function rgba([r, g, b], a) {
  return `rgba(${r},${g},${b},${a})`;
}

function ScannerScene({ active = true }) {
  const canvasRef = useRef(null);
  const activeRef = useRef(active);
  const reducedMotion = useMemo(
    () => window.matchMedia('(prefers-reduced-motion: reduce)').matches,
    [],
  );

  useEffect(() => {
    activeRef.current = active;
  }, [active]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const ctx = canvas.getContext('2d');
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);

    let W = 0;
    let H = 0;
    let horizonY = 0;
    let vpX = 0; // vanishing point x

    function resize() {
      W = Math.round(window.innerWidth * dpr);
      H = Math.round(window.innerHeight * dpr);
      canvas.width = W;
      canvas.height = H;
      horizonY = H * 0.34;
      vpX = W / 2;
    }
    resize();

    // --- scene state (module-lifetime, cheap) ---
    const returns = []; // LiDAR point returns {x, y, born}
    const blips = [];   // hazard blips {x, y, born, color}
    let nextBlipAt = 1500;

    // Precomputed contour tracks (holo isolines near the horizon)
    const contours = [
      { base: 0.16, amp: 26, k: 2.1, speed: 0.020, color: VIOLET },
      { base: 0.22, amp: 34, k: 1.6, speed: -0.014, color: TEAL },
      { base: 0.28, amp: 22, k: 2.7, speed: 0.010, color: VIOLET },
      { base: 0.10, amp: 18, k: 3.3, speed: -0.008, color: TEAL },
    ];

    /** Ground row: parameter z in (0,1], 1 = at viewer. Screen y with easing. */
    function rowY(z) {
      return horizonY + (H - horizonY) * Math.pow(z, 2.1);
    }

    /** Longitudinal line i in [0, N_LONG): x at the bottom edge. */
    function longBottomX(i) {
      const spread = W * 2.6; // wider than screen so edges rake outward
      return vpX - spread / 2 + (spread * i) / (N_LONG - 1);
    }

    function drawScene(tms) {
      ctx.clearRect(0, 0, W, H);

      // Horizon glow (amber instrument wash)
      const glow = ctx.createLinearGradient(0, horizonY - H * 0.12, 0, horizonY + H * 0.10);
      glow.addColorStop(0, 'rgba(245,158,11,0)');
      glow.addColorStop(0.55, 'rgba(245,158,11,0.05)');
      glow.addColorStop(1, 'rgba(245,158,11,0)');
      ctx.fillStyle = glow;
      ctx.fillRect(0, horizonY - H * 0.12, W, H * 0.22);
      ctx.strokeStyle = 'rgba(103,232,249,0.12)';
      ctx.lineWidth = 1 * dpr;
      ctx.beginPath();
      ctx.moveTo(0, horizonY);
      ctx.lineTo(W, horizonY);
      ctx.stroke();

      // --- holo contours above/at the horizon (iridescent isolines) ---
      for (const c of contours) {
        const y0 = horizonY * (1 - c.base);
        ctx.beginPath();
        const seg = 26;
        for (let s = 0; s <= seg; s++) {
          const x = (W * s) / seg;
          const ph = tms * 0.001 * c.speed * 60;
          const y =
            y0 +
            Math.sin((s / seg) * Math.PI * c.k + ph) * c.amp * dpr * 0.5 +
            Math.sin((s / seg) * Math.PI * c.k * 2.7 + ph * 1.7) * c.amp * dpr * 0.2;
          if (s === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = rgba(c.color, 0.09);
        ctx.lineWidth = 1 * dpr;
        ctx.stroke();
      }

      // --- radar sweep azimuth (from below the bottom edge) ---
      const originX = vpX;
      const originY = H * 1.25;
      const phase = (tms % SWEEP_PERIOD) / SWEEP_PERIOD; // 0..1
      const swing = Math.sin(phase * Math.PI * 2); // -1..1 (back & forth)
      const sweepAngle = -Math.PI / 2 + swing * (Math.PI / 3.1); // ±58° around straight-up

      // Sweep wedge (soft light following the beam)
      const wedge = Math.PI / 14;
      const grad = ctx.createRadialGradient(originX, originY, H * 0.2, originX, originY, H * 1.45);
      grad.addColorStop(0, 'rgba(103,232,249,0.10)');
      grad.addColorStop(1, 'rgba(103,232,249,0)');
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(originX, originY);
      ctx.arc(originX, originY, H * 1.5, sweepAngle - wedge, sweepAngle + wedge);
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();
      ctx.restore();

      // Beam line
      ctx.beginPath();
      ctx.moveTo(originX, originY);
      ctx.lineTo(originX + Math.cos(sweepAngle) * H * 1.6, originY + Math.sin(sweepAngle) * H * 1.6);
      ctx.strokeStyle = 'rgba(103,232,249,0.14)';
      ctx.lineWidth = 1.2 * dpr;
      ctx.stroke();

      // --- perspective grid ---
      // Longitudinal lines (straight, converge on the vanishing point)
      ctx.lineWidth = 1 * dpr;
      for (let i = 0; i < N_LONG; i++) {
        const xb = longBottomX(i);
        const centerness = 1 - Math.abs(i - (N_LONG - 1) / 2) / ((N_LONG - 1) / 2);
        ctx.beginPath();
        ctx.moveTo(vpX, horizonY);
        ctx.lineTo(xb, H);
        ctx.strokeStyle = rgba(CYAN, 0.028 + centerness * 0.03);
        ctx.stroke();
      }

      // Lateral rows (scroll toward viewer + warp ripple)
      const drive = (tms % DRIVE_PERIOD) / DRIVE_PERIOD;
      for (let r = 0; r < N_LAT; r++) {
        const z = ((r / N_LAT + drive) % 1) || 1e-4;
        const y = rowY(z);
        const alpha = 0.02 + z * 0.085;
        const rippleA = RIPPLE_AMP * dpr * z; // stronger near the viewer
        ctx.beginPath();
        const seg = 30;
        for (let s = 0; s <= seg; s++) {
          const x = (W * s) / seg;
          const dy =
            Math.sin(x * 0.0016 + tms * 0.0009) * rippleA +
            Math.sin(x * 0.0007 - tms * 0.0005) * rippleA * 0.6;
          if (s === 0) ctx.moveTo(x, y + dy);
          else ctx.lineTo(x, y + dy);
        }
        ctx.strokeStyle = rgba(CYAN, alpha);
        ctx.stroke();
      }

      // --- LiDAR point returns along the beam ---
      if (Math.random() < 0.5 && returns.length < 110) {
        const d = H * (0.35 + Math.random() * 0.75);
        const px = originX + Math.cos(sweepAngle) * d + (Math.random() - 0.5) * 30 * dpr;
        const py = originY + Math.sin(sweepAngle) * d + (Math.random() - 0.5) * 18 * dpr;
        if (py > horizonY + 8 && py < H && px > 0 && px < W) {
          returns.push({ x: px, y: py, born: tms });
        }
      }
      for (let i = returns.length - 1; i >= 0; i--) {
        const p = returns[i];
        const age = (tms - p.born) / 1300;
        if (age >= 1) {
          returns.splice(i, 1);
          continue;
        }
        const a = (1 - age) * 0.55;
        ctx.beginPath();
        ctx.arc(p.x, p.y, (1.2 + age * 1.6) * dpr, 0, Math.PI * 2);
        ctx.fillStyle = rgba(CYAN, a);
        ctx.fill();
      }

      // --- hazard blips ---
      if (tms > nextBlipAt && blips.length < 3) {
        blips.push({
          x: W * (0.12 + Math.random() * 0.76),
          y: rowY(0.35 + Math.random() * 0.55),
          born: tms,
          color: Math.random() < 0.35 ? RED : AMBER,
        });
        nextBlipAt = tms + BLIP_EVERY[0] + Math.random() * (BLIP_EVERY[1] - BLIP_EVERY[0]);
      }
      for (let i = blips.length - 1; i >= 0; i--) {
        const b = blips[i];
        const age = (tms - b.born) / 2100;
        if (age >= 1) {
          blips.splice(i, 1);
          continue;
        }
        const fade = 1 - age;
        // center dot + cross ticks
        ctx.fillStyle = rgba(b.color, 0.72 * fade);
        ctx.beginPath();
        ctx.arc(b.x, b.y, 2.2 * dpr, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = rgba(b.color, 0.4 * fade);
        ctx.lineWidth = 1 * dpr;
        const t = 7 * dpr;
        ctx.beginPath();
        ctx.moveTo(b.x - t, b.y); ctx.lineTo(b.x - t / 2.2, b.y);
        ctx.moveTo(b.x + t / 2.2, b.y); ctx.lineTo(b.x + t, b.y);
        ctx.moveTo(b.x, b.y - t / 1.6); ctx.lineTo(b.x, b.y - t / 3.2);
        ctx.moveTo(b.x, b.y + t / 3.2); ctx.lineTo(b.x, b.y + t / 1.6);
        ctx.stroke();
        // expanding rings (squashed to ground perspective)
        for (const ring of [age, (age + 0.45) % 1]) {
          const rr = 6 * dpr + ring * 46 * dpr;
          ctx.beginPath();
          ctx.ellipse(b.x, b.y, rr, rr * 0.42, 0, 0, Math.PI * 2);
          ctx.strokeStyle = rgba(b.color, 0.4 * (1 - ring) * fade);
          ctx.stroke();
        }
      }
    }

    if (reducedMotion) {
      // Static frame: grid + contours only (no sweep/returns/blips motion)
      drawScene(0);
      const onResizeStatic = () => {
        resize();
        drawScene(0);
      };
      window.addEventListener('resize', onResizeStatic);
      return () => window.removeEventListener('resize', onResizeStatic);
    }

    let raf = 0;
    const start = performance.now();
    function loop(t) {
      if (activeRef.current && document.visibilityState !== 'hidden') {
        drawScene(t - start);
      }
      raf = requestAnimationFrame(loop);
    }
    raf = requestAnimationFrame(loop);

    const onResize = () => resize();
    window.addEventListener('resize', onResize);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', onResize);
    };
  }, [reducedMotion]);

  return <canvas ref={canvasRef} className="w-full h-full" />;
}

/**
 * Container: crossfades between the scanner scene (detection/insights) and
 * the cartoon crew scene (team). The hidden scene's draw loop idles.
 */
export default function BackgroundField({ variant = 'scanner' }) {
  const crew = variant === 'crew';
  return (
    <div className="fixed inset-0 z-0 pointer-events-none" aria-hidden="true">
      <div
        className="absolute inset-0 transition-opacity duration-700 ease-in-out"
        style={{ opacity: crew ? 0 : 1 }}
      >
        <ScannerScene active={!crew} />
      </div>
      <div
        className="absolute inset-0 transition-opacity duration-700 ease-in-out"
        style={{ opacity: crew ? 1 : 0 }}
      >
        <ToonRoadScene active={crew} />
      </div>
      {/* Vignette keeps edges dark and content readable — zero per-frame cost */}
      <div
        className="absolute inset-0"
        style={{
          background:
            'radial-gradient(ellipse at 50% 42%, transparent 0%, transparent 52%, rgba(10,12,16,0.55) 100%)',
        }}
      />
    </div>
  );
}
