import { useEffect, useMemo, useRef } from 'react';

/**
 * ToonRoadScene — the TEAM page backdrop.
 *
 * A cartoonish "survey rover at dusk": twinkling stars, a pale cartoon moon,
 * drifting puffy clouds, two parallax layers of rolling hills, and a road
 * ribbon with a scrolling dashed centerline where a cute rover drives by,
 * bouncing over cartoon potholes with dust puffs. Same palette family as the
 * scanner scene — a new world that still belongs to RoadLens.
 *
 * Perf: one canvas, one rAF, DPR ≤ 1.5, draws only while `active`,
 * pauses on hidden tabs; reduced motion renders a single static frame.
 */

const AMBER = [245, 158, 11];
const AMBER_PALE = [252, 211, 77];
const CYAN = [34, 211, 238];
const VIOLET = [139, 92, 246];
const TEAL = [45, 212, 191];

const ROVER_PERIOD = 13000; // ms between rover crossings
const ROAD_SPEED = 0.16;    // px/ms dash scroll

function rgba([r, g, b], a) {
  return `rgba(${r},${g},${b},${a})`;
}

/** Deterministic pseudo-random from an integer seed. */
function rand(seed) {
  const x = Math.sin(seed * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
}

export default function ToonRoadScene({ active = true }) {
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

    function resize() {
      W = Math.round(window.innerWidth * dpr);
      H = Math.round(window.innerHeight * dpr);
      canvas.width = W;
      canvas.height = H;
    }
    resize();

    const puffs = []; // dust puffs {x, y, born}

    function hillY(x, tms, layer) {
      const speed = layer === 0 ? 0.008 : 0.02; // px/ms scroll
      const base = layer === 0 ? 0.60 : 0.68;
      const amp = layer === 0 ? 0.055 : 0.045;
      const k = layer === 0 ? 0.0016 : 0.0023;
      const xx = x + tms * speed * dpr;
      return (
        H * base +
        (Math.sin(xx * k) * 0.6 + Math.sin(xx * k * 2.3 + 1.7) * 0.4) * H * amp
      );
    }

    function drawScene(tms) {
      ctx.clearRect(0, 0, W, H);

      // --- dusk sky ---
      const sky = ctx.createLinearGradient(0, 0, 0, H * 0.75);
      sky.addColorStop(0, '#0a0c14');
      sky.addColorStop(0.6, '#10131f');
      sky.addColorStop(1, '#141220');
      ctx.fillStyle = sky;
      ctx.fillRect(0, 0, W, H);

      // --- twinkling stars ---
      for (let i = 0; i < 42; i++) {
        const sx = rand(i) * W;
        const sy = rand(i + 100) * H * 0.5;
        const tw = 0.35 + 0.65 * Math.abs(Math.sin(tms * 0.0011 + i * 1.7));
        const col = i % 5 === 0 ? CYAN : i % 7 === 0 ? VIOLET : [230, 235, 245];
        ctx.fillStyle = rgba(col, 0.5 * tw);
        const r = (i % 4 === 0 ? 1.6 : 1.0) * dpr;
        ctx.beginPath();
        ctx.arc(sx, sy, r, 0, Math.PI * 2);
        ctx.fill();
        // occasional 4-point sparkle
        if (i % 11 === 0) {
          ctx.strokeStyle = rgba(col, 0.35 * tw);
          ctx.lineWidth = 0.8 * dpr;
          const s = 4.5 * dpr * tw;
          ctx.beginPath();
          ctx.moveTo(sx - s, sy); ctx.lineTo(sx + s, sy);
          ctx.moveTo(sx, sy - s); ctx.lineTo(sx, sy + s);
          ctx.stroke();
        }
      }

      // --- cartoon moon ---
      const mx = W * 0.78;
      const my = H * 0.18;
      const mr = Math.min(W, H) * 0.065;
      const pulse = 0.85 + 0.15 * Math.sin(tms * 0.0006);
      const halo = ctx.createRadialGradient(mx, my, mr * 0.6, mx, my, mr * 3.2);
      halo.addColorStop(0, rgba(AMBER_PALE, 0.16 * pulse));
      halo.addColorStop(1, rgba(AMBER_PALE, 0));
      ctx.fillStyle = halo;
      ctx.fillRect(mx - mr * 3.4, my - mr * 3.4, mr * 6.8, mr * 6.8);
      ctx.fillStyle = rgba(AMBER_PALE, 0.85);
      ctx.beginPath();
      ctx.arc(mx, my, mr, 0, Math.PI * 2);
      ctx.fill();
      // craters
      ctx.fillStyle = 'rgba(217, 119, 6, 0.18)';
      for (const [cx, cy, cr] of [[-0.3, -0.15, 0.2], [0.25, 0.2, 0.14], [0.05, -0.35, 0.1], [-0.1, 0.32, 0.09]]) {
        ctx.beginPath();
        ctx.arc(mx + cx * mr, my + cy * mr, cr * mr, 0, Math.PI * 2);
        ctx.fill();
      }

      // --- drifting puffy clouds ---
      for (let i = 0; i < 4; i++) {
        const speed = (0.006 + rand(i + 40) * 0.008) * dpr;
        const cw = (140 + rand(i + 50) * 120) * dpr;
        const cx = ((rand(i + 60) * W + tms * speed) % (W + cw * 2)) - cw;
        const cy = H * (0.10 + rand(i + 70) * 0.28);
        ctx.fillStyle = 'rgba(98, 107, 125, 0.10)';
        for (const [ox, oy, orr] of [[0, 0, 0.5], [0.35, 0.08, 0.38], [-0.35, 0.1, 0.36], [0.05, -0.18, 0.34]]) {
          ctx.beginPath();
          ctx.arc(cx + ox * cw, cy + oy * cw * 0.4, orr * cw * 0.5, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      // --- rolling hills (2 parallax layers) ---
      for (const layer of [0, 1]) {
        ctx.beginPath();
        ctx.moveTo(0, H);
        const seg = 40;
        for (let s = 0; s <= seg; s++) {
          const x = (W * s) / seg;
          ctx.lineTo(x, hillY(x, tms, layer));
        }
        ctx.lineTo(W, H);
        ctx.closePath();
        ctx.fillStyle = layer === 0 ? '#151a26' : '#10141d';
        ctx.fill();
        // accent rim light along the crest
        ctx.beginPath();
        for (let s = 0; s <= seg; s++) {
          const x = (W * s) / seg;
          const y = hillY(x, tms, layer);
          if (s === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = layer === 0 ? rgba(VIOLET, 0.22) : rgba(TEAL, 0.18);
        ctx.lineWidth = 1.2 * dpr;
        ctx.stroke();
      }

      // --- road ribbon ---
      const roadTop = H * 0.78;
      const roadBot = H * 0.94;
      ctx.fillStyle = '#0d0f15';
      ctx.fillRect(0, roadTop, W, roadBot - roadTop);
      ctx.strokeStyle = 'rgba(60, 67, 84, 0.8)';
      ctx.lineWidth = 2 * dpr;
      ctx.beginPath();
      ctx.moveTo(0, roadTop); ctx.lineTo(W, roadTop);
      ctx.moveTo(0, roadBot); ctx.lineTo(W, roadBot);
      ctx.stroke();

      // scrolling dashed centerline
      const midY = (roadTop + roadBot) / 2;
      const dash = 46 * dpr;
      const gap = 34 * dpr;
      const offset = (tms * ROAD_SPEED * dpr) % (dash + gap);
      ctx.strokeStyle = rgba(AMBER, 0.75);
      ctx.lineWidth = 4 * dpr;
      ctx.lineCap = 'round';
      ctx.beginPath();
      for (let x = -offset; x < W; x += dash + gap) {
        ctx.moveTo(x, midY);
        ctx.lineTo(x + dash, midY);
      }
      ctx.stroke();
      ctx.lineCap = 'butt';

      // cartoon potholes scrolling with the road
      const phSpacing = W / 2.2;
      const phOffset = (tms * ROAD_SPEED * dpr) % phSpacing;
      const potholeXs = [];
      for (let k = -1; k < 4; k++) {
        const px = k * phSpacing - phOffset + phSpacing * 0.6;
        if (px > -60 && px < W + 60) {
          const py = roadTop + (roadBot - roadTop) * (0.3 + 0.4 * rand(k + 7));
          potholeXs.push([px, py]);
          ctx.beginPath();
          ctx.ellipse(px, py, 20 * dpr, 8 * dpr, 0, 0, Math.PI * 2);
          ctx.fillStyle = '#07080c';
          ctx.fill();
          ctx.strokeStyle = rgba(AMBER, 0.4);
          ctx.lineWidth = 1.5 * dpr;
          ctx.stroke();
        }
      }

      // --- the survey rover ---
      const phase = (tms % ROVER_PERIOD) / ROVER_PERIOD;
      if (phase < 0.55) {
        const rx = -80 * dpr + (W + 160 * dpr) * (phase / 0.55);
        const bounce = Math.sin(tms * 0.02) * 1.6 * dpr;
        // squash near potholes
        let squash = 1;
        for (const [px] of potholeXs) {
          const d = Math.abs(rx - px);
          if (d < 30 * dpr) {
            squash = 1 - 0.22 * (1 - d / (30 * dpr));
            if (d < 8 * dpr && puffs.length < 8 && Math.random() < 0.3) {
              puffs.push({ x: rx - 20 * dpr, y: midY + 8 * dpr, born: tms });
            }
          }
        }
        const ry = midY - 14 * dpr + bounce;
        ctx.save();
        ctx.translate(rx, ry);
        ctx.scale(1, squash);

        // headlight cone
        const cone = ctx.createLinearGradient(24 * dpr, 0, 95 * dpr, 0);
        cone.addColorStop(0, rgba(AMBER_PALE, 0.28));
        cone.addColorStop(1, rgba(AMBER_PALE, 0));
        ctx.fillStyle = cone;
        ctx.beginPath();
        ctx.moveTo(24 * dpr, -6 * dpr);
        ctx.lineTo(95 * dpr, -16 * dpr);
        ctx.lineTo(95 * dpr, 14 * dpr);
        ctx.lineTo(24 * dpr, 6 * dpr);
        ctx.closePath();
        ctx.fill();

        // body
        ctx.fillStyle = '#1b202b';
        ctx.strokeStyle = rgba(CYAN, 0.9);
        ctx.lineWidth = 2 * dpr;
        ctx.beginPath();
        ctx.roundRect(-26 * dpr, -12 * dpr, 52 * dpr, 18 * dpr, 5 * dpr);
        ctx.fill();
        ctx.stroke();
        // cabin
        ctx.beginPath();
        ctx.roundRect(-14 * dpr, -24 * dpr, 22 * dpr, 14 * dpr, 4 * dpr);
        ctx.fill();
        ctx.stroke();
        // window glow
        ctx.fillStyle = rgba(CYAN, 0.35);
        ctx.fillRect(-10 * dpr, -21 * dpr, 14 * dpr, 8 * dpr);
        // scanner dish on top
        ctx.strokeStyle = rgba(VIOLET, 0.9);
        ctx.beginPath();
        ctx.arc(4 * dpr, -26 * dpr, 5 * dpr, Math.PI, 0);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(4 * dpr, -26 * dpr);
        ctx.lineTo(4 * dpr, -31 * dpr);
        ctx.stroke();

        // wheels with rotating spokes
        const wheelAngle = tms * 0.02;
        for (const wx of [-15, 15]) {
          ctx.fillStyle = '#0a0c10';
          ctx.strokeStyle = rgba(CYAN, 0.8);
          ctx.beginPath();
          ctx.arc(wx * dpr, 8 * dpr, 7 * dpr, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(wx * dpr - Math.cos(wheelAngle) * 5 * dpr, 8 * dpr - Math.sin(wheelAngle) * 5 * dpr);
          ctx.lineTo(wx * dpr + Math.cos(wheelAngle) * 5 * dpr, 8 * dpr + Math.sin(wheelAngle) * 5 * dpr);
          ctx.stroke();
        }
        ctx.restore();
      }

      // dust puffs
      for (let i = puffs.length - 1; i >= 0; i--) {
        const p = puffs[i];
        const age = (tms - p.born) / 900;
        if (age >= 1) {
          puffs.splice(i, 1);
          continue;
        }
        ctx.fillStyle = `rgba(151, 160, 177, ${0.28 * (1 - age)})`;
        for (const [ox, oy] of [[0, 0], [-8, -4], [-14, 2]]) {
          ctx.beginPath();
          ctx.arc(p.x + ox * dpr - age * 18 * dpr, p.y + oy * dpr - age * 10 * dpr, (3 + age * 6) * dpr, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      // --- fireflies ---
      for (let i = 0; i < 8; i++) {
        const fx = ((rand(i + 200) * W + Math.sin(tms * 0.0003 + i) * 60 * dpr) + W) % W;
        const fy = H * (0.5 + rand(i + 210) * 0.25) + Math.sin(tms * 0.0007 + i * 2.4) * 22 * dpr;
        const glow = 0.25 + 0.55 * Math.abs(Math.sin(tms * 0.0016 + i * 1.3));
        ctx.fillStyle = rgba(i % 2 ? CYAN : VIOLET, glow * 0.5);
        ctx.beginPath();
        ctx.arc(fx, fy, 1.8 * dpr, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    if (reducedMotion) {
      drawScene(2600); // a pleasant static moment (rover mid-road)
      const onResizeStatic = () => {
        resize();
        drawScene(2600);
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
