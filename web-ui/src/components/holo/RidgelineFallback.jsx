import { useMemo } from 'react';

const ROWS = 14;
const W = 640;
const H = 300;

/**
 * Fallback hologram: the 1-D depth cross-section extruded into a stacked
 * ridgeline (synthetic depth axis). Used when the heatmap decode fails.
 */
export default function RidgelineFallback({ slicePoints }) {
  const paths = useMemo(() => {
    if (!slicePoints || slicePoints.length < 8) return [];
    const pts = slicePoints;
    const xs = pts.map((p) => p.x);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    // Depth relief relative to the extrapolated road surface
    const relief = pts.map((p) => (p.roadSurface ?? 0) - (p.actualDepth ?? 0));
    const maxR = Math.max(...relief.map(Math.abs), 1e-6);

    const out = [];
    for (let r = 0; r < ROWS; r++) {
      const decay = 1 - (r / (ROWS - 1)) * 0.75; // amplitude falls off toward the back
      const yBase = 40 + (r / (ROWS - 1)) * (H - 80);
      let d = '';
      for (let i = 0; i < pts.length; i += 2) {
        const px = 20 + ((pts[i].x - minX) / (maxX - minX || 1)) * (W - 40);
        const py = yBase - (relief[i] / maxR) * 46 * decay;
        d += `${i === 0 ? 'M' : 'L'}${px.toFixed(1)},${py.toFixed(1)}`;
      }
      out.push({ d, t: r / (ROWS - 1) });
    }
    return out.reverse(); // back rows first
  }, [slicePoints]);

  if (paths.length === 0) return null;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full" preserveAspectRatio="xMidYMid meet">
      <defs>
        <linearGradient id="ridgeRamp" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#8b5cf6" />
          <stop offset="55%" stopColor="#2dd4bf" />
          <stop offset="100%" stopColor="#e879f9" />
        </linearGradient>
      </defs>
      {paths.map((p, i) => (
        <path
          key={i}
          d={p.d}
          fill="none"
          stroke="url(#ridgeRamp)"
          strokeWidth={1.4}
          opacity={0.25 + p.t * 0.75}
          style={{ filter: p.t > 0.8 ? 'drop-shadow(0 0 5px rgba(139,92,246,0.5))' : undefined }}
        />
      ))}
    </svg>
  );
}
