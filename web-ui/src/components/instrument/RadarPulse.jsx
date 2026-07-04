/**
 * Circular radar widget: polar grid, rotating conic sweep, expanding ping rings.
 * Pure CSS animation (radar-spin / radar-ping keyframes).
 */
export default function RadarPulse({ size = 120, color = '#22d3ee', active = true }) {
  const ring = 'rgba(103, 232, 249, 0.18)';
  return (
    <div
      className="relative rounded-full shrink-0"
      style={{ width: size, height: size, border: `1px solid ${ring}` }}
      aria-hidden="true"
    >
      {/* Polar grid */}
      <div className="absolute inset-[18%] rounded-full" style={{ border: `1px solid ${ring}` }} />
      <div className="absolute inset-[38%] rounded-full" style={{ border: `1px solid ${ring}` }} />
      <div className="absolute left-1/2 top-0 bottom-0 w-px" style={{ background: ring }} />
      <div className="absolute top-1/2 left-0 right-0 h-px" style={{ background: ring }} />

      {/* Rotating sweep */}
      {active && (
        <div
          className="absolute inset-0 rounded-full overflow-hidden"
          style={{ animation: 'radar-spin 2.6s linear infinite' }}
        >
          <div
            className="absolute inset-0"
            style={{
              background: `conic-gradient(from 0deg, transparent 0deg, transparent 300deg, ${color}55 345deg, ${color} 360deg)`,
              borderRadius: '9999px',
            }}
          />
        </div>
      )}

      {/* Ping rings */}
      {active && (
        <>
          <span
            className="absolute inset-0 rounded-full"
            style={{ border: `1.5px solid ${color}`, animation: 'radar-ping 2.2s ease-out infinite' }}
          />
          <span
            className="absolute inset-0 rounded-full"
            style={{ border: `1.5px solid ${color}`, animation: 'radar-ping 2.2s ease-out 1.1s infinite' }}
          />
        </>
      )}

      {/* Center dot */}
      <span
        className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-1.5 h-1.5 rounded-full"
        style={{ background: color, boxShadow: `0 0 8px ${color}` }}
      />
    </div>
  );
}
