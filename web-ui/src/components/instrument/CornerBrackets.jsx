/**
 * Four viewfinder L-brackets overlaying a container.
 * Pointer-transparent; parent must be `relative`.
 */
export default function CornerBrackets({
  size = 14,
  thickness = 2,
  color = 'rgba(245, 158, 11, 0.8)',
  inset = 6,
  className = '',
}) {
  const s = `${size}px`;
  const t = `${thickness}px`;
  const i = `${inset}px`;
  const corners = [
    { top: i, left: i, borderTop: `${t} solid ${color}`, borderLeft: `${t} solid ${color}` },
    { top: i, right: i, borderTop: `${t} solid ${color}`, borderRight: `${t} solid ${color}` },
    { bottom: i, left: i, borderBottom: `${t} solid ${color}`, borderLeft: `${t} solid ${color}` },
    { bottom: i, right: i, borderBottom: `${t} solid ${color}`, borderRight: `${t} solid ${color}` },
  ];
  return (
    <div className={`absolute inset-0 pointer-events-none ${className}`} aria-hidden="true">
      {corners.map((style, idx) => (
        <span key={idx} className="absolute" style={{ width: s, height: s, ...style }} />
      ))}
    </div>
  );
}
