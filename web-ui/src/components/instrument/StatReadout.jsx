/**
 * Small telemetry block: mono uppercase label over a mono value.
 * The workhorse readout reused across panels.
 */
export default function StatReadout({
  label,
  value,
  unit,
  color = '#e5e7eb',
  size = 'md',
  className = '',
}) {
  const valueSize = size === 'lg' ? 'text-2xl' : size === 'sm' ? 'text-sm' : 'text-lg';
  return (
    <div className={`min-w-0 ${className}`}>
      <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-slate-500 mb-0.5 truncate">
        {label}
      </p>
      <p className={`font-mono font-semibold leading-none ${valueSize}`} style={{ color }}>
        {value}
        {unit && <span className="text-[0.6em] text-slate-500 ml-1">{unit}</span>}
      </p>
    </div>
  );
}
