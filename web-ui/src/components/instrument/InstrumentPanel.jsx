import CornerBrackets from './CornerBrackets';

const ACCENTS = {
  amber: {
    bracket: 'rgba(245, 158, 11, 0.75)',
    dot: '#f59e0b',
    title: 'text-amber-300',
  },
  cyan: {
    bracket: 'rgba(103, 232, 249, 0.6)',
    dot: '#22d3ee',
    title: 'text-cyan-300',
  },
  holo: {
    bracket: 'rgba(139, 92, 246, 0.7)',
    dot: '#a78bfa',
    title: 'holo-text',
  },
  red: {
    bracket: 'rgba(239, 68, 68, 0.75)',
    dot: '#ef4444',
    title: 'text-red-500',
  },
};

/**
 * Universal panel chrome: viewfinder corner brackets, sharp edges,
 * mono header strip with a status dot. Replaces the old .glass-card.
 */
export default function InstrumentPanel({
  title,
  accent = 'amber',
  statusLabel,
  headerRight,
  scanlines = false,
  flicker = true,
  className = '',
  bodyClassName = '',
  children,
}) {
  const a = ACCENTS[accent] || ACCENTS.amber;

  return (
    <section
      className={`relative bg-slate-900/80 border border-slate-700 rounded-[2px] ${flicker ? 'flicker-in' : ''} ${className}`}
    >
      <CornerBrackets color={a.bracket} />
      {title && (
        <header className="flex items-center justify-between gap-3 px-4 py-2.5 border-b border-slate-700/70">
          <div className="flex items-center gap-2.5 min-w-0">
            <span
              className="w-1.5 h-1.5 rounded-full shrink-0"
              style={{ background: a.dot, boxShadow: `0 0 8px ${a.dot}` }}
            />
            <h3 className={`font-mono text-[11px] uppercase tracking-[0.16em] truncate ${a.title}`}>
              {title}
            </h3>
          </div>
          <div className="flex items-center gap-3 shrink-0">
            {statusLabel && (
              <span className="font-mono text-[10px] uppercase tracking-[0.12em] text-slate-500">
                {statusLabel}
              </span>
            )}
            {headerRight}
          </div>
        </header>
      )}
      <div className={`${scanlines ? 'scanline-overlay' : ''} ${bodyClassName}`}>{children}</div>
    </section>
  );
}
