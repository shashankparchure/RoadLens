import { motion as Motion, AnimatePresence } from 'motion/react';

/**
 * Streaming mono stage log for the scan sequence.
 * stages: [{ id, label, detail, status: 'pending' | 'active' | 'done' | 'fail' }]
 * extraLines: strings appended below the stages (heartbeats, warnings).
 */
export default function TelemetryLog({ stages, extraLines = [] }) {
  return (
    <div
      className="font-mono text-[11px] leading-relaxed tracking-wide space-y-1.5"
      aria-live="polite"
      role="log"
    >
      <AnimatePresence initial={false}>
        {stages.map((s) =>
          s.status === 'pending' ? null : (
            <Motion.div
              key={s.id}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.18 }}
              className="flex items-baseline gap-2"
            >
              <span className="text-slate-600 shrink-0">{s.id}</span>
              <span
                className={
                  s.status === 'fail'
                    ? 'text-red-500'
                    : s.status === 'done'
                      ? 'text-slate-400'
                      : 'text-cyan-300'
                }
              >
                {s.label}
              </span>
              <span className="flex-1 border-b border-dotted border-slate-700/60 mx-1 translate-y-[-3px]" />
              {s.status === 'active' && (
                <span className="text-amber-300 shrink-0">
                  RUNNING<span className="blink-caret">▌</span>
                </span>
              )}
              {s.status === 'done' && <span className="text-green-500 shrink-0">OK</span>}
              {s.status === 'fail' && <span className="text-red-500 shrink-0">FAIL</span>}
            </Motion.div>
          ),
        )}
        {extraLines.map((line, i) => (
          <Motion.p
            key={`x-${i}`}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className={line.startsWith('!') ? 'text-amber-400' : 'text-slate-500'}
          >
            {line.startsWith('!') ? line.slice(1) : line}
          </Motion.p>
        ))}
      </AnimatePresence>
    </div>
  );
}
