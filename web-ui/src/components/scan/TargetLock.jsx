import { useEffect, useMemo } from 'react';
import CornerBrackets from '../instrument/CornerBrackets';

const LOCK_MS = 1100;

/**
 * The lock-on beat between scan and dashboard: mask overlay appears,
 * oversized brackets converge, acquisition stamp flashes, then onDone.
 */
export default function TargetLock({ image, potholeCount = 0, onDone }) {
  const reducedMotion = useMemo(
    () => window.matchMedia('(prefers-reduced-motion: reduce)').matches,
    [],
  );

  useEffect(() => {
    if (reducedMotion) {
      onDone();
      return;
    }
    const t = setTimeout(onDone, LOCK_MS);
    return () => clearTimeout(t);
  }, [onDone, reducedMotion]);

  if (reducedMotion) return null;

  const acquired = potholeCount > 0;
  const stamp = acquired
    ? `TARGET ACQUIRED · ${potholeCount} POTHOLE${potholeCount > 1 ? 'S' : ''}`
    : 'NO POTHOLE DETECTED · SURFACE NOMINAL';

  return (
    <div className="max-w-5xl mx-auto">
      <div className="relative bg-slate-950 border border-slate-700 rounded-[2px] overflow-hidden scanline-overlay min-h-[320px] flex items-center justify-center">
        {image && (
          <img src={image} alt="Detection overlay" className="max-h-[420px] w-full object-contain" />
        )}

        {/* Converging brackets */}
        <div style={{ animation: 'target-converge 0.5s cubic-bezier(0.22, 1, 0.36, 1) both' }} className="absolute inset-0">
          <CornerBrackets
            color={acquired ? 'rgba(245,158,11,1)' : 'rgba(34,197,94,0.95)'}
            size={34}
            thickness={3}
            inset={16}
          />
        </div>

        {/* Acquisition stamp */}
        <div
          className="absolute left-1/2 bottom-6 -translate-x-1/2"
          style={{ animation: 'flicker-in 0.4s steps(5, end) 0.35s both' }}
        >
          <span
            className={`font-mono text-xs tracking-[0.22em] px-4 py-2 rounded-[2px] border backdrop-blur-sm ${
              acquired
                ? 'text-amber-300 border-amber-500/60 bg-slate-950/80'
                : 'text-green-500 border-green-500/60 bg-slate-950/80'
            }`}
          >
            {stamp}
          </span>
        </div>
      </div>
    </div>
  );
}
