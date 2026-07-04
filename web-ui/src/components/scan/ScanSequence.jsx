import { useEffect, useRef, useState, useMemo } from 'react';
import InstrumentPanel from '../instrument/InstrumentPanel';
import CornerBrackets from '../instrument/CornerBrackets';
import RadarPulse from '../instrument/RadarPulse';
import TelemetryLog from './TelemetryLog';
import ScanError from './ScanError';

/** Real pipeline stages — honest status checklist, no fake percentages. */
const STAGES = [
  { id: '01', label: 'INGEST', detail: 'frame received, preprocessing' },
  { id: '02', label: 'YOLOV8-SEG', detail: 'pothole instance segmentation' },
  { id: '03', label: 'DEPTH-V2', detail: 'monocular depth estimation' },
  { id: '04', label: 'SFS-GEOMETRY', detail: 'shape-from-shading + curvature' },
  { id: '05', label: 'DINOV2', detail: 'semantic patch verification' },
  { id: '06', label: 'WATER-ENSEMBLE', detail: 'submersion probability' },
  { id: '07', label: 'TEMPORAL', detail: 'age & progression forecast' },
];

const BOOT_MS = 400;
const CADENCE_MS = 700;
const FAST_CADENCE_MS = 150;
const MIN_TOTAL_MS = 2500;
const HEARTBEAT_MS = 8000;
const SLOW_WARN_MS = 90000;

const HEARTBEATS = [
  '… ensemble still computing',
  '… deep models are heavy, hold',
  '… cross-checking depth witnesses',
  '… backend crunching, link alive',
];

/**
 * Cinematic analysis loader. App owns the fetch; this component turns the
 * wait into a scan: laser sweep over the staged frame, streaming stage log,
 * radar pulse. Calls onComplete once the (min-timed) sequence finishes AND
 * the API has responded successfully.
 */
export default function ScanSequence({ image, status, error, onComplete, onRetry, onDismiss }) {
  // The parent mounts this with key={scanId}, so every scan gets fresh state.
  const [tick, setTick] = useState({ now: 0, successAt: null });
  const statusRef = useRef('pending');
  const successAtRef = useRef(null);
  const completedRef = useRef(false);
  const reducedMotion = useMemo(
    () => window.matchMedia('(prefers-reduced-motion: reduce)').matches,
    [],
  );

  // Mirror status into a ref the clock callback can read.
  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  // Single clock; success timestamp is captured inside the interval callback
  // (async — outside render), so cadence compression needs no render-time refs.
  useEffect(() => {
    const start = performance.now();
    const t = setInterval(() => {
      const elapsed = performance.now() - start;
      if (statusRef.current === 'success' && successAtRef.current === null) {
        successAtRef.current = elapsed;
      }
      setTick({ now: elapsed, successAt: successAtRef.current });
    }, 100);
    return () => clearInterval(t);
  }, []);

  const { now, successAt } = tick;

  // Derive revealed-stage count from the timeline.
  const baseElapsed = Math.max(0, (successAt ?? now) - BOOT_MS);
  const baseRevealed = Math.floor(baseElapsed / CADENCE_MS);
  const extraRevealed = successAt !== null ? Math.floor(Math.max(0, now - successAt) / FAST_CADENCE_MS) : 0;
  const revealed = reducedMotion ? STAGES.length : Math.min(STAGES.length, baseRevealed + extraRevealed);
  const allRevealed = revealed >= STAGES.length;

  // Completion: everything revealed, API done, min showtime honored.
  useEffect(() => {
    if (completedRef.current) return;
    const ready =
      status === 'success' && (reducedMotion || (allRevealed && now >= MIN_TOTAL_MS));
    if (ready) {
      completedRef.current = true;
      onComplete();
    }
  }, [status, allRevealed, now, reducedMotion, onComplete]);

  const failed = status === 'error';

  const stages = STAGES.map((s, i) => {
    let st = 'pending';
    if (i < revealed) st = 'done';
    else if (i === revealed && !allRevealed && now > BOOT_MS) st = 'active';
    if (failed && st === 'active') st = 'fail';
    // While awaiting a slow API with all stages shown, keep the last one active.
    if (!failed && allRevealed && status === 'pending' && i === STAGES.length - 1) st = 'active';
    return { ...s, status: st };
  });

  // AWAITING heartbeats + slow-link warning.
  const extraLines = [];
  if (!failed && allRevealed && status === 'pending') {
    const revealAllAt = BOOT_MS + STAGES.length * CADENCE_MS;
    const beats = Math.max(0, Math.floor((now - revealAllAt) / HEARTBEAT_MS));
    for (let i = 0; i < Math.min(beats, HEARTBEATS.length); i++) extraLines.push(HEARTBEATS[i]);
    if (now > SLOW_WARN_MS) extraLines.push('!LINK SLOW — backend may be under load');
  }
  if (failed) extraLines.push('!PIPELINE FAULT — see diagnostic below');

  const booted = now > BOOT_MS || reducedMotion;

  return (
    <div className="max-w-5xl mx-auto space-y-4">
      <InstrumentPanel
        title="Live Scan"
        accent="cyan"
        statusLabel={failed ? 'FAULT' : status === 'success' ? 'FINALIZING' : 'ACQUIRING'}
        bodyClassName="p-4 sm:p-5"
      >
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          {/* Scanner frame */}
          <div className="lg:col-span-2">
            <div
              className={`relative bg-slate-950 border border-slate-700 rounded-[2px] overflow-hidden scanline-overlay min-h-[280px] flex items-center justify-center ${booted ? '' : 'opacity-0'} ${!reducedMotion ? 'flicker-in' : ''}`}
            >
              {image && (
                <img
                  src={image}
                  alt="Frame under analysis"
                  className="max-h-[380px] w-full object-contain"
                />
              )}
              <CornerBrackets color="rgba(103,232,249,0.85)" size={20} inset={10} />

              {/* Laser sweep */}
              {!failed && !reducedMotion && (
                <div
                  className="absolute left-0 right-0 h-[3px] pointer-events-none"
                  style={{
                    background:
                      'linear-gradient(90deg, transparent, rgba(103,232,249,0.9) 20%, rgba(103,232,249,0.9) 80%, transparent)',
                    boxShadow: '0 0 18px rgba(103,232,249,0.8), 0 6px 30px rgba(103,232,249,0.35)',
                    animation: 'scan-sweep-y 2.8s ease-in-out infinite',
                  }}
                />
              )}

              {/* Frame HUD */}
              <span className="absolute top-2.5 left-3 font-mono text-[9px] tracking-[0.18em] text-cyan-300/80 uppercase">
                CH-1 RGB · LIVE
              </span>
              <span className="absolute bottom-2.5 right-3 font-mono text-[9px] tracking-[0.18em] text-slate-500 uppercase">
                {failed ? 'SCAN HALTED' : 'SCANNING SURFACE'}
              </span>
            </div>
          </div>

          {/* Radar + log */}
          <div className="flex flex-col items-center lg:items-start gap-5">
            <RadarPulse size={110} active={!failed && !reducedMotion} color={failed ? '#ef4444' : '#22d3ee'} />
            <div className="w-full min-h-[190px]">
              <TelemetryLog stages={stages} extraLines={extraLines} />
            </div>
          </div>
        </div>
      </InstrumentPanel>

      {failed && <ScanError message={error} onRetry={onRetry} onDismiss={onDismiss} />}
    </div>
  );
}
