import { AlertOctagon, RotateCcw, X } from 'lucide-react';
import InstrumentPanel from '../instrument/InstrumentPanel';

/** Analysis failure panel — replaces alert(). */
export default function ScanError({ message, onRetry, onDismiss }) {
  return (
    <InstrumentPanel title="Pipeline Fault" accent="red" statusLabel="E-01" flicker>
      <div className="h-[3px] hazard-stripes-red stripe-scroll" aria-hidden="true" />
      <div className="p-5 flex flex-col sm:flex-row sm:items-center gap-4">
        <AlertOctagon className="w-8 h-8 text-red-500 shrink-0" strokeWidth={1.6} />
        <div className="flex-1 min-w-0">
          <p className="text-sm text-slate-300">
            {message || 'The analysis pipeline did not return a result.'}
          </p>
          <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-slate-500 mt-1.5">
            Verify the backend is running, then retry the scan.
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <button
            type="button"
            onClick={onRetry}
            className="px-4 py-2 rounded-[2px] bg-red-500/15 border border-red-500/50 text-red-500 font-mono text-[11px] uppercase tracking-[0.14em] hover:bg-red-500/25 transition-colors inline-flex items-center gap-2"
          >
            <RotateCcw className="w-3.5 h-3.5" /> Retry
          </button>
          <button
            type="button"
            onClick={onDismiss}
            className="px-4 py-2 rounded-[2px] border border-slate-700 text-slate-400 font-mono text-[11px] uppercase tracking-[0.14em] hover:text-slate-200 hover:border-slate-600 transition-colors inline-flex items-center gap-2"
          >
            <X className="w-3.5 h-3.5" /> Dismiss
          </button>
        </div>
      </div>
    </InstrumentPanel>
  );
}
