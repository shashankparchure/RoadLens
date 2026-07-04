import {
  Maximize2, ArrowDown, TrendingUp, Activity, BarChart3, ArrowUpRight,
} from 'lucide-react';
import InstrumentPanel from './instrument/InstrumentPanel';

const FEATURE_CARDS = [
  { key: 'pothole_area', label: 'Pothole Area', unit: 'px²', icon: Maximize2, format: (v) => v.toLocaleString() },
  { key: 'max_depth',    label: 'Max Depth',    unit: '',     icon: ArrowDown,    format: (v) => v.toFixed(4) },
  { key: 'mean_depth',   label: 'Mean Depth',   unit: '',     icon: TrendingUp,   format: (v) => v.toFixed(4) },
  { key: 'depth_std',    label: 'Depth Std',    unit: 'σ',    icon: Activity,     format: (v) => v.toFixed(4) },
  { key: 'depth_range',  label: 'Depth Range',  unit: '',     icon: BarChart3,    format: (v) => v.toFixed(4) },
  { key: 'p90_depth',    label: 'P90 Depth',    unit: '',     icon: ArrowUpRight, format: (v) => v.toFixed(4) },
];

function MiniGauge({ value, max = 1.0 }) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div className="w-full h-[3px] bg-slate-700/60 mt-2.5">
      <div
        className="h-full bg-gradient-to-r from-cyan-500/70 to-amber-500"
        style={{ width: `${pct}%`, transition: 'width 0.8s ease-out' }}
      />
    </div>
  );
}

export default function FeatureStrip({ features }) {
  return (
    <InstrumentPanel
      title="Depth Telemetry"
      accent="cyan"
      statusLabel="6 CH"
      bodyClassName="p-3 sm:p-4"
      flicker={false}
    >
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        {FEATURE_CARDS.map(({ key, label, unit, icon: Icon, format }) => {
          const val = features[key] ?? 0;
          const gaugeMax = key === 'pothole_area' ? 50000 : 1.0;
          return (
            <div
              key={key}
              className="bg-slate-950/60 border border-slate-700 rounded-[2px] p-3.5"
            >
              <div className="flex items-center gap-1.5 mb-2">
                <Icon className="w-3.5 h-3.5 text-cyan-400/70" strokeWidth={1.5} />
                <span className="font-mono text-[9px] text-slate-500 uppercase tracking-[0.14em] truncate">
                  {label}
                </span>
              </div>
              <p className="font-mono text-lg font-semibold text-white leading-none">
                {format(val)}
                {unit && <span className="text-[10px] text-slate-500 ml-1">{unit}</span>}
              </p>
              <MiniGauge value={val} max={gaugeMax} />
            </div>
          );
        })}
      </div>
    </InstrumentPanel>
  );
}
