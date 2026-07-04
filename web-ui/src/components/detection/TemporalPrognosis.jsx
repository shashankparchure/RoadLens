import { TrendingUp, ThermometerSnowflake } from 'lucide-react';
import InstrumentPanel from '../instrument/InstrumentPanel';
import StatReadout from '../instrument/StatReadout';
import { severityToken } from '../../theme/severity';

function TimelineNode({ label, days, isCurrent }) {
  const token = severityToken(label);
  return (
    <div className="relative flex flex-col items-center flex-1 min-w-0">
      <div className="font-mono text-[10px] tracking-[0.12em] text-slate-500 mb-2">+{days}D</div>
      <div
        className={`w-3.5 h-3.5 rounded-full z-10 ${isCurrent ? 'ring-4 ring-white/15' : ''}`}
        style={{ background: token.color, boxShadow: `0 0 10px ${token.ring}` }}
      />
      <div
        className="mt-3 px-2.5 py-1 rounded-[2px] border font-mono text-[10px] uppercase tracking-[0.1em] font-semibold"
        style={{ color: token.color, borderColor: token.ring, background: token.bg }}
      >
        {token.label}
      </div>
    </div>
  );
}

export default function TemporalPrognosis({ temporalAnalysis, currentSeverity }) {
  if (!temporalAnalysis) return null;

  const { ageCategory, ageScore, ageDescription, progression } = temporalAnalysis;

  return (
    <InstrumentPanel
      title="Temporal Forecast"
      accent="amber"
      statusLabel="30 / 60 / 90 DAY"
      bodyClassName="p-4 sm:p-5"
      flicker={false}
    >
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {/* Age estimation */}
        <div className="col-span-1 bg-slate-950/60 rounded-[2px] p-4 border border-slate-700">
          <div className="flex justify-between items-start mb-2">
            <span className="font-mono text-[10px] text-slate-500 uppercase tracking-[0.14em]">Estimated age</span>
            <span className="font-mono text-[10px] text-amber-400">SCORE {ageScore.toFixed(2)}</span>
          </div>
          <p className="text-xl font-bold text-white">{ageCategory}</p>
          <p className="text-sm text-slate-400 mt-1 leading-snug">{ageDescription}</p>

          <div className="mt-4 grid grid-cols-2 gap-2">
            <div className="bg-slate-900 rounded-[2px] p-2.5 border border-slate-700">
              <StatReadout label="Edge sharpness" value={temporalAnalysis.edgeSharpness} color="#67e8f9" size="sm" />
            </div>
            <div className="bg-slate-900 rounded-[2px] p-2.5 border border-slate-700">
              <StatReadout label="Crack texture" value={temporalAnalysis.crackTexture} color="#67e8f9" size="sm" />
            </div>
          </div>
        </div>

        {/* Deterioration rail */}
        <div className="col-span-1 md:col-span-2 bg-slate-950/60 rounded-[2px] p-4 border border-slate-700 flex flex-col justify-center">
          <div className="flex items-center justify-between mb-6 flex-wrap gap-2">
            <span className="font-mono text-[10px] text-slate-500 uppercase tracking-[0.14em] flex items-center gap-1.5">
              <TrendingUp className="w-3 h-3" /> Deterioration forecast
            </span>
            <span className="font-mono text-[10px] bg-cyan-500/10 text-cyan-300 border border-cyan-500/25 px-2 py-0.5 rounded-[2px] flex items-center gap-1 uppercase tracking-[0.1em]">
              <ThermometerSnowflake className="w-3 h-3" /> Freeze-thaw physics
            </span>
          </div>

          <div className="relative flex justify-between items-start w-full px-2">
            {/* Rail with tick marks */}
            <div className="absolute left-[8%] right-[8%] top-[30px] h-px bg-slate-700" aria-hidden="true">
              {[0, 25, 50, 75, 100].map((pos) => (
                <span
                  key={pos}
                  className="absolute top-[-3px] w-px h-[7px] bg-slate-600"
                  style={{ left: `${pos}%` }}
                />
              ))}
            </div>

            <TimelineNode label={currentSeverity} days="0" isCurrent />
            <TimelineNode label={progression['30d']} days="30" />
            <TimelineNode label={progression['60d']} days="60" />
            <TimelineNode label={progression['90d']} days="90" />
          </div>
        </div>
      </div>
    </InstrumentPanel>
  );
}
