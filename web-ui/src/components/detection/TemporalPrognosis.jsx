import { Clock, TrendingUp, AlertCircle, ThermometerSnowflake } from 'lucide-react';

function SeverityStep({ label, score, days, isCurrent }) {
  // Score is 0.0 to 1.0 (approx mapping from Shallow to Deep)
  let color = 'text-cyan-400 bg-cyan-400/10 border-cyan-400/30';
  let dotColor = 'bg-cyan-400';
  
  if (label === 'Deep') {
    color = 'text-red-400 bg-red-400/10 border-red-400/30';
    dotColor = 'bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.8)]';
  } else if (label === 'Moderate') {
    color = 'text-amber-400 bg-amber-400/10 border-amber-400/30';
    dotColor = 'bg-amber-400 shadow-[0_0_8px_rgba(251,191,36,0.6)]';
  } else if (label === 'Shallow') {
    color = 'text-yellow-400 bg-yellow-400/10 border-yellow-400/30';
    dotColor = 'bg-yellow-400';
  }

  return (
    <div className="relative flex flex-col items-center flex-1">
      <div className="text-xs text-slate-400 mb-2 font-mono">+{days}d</div>
      
      {/* Node */}
      <div className={`w-4 h-4 rounded-full z-10 ${dotColor} ${isCurrent ? 'ring-4 ring-white/20' : ''}`} />
      
      {/* Label Box */}
      <div className={`mt-3 px-3 py-1 rounded-md border text-xs font-bold ${color}`}>
        {label}
      </div>
    </div>
  );
}

export default function TemporalPrognosis({ temporalAnalysis, currentSeverity }) {
  if (!temporalAnalysis) return null;

  const { ageCategory, ageScore, ageDescription, progression } = temporalAnalysis;

  return (
    <div className="glass-card p-5 fade-in-up">
      <div className="flex items-center gap-2 mb-4">
        <Clock className="w-5 h-5 text-amber-500" />
        <h3 className="text-lg font-bold text-white tracking-tight">Temporal Prognosis</h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Left: Age Estimation */}
        <div className="col-span-1 bg-slate-900/50 rounded-xl p-4 border border-white/5">
          <div className="flex justify-between items-start mb-2">
            <span className="text-xs text-slate-400 uppercase tracking-wider font-semibold">Estimated Age</span>
            <span className="text-xs font-mono text-amber-400">Score: {ageScore.toFixed(2)}</span>
          </div>
          <p className="text-xl font-bold text-white">{ageCategory}</p>
          <p className="text-sm text-slate-400 mt-1">{ageDescription}</p>
          
          <div className="mt-4 flex gap-2">
            <div className="flex-1 bg-slate-800 rounded p-2 border border-white/5 text-center">
              <div className="text-[10px] text-slate-500 uppercase">Edge Sharpness</div>
              <div className="text-sm font-mono text-cyan-400">{temporalAnalysis.edgeSharpness}</div>
            </div>
            <div className="flex-1 bg-slate-800 rounded p-2 border border-white/5 text-center">
              <div className="text-[10px] text-slate-500 uppercase">Crack Texture</div>
              <div className="text-sm font-mono text-cyan-400">{temporalAnalysis.crackTexture}</div>
            </div>
          </div>
        </div>

        {/* Right: Deterioration Timeline */}
        <div className="col-span-1 md:col-span-2 bg-slate-900/50 rounded-xl p-4 border border-white/5 flex flex-col justify-center">
          <div className="flex items-center justify-between mb-6">
            <span className="text-xs text-slate-400 uppercase tracking-wider font-semibold flex items-center gap-1">
              <TrendingUp className="w-3 h-3" /> Deterioration Forecast
            </span>
            <span className="text-xs bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 px-2 py-0.5 rounded flex items-center gap-1">
              <ThermometerSnowflake className="w-3 h-3" /> Freeze-Thaw Physics
            </span>
          </div>

          <div className="relative flex justify-between items-center w-full px-4">
            {/* The Track Line */}
            <div className="absolute left-[10%] right-[10%] top-6 h-1 bg-slate-700/50 rounded-full" />
            
            <SeverityStep label={currentSeverity} score={0.5} days="0" isCurrent={true} />
            <SeverityStep label={progression['30d']} score={0.6} days="30" />
            <SeverityStep label={progression['60d']} score={0.8} days="60" />
            <SeverityStep label={progression['90d']} score={1.0} days="90" />
          </div>
        </div>
      </div>
    </div>
  );
}
