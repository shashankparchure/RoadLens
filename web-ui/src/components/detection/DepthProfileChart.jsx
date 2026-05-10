import { useState, useMemo } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Line, ComposedChart,
} from 'recharts';
import { TrendingDown, ChevronDown } from 'lucide-react';

const SEVERITY_COLORS = {
  'No Pothole': '#22c55e',
  'Shallow':    '#eab308',
  'Moderate':   '#f97316',
  'Deep':       '#ef4444',
};

/**
 * Renders the depth cross-section profile for a selected pothole.
 *
 * Shows the actual depth (filled area) vs the extrapolated road surface
 * (dashed line). The gap between them is the "bowl depth" — the signal
 * that drives severity classification.
 */
export default function DepthProfileChart({ potholes }) {
  const candidates = useMemo(
    () => (potholes || []).filter(p => p.depthProfile?.slicePoints?.length > 0),
    [potholes],
  );

  const [selectedIdx, setSelectedIdx] = useState(0);

  if (candidates.length === 0) {
    return null;
  }

  const pothole = candidates[selectedIdx] || candidates[0];
  const profile = pothole.depthProfile;
  const severity = pothole.consensusSeverity || 'No Pothole';
  const fillColor = SEVERITY_COLORS[severity] || '#8b5cf6';

  // Downsample slice data for smooth charting (cap at ~200 points)
  const raw = profile.slicePoints;
  const step = Math.max(1, Math.floor(raw.length / 200));
  const chartData = raw.filter((_, i) => i % step === 0);

  // Find the pothole region boundaries in the chart data
  const maskStart = chartData.findIndex(p => p.inMask);
  const maskEnd = chartData.length - 1 - [...chartData].reverse().findIndex(p => p.inMask);

  const hasWater = pothole.waterAnalysis?.waterDetected;
  const isDistorted = hasWater;

  return (
    <div className="glass-card-bright p-4 fade-in-up border border-white/10 relative overflow-hidden">
      {/* Header row */}
      <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <TrendingDown className="w-4 h-4 text-amber-400" />
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest">
            Depth Cross-Section
          </h3>
        </div>

        <div className="flex items-center gap-3">
          {/* Bowl depth badge */}
          <span
            className="px-2.5 py-1 rounded-full text-[11px] font-bold tracking-wide"
            style={{
              background: isDistorted ? (profile.predictedBowlDepth ? '#ef444420' : '#3b82f620') : `${fillColor}20`,
              color: isDistorted ? (profile.predictedBowlDepth ? '#f87171' : '#60a5fa') : fillColor,
              border: `1px solid ${isDistorted ? (profile.predictedBowlDepth ? '#ef444440' : '#3b82f640') : fillColor + '40'}`,
            }}
          >
            {isDistorted 
              ? (profile.predictedBowlDepth ? `Predicted Depth: ${(profile.predictedBowlDepth * 100).toFixed(1)}%` : 'Depth: Unknown (Submerged)') 
              : `Bowl Depth: ${(profile.bowlDepth * 100).toFixed(1)}%`}
          </span>

          {/* Pothole selector (when multiple) */}
          {candidates.length > 1 && (
            <div className="relative z-10">
              <select
                value={selectedIdx}
                onChange={e => setSelectedIdx(Number(e.target.value))}
                className="appearance-none bg-slate-800/80 text-slate-300 text-xs
                           pl-2.5 pr-7 py-1.5 rounded-md border border-white/10
                           focus:outline-none focus:border-amber-500/40 cursor-pointer"
              >
                {candidates.map((p, i) => (
                  <option key={p.id} value={i}>
                    Pothole {p.id} — {p.consensusSeverity}
                  </option>
                ))}
              </select>
              <ChevronDown className="w-3 h-3 text-slate-500 absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none" />
            </div>
          )}
        </div>
      </div>

      {/* Warning Banner for Water */}
      {isDistorted && (
        <div className="absolute top-12 left-1/2 -translate-x-1/2 bg-blue-900/80 border border-blue-500/50 text-blue-200 text-[10px] px-3 py-1 rounded-full z-10 backdrop-blur-sm flex items-center gap-1.5">
          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          Water refraction is flattening the depth profile
        </div>
      )}

      {/* Chart */}
      <div className="h-[180px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
            <defs>
              <linearGradient id="depthFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={isDistorted ? '#3b82f6' : fillColor} stopOpacity={isDistorted ? 0.3 : 0.5} />
                <stop offset="100%" stopColor={isDistorted ? '#3b82f6' : fillColor} stopOpacity={0.05} />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />

            <XAxis
              dataKey="x"
              tick={{ fontSize: 10, fill: '#94a3b8' }}
              axisLine={{ stroke: '#334155' }}
              tickLine={false}
              label={{
                value: 'Position (px)',
                position: 'insideBottomRight',
                offset: -2,
                style: { fontSize: 10, fill: '#64748b' },
              }}
            />
            <YAxis
              domain={['auto', 'auto']}
              tick={{ fontSize: 10, fill: '#94a3b8' }}
              axisLine={{ stroke: '#334155' }}
              tickLine={false}
              label={{
                value: 'Depth',
                angle: -90,
                position: 'insideLeft',
                style: { fontSize: 10, fill: '#64748b' },
              }}
            />

            <Tooltip
              contentStyle={{
                background: 'rgba(15,23,42,0.95)',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: 8,
                fontSize: 11,
                color: '#e2e8f0',
              }}
              formatter={(value, name) => {
                const labels = {
                  actualDepth: isDistorted ? 'Water Surface' : 'Actual Depth',
                  roadSurface: 'Road Surface',
                  predictedDepth: 'Predicted Submerged Depth'
                };
                return [typeof value === 'number' ? value.toFixed(4) : value, labels[name] || name];
              }}
              labelFormatter={v => `Position: ${v}px`}
            />

            {/* Pothole region highlight */}
            {maskStart >= 0 && maskEnd >= 0 && (
              <ReferenceLine
                x={chartData[maskStart]?.x}
                stroke={isDistorted ? '#3b82f6' : fillColor}
                strokeDasharray="4 2"
                strokeOpacity={0.4}
                label={{
                  value: '◀ pothole',
                  position: 'top',
                  style: { fontSize: 9, fill: isDistorted ? '#60a5fa' : fillColor },
                }}
              />
            )}
            {maskStart >= 0 && maskEnd >= 0 && (
              <ReferenceLine
                x={chartData[maskEnd]?.x}
                stroke={isDistorted ? '#3b82f6' : fillColor}
                strokeDasharray="4 2"
                strokeOpacity={0.4}
                label={{
                  value: 'pothole ▶',
                  position: 'top',
                  style: { fontSize: 9, fill: isDistorted ? '#60a5fa' : fillColor },
                }}
              />
            )}

            {/* Road surface — dashed reference line */}
            <Line
              type="monotone"
              dataKey="roadSurface"
              stroke="#38bdf8"
              strokeWidth={2}
              strokeDasharray="6 3"
              dot={false}
              name="roadSurface"
              isAnimationActive={false}
            />
            
            {/* Predicted Submerged Depth (Visible only if computed) */}
            {isDistorted && profile.predictedBowlDepth && (
              <Line
                type="monotone"
                dataKey="predictedDepth"
                stroke="#f87171"
                strokeWidth={2}
                strokeDasharray="4 4"
                dot={false}
                name="predictedDepth"
                isAnimationActive={false}
              />
            )}

            {/* Actual depth (or water surface) — filled area */}
            <Area
              type="monotone"
              dataKey="actualDepth"
              stroke={isDistorted ? '#60a5fa' : fillColor}
              strokeWidth={isDistorted ? 2 : 2}
              strokeDasharray={isDistorted ? "4 4" : "0"}
              fill="url(#depthFill)"
              name="actualDepth"
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-5 mt-2 ml-1 flex-wrap">
        <div className="flex items-center gap-1.5">
          <div className="w-5 h-0.5 rounded" style={{ background: isDistorted ? '#60a5fa' : fillColor, borderTop: isDistorted ? '2px dashed #60a5fa' : 'none' }} />
          <span className="text-[10px] text-slate-500">
            {isDistorted ? 'Water Surface (Apparent)' : 'Actual Depth'}
          </span>
        </div>
        
        {isDistorted && profile.predictedBowlDepth && (
          <div className="flex items-center gap-1.5">
            <div className="w-5 h-0.5 rounded border-t-2 border-dashed border-red-400" />
            <span className="text-[10px] text-slate-500">Predicted Depth (Extrapolated)</span>
          </div>
        )}
        
        <div className="flex items-center gap-1.5">
          <div className="w-5 h-0.5 rounded border-t-2 border-dashed border-sky-400" />
          <span className="text-[10px] text-slate-500">Road Surface (extrapolated)</span>
        </div>
        
        {!isDistorted && (
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full" style={{ background: `${fillColor}50` }} />
            <span className="text-[10px] text-slate-500">Gap = Bowl Depth ({severity})</span>
          </div>
        )}
      </div>
    </div>
  );
}
