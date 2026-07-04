import { useState, useMemo } from 'react';
import {
  Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Line, ComposedChart,
} from 'recharts';
import { ChevronDown, Info } from 'lucide-react';
import InstrumentPanel from '../instrument/InstrumentPanel';
import { severityToken } from '../../theme/severity';

const WATER = { line: '#5eead4', fill: '#2dd4bf', soft: '#2dd4bf' };
const MONO_TICK = { fontSize: 10, fill: '#97a0b1', fontFamily: 'IBM Plex Mono, monospace' };

/**
 * Depth cross-section through the pothole centroid: measured depth vs the
 * extrapolated road surface; the gap is bowl depth. Water mode swaps to the
 * iridescent palette and shows the predicted submerged floor.
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
  const fillColor = severityToken(severity).color;

  // Downsample slice data for smooth charting (cap at ~200 points)
  const raw = profile.slicePoints;
  const step = Math.max(1, Math.floor(raw.length / 200));
  const chartData = raw.filter((_, i) => i % step === 0);

  // Find the pothole region boundaries in the chart data
  const maskStart = chartData.findIndex(p => p.inMask);
  const maskEnd = chartData.length - 1 - [...chartData].reverse().findIndex(p => p.inMask);

  const hasWater = pothole.waterAnalysis?.waterDetected;
  const isDistorted = hasWater;

  const badge = isDistorted
    ? profile.predictedBowlDepth
      ? { text: `PREDICTED DEPTH ${(profile.predictedBowlDepth * 100).toFixed(1)}%`, color: '#f87171' }
      : { text: 'DEPTH UNKNOWN · SUBMERGED', color: WATER.line }
    : { text: `BOWL DEPTH ${(profile.bowlDepth * 100).toFixed(1)}%`, color: fillColor };

  return (
    <InstrumentPanel
      title="Cross-Section Profile"
      accent={isDistorted ? 'holo' : 'amber'}
      statusLabel={isDistorted ? 'WATER MODE' : severityToken(severity).label}
      flicker={false}
      headerRight={
        <div className="flex items-center gap-3">
          <span
            className="px-2.5 py-1 rounded-[2px] font-mono text-[10px] tracking-[0.1em] font-semibold border"
            style={{ background: `${badge.color}18`, color: badge.color, borderColor: `${badge.color}45` }}
          >
            {badge.text}
          </span>
          {candidates.length > 1 && (
            <div className="relative">
              <select
                value={selectedIdx}
                onChange={e => setSelectedIdx(Number(e.target.value))}
                className="appearance-none bg-slate-800 text-slate-300 font-mono text-[10px] tracking-[0.08em]
                           pl-2.5 pr-7 py-1.5 rounded-[2px] border border-slate-700
                           focus:outline-none focus:border-amber-500/50 cursor-pointer"
              >
                {candidates.map((p, i) => (
                  <option key={p.id} value={i}>
                    P-{String(p.id).padStart(2, '0')} · {p.consensusSeverity}
                  </option>
                ))}
              </select>
              <ChevronDown className="w-3 h-3 text-slate-500 absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none" />
            </div>
          )}
        </div>
      }
      bodyClassName="p-4 relative"
    >
      {/* Water distortion notice */}
      {isDistorted && (
        <div className="absolute top-3 left-1/2 -translate-x-1/2 bg-slate-950/85 border border-holo-teal/40 text-holo-teal-bright font-mono text-[9px] uppercase tracking-[0.12em] px-3 py-1 rounded-[2px] z-10 backdrop-blur-sm flex items-center gap-1.5">
        <Info className="w-3 h-3" />
          Water refraction is flattening the depth profile
        </div>
      )}

      {/* Chart */}
      <div className="h-[190px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
            <defs>
              <linearGradient id="depthFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={isDistorted ? WATER.fill : fillColor} stopOpacity={isDistorted ? 0.3 : 0.5} />
                <stop offset="100%" stopColor={isDistorted ? WATER.fill : fillColor} stopOpacity={0.05} />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="#29303f" />

            <XAxis
              dataKey="x"
              tick={MONO_TICK}
              axisLine={{ stroke: '#29303f' }}
              tickLine={false}
              label={{
                value: 'POSITION (PX)',
                position: 'insideBottomRight',
                offset: -2,
                style: { fontSize: 9, fill: '#626b7d', fontFamily: 'IBM Plex Mono, monospace', letterSpacing: '0.1em' },
              }}
            />
            <YAxis
              domain={['auto', 'auto']}
              tick={MONO_TICK}
              axisLine={{ stroke: '#29303f' }}
              tickLine={false}
              label={{
                value: 'DEPTH',
                angle: -90,
                position: 'insideLeft',
                style: { fontSize: 9, fill: '#626b7d', fontFamily: 'IBM Plex Mono, monospace', letterSpacing: '0.1em' },
              }}
            />

            <Tooltip
              contentStyle={{
                background: 'rgba(10,12,16,0.95)',
                border: '1px solid #3c4354',
                borderRadius: 2,
                fontSize: 11,
                fontFamily: 'IBM Plex Mono, monospace',
                color: '#cdd3dd',
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

            {/* Pothole region boundaries */}
            {maskStart >= 0 && maskEnd >= 0 && (
              <ReferenceLine
                x={chartData[maskStart]?.x}
                stroke={isDistorted ? WATER.soft : fillColor}
                strokeDasharray="4 2"
                strokeOpacity={0.4}
                label={{
                  value: '◀ pothole',
                  position: 'top',
                  style: { fontSize: 9, fill: isDistorted ? WATER.line : fillColor, fontFamily: 'IBM Plex Mono, monospace' },
                }}
              />
            )}
            {maskStart >= 0 && maskEnd >= 0 && (
              <ReferenceLine
                x={chartData[maskEnd]?.x}
                stroke={isDistorted ? WATER.soft : fillColor}
                strokeDasharray="4 2"
                strokeOpacity={0.4}
                label={{
                  value: 'pothole ▶',
                  position: 'top',
                  style: { fontSize: 9, fill: isDistorted ? WATER.line : fillColor, fontFamily: 'IBM Plex Mono, monospace' },
                }}
              />
            )}

            {/* Road surface — dashed reference line */}
            <Line
              type="monotone"
              dataKey="roadSurface"
              stroke="#67e8f9"
              strokeWidth={2}
              strokeDasharray="6 3"
              dot={false}
              name="roadSurface"
              isAnimationActive={false}
            />

            {/* Predicted submerged depth (water only) */}
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
              stroke={isDistorted ? WATER.line : fillColor}
              strokeWidth={2}
              strokeDasharray={isDistorted ? '4 4' : '0'}
              fill="url(#depthFill)"
              name="actualDepth"
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-5 mt-2 ml-1 flex-wrap font-mono text-[9px] uppercase tracking-[0.1em] text-slate-500">
        <div className="flex items-center gap-1.5">
          <div className="w-5 h-0.5" style={{ background: isDistorted ? WATER.line : fillColor }} />
          <span>{isDistorted ? 'Water surface (apparent)' : 'Actual depth'}</span>
        </div>

        {isDistorted && profile.predictedBowlDepth && (
          <div className="flex items-center gap-1.5">
            <div className="w-5 h-0.5 border-t-2 border-dashed border-red-400" />
            <span>Predicted depth (extrapolated)</span>
          </div>
        )}

        <div className="flex items-center gap-1.5">
          <div className="w-5 h-0.5 border-t-2 border-dashed border-cyan-300" />
          <span>Road surface (extrapolated)</span>
        </div>

        {!isDistorted && (
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full" style={{ background: `${fillColor}50` }} />
            <span>Gap = bowl depth ({severity})</span>
          </div>
        )}
      </div>
    </InstrumentPanel>
  );
}
