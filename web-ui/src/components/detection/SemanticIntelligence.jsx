import { BrainCircuit, SunDim } from 'lucide-react';
import InstrumentPanel from '../instrument/InstrumentPanel';

/** Semicircular arc gauge with a mono value — instrument dial. */
function ArcGauge({ label, value, max = 1.0, color = '#a78bfa' }) {
  const pct = Math.max(0, Math.min(value / max, 1));
  const R = 34;
  const C = Math.PI * R; // semicircle length
  return (
    <div className="flex items-center gap-4">
      <svg width="88" height="52" viewBox="0 0 88 52" aria-hidden="true" className="shrink-0">
        <path
          d={`M 10 46 A ${R} ${R} 0 0 1 78 46`}
          fill="none"
          stroke="#29303f"
          strokeWidth="6"
          strokeLinecap="round"
        />
        <path
          d={`M 10 46 A ${R} ${R} 0 0 1 78 46`}
          fill="none"
          stroke={color}
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={`${C * pct} ${C}`}
          style={{ filter: `drop-shadow(0 0 4px ${color}66)`, transition: 'stroke-dasharray 0.8s ease-out' }}
        />
        {/* Quadrant ticks */}
        <line x1="44" y1="6" x2="44" y2="12" stroke="#3c4354" strokeWidth="1.5" />
        <line x1="10" y1="46" x2="14" y2="46" stroke="#3c4354" strokeWidth="1.5" />
        <line x1="74" y1="46" x2="78" y2="46" stroke="#3c4354" strokeWidth="1.5" />
      </svg>
      <div className="min-w-0">
        <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-slate-500">{label}</p>
        <p className="font-mono text-lg font-semibold" style={{ color }}>
          {value.toFixed(4)}
        </p>
      </div>
    </div>
  );
}

export default function SemanticIntelligence({ geometryAnalysis }) {
  if (!geometryAnalysis) return null;

  const { foundationFeatures, curvatureFeatures } = geometryAnalysis;
  const sfsAnomaly = curvatureFeatures?.mean_normal_deviation || 0;

  return (
    <InstrumentPanel
      title="Semantic Verification"
      accent="holo"
      statusLabel="DINOV2 · SFS"
      bodyClassName="p-4 sm:p-5"
      flicker={false}
    >
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Foundation Features (DINOv2) */}
        <div>
          <div className="flex items-center gap-2 mb-4">
            <div className="p-1.5 rounded-[2px] bg-holo-violet/10 border border-holo-violet/30">
              <BrainCircuit className="w-4 h-4 text-holo-violet-bright" />
            </div>
            <h3 className="text-sm font-semibold text-slate-300">Vision foundation — is it a real crater?</h3>
          </div>

          <div className="bg-slate-950/60 rounded-[2px] p-4 border border-slate-700 min-h-[150px] space-y-4">
            {foundationFeatures ? (
              <>
                <ArcGauge
                  label="Semantic dissimilarity"
                  value={foundationFeatures.dissimilarity}
                  max={1.0}
                  color="#a78bfa"
                />
                <ArcGauge
                  label="Interior variance"
                  value={foundationFeatures.insideVariance}
                  max={2.0}
                  color="#e879f9"
                />
              </>
            ) : (
              <div className="h-full min-h-[120px] flex items-center justify-center font-mono text-[11px] text-slate-500 text-center uppercase tracking-wider">
                Foundation model offline
              </div>
            )}
          </div>
        </div>

        {/* Shape from Shading */}
        <div>
          <div className="flex items-center gap-2 mb-4">
            <div className="p-1.5 rounded-[2px] bg-holo-teal/10 border border-holo-teal/30">
              <SunDim className="w-4 h-4 text-holo-teal-bright" />
            </div>
            <h3 className="text-sm font-semibold text-slate-300">Shape-from-shading — wall steepness</h3>
          </div>

          <div className="bg-slate-950/60 rounded-[2px] p-4 border border-slate-700 min-h-[150px]">
            <ArcGauge
              label="Surface normal anomaly"
              value={sfsAnomaly}
              max={1.0}
              color="#5eead4"
            />
            <p className="mt-4 text-xs text-slate-500 leading-relaxed">
              Steep lighting gradients and deviant surface normals mark real cavity walls —
              a depth witness that ignores 2D shadows entirely.
            </p>
          </div>
        </div>
      </div>
    </InstrumentPanel>
  );
}
