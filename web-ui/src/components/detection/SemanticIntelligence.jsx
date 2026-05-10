import { BrainCircuit, SunDim } from 'lucide-react';

function StatGauge({ label, value, max = 1.0, isInverse = false }) {
  // If isInverse is true, higher value = worse (red). If false, higher = better (green)
  // For most of our metrics (variance, dissimilarity, anomaly), higher = worse (redder)
  const pct = Math.min((value / max) * 100, 100);
  
  return (
    <div className="mb-4">
      <div className="flex justify-between items-end mb-1">
        <span className="text-xs text-slate-400 uppercase tracking-wider">{label}</span>
        <span className="text-sm font-mono text-white">{value.toFixed(4)}</span>
      </div>
      <div className="w-full h-1.5 rounded-full bg-slate-800">
        <div 
          className="h-full rounded-full bg-gradient-to-r from-amber-500 to-red-500" 
          style={{ width: `${pct}%` }} 
        />
      </div>
    </div>
  );
}

export default function SemanticIntelligence({ geometryAnalysis }) {
  if (!geometryAnalysis) return null;

  const { foundationFeatures, curvatureFeatures } = geometryAnalysis;
  const sfsAnomaly = curvatureFeatures?.mean_normal_deviation || 0;
  const sfsRange = curvatureFeatures?.depth_range || 0; // Using depth range as proxy if SfS missing

  return (
    <div className="glass-card p-5 fade-in-up" style={{ animationDelay: '0.1s' }}>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        {/* Foundation Features (DINOv2) */}
        <div>
          <div className="flex items-center gap-2 mb-4">
            <div className="p-1.5 rounded bg-purple-500/10 border border-purple-500/20">
              <BrainCircuit className="w-4 h-4 text-purple-400" />
            </div>
            <h3 className="text-sm font-bold text-slate-200">Vision Foundation (DINOv2)</h3>
          </div>
          
          <div className="bg-slate-900/40 rounded-lg p-4 border border-white/5 h-[140px]">
            {foundationFeatures ? (
              <>
                <StatGauge label="Semantic Dissimilarity" value={foundationFeatures.dissimilarity} max={1.0} />
                <StatGauge label="Internal Chaos (Var)" value={foundationFeatures.insideVariance} max={2.0} />
              </>
            ) : (
              <div className="h-full flex items-center justify-center text-xs text-slate-500 text-center">
                Foundation model not loaded.<br/>Install transformers to enable DINOv2.
              </div>
            )}
          </div>
        </div>

        {/* Shape from Shading */}
        <div>
          <div className="flex items-center gap-2 mb-4">
            <div className="p-1.5 rounded bg-orange-500/10 border border-orange-500/20">
              <SunDim className="w-4 h-4 text-orange-400" />
            </div>
            <h3 className="text-sm font-bold text-slate-200">Shape-from-Shading (SfS)</h3>
          </div>
          
          <div className="bg-slate-900/40 rounded-lg p-4 border border-white/5 h-[140px]">
            <StatGauge label="Surface Normal Anomaly" value={sfsAnomaly} max={1.0} />
            <div className="mt-3 text-xs text-slate-400 leading-relaxed">
              Detects steep lighting gradients and inconsistent surface normals indicative of deep cavities regardless of 2D shadows.
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
