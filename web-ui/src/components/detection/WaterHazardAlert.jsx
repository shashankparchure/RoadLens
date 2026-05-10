import { Droplets, AlertTriangle } from 'lucide-react';

export default function WaterHazardAlert({ waterAnalysis }) {
  if (!waterAnalysis?.waterDetected) return null;

  return (
    <div className="relative overflow-hidden glass-card-bright border-red-500/50 p-4 mb-6 fade-in-up glow-red-pulse">
      {/* Background Warning Stripes */}
      <div className="absolute inset-0 opacity-[0.03] pointer-events-none" 
           style={{ backgroundImage: 'repeating-linear-gradient(45deg, transparent, transparent 10px, #ef4444 10px, #ef4444 20px)' }} />
      
      <div className="relative flex items-start gap-4">
        <div className="p-3 rounded-xl bg-red-500/20 border border-red-500/30 text-red-400">
          <AlertTriangle className="w-6 h-6 animate-pulse" />
        </div>
        
        <div className="flex-1">
          <h3 className="text-lg font-bold text-red-400 flex items-center gap-2">
            Water Hazard Detected
            <Droplets className="w-4 h-4 text-cyan-400" />
          </h3>
          <p className="text-sm text-slate-300 mt-1">
            The AI has detected water filling this pothole ({(waterAnalysis.waterProbability * 100).toFixed(0)}% probability, {waterAnalysis.confidenceLevel} confidence). 
            <strong className="text-red-300 ml-1">Depth and severity metrics may be artificially distorted due to water refraction.</strong>
          </p>
          
          {waterAnalysis.inconsistencyScore > 0.6 && (
            <div className="mt-2 text-xs font-mono text-red-400 bg-red-950/40 px-2 py-1 rounded inline-block border border-red-900/50">
              High shape-from-shading inconsistency ({waterAnalysis.inconsistencyScore.toFixed(2)})
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
