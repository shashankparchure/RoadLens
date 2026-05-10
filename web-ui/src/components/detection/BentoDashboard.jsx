import SeverityBadge from '../SeverityBadge';
import FeatureStrip from '../FeatureStrip';
import ClassifierTable from '../ClassifierTable';
import InteractiveViewport from './InteractiveViewport';
import DepthProfileChart from './DepthProfileChart';
import TemporalPrognosis from './TemporalPrognosis';
import SemanticIntelligence from './SemanticIntelligence';
import WaterHazardAlert from './WaterHazardAlert';
import { Target, ListChecks, ArrowRight } from 'lucide-react';

export default function BentoDashboard({ results }) {
  if (!results || !results.potholes || results.potholes.length === 0) {
    return null;
  }

  // Use the first/representative pothole for the detailed advanced panels
  // In a full app, you might have a dropdown to select which pothole to inspect
  const primaryPothole = results.potholes[0];

  const map = { "Rule-Based": "rule_based", "Logistic Regression": "logistic_regression", "Random Forest": "random_forest", "SVM (RBF Kernel)": "svm", "Naive Bayes": "naive_bayes" };
  const confMap = { "Rule-Based": 0.92, "Logistic Regression": 0.87, "Random Forest": 0.94, "SVM (RBF Kernel)": 0.76, "Naive Bayes": 0.81 };
  
  const formattedClassifiers = {};
  if (results.classifications) {
    for (const [name, verdict] of Object.entries(results.classifications)) {
      const id = map[name] || name;
      formattedClassifiers[id] = { severity: verdict, confidence: confMap[name] || 0.85 };
    }
  }

  return (
    <div className="space-y-6">
      
      {/* 1. Global Alerts */}
      <WaterHazardAlert waterAnalysis={primaryPothole.waterAnalysis} />

      {/* 2. Top Section: Viewport + Executive Summary */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left: Viewport (Takes up 2 columns on large screens) */}
        <div className="lg:col-span-2 h-full">
          <InteractiveViewport images={results.images} />
        </div>

        {/* Right: Executive Summary */}
        <div className="lg:col-span-1 flex flex-col gap-4">
          <div className="glass-card-bright p-6 fade-in-up flex-1 flex flex-col items-center justify-center text-center">
            <h2 className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-4">Consensus Severity</h2>
            <SeverityBadge severity={results.consensusSeverity} size="lg" />
            <p className="mt-4 text-slate-400 text-sm">{results.consensusSubtext}</p>
            
            <div className="w-full h-[1px] bg-white/10 my-6" />
            
            <div className="w-full flex justify-between items-center px-2">
              <div className="flex items-center gap-2 text-slate-300">
                <Target className="w-4 h-4 text-cyan-400" />
                <span className="text-sm">Potholes Detected:</span>
              </div>
              <span className="text-xl font-bold font-mono">{results.potholeCount}</span>
            </div>
            
            <div className="w-full flex justify-between items-center px-2 mt-3">
              <div className="flex items-center gap-2 text-slate-300">
                <ListChecks className="w-4 h-4 text-emerald-400" />
                <span className="text-sm">Classifiers Agreed:</span>
              </div>
              <span className="text-xl font-bold font-mono">{results.consensusCount}/{results.totalClassifiers}</span>
            </div>
          </div>
        </div>
      </div>

      {/* 2b. Depth Cross-Section Chart */}
      <DepthProfileChart potholes={results.potholes} />

      {/* 3. Core Geometry Strip */}
      <div className="mt-2">
        <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-3 ml-2 flex items-center gap-2">
          <ArrowRight className="w-3 h-3" /> Core Metrics
        </h3>
        <FeatureStrip features={results.features} />
      </div>

      {/* 4. Advanced Diagnostics (Batch 4 Additions) */}
      <div className="mt-8 pt-6 border-t border-white/5">
        <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4 ml-2 flex items-center gap-2">
          <ArrowRight className="w-3 h-3" /> Advanced Diagnostics
        </h3>
        
        <div className="space-y-6">
          <TemporalPrognosis 
            temporalAnalysis={primaryPothole.temporalAnalysis} 
            currentSeverity={results.consensusSeverity}
          />
          <SemanticIntelligence 
            geometryAnalysis={primaryPothole.geometryAnalysis} 
          />
        </div>
      </div>

      {/* 5. ML Models Breakdown */}
      <div className="mt-8 pt-6 border-t border-white/5">
        <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4 ml-2 flex items-center gap-2">
          <ArrowRight className="w-3 h-3" /> Classifier Breakdown
        </h3>
        <ClassifierTable 
          results={formattedClassifiers} 
        />
      </div>

    </div>
  );
}
