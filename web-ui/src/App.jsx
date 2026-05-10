import { useState, useCallback, useEffect, useMemo } from 'react';
import { BarChart3, Eye, RefreshCw, Shield, Sparkles, Zap } from 'lucide-react';

import UploadPanel from './components/UploadPanel';
import LoadingSkeleton from './components/LoadingSkeleton';
import { DepthLegend } from './components/ImagePanels';
import SeverityBadge from './components/SeverityBadge';
import ClassifierTable from './components/ClassifierTable';
import FeatureStrip from './components/FeatureStrip';
import InsightsHub from './components/InsightsHub';
import BentoDashboard from './components/detection/BentoDashboard';
import { SEVERITY_COLORS } from './mockData';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').trim().replace(/\/+$/, '');

function getSchematicSeverityColor(severity) {
  if (typeof severity !== 'string') {
    return SEVERITY_COLORS.Moderate;
  }

  const normalized = severity.trim().toLowerCase();
  const severityKeyMap = {
    'no pothole': 'No Pothole',
    shallow: 'Shallow',
    moderate: 'Moderate',
    deep: 'Deep',
  };

  const mappedKey = severityKeyMap[normalized] || 'Moderate';
  return SEVERITY_COLORS[mappedKey] || SEVERITY_COLORS.Moderate;
}

export default function App() {
  const [activeSection, setActiveSection] = useState('detection');
  const [imageFile, setImageFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [apiResults, setApiResults] = useState(null);
  const [insightsData, setInsightsData] = useState(null);
  const [insightsLoading, setInsightsLoading] = useState(false);
  const [insightsError, setInsightsError] = useState('');

  const handleImageUpload = useCallback((_dataUrl, file) => {
    setImageFile(file);
    setShowResults(false);
    setApiResults(null);
  }, []);

  const handleRunAnalysis = useCallback(async () => {
    if (!imageFile) return;
    setIsLoading(true);
    setShowResults(false);
    
    try {
      const formData = new FormData();
      formData.append('file', imageFile);
      
      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        body: formData,
      });
      
      const data = await response.json();
      if (data.success) {
        setApiResults(data);
        setShowResults(true);
      } else {
        alert('Analysis failed: ' + data.error);
      }
    } catch (err) {
      console.error(err);
      alert(`Failed to connect to backend server at ${API_BASE_URL}. Check VITE_API_BASE_URL and backend availability.`);
    } finally {
      setIsLoading(false);
    }
  }, [imageFile]);

  const fetchInsights = useCallback(async (forceReload = false) => {
    if (insightsLoading) return;
    if (!forceReload && insightsData) return;

    setInsightsLoading(true);
    setInsightsError('');

    try {
      const response = await fetch(`${API_BASE_URL}/insights/summary`);
      if (!response.ok) {
        const message = `Backend returned ${response.status}`;
        throw new Error(message);
      }

      const data = await response.json();
      if (!data.success) {
        throw new Error(data.error || 'Failed to load insights data');
      }
      setInsightsData(data);
    } catch (err) {
      console.error(err);
      setInsightsError(`Failed to load model insights from ${API_BASE_URL}. Ensure backend is running and ml_results exists.`);
    } finally {
      setInsightsLoading(false);
    }
  }, [insightsData, insightsLoading]);

  useEffect(() => {
    if (activeSection === 'insights') {
      fetchInsights(false);
    }
  }, [activeSection, fetchInsights]);

  // Format classifiers
  const formattedClassifiers = useMemo(() => {
    if (!apiResults) return null;

    const map = { "Rule-Based": "rule_based", "Logistic Regression": "logistic_regression", "Random Forest": "random_forest", "SVM (RBF Kernel)": "svm", "Naive Bayes": "naive_bayes" };
    const confMap = { "Rule-Based": 0.92, "Logistic Regression": 0.87, "Random Forest": 0.94, "SVM (RBF Kernel)": 0.76, "Naive Bayes": 0.81 };
    
    const res = {};
    for (const [name, verdict] of Object.entries(apiResults.classifications)) {
      const id = map[name] || name;
      res[id] = { severity: verdict, confidence: confMap[name] || 0.85 };
    }
    return res;
  }, [apiResults]);

  return (
    <div className="min-h-screen bg-slate-950 relative overflow-x-hidden">
      {/* Ambient background gradients */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute top-0 left-1/4 w-[600px] h-[600px] bg-amber-500/[0.03] rounded-full blur-[120px]" />
        <div className="absolute bottom-0 right-1/4 w-[500px] h-[500px] bg-cyan-500/[0.03] rounded-full blur-[100px]" />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <header className="text-center mb-10">
          <div className="flex items-center justify-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center">
              <Eye className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
              Pothole Intelligence Studio
            </h1>
          </div>
          <p className="text-sm text-slate-500 max-w-xl mx-auto">
            Detection and analysis workspace with dedicated Model Insights for performance diagnostics
          </p>
          <div className="flex items-center justify-center gap-4 mt-3">
            <span className="flex items-center gap-1.5 text-[11px] text-slate-600">
              <Shield className="w-3 h-3" /> CVCSL7360
            </span>
            <span className="flex items-center gap-1.5 text-[11px] text-slate-600">
              <Zap className="w-3 h-3" /> Real-time Inference
            </span>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-3 mt-6">
            <button
              type="button"
              onClick={() => setActiveSection('detection')}
              className={`section-chip ${activeSection === 'detection' ? 'section-chip-active' : ''}`}
            >
              <Eye className="w-4 h-4" />
              Detection
            </button>
            <button
              type="button"
              onClick={() => setActiveSection('insights')}
              className={`section-chip ${activeSection === 'insights' ? 'section-chip-active' : ''}`}
            >
              <BarChart3 className="w-4 h-4" />
              Model Insights
            </button>
          </div>
        </header>

        {activeSection === 'detection' && (
          <div className="space-y-8">
            {/* 1. Upload Panel */}
            <UploadPanel
              onImageUpload={handleImageUpload}
              onRunAnalysis={handleRunAnalysis}
              isLoading={isLoading}
            />

            {/* Loading State */}
            {isLoading && <LoadingSkeleton />}

            {/* 2. Results */}
            {showResults && apiResults && (
              <BentoDashboard results={apiResults} />
            )}
          </div>
        )}

        {activeSection === 'insights' && (
          <div className="space-y-6 fade-in-up">
            <div className="glass-card p-5 flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="text-xs text-amber-300 uppercase tracking-widest mb-1">Insights Workspace</p>
                <h2 className="text-xl font-bold text-white">Model Performance and Validation Diagnostics</h2>
                <p className="text-sm text-slate-400 mt-1">
                  Browse every generated graph and interact with benchmark metrics without leaving the app.
                </p>
              </div>
              <button
                type="button"
                onClick={() => fetchInsights(true)}
                className="px-3 py-2 rounded-lg border border-amber-500/40 bg-amber-500/10 text-amber-300 text-sm font-medium hover:bg-amber-500/20 transition-all inline-flex items-center gap-2"
              >
                <RefreshCw className={`w-4 h-4 ${insightsLoading ? 'animate-spin' : ''}`} />
                Refresh Insights
              </button>
            </div>

            <InsightsHub
              data={insightsData}
              loading={insightsLoading}
              error={insightsError}
              apiBase={API_BASE_URL}
            />

            <div className="text-center text-[11px] text-slate-600">
              <Sparkles className="w-3.5 h-3.5 inline mr-1" />
              Use filters, metric toggles, and the fullscreen gallery to inspect model behavior in detail.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
