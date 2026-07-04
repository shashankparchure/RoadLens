import { useEffect, useState } from 'react';
import { motion as Motion } from 'motion/react';
import { Radar } from 'lucide-react';

const TABS = [
  { id: 'detection', label: 'DETECTION' },
  { id: 'insights', label: 'INSIGHTS' },
  { id: 'team', label: 'TEAM' },
];

function useClock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);
  return now.toLocaleTimeString('en-GB', { hour12: false });
}

/**
 * Instrument top-bar: wordmark, section tabs with a sliding amber
 * indicator, and live telemetry chips. Sticky; hazard-stripe base edge.
 */
export default function TelemetryBar({ activeSection, onNavigate, apiOnline }) {
  const clock = useClock();

  const apiDot =
    apiOnline === null ? '#626b7d' : apiOnline ? '#22c55e' : '#ef4444';
  const apiText =
    apiOnline === null ? 'API :8000' : apiOnline ? 'API LINKED' : 'API DOWN';

  return (
    <div className="sticky top-0 z-40">
      <div className="bg-slate-950/85 backdrop-blur-sm border-b border-slate-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between gap-4 h-14">
            {/* Wordmark */}
            <div className="flex items-center gap-3 min-w-0">
              <div className="w-8 h-8 bg-amber-500 rounded-[2px] flex items-center justify-center shrink-0">
                <Radar className="w-4.5 h-4.5 text-slate-950" strokeWidth={2.2} />
              </div>
              <div className="min-w-0 leading-tight">
                <p className="font-bold text-white tracking-[0.22em] text-sm">ROADLENS</p>
                <p className="font-mono text-[9px] uppercase tracking-[0.18em] text-slate-500 truncate">
                  Road Survey Instrument · CVCSL7360
                </p>
              </div>
            </div>

            {/* Section tabs */}
            <nav className="flex items-center gap-1" aria-label="Sections">
              {TABS.map((tab) => {
                const active = activeSection === tab.id;
                return (
                  <button
                    key={tab.id}
                    type="button"
                    onClick={() => onNavigate(tab.id)}
                    className={`relative px-3 sm:px-4 py-2 font-mono text-[11px] tracking-[0.14em] transition-colors duration-150 ${
                      active ? 'text-amber-300' : 'text-slate-500 hover:text-slate-300'
                    }`}
                    aria-current={active ? 'page' : undefined}
                  >
                    <span className="relative z-10">{tab.label}</span>
                    {active && (
                      <Motion.span
                        layoutId="tab-indicator"
                        className="absolute inset-x-1 -bottom-[1px] h-[2px] bg-amber-500"
                        style={{ boxShadow: '0 0 10px rgba(245,158,11,0.7)' }}
                        transition={{ type: 'spring', stiffness: 500, damping: 40 }}
                      />
                    )}
                  </button>
                );
              })}
            </nav>

            {/* Telemetry chips */}
            <div className="hidden md:flex items-center gap-4 font-mono text-[10px] tracking-[0.12em] text-slate-500">
              <span className="flex items-center gap-1.5">
                <span
                  className="w-1.5 h-1.5 rounded-full"
                  style={{ background: apiDot, boxShadow: `0 0 6px ${apiDot}` }}
                />
                {apiText}
              </span>
              <span className="text-slate-600">|</span>
              <span className="text-cyan-300/80 tabular-nums">{clock}</span>
            </div>
          </div>
        </div>
      </div>
      {/* Hazard base edge */}
      <div className="h-[3px] hazard-stripes opacity-60" aria-hidden="true" />
    </div>
  );
}
