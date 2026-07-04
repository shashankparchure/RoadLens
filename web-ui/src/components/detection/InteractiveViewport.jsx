import { useState } from 'react';
import InstrumentPanel from '../instrument/InstrumentPanel';
import CornerBrackets from '../instrument/CornerBrackets';

const LAYERS = [
  { id: 'original',       ch: 'CH-1', label: 'RGB' },
  { id: 'maskOverlay',    ch: 'CH-2', label: 'SEG' },
  { id: 'depthHeatmap',   ch: 'CH-3', label: 'DEPTH' },
  { id: 'depthAnnotated', ch: 'CH-4', label: 'ANNOT' },
  { id: 'schematic',      ch: 'CH-5', label: 'SCHEM' },
];

export default function InteractiveViewport({ images, potholeCount }) {
  const [activeLayer, setActiveLayer] = useState('original');

  if (!images) return null;

  const current = LAYERS.find((l) => l.id === activeLayer);
  const src = images[activeLayer];

  return (
    <InstrumentPanel
      title="Optical Feed"
      accent="cyan"
      statusLabel={`${current.ch} ACTIVE`}
      flicker={false}
      className="h-full flex flex-col"
      bodyClassName="flex flex-col flex-1"
    >
      {/* Channel tabs */}
      <div className="flex items-center gap-1 px-3 py-2 border-b border-slate-700/70 flex-wrap">
        {LAYERS.map((layer) => {
          const isActive = activeLayer === layer.id;
          return (
            <button
              key={layer.id}
              onClick={() => setActiveLayer(layer.id)}
              className={`px-2.5 py-1.5 rounded-[2px] font-mono text-[10px] tracking-[0.1em] transition-all border ${
                isActive
                  ? 'bg-amber-500/15 text-amber-300 border-amber-500/40'
                  : 'text-slate-500 hover:text-slate-300 border-transparent hover:border-slate-700'
              }`}
            >
              <span className={isActive ? 'text-amber-500' : 'text-slate-600'}>{layer.ch}</span>{' '}
              {layer.label}
            </button>
          );
        })}
      </div>

      {/* Feed */}
      <div className="relative w-full flex-1 min-h-[300px] sm:min-h-[380px] bg-slate-950 flex items-center justify-center p-4 overflow-hidden scanline-overlay">
        {src ? (
          <img
            src={src}
            alt={`${current.label} layer`}
            className="max-w-full max-h-[420px] object-contain transition-opacity duration-300"
          />
        ) : (
          <div className="font-mono text-[11px] uppercase tracking-wider text-slate-500">
            Channel offline
          </div>
        )}

        <CornerBrackets color="rgba(103,232,249,0.5)" size={16} inset={8} />

        {/* Center crosshair — decorative reticle, not a detection marker */}
        <div
          className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none crosshair-pulse"
          aria-hidden="true"
        >
          <svg width="44" height="44" viewBox="0 0 44 44" fill="none">
            <circle cx="22" cy="22" r="14" stroke="rgba(103,232,249,0.35)" strokeWidth="1" />
            <line x1="22" y1="2" x2="22" y2="10" stroke="rgba(103,232,249,0.5)" strokeWidth="1" />
            <line x1="22" y1="34" x2="22" y2="42" stroke="rgba(103,232,249,0.5)" strokeWidth="1" />
            <line x1="2" y1="22" x2="10" y2="22" stroke="rgba(103,232,249,0.5)" strokeWidth="1" />
            <line x1="34" y1="22" x2="42" y2="22" stroke="rgba(103,232,249,0.5)" strokeWidth="1" />
          </svg>
        </div>

        {/* HUD readouts */}
        <span className="absolute top-2.5 left-3 font-mono text-[9px] tracking-[0.18em] text-cyan-300/80 uppercase">
          {current.ch} · {current.label}
        </span>
        {Number.isFinite(potholeCount) && (
          <span className="absolute top-2.5 right-3 font-mono text-[9px] tracking-[0.18em] text-amber-300/90 uppercase">
            {potholeCount} TARGET{potholeCount === 1 ? '' : 'S'}
          </span>
        )}
        <span className="absolute bottom-2.5 right-3 font-mono text-[9px] tracking-[0.18em] text-slate-500 uppercase">
          FEED STABLE
        </span>
      </div>
    </InstrumentPanel>
  );
}
