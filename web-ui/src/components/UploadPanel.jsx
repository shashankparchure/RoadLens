import { useState, useRef, useCallback } from 'react';
import { Crosshair, ScanLine } from 'lucide-react';
import InstrumentPanel from './instrument/InstrumentPanel';
import CornerBrackets from './instrument/CornerBrackets';

/** Real pipeline stages — fixed status chips, not toggles. */
const PIPELINE_MODULES = ['YOLOV8-SEG', 'DEPTH-V2', 'SFS', 'DINOV2', 'ENSEMBLE'];

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return '—';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export default function UploadPanel({ onImageUpload, onRunAnalysis, isLoading }) {
  const [preview, setPreview] = useState(null);
  const [meta, setMeta] = useState(null); // { name, size, w, h }
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target.result;
      setPreview(dataUrl);
      const img = new Image();
      img.onload = () => setMeta({ name: file.name, size: file.size, w: img.naturalWidth, h: img.naturalHeight });
      img.src = dataUrl;
      onImageUpload(dataUrl, file);
    };
    reader.readAsDataURL(file);
  }, [onImageUpload]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => setIsDragging(false), []);

  const loadDemo = useCallback(async () => {
    const res = await fetch('/sample_pothole.png');
    const blob = await res.blob();
    const file = new File([blob], 'sample_pothole.png', { type: blob.type });
    handleFile(file);
  }, [handleFile]);

  return (
    <InstrumentPanel
      title="Specimen Intake"
      statusLabel={preview ? 'FRAME STAGED' : 'AWAITING FRAME'}
      accent="amber"
      bodyClassName="p-4 sm:p-5"
      className="max-w-5xl mx-auto"
    >
      {/* Scanner tray */}
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => fileInputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') fileInputRef.current?.click(); }}
        aria-label="Upload a road-surface image"
        className={`
          relative cursor-pointer instrument-grid rounded-[2px]
          border transition-all duration-300 ease-out
          flex flex-col items-center justify-center
          min-h-[260px] p-8 bg-slate-950/60
          ${isDragging ? 'border-amber-500 bg-amber-500/[0.05]' : 'border-slate-700 hover:border-amber-500/50'}
        `}
      >
        <CornerBrackets
          color={isDragging ? 'rgba(245,158,11,1)' : 'rgba(245,158,11,0.45)'}
          size={18}
          inset={8}
        />

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => handleFile(e.target.files[0])}
        />

        {preview ? (
          <div className="flex flex-col sm:flex-row items-center gap-6 w-full justify-center">
            <div className="relative">
              <img
                src={preview}
                alt="Staged road frame"
                className="max-h-[190px] rounded-[2px] object-contain border border-slate-700"
              />
              <CornerBrackets color="rgba(103,232,249,0.8)" size={12} inset={-4} />
            </div>
            <div className="space-y-2.5 font-mono text-[11px] tracking-wider text-left">
              <p className="text-slate-500 uppercase text-[10px]">Frame metadata</p>
              <p className="text-slate-300 max-w-[240px] truncate">{meta?.name || 'frame.jpg'}</p>
              <p className="text-cyan-300">{meta ? `${meta.w} × ${meta.h} PX` : '— × — PX'}</p>
              <p className="text-slate-400">{formatBytes(meta?.size)}</p>
              <p className="text-slate-600 text-[10px] uppercase">Click or drop to replace</p>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-4 text-center">
            <div className="relative w-16 h-16 flex items-center justify-center">
              <Crosshair className="w-9 h-9 text-amber-500 crosshair-pulse" strokeWidth={1.4} />
            </div>
            <div>
              <p className="font-mono text-sm tracking-[0.18em] text-slate-300 uppercase">
                Drop road-surface frame
              </p>
              <p className="font-mono text-[10px] tracking-[0.14em] text-slate-500 uppercase mt-2">
                or click to browse · PNG / JPG ≤ 20 MB
              </p>
              <button
                onClick={(e) => { e.stopPropagation(); loadDemo(); }}
                className="mt-3 font-mono text-[11px] tracking-[0.12em] uppercase text-amber-500 hover:text-amber-300 underline underline-offset-4 transition-colors"
              >
                Load sample frame →
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Pipeline modules — honest status row */}
      <div className="mt-4 flex flex-wrap items-center justify-center gap-x-5 gap-y-2">
        <span className="font-mono text-[9px] uppercase tracking-[0.2em] text-slate-600">
          Pipeline modules
        </span>
        {PIPELINE_MODULES.map((mod) => (
          <span key={mod} className="flex items-center gap-1.5 font-mono text-[10px] tracking-[0.12em] text-slate-400">
            <span className="w-1 h-1 rounded-full bg-cyan-400" style={{ boxShadow: '0 0 5px #22d3ee' }} />
            {mod}
            <span className="text-slate-600">ARMED</span>
          </span>
        ))}
      </div>

      {/* Initiate scan */}
      <button
        onClick={onRunAnalysis}
        disabled={!preview || isLoading}
        className={`
          mt-4 w-full py-3.5 rounded-[2px] font-mono text-sm tracking-[0.22em] uppercase
          transition-all duration-300 flex items-center justify-center gap-3 relative overflow-hidden
          ${preview && !isLoading
            ? 'bg-amber-500 text-slate-950 font-semibold hover:bg-amber-400 cursor-pointer'
            : 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
          }
        `}
      >
        {preview && !isLoading && (
          <span className="absolute left-0 top-0 bottom-0 w-2 hazard-stripes" aria-hidden="true" />
        )}
        {isLoading ? (
          <>
            <ScanLine className="w-4 h-4 animate-pulse" />
            Scan in progress
          </>
        ) : (
          <>
            <ScanLine className="w-4 h-4" />
            Initiate scan
          </>
        )}
      </button>
    </InstrumentPanel>
  );
}
