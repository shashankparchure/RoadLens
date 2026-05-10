import { useState } from 'react';
import { Layers, Map, Eye, Grid, Crosshair } from 'lucide-react';

export default function InteractiveViewport({ images }) {
  const [activeLayer, setActiveLayer] = useState('original');
  
  if (!images) return null;

  const layers = [
    { id: 'original', label: 'RGB Image', icon: Eye, src: images.original },
    { id: 'maskOverlay', label: 'YOLO Mask', icon: Layers, src: images.maskOverlay },
    { id: 'depthHeatmap', label: 'Depth Heatmap', icon: Map, src: images.depthHeatmap },
    { id: 'depthAnnotated', label: 'Depth Analysis', icon: Crosshair, src: images.depthAnnotated },
    { id: 'schematic', label: 'Schematic', icon: Grid, src: images.schematic },
  ];

  const currentLayer = layers.find(l => l.id === activeLayer);

  return (
    <div className="glass-card-bright overflow-hidden flex flex-col fade-in-up border border-white/10 h-[300px] sm:h-[400px] md:h-[450px]">
      
      {/* Viewport Toolbar */}
      <div className="flex items-center justify-between p-3 border-b border-white/5 bg-slate-900/50 flex-wrap gap-2">
        <div className="flex items-center gap-2">
          {layers.map(layer => {
            const Icon = layer.icon;
            const isActive = activeLayer === layer.id;
            return (
              <button
                key={layer.id}
                onClick={() => setActiveLayer(layer.id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  isActive 
                    ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30' 
                    : 'text-slate-400 hover:text-white hover:bg-white/5 border border-transparent'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                {layer.label}
              </button>
            )
          })}
        </div>
      </div>

      {/* Image Display */}
      <div className="relative w-full h-full bg-black/60 flex items-center justify-center p-4 overflow-hidden">
        {currentLayer?.src ? (
          <img 
            src={currentLayer.src} 
            alt={activeLayer}
            className="max-w-full max-h-full object-contain rounded drop-shadow-2xl transition-opacity duration-300"
          />
        ) : (
          <div className="text-slate-500 text-sm">Image layer not available</div>
        )}
      </div>
    </div>
  );
}
