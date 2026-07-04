import { useRef, useState } from 'react';
import {
  motion as Motion,
  useMotionValue,
  useSpring,
  useMotionTemplate,
  useReducedMotion,
} from 'motion/react';
// (useMotionValue drives the glare position; useSpring the tilt)
import { Link2 as Linkedin, Camera as Instagram, GitBranch as Github, RotateCw } from 'lucide-react';
import CornerBrackets from '../instrument/CornerBrackets';
import { panelCascade, panelEnter } from '../../theme/motion';
import atharvaImage from '../images/atharva.jpeg';
import vyankateshImage from '../images/Vyankatesh.jpeg';
import shashankImage from '../images/shashank.jpeg';

const OPERATORS = [
  {
    id: 'OP-01',
    name: 'Atharva Honparkhe',
    avatar: atharvaImage,
    accent: '#22d3ee',
    role: 'Core Model & UI Developer',
    oneLiner: 'Turns noisy road visuals into sharp ML signals, one edge case at a time.',
    record: [
      'Led core severity-model design, feature engineering, and classifier fusion',
      'Built the full UI/UX across detection, insights, and team modules',
      'Integrated inference pipeline outputs with end-to-end interactive visualization',
    ],
    social: {
      linkedin: 'https://www.linkedin.com/in/atharva-honparkhe-6ba05a2aa/',
      instagram: 'https://www.instagram.com/atharva_02_05/',
      github: 'https://github.com/RepoRogue123',
    },
  },
  {
    id: 'OP-02',
    name: 'Vyankatesh Deshpande',
    avatar: vyankateshImage,
    accent: '#f59e0b',
    role: 'YOLOv8 Segmentation & Data Engineer',
    oneLiner: 'Brings calm engineering and fast iteration to every stage of the pipeline.',
    record: [
      'Prepared dataset splits, annotation checks, and preprocessing pipelines',
      'Trained and tuned baseline YOLOv8 segmentation experiments',
      'Managed augmentation and data-quality validation scripts',
    ],
    social: {
      linkedin: 'https://www.linkedin.com/in/vyankatesh-deshpande/',
      instagram: 'https://www.instagram.com/vyankatesh_206/',
      github: 'https://github.com/VyankateshD206',
    },
  },
  {
    id: 'OP-03',
    name: 'Shashank Parchure',
    avatar: shashankImage,
    accent: '#8b5cf6',
    role: 'Data & Evaluation Support',
    oneLiner: 'Keeps the product vision grounded in usability while pushing model quality forward.',
    record: [
      'Assisted YOLOv8 mask review and annotation consistency checks',
      'Built evaluation scripts for train/valid/test metric tracking',
      'Supported feature-validation reports and benchmark comparisons',
    ],
    social: {
      linkedin: 'https://www.linkedin.com/in/shashank-parchure-360bb6a7/',
      instagram: 'https://www.instagram.com/snparchure/',
      github: 'https://github.com/shashankparchure',
    },
  },
];

const SOCIAL_ICONS = { linkedin: Linkedin, instagram: Instagram, github: Github };

function hexToRgba(hex, alpha) {
  const n = Number.parseInt(hex.replace('#', ''), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

/** One holographic operator dossier: mouse-tilt, glare, click-to-flip. */
function DossierCard({ op, index, reducedMotion }) {
  const [flipped, setFlipped] = useState(false);
  const cardRef = useRef(null);

  // Parallax tilt springs (±9°) + cursor-tracking glare
  const rx = useSpring(0, { stiffness: 180, damping: 18 });
  const ry = useSpring(0, { stiffness: 180, damping: 18 });
  const gx = useMotionValue(50);
  const gy = useMotionValue(40);
  const glare = useMotionTemplate`radial-gradient(320px circle at ${gx}% ${gy}%, ${hexToRgba(op.accent, 0.13)}, transparent 65%)`;

  function onMove(e) {
    if (reducedMotion || !cardRef.current) return;
    const r = cardRef.current.getBoundingClientRect();
    const px = (e.clientX - r.left) / r.width;   // 0..1
    const py = (e.clientY - r.top) / r.height;
    ry.set((px - 0.5) * 18);
    rx.set((0.5 - py) * 14);
    gx.set(px * 100);
    gy.set(py * 100);
  }
  function onLeave() {
    ry.set(0);
    rx.set(0);
    gx.set(50);
    gy.set(40);
  }
  function toggle() {
    setFlipped((f) => !f);
  }

  const flipTransition = reducedMotion
    ? { duration: 0 }
    : { type: 'spring', stiffness: 160, damping: 19 };

  return (
    <Motion.div variants={panelEnter} className="w-full max-w-sm mx-auto">
      {/* Float layer (CSS transform only — kept separate from motion transforms) */}
      <div
        className={reducedMotion ? '' : 'card-float'}
        style={{ animationDelay: `${index * 1.7}s` }}
      >
        {/* Tilt layer */}
        <Motion.div
          ref={cardRef}
          style={{ rotateX: rx, rotateY: ry, transformStyle: 'preserve-3d' }}
          onMouseMove={onMove}
          onMouseLeave={onLeave}
          onClick={toggle}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              toggle();
            }
          }}
          role="button"
          tabIndex={0}
          aria-pressed={flipped}
          aria-label={`${op.name} dossier — press to flip`}
          className="cursor-pointer select-none outline-offset-4"
        >
          {/* Flip layer */}
          <Motion.div
            animate={{ rotateY: flipped ? 180 : 0 }}
            transition={flipTransition}
            style={{ transformStyle: 'preserve-3d' }}
            className="relative h-[460px]"
          >
            {/* ---------- FRONT ---------- */}
            <div
              className="absolute inset-0 rounded-[3px] border bg-slate-900/90 overflow-hidden flex flex-col"
              style={{
                backfaceVisibility: 'hidden',
                borderColor: hexToRgba(op.accent, 0.45),
                boxShadow: `0 0 34px ${hexToRgba(op.accent, 0.14)}, inset 0 0 60px rgba(10,12,16,0.6)`,
              }}
            >
              <CornerBrackets color={hexToRgba(op.accent, 0.85)} size={16} inset={8} />
              {/* Glare */}
              <Motion.div className="absolute inset-0 pointer-events-none z-10" style={{ background: glare }} />

              {/* ID strip */}
              <div className="flex items-center justify-between px-4 py-2.5 border-b" style={{ borderColor: hexToRgba(op.accent, 0.25) }}>
                <span className="font-mono text-[10px] tracking-[0.2em]" style={{ color: op.accent }}>
                  {op.id} · OPERATOR
                </span>
                <span className="flex items-center gap-1.5 font-mono text-[9px] tracking-[0.16em] text-slate-500">
                  <span className="w-1.5 h-1.5 rounded-full bg-green-500" style={{ boxShadow: '0 0 6px #22c55e' }} />
                  ACTIVE
                </span>
              </div>

              {/* Avatar port — blurred backdrop + full contained photo (never cropped) */}
              <div className="relative mx-4 mt-4 h-[210px] rounded-[2px] overflow-hidden border border-slate-700 scanline-overlay bg-slate-950">
                <img
                  src={op.avatar}
                  alt=""
                  aria-hidden="true"
                  className="absolute inset-0 w-full h-full object-cover blur-md scale-110 opacity-40"
                />
                <img src={op.avatar} alt={op.name} className="relative w-full h-full object-contain" />
                <div
                  className="absolute inset-0"
                  style={{ background: `linear-gradient(180deg, transparent 55%, ${hexToRgba(op.accent, 0.16)} 100%)` }}
                />
                <CornerBrackets color={hexToRgba(op.accent, 0.7)} size={12} inset={5} />
              </div>

              {/* Identity */}
              <div className="px-5 pt-4 pb-3 flex-1 flex flex-col">
                <h3 className="text-xl font-bold text-white leading-tight">{op.name}</h3>
                <span
                  className="mt-2 self-start px-2 py-0.5 rounded-[2px] border font-mono text-[9px] uppercase tracking-[0.14em]"
                  style={{ color: op.accent, borderColor: hexToRgba(op.accent, 0.4), background: hexToRgba(op.accent, 0.08) }}
                >
                  {op.role}
                </span>
                <p className="mt-3 text-[13px] text-slate-400 leading-relaxed italic">“{op.oneLiner}”</p>
                <p className="mt-auto pt-3 font-mono text-[9px] uppercase tracking-[0.18em] text-slate-600 flex items-center gap-1.5">
                  <RotateCw className="w-3 h-3" /> Click to flip · service record
                </p>
              </div>
            </div>

            {/* ---------- BACK ---------- */}
            <div
              className="absolute inset-0 rounded-[3px] border bg-slate-950/95 overflow-hidden flex flex-col instrument-grid"
              style={{
                backfaceVisibility: 'hidden',
                transform: 'rotateY(180deg)',
                borderColor: hexToRgba(op.accent, 0.45),
                boxShadow: `0 0 34px ${hexToRgba(op.accent, 0.14)}`,
              }}
            >
              <CornerBrackets color={hexToRgba(op.accent, 0.85)} size={16} inset={8} />

              <div className="flex items-center justify-between px-4 py-2.5 border-b" style={{ borderColor: hexToRgba(op.accent, 0.25) }}>
                <span className="font-mono text-[10px] tracking-[0.2em]" style={{ color: op.accent }}>
                  {op.id} · SERVICE RECORD
                </span>
                <span className="font-mono text-[9px] tracking-[0.16em] text-slate-500">CLEARANCE Δ</span>
              </div>

              <div className="px-5 py-4 flex-1 flex flex-col gap-3">
                {op.record.map((line, i) => (
                  <div key={i} className="flex gap-2.5 items-start">
                    <span className="font-mono text-[10px] mt-0.5 shrink-0" style={{ color: op.accent }}>
                      {String(i + 1).padStart(2, '0')}
                    </span>
                    <p className="text-[13px] text-slate-300 leading-snug">{line}</p>
                  </div>
                ))}

                {/* Console link buttons */}
                <div className="mt-auto grid grid-cols-3 gap-2 pt-2">
                  {Object.entries(op.social).map(([key, url]) => {
                    const Icon = SOCIAL_ICONS[key];
                    return (
                      <a
                        key={key}
                        href={url}
                        target="_blank"
                        rel="noopener noreferrer"
                        onClick={(e) => e.stopPropagation()}
                        className="flex items-center justify-center gap-1.5 py-2 rounded-[2px] border border-slate-700 font-mono text-[9px] uppercase tracking-[0.12em] text-slate-400 transition-colors hover:text-white"
                        onMouseEnter={(e) => { e.currentTarget.style.borderColor = op.accent; }}
                        onMouseLeave={(e) => { e.currentTarget.style.borderColor = ''; }}
                      >
                        {Icon && <Icon className="w-3.5 h-3.5" />}
                        {key}
                      </a>
                    );
                  })}
                </div>

                {/* Barcode strip */}
                <div
                  className="h-7 rounded-[1px] opacity-60 mt-2"
                  style={{
                    background:
                      'repeating-linear-gradient(90deg, #cdd3dd 0px, #cdd3dd 1.5px, transparent 1.5px, transparent 4px, #cdd3dd 4px, #cdd3dd 7px, transparent 7px, transparent 9px)',
                  }}
                  aria-hidden="true"
                />
                <p className="font-mono text-[9px] uppercase tracking-[0.18em] text-slate-600 flex items-center gap-1.5">
                  <RotateCw className="w-3 h-3" /> Click to return
                </p>
              </div>
            </div>
          </Motion.div>
        </Motion.div>
      </div>
    </Motion.div>
  );
}

export default function TeamManifest() {
  const reducedMotion = useReducedMotion();

  return (
    <div className="space-y-8">
      {/* Manifest header rail */}
      <div className="text-center space-y-3">
        <div className="flex items-center justify-center gap-3" aria-hidden="true">
          <span className="w-10 h-px bg-slate-700" />
          <span className="font-mono text-[10px] uppercase tracking-[0.3em] text-slate-500">
            Crew Manifest
          </span>
          <span className="w-10 h-px bg-slate-700" />
        </div>
        <h2 className="text-3xl sm:text-4xl font-bold text-white tracking-tight">
          The operators behind <span className="holo-text">RoadLens</span>
        </h2>
        <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-cyan-300/70">
          3 operators · all channels active · click a badge to flip its dossier
        </p>
      </div>

      {/* Perspective scene */}
      <Motion.div
        variants={panelCascade}
        initial="hidden"
        animate="show"
        className="grid grid-cols-1 md:grid-cols-3 gap-8 lg:gap-10 pt-2 pb-10"
        style={{ perspective: '1400px' }}
      >
        {OPERATORS.map((op, i) => (
          <DossierCard key={op.id} op={op} index={i} reducedMotion={reducedMotion} />
        ))}
      </Motion.div>
    </div>
  );
}
