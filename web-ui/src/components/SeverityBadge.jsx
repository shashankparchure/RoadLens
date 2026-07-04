import { severityToken, normalizeSeverity } from '../theme/severity';

export default function SeverityBadge({ severity, consensusCount, total }) {
  const token = severityToken(severity);
  const key = normalizeSeverity(severity);
  const isDeep = key === 'Deep';
  const hasConsensus = Number.isFinite(consensusCount) && Number.isFinite(total);

  return (
    <div className="flex flex-col items-center gap-3">
      <div
        className={`
          w-36 h-36 rounded-full flex flex-col items-center justify-center
          border-4 transition-all duration-500
          ${isDeep ? 'hazard-pulse' : ''}
        `}
        style={{
          borderColor: token.color,
          boxShadow: isDeep ? undefined : `0 0 30px ${token.ring}, 0 0 80px ${token.ring}`,
          background: `radial-gradient(circle at center, ${token.color}18, transparent 70%)`,
        }}
      >
        <span
          className="w-3 h-3 rounded-full mb-2"
          style={{ background: token.color, boxShadow: `0 0 12px ${token.color}` }}
        />
        <span className="text-lg font-bold text-white tracking-wide">{severity}</span>
      </div>
      {hasConsensus && (
        <p className="text-sm text-slate-400 font-mono">
          <span className="text-amber-400 font-semibold">{consensusCount}</span>/{total} AGREE
        </p>
      )}
    </div>
  );
}
