/**
 * Unified severity tokens — the single source of truth.
 * Mirrors the --color-severity-* values in index.css @theme.
 * Replaces the three divergent palettes previously in mockData.js,
 * DepthProfileChart.jsx and InsightsHub.jsx.
 */

export const SEVERITY = {
  'No Pothole': {
    color: '#22c55e',
    bg: 'rgba(34, 197, 94, 0.15)',
    ring: 'rgba(34, 197, 94, 0.4)',
    label: 'NO POTHOLE',
  },
  Shallow: {
    color: '#eab308',
    bg: 'rgba(234, 179, 8, 0.15)',
    ring: 'rgba(234, 179, 8, 0.4)',
    label: 'SHALLOW',
  },
  Moderate: {
    color: '#f97316',
    bg: 'rgba(249, 115, 22, 0.15)',
    ring: 'rgba(249, 115, 22, 0.4)',
    label: 'MODERATE',
  },
  Deep: {
    color: '#ef4444',
    bg: 'rgba(239, 68, 68, 0.15)',
    ring: 'rgba(239, 68, 68, 0.5)',
    label: 'DEEP',
  },
  Unknown: {
    color: '#97a0b1',
    bg: 'rgba(151, 160, 177, 0.15)',
    ring: 'rgba(151, 160, 177, 0.35)',
    label: 'UNKNOWN',
  },
};

const KEY_MAP = {
  'no pothole': 'No Pothole',
  none: 'No Pothole',
  shallow: 'Shallow',
  moderate: 'Moderate',
  deep: 'Deep',
};

/** Normalize any severity string from the API to a canonical SEVERITY key. */
export function normalizeSeverity(severity) {
  if (typeof severity !== 'string') return 'Unknown';
  return KEY_MAP[severity.trim().toLowerCase()] || 'Unknown';
}

/** Convenience: token entry for any severity string (never undefined). */
export function severityToken(severity) {
  return SEVERITY[normalizeSeverity(severity)];
}
