/**
 * Shared motion/react variants for the instrument UI.
 * Continuous ambient animation lives in CSS keyframes (index.css);
 * these variants handle orchestrated entrances and section swaps.
 */

/** Container: staggers panel entrances (post target-lock cascade). */
export const panelCascade = {
  hidden: {},
  show: {
    transition: { staggerChildren: 0.08, delayChildren: 0.05 },
  },
};

/** Child panel entrance: rise + unblur, like an instrument channel coming online. */
export const panelEnter = {
  hidden: { opacity: 0, y: 24, filter: 'blur(4px)' },
  show: {
    opacity: 1,
    y: 0,
    filter: 'blur(0px)',
    transition: { duration: 0.45, ease: [0.22, 1, 0.36, 1] },
  },
};

/** Section swap (Detection / Insights / Team) under AnimatePresence mode="wait". */
export const sectionSwap = {
  initial: { opacity: 0, y: 14 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.3, ease: 'easeOut' } },
  exit: { opacity: 0, y: -10, transition: { duration: 0.18, ease: 'easeIn' } },
};
