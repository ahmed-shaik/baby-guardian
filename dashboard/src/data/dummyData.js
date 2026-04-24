export const STATUS = {
  SAFE: 'safe',
  WARNING: 'warning',
  DANGER: 'danger',
};

export const statusConfig = {
  safe: {
    label: 'Safe',
    bg: 'bg-safe-500/10',
    text: 'text-safe-400',
    border: 'border-safe-500/20',
    dot: 'bg-safe-400',
    glow: 'glow-safe',
  },
  warning: {
    label: 'Warning',
    bg: 'bg-warning-500/10',
    text: 'text-warning-400',
    border: 'border-warning-500/20',
    dot: 'bg-warning-400',
    glow: 'glow-warning',
  },
  danger: {
    label: 'Danger',
    bg: 'bg-danger-500/10',
    text: 'text-danger-400',
    border: 'border-danger-500/20',
    dot: 'bg-danger-400',
    glow: 'glow-danger',
  },
};
