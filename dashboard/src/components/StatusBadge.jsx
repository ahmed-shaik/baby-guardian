import { motion, AnimatePresence } from 'framer-motion';
import { Shield, ShieldAlert, AlertTriangle } from 'lucide-react';
import { statusConfig } from '../data/dummyData';

const icons = {
  safe: Shield,
  warning: AlertTriangle,
  danger: ShieldAlert,
};

export default function StatusBadge({ status = 'safe', size = 'lg' }) {
  const config = statusConfig[status];
  const Icon = icons[status];

  const sizeClasses = {
    sm: 'px-2.5 py-1.5 text-xs gap-1.5',
    md: 'px-3.5 py-2 text-sm gap-2',
    lg: 'px-5 py-2.5 text-base gap-2',
  };
  const iconSize = { sm: 13, md: 16, lg: 20 };

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={status}
        initial={{ opacity: 0, scale: 0.92, filter: 'blur(4px)' }}
        animate={{ opacity: 1, scale: 1, filter: 'blur(0px)' }}
        exit={{ opacity: 0, scale: 0.92, filter: 'blur(4px)' }}
        transition={{ duration: 0.3, ease: [0.25, 0.46, 0.45, 0.94] }}
        className={`
          inline-flex items-center font-semibold rounded-xl
          ${config.bg} ${config.text} ${config.border} border
          ${config.glow} ${sizeClasses[size]}
          backdrop-blur-md
        `}
      >
        <motion.div initial={{ rotate: -10 }} animate={{ rotate: 0 }} transition={{ duration: 0.3 }}>
          <Icon size={iconSize[size]} />
        </motion.div>
        <span className="font-bold tracking-tight">{config.label}</span>
        <span className="relative flex h-2 w-2 ml-0.5">
          <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${config.dot} opacity-30`} />
          <span className={`relative inline-flex rounded-full h-2 w-2 ${config.dot}`} />
        </span>
      </motion.div>
    </AnimatePresence>
  );
}
