import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, ShieldAlert, X, Volume2 } from 'lucide-react';
import { useState } from 'react';

export default function AlertBanner({ alert, onDismiss }) {
  const [visible, setVisible] = useState(true);
  if (!alert || !visible) return null;
  const isDanger = alert.severity === 'danger';

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          initial={{ opacity: 0, y: -16, scale: 0.98 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -12, scale: 0.98 }}
          transition={{ duration: 0.35, ease: [0.25, 0.46, 0.45, 0.94] }}
          className={`
            relative overflow-hidden rounded-2xl border px-5 py-4 flex items-start gap-3.5
            ${isDanger
              ? 'bg-danger-500/8 border-danger-500/20 text-danger-400 glow-ambient-danger'
              : 'bg-warning-500/8 border-warning-500/20 text-warning-400 glow-ambient-warning'
            }
          `}
        >
          {/* Animated background pulse */}
          <motion.div
            className={`absolute inset-0 ${isDanger ? 'bg-danger-500/5' : 'bg-warning-500/5'}`}
            animate={{ opacity: [0, 0.5, 0] }}
            transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
          />

          {/* Left accent bar */}
          <div className={`absolute left-0 top-0 bottom-0 w-1 ${isDanger ? 'bg-danger-500' : 'bg-warning-500'}`} />

          <motion.div initial={{ scale: 0.5, rotate: -15 }} animate={{ scale: 1, rotate: 0 }} transition={{ duration: 0.35, delay: 0.1 }} className="relative mt-0.5 shrink-0">
            {isDanger ? <ShieldAlert size={20} /> : <AlertTriangle size={20} />}
          </motion.div>
          <div className="relative flex-1 min-w-0">
            <p className="font-bold text-sm leading-snug">{alert.message}</p>
            {alert.detail && (
              <motion.p initial={{ opacity: 0 }} animate={{ opacity: 0.7 }} transition={{ delay: 0.15 }}
                className="text-xs mt-1 leading-relaxed">{alert.detail}</motion.p>
            )}
          </div>
          <div className="relative flex items-center gap-1">
            <button onClick={() => { setVisible(false); onDismiss?.(); }}
              className="p-1.5 rounded-lg hover:bg-white/10 active:scale-90 transition-all duration-150 cursor-pointer" aria-label="Dismiss alert">
              <X size={15} />
            </button>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
