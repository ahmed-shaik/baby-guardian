import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ShieldAlert, AlertTriangle, ShieldCheck, Clock, Filter } from 'lucide-react';
import Card from './Card';

const severityConfig = {
  danger: { icon: ShieldAlert, bg: 'bg-danger-500/8', text: 'text-danger-400', border: 'border-danger-500/10', dot: 'bg-danger-400' },
  warning: { icon: AlertTriangle, bg: 'bg-warning-500/8', text: 'text-warning-400', border: 'border-warning-500/10', dot: 'bg-warning-400' },
  safe: { icon: ShieldCheck, bg: 'bg-safe-500/8', text: 'text-safe-400', border: 'border-safe-500/10', dot: 'bg-safe-400' },
};

const filters = [
  { key: 'all', label: 'All' },
  { key: 'danger', label: 'Danger' },
  { key: 'warning', label: 'Warning' },
  { key: 'safe', label: 'Safe' },
];

export default function AlertHistory({ alerts = [] }) {
  const [activeFilter, setActiveFilter] = useState('all');

  const filtered = activeFilter === 'all' ? alerts : alerts.filter(a => a.severity === activeFilter);

  return (
    <Card delay={0.2}>
      <div className="p-5">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className="w-1 h-4 rounded-full bg-gradient-to-b from-warning-400 to-danger-500" />
            <h3 className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Alert History</h3>
          </div>
          <span className="text-[10px] bg-white/[0.06] text-slate-400 font-bold px-2 py-0.5 rounded-full tabular-nums border border-white/[0.04]">
            {alerts.length}
          </span>
        </div>

        {/* Filter tabs */}
        <div className="flex items-center gap-1 mb-3 p-0.5 bg-white/[0.02] rounded-lg border border-white/[0.03]">
          {filters.map((f) => (
            <button
              key={f.key}
              onClick={() => setActiveFilter(f.key)}
              className={`
                relative flex-1 text-[10px] font-semibold py-1.5 rounded-md cursor-pointer
                transition-all duration-200
                ${activeFilter === f.key
                  ? 'text-white'
                  : 'text-slate-500 hover:text-slate-400'
                }
              `}
            >
              {activeFilter === f.key && (
                <motion.div
                  layoutId="alert-filter"
                  className="absolute inset-0 bg-white/[0.06] rounded-md border border-white/[0.06]"
                  transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                />
              )}
              <span className="relative">{f.label}</span>
            </button>
          ))}
        </div>

        {filtered.length === 0 ? (
          <div className="py-6 text-center">
            <div className="w-10 h-10 rounded-xl bg-white/[0.03] flex items-center justify-center mx-auto mb-2 border border-white/[0.04]">
              <ShieldCheck size={18} className="text-slate-600" />
            </div>
            <p className="text-xs text-slate-500 font-medium">
              {alerts.length === 0 ? 'No alerts yet' : 'No matching alerts'}
            </p>
          </div>
        ) : (
          <div className="space-y-1 max-h-[300px] overflow-y-auto pr-0.5">
            <AnimatePresence initial={false}>
              {filtered.map((alert, i) => {
                const config = severityConfig[alert.severity] || severityConfig.safe;
                const Icon = config.icon;
                return (
                  <motion.div key={alert.id} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, x: -16 }}
                    transition={{ duration: 0.3, delay: i * 0.03, ease: [0.25, 0.46, 0.45, 0.94] }}
                    className="flex items-start gap-3 p-3 rounded-xl hover:bg-white/[0.02] transition-colors duration-200 cursor-default group"
                  >
                    <div className={`w-7 h-7 rounded-lg ${config.bg} flex items-center justify-center shrink-0 border ${config.border}`}>
                      <Icon size={13} className={config.text} strokeWidth={2} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-[13px] font-semibold text-slate-200 leading-snug">{alert.message}</p>
                      {alert.detail && <p className="text-[11px] text-slate-500 mt-0.5 truncate leading-relaxed">{alert.detail}</p>}
                    </div>
                    <div className="flex items-center gap-1 text-[10px] text-slate-600 shrink-0 mt-1 group-hover:text-slate-400 transition-colors">
                      <Clock size={9} /><span className="tabular-nums">{alert.time}</span>
                    </div>
                  </motion.div>
                );
              })}
            </AnimatePresence>
          </div>
        )}
      </div>
    </Card>
  );
}
