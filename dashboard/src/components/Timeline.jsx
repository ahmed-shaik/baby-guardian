import { motion } from 'framer-motion';
import Card from './Card';

const typeConfig = {
  safe:    { dot: 'bg-safe-400',    ring: 'ring-safe-500/20',    text: 'text-safe-400',    glow: '#22c55e' },
  warning: { dot: 'bg-warning-400', ring: 'ring-warning-500/20', text: 'text-warning-400', glow: '#f59e0b' },
  danger:  { dot: 'bg-danger-400',  ring: 'ring-danger-500/20',  text: 'text-danger-400',  glow: '#ef4444' },
  info:    { dot: 'bg-primary-400', ring: 'ring-primary-500/20', text: 'text-primary-400', glow: '#3b82f6' },
};

export default function Timeline({ events = [] }) {
  return (
    <Card delay={0.3}>
      <div className="p-5">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-4 rounded-full bg-gradient-to-b from-primary-400 to-violet-500" />
          <h3 className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Activity Timeline</h3>
        </div>
        {events.length === 0 ? (
          <div className="py-6 text-center">
            <p className="text-xs text-slate-500 font-medium">No activity yet</p>
            <p className="text-[10px] text-slate-600 mt-1">Events will appear here</p>
          </div>
        ) : (
          <div className="relative pl-6">
            {/* Animated line */}
            <motion.div initial={{ height: 0 }} animate={{ height: '100%' }}
              transition={{ duration: 1, delay: 0.3, ease: [0.25, 0.46, 0.45, 0.94] }}
              className="absolute left-[5px] top-2 bottom-2 w-px bg-gradient-to-b from-[var(--color-border)] via-[var(--color-border)] to-transparent" />

            <div className="space-y-0.5">
              {events.map((event, i) => {
                const config = typeConfig[event.type] || typeConfig.info;
                return (
                  <motion.div key={event.id} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.35, delay: 0.2 + i * 0.06, ease: [0.25, 0.46, 0.45, 0.94] }}
                    className="relative flex items-start gap-4 py-2.5 group">
                    {/* Dot with glow */}
                    <div className="absolute -left-6 mt-[5px]">
                      <div className={`w-[10px] h-[10px] rounded-full ${config.dot} ring-[3px] ${config.ring}`}
                        style={{ boxShadow: `0 0 8px ${config.glow}25` }} />
                    </div>

                    <div className="flex-1 min-w-0">
                      <p className={`text-[13px] font-semibold leading-snug ${config.text}`}>{event.event}</p>
                      <p className="text-[10px] text-slate-600 mt-0.5 font-medium tabular-nums">{event.time}</p>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}
