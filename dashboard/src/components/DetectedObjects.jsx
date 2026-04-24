import { motion } from 'framer-motion';
import { Box, User, BedDouble, Baby, Milk, AlertTriangle, ShieldAlert, Shield } from 'lucide-react';
import Card from './Card';

// Per-class icons
const classIcons = {
  person: User,
  'teddy bear': Baby,
  bed: BedDouble,
  bottle: Milk,
};

// Risk level → visual style
const riskStyles = {
  danger: {
    bg: 'bg-danger-500/15',
    text: 'text-danger-400',
    border: 'border-danger-500/30',
    badge: 'bg-danger-500/20 text-danger-400 border-danger-500/30',
    badgeLabel: 'DANGER',
    badgeIcon: ShieldAlert,
    rowHover: 'hover:bg-danger-500/5',
  },
  hazard: {
    bg: 'bg-warning-500/15',
    text: 'text-warning-400',
    border: 'border-warning-500/30',
    badge: 'bg-warning-500/20 text-warning-400 border-warning-500/30',
    badgeLabel: 'HAZARD',
    badgeIcon: AlertTriangle,
    rowHover: 'hover:bg-warning-500/5',
  },
  safe: {
    bg: 'bg-white/[0.04]',
    text: 'text-slate-400',
    border: 'border-white/[0.06]',
    badge: 'bg-white/[0.06] text-slate-400 border-white/[0.08]',
    badgeLabel: 'SAFE',
    badgeIcon: Shield,
    rowHover: 'hover:bg-white/[0.03]',
  },
};

function ConfidenceBar({ value }) {
  const barColor = value >= 0.75 ? 'from-safe-400 to-safe-500'
    : value >= 0.5 ? 'from-warning-400 to-warning-500'
    : 'from-danger-400 to-danger-500';
  return (
    <div className="w-16 h-1.5 bg-white/[0.04] rounded-full overflow-hidden">
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${value * 100}%` }}
        transition={{ duration: 0.8, delay: 0.2, ease: [0.25, 0.46, 0.45, 0.94] }}
        className={`h-full rounded-full bg-gradient-to-r ${barColor}`}
      />
    </div>
  );
}

export default function DetectedObjects({ detections = [] }) {
  // Sort: danger first, then hazard, then safe
  const sorted = [...detections].sort((a, b) => {
    const order = { danger: 0, hazard: 1, safe: 2 };
    return (order[a.riskLevel] ?? 2) - (order[b.riskLevel] ?? 2);
  });

  const dangerCount = detections.filter(d => d.riskLevel === 'danger').length;
  const hazardCount = detections.filter(d => d.riskLevel === 'hazard').length;

  return (
    <Card delay={0.15}>
      <div className="p-5">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <div className="w-1 h-4 rounded-full bg-gradient-to-b from-primary-400 to-primary-600" />
            <h3 className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Detected Objects</h3>
          </div>
          <div className="flex items-center gap-1.5">
            {dangerCount > 0 && (
              <span className="text-[10px] bg-danger-500/20 text-danger-400 font-bold px-2 py-0.5 rounded-full border border-danger-500/30">
                {dangerCount} danger
              </span>
            )}
            {hazardCount > 0 && (
              <span className="text-[10px] bg-warning-500/20 text-warning-400 font-bold px-2 py-0.5 rounded-full border border-warning-500/30">
                {hazardCount} hazard
              </span>
            )}
            <span className="text-[10px] bg-white/[0.06] text-slate-400 font-bold px-2 py-0.5 rounded-full tabular-nums border border-white/[0.04]">
              {detections.length}
            </span>
          </div>
        </div>

        {detections.length === 0 ? (
          <div className="py-8 text-center">
            <div className="w-12 h-12 rounded-2xl bg-white/[0.03] flex items-center justify-center mx-auto mb-3 border border-white/[0.04]">
              <Box size={20} className="text-slate-600" />
            </div>
            <p className="text-xs text-slate-500 font-medium">No objects detected</p>
            <p className="text-[10px] text-slate-600 mt-1">Waiting for analysis...</p>
          </div>
        ) : (
          <div className="space-y-1">
            {sorted.map((det, i) => {
              const Icon = classIcons[det.className] || Box;
              const style = riskStyles[det.riskLevel] || riskStyles.safe;
              const BadgeIcon = style.badgeIcon;
              return (
                <motion.div
                  key={`${det.id}-${det.className}`}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3, delay: 0.05 + i * 0.05 }}
                  className={`flex items-center justify-between py-2.5 px-3 rounded-xl ${style.rowHover} transition-all duration-200 cursor-default group`}
                >
                  {/* Icon + name */}
                  <div className="flex items-center gap-2.5">
                    <div className={`w-8 h-8 rounded-lg ${style.bg} flex items-center justify-center border ${style.border} group-hover:scale-105 transition-transform duration-200`}>
                      <Icon size={14} className={style.text} strokeWidth={2} />
                    </div>
                    <div>
                      <p className="text-sm font-semibold text-slate-200 capitalize leading-tight">{det.className}</p>
                      <p className="text-[10px] text-slate-600 font-medium">Track #{det.trackId ?? '—'}</p>
                    </div>
                  </div>

                  {/* Right side: risk badge + confidence */}
                  <div className="flex items-center gap-2">
                    {/* Risk badge */}
                    <span className={`flex items-center gap-1 text-[9px] font-bold px-1.5 py-0.5 rounded-md border ${style.badge}`}>
                      <BadgeIcon size={9} strokeWidth={2.5} />
                      {style.badgeLabel}
                    </span>

                    {/* Confidence */}
                    <div className="flex flex-col items-end gap-1">
                      <ConfidenceBar value={det.confidence} />
                      <span className="text-[10px] font-mono font-bold text-slate-500 tabular-nums">
                        {(det.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </div>
        )}
      </div>
    </Card>
  );
}
