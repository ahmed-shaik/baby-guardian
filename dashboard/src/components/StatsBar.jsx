import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Clock, Gauge, Bell, ShieldAlert, TrendingUp, TrendingDown, Minus } from 'lucide-react';

function AnimatedNumber({ value, duration = 800 }) {
  const [display, setDisplay] = useState(0);
  const numericValue = typeof value === 'number' ? value : parseFloat(value);
  useEffect(() => {
    if (isNaN(numericValue)) { setDisplay(value); return; }
    const start = Date.now();
    const step = () => {
      const progress = Math.min((Date.now() - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(Math.round(numericValue * eased * 10) / 10);
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, [numericValue, duration]);
  if (typeof value === 'string') return value;
  return !Number.isInteger(numericValue) ? display.toFixed(1) : Math.round(display);
}

const stats = [
  {
    key: 'uptime',
    label: 'Uptime',
    icon: Clock,
    gradient: 'from-primary-500/15 to-primary-600/5',
    iconBg: 'bg-primary-500/10',
    iconColor: 'text-primary-400',
    glowColor: 'shadow-primary-500/5',
  },
  {
    key: 'fps',
    label: 'Frame Rate',
    icon: Gauge,
    gradient: 'from-safe-500/15 to-safe-600/5',
    iconBg: 'bg-safe-500/10',
    iconColor: 'text-safe-400',
    glowColor: 'shadow-safe-500/5',
  },
  {
    key: 'totalAlerts',
    label: 'Total Alerts',
    icon: Bell,
    gradient: 'from-warning-500/15 to-warning-600/5',
    iconBg: 'bg-warning-500/10',
    iconColor: 'text-warning-400',
    glowColor: 'shadow-warning-500/5',
  },
  {
    key: 'dangerAlerts',
    label: 'Danger',
    icon: ShieldAlert,
    gradient: 'from-danger-500/15 to-danger-600/5',
    iconBg: 'bg-danger-500/10',
    iconColor: 'text-danger-400',
    glowColor: 'shadow-danger-500/5',
  },
];

export default function StatsBar({ data = {} }) {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      {stats.map((stat, i) => {
        const Icon = stat.icon;
        const value = data[stat.key];
        const isDanger = stat.key === 'dangerAlerts' && value > 0;

        return (
          <motion.div
            key={stat.key}
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, delay: 0.05 + i * 0.08, ease: [0.25, 0.46, 0.45, 0.94] }}
            whileHover={{ y: -2, transition: { duration: 0.2 } }}
            className={`
              relative overflow-hidden flex items-center gap-3 px-4 py-4 rounded-2xl
              border bg-[var(--color-card)]
              ${isDanger ? 'border-danger-500/20 glow-ambient-danger' : 'border-[var(--color-border)]'}
              hover:border-[#283048] transition-all duration-300 cursor-default group
            `}
          >
            {/* Subtle gradient overlay */}
            <div className={`absolute inset-0 bg-gradient-to-br ${stat.gradient} opacity-0 group-hover:opacity-100 transition-opacity duration-500`} />

            {/* Icon */}
            <div className={`relative w-10 h-10 rounded-xl ${stat.iconBg} flex items-center justify-center shrink-0 border border-white/[0.04]`}>
              <Icon size={17} className={stat.iconColor} strokeWidth={1.8} />
            </div>

            {/* Content */}
            <div className="relative min-w-0">
              <p className="text-[10px] text-slate-500 font-semibold uppercase tracking-wider">{stat.label}</p>
              <div className="flex items-baseline gap-1.5">
                <p className={`text-lg font-bold tabular-nums tracking-tight ${isDanger ? 'text-danger-400' : 'text-white'}`}>
                  {value !== undefined ? (typeof value === 'string' ? value : <AnimatedNumber value={value} />) : '--'}
                </p>
                {stat.key === 'fps' && typeof value === 'number' && value > 0 && (
                  <span className="text-[10px] text-slate-600 font-medium">fps</span>
                )}
              </div>
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}
