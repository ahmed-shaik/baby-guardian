import { motion } from 'framer-motion';
import { Activity, Eye, AlertCircle, ShieldCheck } from 'lucide-react';
import Card from './Card';
import { statusConfig } from '../data/dummyData';

function CircularMeter({ score, label, size = 72 }) {
  const config = statusConfig[label] || statusConfig.safe;
  const percentage = Math.round(score * 100);
  const radius = (size - 10) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference * (1 - score);

  const gradientId = `meter-gradient-${label}`;
  const colors = {
    danger: ['#f87171', '#dc2626'],
    warning: ['#fbbf24', '#d97706'],
    safe: ['#4ade80', '#16a34a'],
  };
  const [c1, c2] = colors[label] || colors.safe;

  return (
    <div className="flex items-center gap-4">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          <defs>
            <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor={c1} />
              <stop offset="100%" stopColor={c2} />
            </linearGradient>
          </defs>
          {/* Track */}
          <circle cx={size/2} cy={size/2} r={radius} fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth={5} />
          {/* Progress */}
          <motion.circle cx={size/2} cy={size/2} r={radius} fill="none" stroke={`url(#${gradientId})`} strokeWidth={5} strokeLinecap="round"
            strokeDasharray={circumference} initial={{ strokeDashoffset: circumference }} animate={{ strokeDashoffset }}
            transition={{ duration: 1.2, delay: 0.3, ease: [0.25, 0.46, 0.45, 0.94] }}
            style={{ filter: `drop-shadow(0 0 8px ${c1}30)` }} />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className={`text-base font-bold tabular-nums ${config.text}`}>{percentage}%</span>
        </div>
      </div>
      <div>
        <p className="text-[10px] text-slate-500 font-semibold uppercase tracking-wider">Risk Level</p>
        <p className={`text-sm font-bold ${config.text}`}>{config.label}</p>
        <p className="text-[10px] text-slate-600 mt-0.5">
          {label === 'safe' ? 'All clear' : label === 'warning' ? 'Monitor closely' : 'Attention needed'}
        </p>
      </div>
    </div>
  );
}

export default function PoseAnalysis({ pose = {} }) {
  const { confidence = 0, visibleLandmarks = 0, totalLandmarks = 33, riskScore = 0, riskLabel = 'safe', activeRules = [] } = pose;

  const glowClass = riskLabel === 'danger' ? 'glow-ambient-danger' : riskLabel === 'warning' ? 'glow-ambient-warning' : '';

  return (
    <Card delay={0.2} glow={glowClass}>
      <div className="p-5">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-4 rounded-full bg-gradient-to-b from-violet-400 to-violet-600" />
          <h3 className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Pose Analysis</h3>
        </div>
        <div className="space-y-4">
          <CircularMeter score={riskScore} label={riskLabel} />

          {/* Stats grid */}
          <div className="grid grid-cols-2 gap-2">
            <div className="flex items-center gap-2.5 px-3 py-3 bg-white/[0.02] rounded-xl border border-white/[0.04] hover:bg-white/[0.04] transition-colors duration-200">
              <div className="w-7 h-7 rounded-lg bg-primary-500/10 flex items-center justify-center">
                <Activity size={13} className="text-primary-400" strokeWidth={2} />
              </div>
              <div>
                <p className="text-[9px] text-slate-500 font-semibold uppercase tracking-wider">Confidence</p>
                <p className="text-sm font-bold text-white tabular-nums">{(confidence * 100).toFixed(0)}%</p>
              </div>
            </div>
            <div className="flex items-center gap-2.5 px-3 py-3 bg-white/[0.02] rounded-xl border border-white/[0.04] hover:bg-white/[0.04] transition-colors duration-200">
              <div className="w-7 h-7 rounded-lg bg-violet-500/10 flex items-center justify-center">
                <Eye size={13} className="text-violet-400" strokeWidth={2} />
              </div>
              <div>
                <p className="text-[9px] text-slate-500 font-semibold uppercase tracking-wider">Landmarks</p>
                <p className="text-sm font-bold text-white tabular-nums">{visibleLandmarks}<span className="text-slate-600">/{totalLandmarks}</span></p>
              </div>
            </div>
          </div>

          {/* Active rules */}
          {activeRules.length > 0 && (
            <div className="space-y-1.5 pt-1">
              <p className="text-[9px] text-slate-600 font-semibold uppercase tracking-wider mb-2">Active Rules</p>
              {activeRules.map((rule, i) => {
                const isSafe = rule.toLowerCase().includes('no risk') || rule.toLowerCase().includes('waiting') || rule.toLowerCase().includes('not connected');
                return (
                  <motion.div key={i} initial={{ opacity: 0, x: -4 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.4 + i * 0.1 }}
                    className={`flex items-start gap-2 text-xs px-2.5 py-2 rounded-lg ${isSafe ? 'bg-safe-500/5' : 'bg-warning-500/5'}`}>
                    {isSafe ? <ShieldCheck size={12} className="mt-0.5 shrink-0 text-safe-400" /> : <AlertCircle size={12} className="mt-0.5 shrink-0 text-warning-400" />}
                    <span className={isSafe ? 'text-safe-400/80' : 'text-slate-400'}>{rule}</span>
                  </motion.div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}
