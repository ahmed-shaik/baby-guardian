import { motion } from 'framer-motion';
import { Camera, Maximize2, Volume2, WifiOff, Radio } from 'lucide-react';
import StatusBadge from './StatusBadge';

export default function CameraFeed({ status = 'safe', fps = 0, frameUrl = null, connected = false }) {
  const now = new Date();
  const timeStr = now.toLocaleTimeString('en-US', { hour12: false });

  const statusGlow = status === 'danger'
    ? 'glow-ambient-danger'
    : status === 'warning'
    ? 'glow-ambient-warning'
    : '';

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.25, 0.46, 0.45, 0.94] }}
      className={`relative w-full aspect-video rounded-2xl overflow-hidden group border border-[var(--color-border)] shadow-[0_0_60px_rgba(0,0,0,0.5)] ${statusGlow}`}
    >
      {frameUrl ? (
        <img src={frameUrl} alt="Live camera feed with AI annotations" className="absolute inset-0 w-full h-full object-contain bg-black" />
      ) : (
        <>
          {/* Sophisticated empty state */}
          <div className="absolute inset-0 bg-gradient-to-br from-[#080c18] via-[#0a0e1a] to-[#06080f]" />

          {/* Grid pattern */}
          <div className="absolute inset-0 opacity-[0.03]"
            style={{ backgroundImage: 'radial-gradient(circle, #fff 1px, transparent 1px)', backgroundSize: '32px 32px' }} />

          {/* Animated gradient orb */}
          <div className="absolute inset-0 overflow-hidden">
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[300px] h-[300px] rounded-full bg-primary-500/5 blur-[100px] animate-breathe" />
          </div>

          {/* Scanline */}
          <div className="absolute inset-0 overflow-hidden pointer-events-none">
            <div className="animate-scanline w-full h-[20%] bg-gradient-to-b from-transparent via-white/[0.012] to-transparent" />
          </div>

          {/* Center content */}
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              {connected ? (
                <>
                  <div className="animate-float mx-auto mb-4 w-16 h-16 rounded-2xl glass flex items-center justify-center">
                    <Camera size={26} className="text-slate-400" />
                  </div>
                  <p className="text-slate-300 text-sm font-semibold">Waiting for frames...</p>
                  <p className="text-slate-600 text-xs mt-1.5">Pipeline is running</p>
                  <div className="flex items-center justify-center gap-1 mt-3">
                    {[0, 1, 2].map(i => (
                      <motion.div key={i} className="w-1.5 h-1.5 rounded-full bg-primary-500"
                        animate={{ opacity: [0.3, 1, 0.3] }}
                        transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.2 }} />
                    ))}
                  </div>
                </>
              ) : (
                <>
                  <div className="mx-auto mb-4 w-16 h-16 rounded-2xl bg-white/[0.03] flex items-center justify-center border border-white/[0.04]">
                    <WifiOff size={26} className="text-slate-600" />
                  </div>
                  <p className="text-slate-400 text-sm font-semibold">No connection</p>
                  <p className="text-slate-600 text-xs mt-1.5 max-w-[260px] leading-relaxed">
                    Start pipeline: <code className="text-primary-400 bg-primary-500/10 px-2 py-0.5 rounded-md text-[11px] font-mono border border-primary-500/10">python server.py</code>
                  </p>
                </>
              )}
            </div>
          </div>
        </>
      )}

      {/* Top overlay */}
      <div className="absolute top-0 left-0 right-0 p-3.5 sm:p-4 flex items-start justify-between bg-gradient-to-b from-black/70 via-black/30 to-transparent">
        <StatusBadge status={status} size="sm" />
        {connected && (
          <div className="flex items-center gap-1.5 glass-subtle text-white text-[11px] font-mono px-2.5 py-1.5 rounded-lg border border-white/[0.08]">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-500 opacity-40" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500" />
            </span>
            REC
          </div>
        )}
      </div>

      {/* Bottom overlay */}
      <div className="absolute bottom-0 left-0 right-0 p-3.5 sm:p-4 bg-gradient-to-t from-black/70 via-black/30 to-transparent">
        <div className="flex items-end justify-between">
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-mono text-slate-300 glass-subtle px-2.5 py-1 rounded-lg border border-white/[0.06]">{timeStr}</span>
            {fps > 0 && (
              <span className="text-[11px] font-mono text-slate-300 glass-subtle px-2.5 py-1 rounded-lg border border-white/[0.06]">
                <span className="text-safe-400">{fps.toFixed(1)}</span> FPS
              </span>
            )}
            {connected && (
              <span className="text-[11px] font-mono text-primary-400 glass-subtle px-2.5 py-1 rounded-lg border border-white/[0.06] flex items-center gap-1">
                <Radio size={10} />AI
              </span>
            )}
          </div>
          <div className="flex items-center gap-1.5 opacity-0 group-hover:opacity-100 transition-opacity duration-250">
            <button className="p-2 glass-subtle rounded-lg hover:bg-white/10 active:scale-95 transition-all duration-150 cursor-pointer border border-white/[0.06]" aria-label="Toggle audio">
              <Volume2 size={14} className="text-white" />
            </button>
            <button className="p-2 glass-subtle rounded-lg hover:bg-white/10 active:scale-95 transition-all duration-150 cursor-pointer border border-white/[0.06]" aria-label="Fullscreen">
              <Maximize2 size={14} className="text-white" />
            </button>
          </div>
        </div>
      </div>

      {/* Dynamic status ring */}
      <motion.div key={status} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.4 }}
        className={`absolute inset-0 pointer-events-none rounded-2xl ring-[1.5px] ring-inset ${
          status === 'danger' ? 'ring-danger-500/40' : status === 'warning' ? 'ring-warning-500/30' : 'ring-safe-500/10'
        }`}
      />
    </motion.div>
  );
}
