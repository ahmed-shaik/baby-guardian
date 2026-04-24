import { motion } from 'framer-motion';
import { Search, Bell, Wifi, WifiOff, CircleUser, Sparkles } from 'lucide-react';
import { useState } from 'react';

export default function Header({ connected = false }) {
  const [searchFocused, setSearchFocused] = useState(false);
  const now = new Date();
  const dateStr = now.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });

  return (
    <motion.header
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] }}
      className="sticky top-0 z-40 glass border-b border-[var(--color-border)]"
    >
      <div className="flex items-center justify-between px-5 sm:px-6 py-3">
        {/* Left: Page title + breadcrumb */}
        <div className="flex items-center gap-4">
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-[15px] font-bold text-white tracking-tight">Dashboard</h1>
              <div className={`flex items-center gap-1.5 text-[10px] font-semibold px-2 py-0.5 rounded-full border ${
                connected
                  ? 'text-safe-400 bg-safe-500/10 border-safe-500/15'
                  : 'text-slate-500 bg-white/5 border-white/5'
              }`}>
                <span className={`h-1.5 w-1.5 rounded-full ${connected ? 'bg-safe-400 animate-pulse' : 'bg-slate-600'}`} />
                {connected ? 'Live' : 'Offline'}
              </div>
            </div>
            <p className="text-[11px] text-slate-500 font-medium mt-0.5">
              Real-time baby safety monitoring &middot; {dateStr}
            </p>
          </div>
        </div>

        {/* Center: Search */}
        <div className="hidden md:flex items-center flex-1 max-w-sm mx-8">
          <div className={`
            relative w-full flex items-center gap-2 px-3.5 py-2 rounded-xl
            bg-white/[0.03] border transition-all duration-200
            ${searchFocused ? 'border-primary-500/30 bg-white/[0.05] shadow-[0_0_20px_-6px_rgba(59,130,246,0.15)]' : 'border-[var(--color-border)]'}
          `}>
            <Search size={14} className={`shrink-0 transition-colors duration-200 ${searchFocused ? 'text-primary-400' : 'text-slate-500'}`} />
            <input
              type="text"
              placeholder="Search alerts, detections..."
              className="w-full bg-transparent text-sm text-slate-300 placeholder-slate-600 outline-none"
              onFocus={() => setSearchFocused(true)}
              onBlur={() => setSearchFocused(false)}
            />
            <kbd className="hidden lg:inline-flex items-center gap-0.5 text-[10px] text-slate-600 bg-white/[0.04] border border-white/[0.06] rounded px-1.5 py-0.5 font-mono">
              ⌘K
            </kbd>
          </div>
        </div>

        {/* Right: Actions */}
        <div className="flex items-center gap-1.5">
          {/* Connection indicator */}
          <div className={`hidden sm:flex items-center gap-1.5 text-[11px] font-medium mr-1 px-2.5 py-1.5 rounded-lg border transition-colors ${
            connected
              ? 'text-safe-400 bg-safe-500/8 border-safe-500/15'
              : 'text-slate-500 bg-white/[0.03] border-white/5'
          }`}>
            {connected ? <Wifi size={12} /> : <WifiOff size={12} />}
            {connected ? 'Connected' : 'Offline'}
          </div>

          {/* AI Status */}
          <div className="hidden sm:flex items-center gap-1.5 text-[11px] font-medium px-2.5 py-1.5 rounded-lg bg-primary-500/8 border border-primary-500/15 text-primary-400 mr-1">
            <Sparkles size={12} />
            AI Active
          </div>

          {/* Notifications */}
          <button className="relative p-2 rounded-xl hover:bg-white/[0.05] active:scale-95 transition-all duration-150 cursor-pointer group" aria-label="Notifications">
            <Bell size={17} className="text-slate-400 group-hover:text-slate-300 transition-colors" />
            <span className="absolute top-1.5 right-1.5 h-2 w-2 rounded-full bg-danger-500 border-2 border-[var(--color-bg)]" />
          </button>

          {/* User avatar */}
          <button className="flex items-center gap-2 p-1.5 rounded-xl hover:bg-white/[0.05] active:scale-95 transition-all duration-150 cursor-pointer" aria-label="Profile">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-primary-500 to-violet-500 flex items-center justify-center">
              <CircleUser size={15} className="text-white" />
            </div>
          </button>
        </div>
      </div>
    </motion.header>
  );
}
