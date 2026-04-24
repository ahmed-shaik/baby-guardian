import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  LayoutDashboard, Camera, Bell, Activity, Settings,
  ChevronLeft, ChevronRight, Shield, BarChart3, Baby
} from 'lucide-react';

const navItems = [
  { id: 'dashboard', icon: LayoutDashboard, label: 'Dashboard' },
  { id: 'cameras', icon: Camera, label: 'Cameras' },
  { id: 'alerts', icon: Bell, label: 'Alerts' },
  { id: 'analytics', icon: BarChart3, label: 'Analytics' },
  { id: 'activity', icon: Activity, label: 'Activity' },
  { id: 'safety', icon: Shield, label: 'Safety Rules' },
];

export default function Sidebar({ collapsed, onToggle }) {
  const [active, setActive] = useState('dashboard');

  return (
    <motion.aside
      initial={false}
      animate={{ width: collapsed ? 64 : 220 }}
      transition={{ duration: 0.3, ease: [0.25, 0.46, 0.45, 0.94] }}
      className="fixed left-0 top-0 bottom-0 z-50 flex flex-col bg-[var(--color-sidebar)] border-r border-[var(--color-border)]"
    >
      {/* Logo */}
      <div className="flex items-center gap-2.5 px-4 h-[60px] border-b border-[var(--color-border)]">
        <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center shadow-lg shadow-primary-500/20 shrink-0">
          <Baby size={16} className="text-white" />
        </div>
        <AnimatePresence>
          {!collapsed && (
            <motion.span
              initial={{ opacity: 0, width: 0 }}
              animate={{ opacity: 1, width: 'auto' }}
              exit={{ opacity: 0, width: 0 }}
              transition={{ duration: 0.2 }}
              className="text-sm font-bold text-white whitespace-nowrap overflow-hidden"
            >
              Baby Guardian
            </motion.span>
          )}
        </AnimatePresence>
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-3 px-2.5 space-y-0.5 overflow-y-auto">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = active === item.id;

          return (
            <button
              key={item.id}
              onClick={() => setActive(item.id)}
              className={`
                relative w-full flex items-center gap-3 rounded-xl cursor-pointer
                transition-all duration-200 group
                ${collapsed ? 'justify-center px-0 py-2.5' : 'px-3 py-2.5'}
                ${isActive
                  ? 'bg-primary-500/10 text-primary-400'
                  : 'text-slate-500 hover:text-slate-300 hover:bg-white/[0.03]'
                }
              `}
            >
              {isActive && (
                <motion.div
                  layoutId="sidebar-active"
                  className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 rounded-r-full bg-primary-500"
                  transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                />
              )}
              <Icon size={18} strokeWidth={isActive ? 2 : 1.5} className="shrink-0" />
              <AnimatePresence>
                {!collapsed && (
                  <motion.span
                    initial={{ opacity: 0, width: 0 }}
                    animate={{ opacity: 1, width: 'auto' }}
                    exit={{ opacity: 0, width: 0 }}
                    transition={{ duration: 0.2 }}
                    className="text-[13px] font-medium whitespace-nowrap overflow-hidden"
                  >
                    {item.label}
                  </motion.span>
                )}
              </AnimatePresence>

              {/* Tooltip when collapsed */}
              {collapsed && (
                <div className="absolute left-full ml-2 px-2.5 py-1.5 rounded-lg bg-[var(--color-card)] border border-[var(--color-border)] text-xs font-medium text-slate-300 whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-150 pointer-events-none z-50 shadow-xl">
                  {item.label}
                </div>
              )}
            </button>
          );
        })}
      </nav>

      {/* Bottom section */}
      <div className="px-2.5 pb-3 space-y-0.5">
        <button className={`
          w-full flex items-center gap-3 rounded-xl cursor-pointer
          text-slate-500 hover:text-slate-300 hover:bg-white/[0.03]
          transition-all duration-200
          ${collapsed ? 'justify-center px-0 py-2.5' : 'px-3 py-2.5'}
        `}>
          <Settings size={18} strokeWidth={1.5} className="shrink-0" />
          <AnimatePresence>
            {!collapsed && (
              <motion.span
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="text-[13px] font-medium"
              >
                Settings
              </motion.span>
            )}
          </AnimatePresence>
        </button>

        {/* Collapse toggle */}
        <button
          onClick={onToggle}
          className={`
            w-full flex items-center gap-3 rounded-xl cursor-pointer
            text-slate-600 hover:text-slate-400 hover:bg-white/[0.03]
            transition-all duration-200
            ${collapsed ? 'justify-center px-0 py-2.5' : 'px-3 py-2.5'}
          `}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
          <AnimatePresence>
            {!collapsed && (
              <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="text-[12px] font-medium">
                Collapse
              </motion.span>
            )}
          </AnimatePresence>
        </button>
      </div>
    </motion.aside>
  );
}
