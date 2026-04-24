import { motion } from 'framer-motion';

export default function Card({ children, className = '', delay = 0, hoverable = true, glow = '' }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay, ease: [0.25, 0.46, 0.45, 0.94] }}
      whileHover={hoverable ? { y: -2, transition: { duration: 0.2 } } : undefined}
      className={`
        relative overflow-hidden
        bg-[var(--color-card)] rounded-2xl
        border border-[var(--color-border)]
        hover:border-[#283048]
        hover:shadow-[0_8px_32px_rgba(0,0,0,0.3)]
        transition-all duration-300
        ${glow}
        ${className}
      `}
    >
      {/* Subtle top highlight */}
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/[0.06] to-transparent" />
      {children}
    </motion.div>
  );
}
