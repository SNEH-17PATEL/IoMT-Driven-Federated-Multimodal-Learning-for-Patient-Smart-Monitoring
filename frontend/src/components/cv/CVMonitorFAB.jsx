import { motion, AnimatePresence } from 'framer-motion'
import { useState } from 'react'

const a = (c, p) => `color-mix(in srgb, ${c} ${p}%, transparent)`
const PURPLE = '#a78bfa'

function EyeIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor"
      strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M1 9C1 9 4 3 9 3s8 6 8 6-3 6-8 6S1 9 1 9z"/>
      <circle cx="9" cy="9" r="2.5"/>
    </svg>
  )
}

export default function CVMonitorFAB({ onClick, isActive }) {
  const [hovered, setHovered] = useState(false)

  return (
    <motion.button
      onClick={onClick}
      onHoverStart={() => setHovered(true)}
      onHoverEnd={() => setHovered(false)}
      whileHover={{ scale: 1.06 }}
      whileTap={{ scale: 0.92 }}
      transition={{ type: 'spring', stiffness: 380, damping: 26 }}
      title="Open Live CV Monitor"
      aria-label="Open Live CV Monitor"
      style={{
        position: 'fixed',
        bottom: 88,
        right: 28,
        zIndex: 800,
        display: 'flex',
        alignItems: 'center',
        gap: 9,
        padding: hovered ? '12px 18px 12px 14px' : '12px',
        borderRadius: 50,
        background: `linear-gradient(145deg, var(--s2), var(--s1))`,
        border: `1.5px solid ${a(PURPLE, 28)}`,
        color: isActive ? PURPLE : PURPLE,
        cursor: 'pointer',
        boxShadow: isActive
          ? `0 8px 28px rgba(0,0,0,0.45), 0 0 0 1px ${a(PURPLE, 10)}, 0 0 28px ${a(PURPLE, 25)}`
          : `0 8px 28px rgba(0,0,0,0.45), 0 0 0 1px ${a(PURPLE, 10)}, 0 0 22px ${a(PURPLE, 14)}`,
        overflow: 'hidden',
        transition: 'padding 0.22s cubic-bezier(0.34,1.56,0.64,1), border-radius 0.22s cubic-bezier(0.34,1.56,0.64,1)',
        outline: 'none',
      }}
    >
      {/* Glow ring on hover / active */}
      <AnimatePresence>
        {(hovered || isActive) && (
          <motion.span
            key="glow"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            style={{
              position: 'absolute', inset: -4,
              borderRadius: 'inherit',
              background: a(PURPLE, isActive ? 10 : 6),
              pointerEvents: 'none',
              zIndex: -1,
            }}
          />
        )}
      </AnimatePresence>

      {/* Live indicator dot when active */}
      {isActive && (
        <span style={{
          position: 'absolute', top: 6, right: 6,
          width: 7, height: 7, borderRadius: '50%',
          background: '#ef4444',
          boxShadow: '0 0 6px #ef4444',
          animation: 'pulse 1.6s ease-in-out infinite',
        }} />
      )}

      {/* Eye icon — blinks slowly */}
      <span style={{
        animation: isActive
          ? 'eyeBlink 2.0s ease-in-out infinite'
          : 'eyeBlink 4.5s ease-in-out infinite',
        display: 'flex',
        flexShrink: 0,
      }}>
        <EyeIcon />
      </span>

      {/* Expanding label */}
      <AnimatePresence>
        {hovered && (
          <motion.span
            key="label"
            initial={{ opacity: 0, width: 0 }}
            animate={{ opacity: 1, width: 'auto' }}
            exit={{ opacity: 0, width: 0 }}
            transition={{ duration: 0.18 }}
            style={{
              fontSize: '12px', fontWeight: 700,
              fontFamily: 'var(--font-display)',
              letterSpacing: '0.04em',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              color: PURPLE,
            }}
          >
            {isActive ? 'CV Live — Click to Close' : 'Live CV Monitor'}
          </motion.span>
        )}
      </AnimatePresence>
    </motion.button>
  )
}
