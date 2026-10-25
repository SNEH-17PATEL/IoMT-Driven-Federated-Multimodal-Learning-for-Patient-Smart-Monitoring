import { motion, AnimatePresence } from 'framer-motion'
import { useState } from 'react'

const a = (c, p) => `color-mix(in srgb, ${c} ${p}%, transparent)`
const CYAN = '#38bdf8'

function BarChartIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor"
      strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="1" y="10" width="3.5" height="6" rx="0.8"/>
      <rect x="7.25" y="6" width="3.5" height="10" rx="0.8"/>
      <rect x="13.5" y="2" width="3.5" height="14" rx="0.8"/>
    </svg>
  )
}

export default function SOFAValidationFAB({ onClick }) {
  const [hovered, setHovered] = useState(false)

  return (
    <motion.button
      onClick={onClick}
      onHoverStart={() => setHovered(true)}
      onHoverEnd={() => setHovered(false)}
      whileHover={{ scale: 1.06 }}
      whileTap={{ scale: 0.92 }}
      transition={{ type: 'spring', stiffness: 380, damping: 26 }}
      title="SOFA Score Validation"
      aria-label="Open SOFA Score Validation"
      style={{
        position: 'fixed',
        bottom: 208,
        right: 28,
        zIndex: 800,
        display: 'flex',
        alignItems: 'center',
        gap: 9,
        padding: hovered ? '12px 18px 12px 14px' : '12px',
        borderRadius: 50,
        background: `linear-gradient(145deg, var(--s2), var(--s1))`,
        border: `1.5px solid ${a(CYAN, 28)}`,
        color: CYAN,
        cursor: 'pointer',
        boxShadow: `0 8px 28px rgba(0,0,0,0.45), 0 0 0 1px ${a(CYAN, 10)}, 0 0 22px ${a(CYAN, 14)}`,
        overflow: 'hidden',
        transition: 'padding 0.22s cubic-bezier(0.34,1.56,0.64,1), border-radius 0.22s cubic-bezier(0.34,1.56,0.64,1)',
        outline: 'none',
      }}
    >
      {/* Glow ring on hover */}
      <AnimatePresence>
        {hovered && (
          <motion.span
            key="glow"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            style={{
              position: 'absolute', inset: -4,
              borderRadius: 'inherit',
              background: a(CYAN, 7),
              pointerEvents: 'none',
              zIndex: -1,
            }}
          />
        )}
      </AnimatePresence>

      {/* Bar chart icon — bars grow up on each tick */}
      <span style={{ animation: 'barRise 2.2s ease-in-out infinite', display: 'flex', flexShrink: 0 }}>
        <BarChartIcon />
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
              color: CYAN,
            }}
          >
            SOFA Validation
          </motion.span>
        )}
      </AnimatePresence>
    </motion.button>
  )
}
