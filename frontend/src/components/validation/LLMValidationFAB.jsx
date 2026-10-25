import { motion, AnimatePresence } from 'framer-motion'
import { useState } from 'react'

const a = (c, p) => `color-mix(in srgb, ${c} ${p}%, transparent)`
const PINK = '#f472b6'

function ShieldCheckIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor"
      strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M9 1.5 L15.5 4.5 L15.5 9 C15.5 12.5 12.5 15.5 9 16.5 C5.5 15.5 2.5 12.5 2.5 9 L2.5 4.5 Z"/>
      <polyline points="6.5,9 8.2,10.8 11.5,7.2"/>
    </svg>
  )
}

export default function LLMValidationFAB({ onClick, hasData }) {
  const [hovered, setHovered] = useState(false)

  return (
    <motion.button
      onClick={onClick}
      onHoverStart={() => setHovered(true)}
      onHoverEnd={() => setHovered(false)}
      whileHover={{ scale: 1.06 }}
      whileTap={{ scale: 0.92 }}
      transition={{ type: 'spring', stiffness: 380, damping: 26 }}
      title="AI Report Validation"
      aria-label="Open AI Report Validation"
      style={{
        position: 'fixed',
        bottom: 148,
        right: 28,
        zIndex: 800,
        display: 'flex',
        alignItems: 'center',
        gap: 9,
        padding: hovered ? '12px 18px 12px 14px' : '12px',
        borderRadius: 50,
        background: `linear-gradient(145deg, var(--s2), var(--s1))`,
        border: `1.5px solid ${a(PINK, 28)}`,
        color: PINK,
        cursor: 'pointer',
        boxShadow: `0 8px 28px rgba(0,0,0,0.45), 0 0 0 1px ${a(PINK, 10)}, 0 0 22px ${a(PINK, 14)}`,
        overflow: 'hidden',
        transition: 'padding 0.22s cubic-bezier(0.34,1.56,0.64,1), border-radius 0.22s cubic-bezier(0.34,1.56,0.64,1)',
        outline: 'none',
        opacity: hasData ? 1 : 0.45,
        pointerEvents: hasData ? 'auto' : 'none',
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
              background: a(PINK, 7),
              pointerEvents: 'none',
              zIndex: -1,
            }}
          />
        )}
      </AnimatePresence>

      {/* Shield icon — pulses softly */}
      <span style={{ animation: 'shieldPulse 3s ease-in-out infinite', display: 'flex', flexShrink: 0 }}>
        <ShieldCheckIcon />
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
              color: PINK,
            }}
          >
            {hasData ? 'AI Validation' : 'Awaiting Report…'}
          </motion.span>
        )}
      </AnimatePresence>
    </motion.button>
  )
}
