import { motion, AnimatePresence } from 'framer-motion'
import { useState } from 'react'

const a = (c, p) => `color-mix(in srgb, ${c} ${p}%, transparent)`

function HistoryClockIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor"
      strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M2.5 9a6.5 6.5 0 1 1 1.7 4.4"/>
      <path d="M2.5 4.5V9H7"/>
      <path d="M9 6v3.5l2.5 1.5"/>
    </svg>
  )
}

export default function HistoryFAB({ onClick, recordCount }) {
  const [hovered, setHovered] = useState(false)

  return (
    <motion.button
      onClick={onClick}
      onHoverStart={() => setHovered(true)}
      onHoverEnd={() => setHovered(false)}
      whileHover={{ scale: 1.06 }}
      whileTap={{ scale: 0.92 }}
      transition={{ type: 'spring', stiffness: 380, damping: 26 }}
      title="Open patient history"
      aria-label="Open patient history"
      style={{
        position: 'fixed',
        bottom: 28,
        right: 28,
        zIndex: 800,
        display: 'flex',
        alignItems: 'center',
        gap: 9,
        padding: hovered ? '12px 18px 12px 14px' : '12px',
        borderRadius: 50,
        background: `linear-gradient(145deg, var(--s2), var(--s1))`,
        border: `1.5px solid ${a('var(--cyan-l)', 28)}`,
        color: 'var(--cyan-l)',
        cursor: 'pointer',
        boxShadow: `0 8px 28px rgba(0,0,0,0.45), 0 0 0 1px ${a('var(--cyan-l)', 10)}, 0 0 22px ${a('var(--cyan-l)', 14)}`,
        overflow: 'hidden',
        transition: 'padding 0.22s cubic-bezier(0.34,1.56,0.64,1), border-radius 0.22s cubic-bezier(0.34,1.56,0.64,1)',
        outline: 'none',
      }}
    >
      {/* Subtle glow ring on hover */}
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
              background: a('var(--cyan-l)', 6),
              pointerEvents: 'none',
              zIndex: -1,
            }}
          />
        )}
      </AnimatePresence>

      {/* Clock icon — always spins slowly */}
      <span style={{ animation: 'mapSpin 12s linear infinite', display: 'flex', flexShrink: 0 }}>
        <HistoryClockIcon />
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
              color: 'var(--cyan-l)',
            }}
          >
            Patient History
            {recordCount > 0 && (
              <span style={{
                marginLeft: 7, fontSize: '10px', fontWeight: 700,
                background: a('var(--cyan-l)', 15),
                border: `1px solid ${a('var(--cyan-l)', 25)}`,
                borderRadius: 999, padding: '1px 6px',
                color: 'var(--cyan-l)',
              }}>
                {recordCount}
              </span>
            )}
          </motion.span>
        )}
      </AnimatePresence>
    </motion.button>
  )
}
