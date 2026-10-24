import { motion } from 'framer-motion'

export default function AlertBanner({ predictData }) {
  if (!predictData) return null
  const { risk_level, sofa_score } = predictData
  if (risk_level === 'Low Risk') return null

  const isHigh  = risk_level === 'High Risk'
  const border  = isHigh ? '#dc2626' : '#d97706'
  const glow    = isHigh ? 'rgba(220,38,38,0.22)' : 'rgba(217,119,6,0.16)'
  const label   = isHigh ? 'CRITICAL ALERT'        : 'ELEVATED RISK'
  const msg     = isHigh
    ? `SOFA ${sofa_score?.toFixed(1)} — Immediate clinical review required`
    : `SOFA ${sofa_score?.toFixed(1)} — Close observation warranted`

  return (
    <motion.div
      role="alert"
      aria-live={isHigh ? 'assertive' : 'polite'}
      className="flex items-center gap-4"
      initial={{ opacity: 0, y: -48, scaleY: 0.8 }}
      animate={{ opacity: 1, y: 0, scaleY: 1 }}
      exit={{ opacity: 0, y: -40, scaleY: 0.85, transition: { duration: 0.22, ease: [0.22, 1, 0.36, 1] } }}
      transition={{ type: 'spring', stiffness: 400, damping: 30, mass: 0.8 }}
      style={{
        background: isHigh
          ? 'linear-gradient(to right, rgba(220,38,38,0.14), rgba(220,38,38,0.06))'
          : 'linear-gradient(to right, rgba(217,119,6,0.12), rgba(217,119,6,0.05))',
        borderBottom: `1px solid ${border}40`,
        borderLeft: `4px solid ${border}`,
        padding: '10px 20px',
        boxShadow: `0 4px 20px ${glow}`,
      }}
    >
      {/* Pulse dot */}
      <span className="relative flex h-2.5 w-2.5 flex-shrink-0">
        <span
          className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-70"
          style={{ background: border }}
        />
        <span
          className="relative inline-flex h-2.5 w-2.5 rounded-full"
          style={{ background: border, boxShadow: `0 0 8px ${border}` }}
        />
      </span>

      {/* Label chip */}
      <span
        className="font-black tracking-[0.12em] px-3 py-1.5 rounded-lg flex-shrink-0"
        style={{
          background: `${border}20`,
          color: border,
          fontSize: '10px',
          border: `1px solid ${border}50`,
          fontFamily: 'var(--font-display)',
          letterSpacing: '0.14em',
          boxShadow: `0 2px 8px ${border}20`,
        }}
      >
        {label}
      </span>

      {/* Message */}
      <span
        className="text-sm font-semibold"
        style={{ color: border, fontFamily: 'var(--font-display)' }}
      >
        {msg}
      </span>

      {/* Right: urgency icon for critical */}
      {isHigh && (
        <motion.div
          className="ml-auto flex-shrink-0 flex items-center gap-2"
          animate={{ opacity: [1, 0.4, 1] }}
          transition={{ duration: 1.4, repeat: Infinity, ease: 'easeInOut' }}
        >
          <svg
            width="18" height="18" viewBox="0 0 20 20"
            fill="none" stroke={border} strokeWidth="1.8"
            strokeLinecap="round" strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="M10 2L2 17h16L10 2z"/>
            <line x1="10" y1="9" x2="10" y2="12"/>
            <circle cx="10" cy="14.5" r="0.5" fill={border}/>
          </svg>
        </motion.div>
      )}
    </motion.div>
  )
}
