import { motion, useReducedMotion } from 'framer-motion'
import { useRef, useEffect } from 'react'

/* color-mix helper — works with both CSS vars and hex colors, unlike ${color}XX hex-alpha hacks */
const a = (color, pct) => `color-mix(in srgb, ${color} ${pct}%, transparent)`

/* Always-on per-icon animation — every icon moves at all times; abnormal = faster */
function iconAnimation(label, isAbnormal) {
  const l = label.toLowerCase()
  const spd = isAbnormal ? 0.65 : 1   // speed multiplier — tighter when alert
  if (l.includes('heart'))             return `heartbeat   ${(1.9  * spd).toFixed(1)}s ease-in-out infinite`
  if (l.includes('spo') || l.includes('o₂')) return `dropSway ${(3.4 * spd).toFixed(1)}s ease-in-out infinite`
  if (l.includes('resp'))              return `breathe     ${(4.0  * spd).toFixed(1)}s ease-in-out infinite`
  if (l.includes('temp'))              return `floatY      ${(3.6  * spd).toFixed(1)}s ease-in-out infinite`
  if (l.includes('sys') || l.includes('sbp')) return `bpNudgeUp ${(2.6 * spd).toFixed(1)}s ease-in-out infinite`
  if (l.includes('map'))               return `mapSpin     ${(9.0  * spd).toFixed(1)}s linear infinite`
  if (l.includes('dia') || l.includes('dbp')) return `bpNudgeDn ${(2.6 * spd).toFixed(1)}s ease-in-out infinite`
  return `iconPulse ${(3.0 * spd).toFixed(1)}s ease-in-out infinite`
}

const STATUS = {
  normal: { color: 'var(--ok-l)',     label: 'Normal',       arrow: null },
  high:   { color: 'var(--danger-l)', label: 'Above Normal', arrow: '↑'  },
  low:    { color: 'var(--warn-l)',   label: 'Below Normal', arrow: '↓'  },
}

/* SVG trend arrow — precise, no emoji */
function TrendArrow({ direction }) {
  if (!direction) return null
  const isUp = direction === '↑'
  return (
    <svg width="11" height="11" viewBox="0 0 12 12" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
      aria-hidden="true" style={{ display: 'inline-block', flexShrink: 0 }}>
      {isUp ? <path d="M6 10V2M2 6l4-4 4 4"/> : <path d="M6 2v8M2 6l4 4 4-4"/>}
    </svg>
  )
}

/* Mini sparkline — 5 relative readings showing trajectory */
function MiniSparkline({ value, normalLo, normalHi, color }) {
  if (value == null) return null
  const mid   = (normalLo + normalHi) / 2
  const range = Math.max((normalHi - normalLo) * 1.8, 1)

  /* Simulate slight trail toward current value */
  const offsets = [-0.35, -0.20, -0.25, -0.08, 0]
  const pts = offsets.map((off, i) => {
    const v = 50 + ((value + off * range * 0.18 - mid) / range) * 80
    const clamped = Math.max(5, Math.min(95, v))
    return `${i * 25},${50 - (clamped - 50) * 0.45}`
  })

  return (
    <svg viewBox="0 0 100 20" width="56" height="12" aria-hidden="true"
      style={{ opacity: 0.50, flexShrink: 0 }}>
      <polyline points={pts.join(' ')} fill="none" stroke={color}
        strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

/* ─── Wide variant — HR and SpO₂ in the bento grid ─────────────────────── */
function WideVitalCard({ Icon, label, value, unit, status, normalLo, normalHi }) {
  const { color, label: statusLabel, arrow } = STATUS[status] ?? STATUS.normal
  const isAbnormal = status !== 'normal'
  const prefersReduced = useReducedMotion()
  const valueRef = useValuePop(value)

  const displayValue = value != null
    ? (Number.isInteger(value) ? value : value.toFixed(1))
    : '—'

  const iconAnim = iconAnimation(label, isAbnormal)

  return (
    <motion.div
      className="flex flex-col sm:flex-row items-start sm:items-center gap-5"
      whileHover={prefersReduced ? {} : { y: -3, scale: 1.008 }}
      whileTap={prefersReduced ? {} : { scale: 0.975 }}
      transition={{ type: 'spring', stiffness: 340, damping: 26, mass: 0.9 }}
      style={{
        background: isAbnormal
          ? `linear-gradient(145deg, ${a(color, 5)} 0%, var(--s1) 60%)`
          : 'linear-gradient(145deg, var(--s2) 0%, var(--s1) 60%)',
        border: `1px solid ${isAbnormal ? a(color, 22) : 'var(--b1)'}`,
        borderRadius: 'var(--r-xl)',
        padding: '20px 24px',
        boxShadow: isAbnormal
          ? `var(--sh-sm), 0 0 0 3px ${a(color, 7)}, inset 0 1px 0 ${a(color, 8)}`
          : 'var(--sh-sm)',
        position: 'relative',
        overflow: 'hidden',
        cursor: 'default',
      }}
    >
      {/* Top color band */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: 3,
        background: isAbnormal
          ? `linear-gradient(to right, ${color}, ${a(color, 40)}, transparent 80%)`
          : `linear-gradient(to right, var(--b1), transparent)`,
        borderRadius: 'var(--r-xl) var(--r-xl) 0 0',
      }} />

      {/* Icon */}
      <div
        className="flex-shrink-0 flex items-center justify-center mt-1"
        style={{
          width: 56, height: 56,
          borderRadius: 16,
          background: `linear-gradient(145deg, ${a(color, 14)}, ${a(color, 5)})`,
          border: `1.5px solid ${a(color, 18)}`,
          color,
          animation: iconAnim,
          boxShadow: isAbnormal ? `0 6px 18px ${a(color, 14)}, inset 0 1px 0 ${a(color, 22)}` : `inset 0 1px 0 ${a(color, 14)}`,
        }}
      >
        <div style={{ width: 26, height: 26 }}><Icon /></div>
      </div>

      {/* Label + big value */}
      <div className="flex-1 flex flex-col min-w-0">
        <span
          className="text-xs font-semibold mb-1 uppercase tracking-wider"
          style={{ color: 'var(--t3)', fontFamily: 'var(--font-display)', letterSpacing: '0.10em' }}
        >
          {label}
        </span>
        <div className="flex items-end gap-2">
          <span
            ref={valueRef}
            className="num font-bold leading-none"
            style={{
              fontSize: '3.8rem',
              fontWeight: 800,
              color: isAbnormal ? color : 'var(--t1)',
              transition: 'color 0.35s',
              letterSpacing: '-0.05em',
              fontOpticalSizing: 'auto',
            }}
          >
            {displayValue}
          </span>
          <span className="text-sm mb-1.5 leading-none font-medium" style={{ color: 'var(--t3)' }}>
            {unit}
          </span>
        </div>
      </div>

      {/* Right cluster: status + sparkline + range */}
      <div className="flex flex-col items-end gap-2 flex-shrink-0">
        <span
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-bold"
          style={{
            color,
            background: a(color, 10),
            border: `1px solid ${a(color, 20)}`,
            fontFamily: 'var(--font-display)',
            letterSpacing: '0.04em',
          }}
        >
          <TrendArrow direction={arrow} />
          {statusLabel}
        </span>
        <MiniSparkline value={value} normalLo={normalLo} normalHi={normalHi} color={color} />
        <span
          className="num text-[11px] px-2 py-0.5 rounded-md"
          style={{ color: 'var(--t3)', background: 'var(--s2)', border: '1px solid var(--b1)' }}
        >
          {normalLo}–{normalHi}
        </span>
      </div>
    </motion.div>
  )
}

/* ─── Value pop — brief glow when value changes ──────────────────────────── */
function useValuePop(value) {
  const ref = useRef(null)
  const prev = useRef(value)
  useEffect(() => {
    if (prev.current !== value && ref.current) {
      ref.current.classList.remove('animate-value-pop')
      void ref.current.offsetWidth
      ref.current.classList.add('animate-value-pop')
    }
    prev.current = value
  }, [value])
  return ref
}

/* ─── Standard card ─────────────────────────────────────────────────────── */
export default function VitalCard({ Icon, label, value, unit, status, normalLo, normalHi, wide }) {
  const prefersReduced = useReducedMotion()

  if (wide) {
    return <WideVitalCard
      Icon={Icon} label={label} value={value} unit={unit}
      status={status} normalLo={normalLo} normalHi={normalHi}
    />
  }

  const { color, label: statusLabel, arrow } = STATUS[status] ?? STATUS.normal
  const isAbnormal = status !== 'normal'
  const valueRef = useValuePop(value)

  const displayValue = value != null
    ? (Number.isInteger(value) ? value : value.toFixed(1))
    : '—'

  const iconAnim = iconAnimation(label, isAbnormal)

  return (
    <motion.div
      className="flex flex-col"
      whileHover={prefersReduced ? {} : { y: -4, scale: 1.012 }}
      whileTap={prefersReduced ? {} : { scale: 0.97 }}
      transition={{ type: 'spring', stiffness: 360, damping: 26, mass: 0.85 }}
      style={{
        background: isAbnormal
          ? `linear-gradient(160deg, ${a(color, 6)} 0%, var(--s1) 55%)`
          : 'linear-gradient(160deg, var(--s2) 0%, var(--s1) 55%)',
        border: `1px solid ${isAbnormal ? a(color, 22) : 'var(--b1)'}`,
        borderRadius: 'var(--r-xl)',
        padding: '18px 18px 16px',
        boxShadow: isAbnormal
          ? `var(--sh-sm), 0 0 0 3px ${a(color, 7)}, inset 0 1px 0 ${a(color, 8)}`
          : 'var(--sh-sm)',
        position: 'relative',
        overflow: 'hidden',
        cursor: 'default',
      }}
    >
      {/* Top color band */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: 3,
        background: isAbnormal
          ? `linear-gradient(to right, ${color}, ${a(color, 40)})`
          : `linear-gradient(to right, var(--b1), transparent)`,
        borderRadius: 'var(--r-xl) var(--r-xl) 0 0',
      }} />

      {/* Icon + label row */}
      <div className="flex items-center gap-3 mb-4 mt-1">
        <div
          className="flex-shrink-0 flex items-center justify-center"
          style={{
            width: 46, height: 46,
            borderRadius: 14,
            background: `linear-gradient(145deg, ${a(color, 12)}, ${a(color, 4)})`,
            border: `1.5px solid ${a(color, 16)}`,
            color,
            animation: iconAnim,
            boxShadow: isAbnormal
              ? `0 4px 14px ${a(color, 12)}, inset 0 1px 0 ${a(color, 20)}`
              : `inset 0 1px 0 ${a(color, 12)}`,
          }}
        >
          <div style={{ width: 22, height: 22 }}><Icon /></div>
        </div>

        <span
          className="text-[11px] font-bold leading-tight uppercase tracking-wider"
          style={{ color: 'var(--t3)', fontFamily: 'var(--font-display)', letterSpacing: '0.09em' }}
        >
          {label}
        </span>
      </div>

      {/* Big value — the hero */}
      <div className="flex items-end gap-1.5 mb-4">
        <span
          ref={valueRef}
          className="num font-bold leading-none"
          style={{
            fontSize: '3.1rem',
            fontWeight: 800,
            color: isAbnormal ? color : 'var(--t1)',
            transition: 'color 0.35s',
            letterSpacing: '-0.04em',
            fontOpticalSizing: 'auto',
          }}
        >
          {displayValue}
        </span>
        <span
          className="text-sm mb-1 leading-none font-medium"
          style={{ color: 'var(--t3)' }}
        >
          {unit}
        </span>
      </div>

      {/* Status footer */}
      <div className="flex items-center justify-between">
        <span
          className="flex items-center gap-1 text-[11px] font-bold"
          style={{ color, fontFamily: 'var(--font-display)' }}
        >
          <TrendArrow direction={arrow} />
          {statusLabel}
        </span>
        <span
          className="num text-[11px] px-2 py-0.5 rounded-md"
          style={{ color: 'var(--t3)', background: 'var(--s2)', border: '1px solid var(--b1)' }}
        >
          {normalLo}–{normalHi}
        </span>
      </div>
    </motion.div>
  )
}
