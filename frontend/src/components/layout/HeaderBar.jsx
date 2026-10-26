import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { useReducedMotion } from 'framer-motion'
import ThemeToggle from './ThemeToggle.jsx'

function formatTime(date) {
  if (!date) return '—'
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

/* ── ECG waveform ─────────────────────────────────────────────────────────── */
function EcgLine({ color = 'currentColor', width = 80, height = 22 }) {
  return (
    <svg viewBox="0 0 80 22" width={width} height={height} aria-hidden="true" style={{ overflow: 'visible' }}>
      <path
        d="M0,11 L16,11 L20,10 L25,3 L30,20 L35,11 L40,13 L45,11 L80,11"
        fill="none"
        stroke={color}
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
        style={{
          strokeDasharray: '40 120',
          strokeDashoffset: 0,
          animation: 'ecgScroll 1.8s linear infinite',
          opacity: 0.85,
        }}
      />
    </svg>
  )
}

/* ── Brand mark ───────────────────────────────────────────────────────────── */
function BrandMark() {
  return (
    <div className="flex items-center gap-3 flex-shrink-0">
      {/* Logo badge */}
      <div
        style={{
          position: 'relative',
          width: 46,
          height: 46,
          borderRadius: 14,
          background: 'linear-gradient(135deg, var(--blue-l) 0%, var(--cyan-l) 100%)',
          boxShadow:
            '0 0 0 1px rgba(255,255,255,0.18) inset, 0 6px 24px rgba(32,112,204,0.48), 0 2px 8px rgba(14,165,201,0.30)',
          flexShrink: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <svg width="22" height="22" viewBox="0 0 22 22" fill="none" aria-hidden="true">
          <path
            d="M1,11 L6,11 L8,6 L10.5,17 L13,11 L16,11 L18,8 L19.5,11 L21,11"
            stroke="white"
            strokeWidth="2.2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        {/* Glint */}
        <div style={{
          position: 'absolute',
          top: 4, left: 6, right: 14,
          height: 4,
          borderRadius: 4,
          background: 'rgba(255,255,255,0.30)',
          filter: 'blur(1px)',
        }} />
      </div>

      {/* Word-mark */}
      <div className="flex flex-col" style={{ gap: 1 }}>
        <span
          style={{
            fontFamily: 'var(--font-display)',
            fontSize: '20px',
            fontWeight: 800,
            letterSpacing: '-0.04em',
            lineHeight: 1,
            color: 'var(--t1)',
          }}
        >
          ClinIQ
        </span>
        <span
          style={{
            fontFamily: 'var(--font-display)',
            fontSize: '9.5px',
            fontWeight: 600,
            letterSpacing: '0.14em',
            textTransform: 'uppercase',
            color: 'var(--t4)',
            lineHeight: 1,
          }}
        >
          ICU Intelligence
        </span>
      </div>
    </div>
  )
}

/* ── Live indicator ───────────────────────────────────────────────────────── */
function LiveChip({ isRefreshing, countdown, connectionLost }) {
  if (connectionLost) {
    return (
      <div
        className="flex items-center gap-2 px-3.5 py-2 rounded-xl"
        style={{
          background: 'rgba(251,146,60,0.10)',
          border: '1px solid rgba(251,146,60,0.30)',
          color: '#fb923c',
        }}
      >
        <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
          <polygon points="8,2 14.5,14 1.5,14"/>
          <line x1="8" y1="7" x2="8" y2="10"/>
          <circle cx="8" cy="12.5" r="0.5" fill="currentColor" stroke="none"/>
        </svg>
        <span style={{ fontFamily: 'var(--font-display)', fontSize: '12px', fontWeight: 700, letterSpacing: '0.04em' }}>
          Connection lost
        </span>
      </div>
    )
  }

  return (
    <div
      className="flex items-center gap-2.5 px-4 py-2 rounded-xl"
      style={{
        background: 'rgba(220,38,38,0.10)',
        border: '1px solid rgba(220,38,38,0.25)',
        boxShadow: '0 0 16px rgba(220,38,38,0.10)',
      }}
    >
      {/* Ping dot */}
      <span className="relative flex flex-shrink-0" style={{ width: 9, height: 9 }}>
        <span
          className="animate-ping absolute inline-flex h-full w-full rounded-full"
          style={{ background: '#e53535', opacity: 0.5 }}
        />
        <span
          className="relative inline-flex rounded-full"
          style={{ width: 9, height: 9, background: '#e53535', boxShadow: '0 0 8px #e53535' }}
        />
      </span>

      {/* ECG */}
      <EcgLine color="#e53535" width={64} height={18} />

      {/* LIVE text */}
      <span
        style={{
          fontFamily: 'var(--font-display)',
          fontSize: '11px',
          fontWeight: 900,
          letterSpacing: '0.16em',
          color: '#e53535',
        }}
      >
        LIVE
      </span>

      {/* Countdown */}
      <span
        className="hidden sm:block"
        style={{
          fontFamily: 'var(--mono)',
          fontSize: '11px',
          color: 'var(--t3)',
          letterSpacing: '0.04em',
        }}
      >
        {isRefreshing ? '···' : `${countdown}s`}
      </span>
    </div>
  )
}

/* ── Patient context pill ─────────────────────────────────────────────────── */
function PatientPill({ patient, predictData }) {
  if (!patient) return null
  const riskColor = predictData?.risk_color
  const riskLevel = predictData?.risk_level

  return (
    <AnimatePresence>
      <motion.div
        key="patient-pill"
        initial={{ opacity: 0, x: -10 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: -10 }}
        transition={{ type: 'spring', stiffness: 380, damping: 28 }}
        className="hidden md:flex items-center gap-3"
        style={{
          padding: '8px 14px',
          borderRadius: 12,
          background: riskColor ? `${riskColor}0d` : 'var(--s2)',
          border: `1px solid ${riskColor ? riskColor + '28' : 'var(--b1)'}`,
        }}
      >
        {/* Separator */}
        <span style={{ color: 'var(--b2)', fontSize: '18px', lineHeight: 1, opacity: 0.6 }}>/</span>

        {/* Patient avatar initial */}
        <div
          style={{
            width: 28, height: 28,
            borderRadius: 8,
            background: 'linear-gradient(135deg, var(--blue-l), var(--cyan-l))',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '11px', fontWeight: 800, color: 'white',
            fontFamily: 'var(--font-display)',
            flexShrink: 0,
          }}
        >
          {patient.name?.[0] ?? 'P'}
        </div>

        <div className="flex flex-col" style={{ gap: 1 }}>
          <span style={{
            fontFamily: 'var(--font-display)',
            fontSize: '13px',
            fontWeight: 700,
            color: 'var(--t1)',
            lineHeight: 1,
            whiteSpace: 'nowrap',
            maxWidth: 160,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}>
            {patient.name}
          </span>
          {patient.age && (
            <span style={{ fontFamily: 'var(--font-display)', fontSize: '10px', color: 'var(--t4)', lineHeight: 1 }}>
              {patient.age}y · {patient.gender ?? ''}
            </span>
          )}
        </div>

        {/* Risk badge */}
        {predictData && riskColor && (
          <div
            className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded-full"
            style={{
              background: `${riskColor}15`,
              border: `1px solid ${riskColor}38`,
            }}
          >
            <span style={{
              width: 6, height: 6, borderRadius: '50%',
              background: riskColor,
              display: 'inline-block',
              boxShadow: `0 0 6px ${riskColor}`,
              flexShrink: 0,
            }} />
            <span style={{
              fontFamily: 'var(--font-display)',
              fontSize: '10px',
              fontWeight: 800,
              letterSpacing: '0.07em',
              color: riskColor,
              textTransform: 'uppercase',
            }}>
              {riskLevel}
            </span>
          </div>
        )}
      </motion.div>
    </AnimatePresence>
  )
}

/* ── Main header ──────────────────────────────────────────────────────────── */
export default function HeaderBar({ patient, predictData, isRefreshing, lastUpdated, countdown, error }) {
  const navigate = useNavigate()
  const prefersReduced = useReducedMotion()
  const connectionLost = Boolean(error) && !isRefreshing
  const riskColor = predictData?.risk_color

  /* Decide top accent gradient color */
  const accentColor = riskColor ?? 'var(--blue-l)'
  const isHighRisk = riskColor && riskColor !== '#0d8a52' && riskColor !== '#18c872' && riskColor !== 'var(--ok-l)'

  return (
    <header
      className="sticky top-0 z-50"
      style={{
        background: 'var(--s0)',
        backdropFilter: 'blur(24px) saturate(180%)',
        WebkitBackdropFilter: 'blur(24px) saturate(180%)',
        borderBottom: `1px solid ${isHighRisk ? riskColor + '22' : 'var(--b1)'}`,
        boxShadow: isHighRisk
          ? `0 1px 0 ${riskColor}18, 0 4px 32px rgba(0,0,0,0.28)`
          : '0 4px 32px rgba(0,0,0,0.22)',
      }}
    >
      {/* Top glow accent strip */}
      <div
        style={{
          position: 'absolute',
          top: 0, left: 0, right: 0,
          height: 2,
          background: isHighRisk
            ? `linear-gradient(90deg, transparent 0%, ${accentColor} 30%, ${accentColor}80 60%, transparent 100%)`
            : `linear-gradient(90deg, transparent 0%, var(--blue-l) 25%, var(--cyan-l) 55%, transparent 100%)`,
          opacity: isHighRisk ? 0.9 : 0.55,
          transition: 'background 0.5s ease, opacity 0.5s ease',
          zIndex: 1,
        }}
      />

      {/* Main bar — taller, more breathing room */}
      <div
        className="flex items-center justify-between gap-4"
        style={{ padding: '0 24px', height: 68 }}
      >
        {/* ── Left: brand + patient ──────────────────────────────────────── */}
        <div className="flex items-center gap-3 min-w-0 flex-shrink-0">
          <BrandMark />
          <PatientPill patient={patient} predictData={predictData} />
        </div>

        {/* ── Center: live chip (absolute-centered on large screens) ──────── */}
        <div className="flex items-center justify-center flex-1">
          <LiveChip
            isRefreshing={isRefreshing}
            countdown={countdown}
            connectionLost={connectionLost}
          />
        </div>

        {/* ── Right: time + theme + exit ─────────────────────────────────── */}
        <div className="flex items-center gap-3 flex-shrink-0">
          {/* Last updated time */}
          {lastUpdated && (
            <div
              className="hidden lg:flex flex-col items-end"
              style={{ gap: 1 }}
            >
              <span style={{ fontFamily: 'var(--font-display)', fontSize: '9px', fontWeight: 600, letterSpacing: '0.10em', textTransform: 'uppercase', color: 'var(--t4)' }}>
                Updated
              </span>
              <span className="num" style={{ fontSize: '12px', color: 'var(--t3)', letterSpacing: '-0.02em', fontFamily: 'var(--mono)' }}>
                {formatTime(lastUpdated)}
              </span>
            </div>
          )}

          {/* Divider */}
          <div className="hidden lg:block" style={{ width: 1, height: 28, background: 'var(--b1)' }} />

          {/* Theme toggle */}
          <ThemeToggle compact />

          {/* Exit button */}
          <motion.button
            onClick={() => navigate('/')}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-semibold"
            style={{
              border: '1px solid var(--b2)',
              color: 'var(--t2)',
              background: 'var(--s2)',
              fontFamily: 'var(--font-display)',
              cursor: 'pointer',
              fontSize: '12px',
              fontWeight: 700,
              letterSpacing: '0.02em',
              transition: 'border-color 0.18s, color 0.18s, background 0.18s, box-shadow 0.18s',
            }}
            whileHover={prefersReduced ? {} : { scale: 1.03 }}
            whileTap={prefersReduced ? {} : { scale: 0.95 }}
            transition={{ type: 'spring', stiffness: 440, damping: 28 }}
            onMouseEnter={e => {
              e.currentTarget.style.borderColor = 'var(--blue-l)'
              e.currentTarget.style.color = 'var(--blue-l)'
              e.currentTarget.style.background = 'rgba(32,112,204,0.08)'
              e.currentTarget.style.boxShadow = '0 0 0 3px rgba(32,112,204,0.12)'
            }}
            onMouseLeave={e => {
              e.currentTarget.style.borderColor = 'var(--b2)'
              e.currentTarget.style.color = 'var(--t2)'
              e.currentTarget.style.background = 'var(--s2)'
              e.currentTarget.style.boxShadow = 'none'
            }}
          >
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden="true">
              <path d="M13 8H3M7 4L3 8l4 4"/>
            </svg>
            Exit
          </motion.button>
        </div>
      </div>

      {/* ── Sub-bar: system status strip ─────────────────────────────────── */}
      <div
        style={{
          borderTop: '1px solid var(--b1)',
          padding: '6px 24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 28,
          background: 'var(--s0)',
          backdropFilter: 'blur(12px)',
        }}
      >
        {[
          connectionLost
            ? { dot: '#ef4444', label: 'Connection Issue' }
            : { dot: '#22c55e', label: 'All Systems Operational' },
          { dot: 'var(--t3)', label: patient ? '1 Active Patient' : 'No Active Patient' },
          { dot: 'var(--t3)', label: 'ICU Ward A' },
        ].map(({ dot, label }, i) => (
          <div
            key={i}
            className="flex items-center gap-1.5"
            style={{ color: 'var(--t3)', fontSize: '11px', fontFamily: 'var(--font-display)', fontWeight: 500, letterSpacing: '0.01em' }}
          >
            <span style={{
              width: 6, height: 6, borderRadius: '50%',
              background: dot,
              display: 'inline-block',
              flexShrink: 0,
              boxShadow: i === 0 ? '0 0 6px #22c55e80' : 'none',
            }} />
            {label}
          </div>
        ))}
      </div>
    </header>
  )
}
