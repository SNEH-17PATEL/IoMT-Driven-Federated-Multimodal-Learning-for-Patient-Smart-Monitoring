import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import React, { useEffect, useRef } from 'react'
import { motion, useReducedMotion } from 'framer-motion'
import { createTimeline } from 'animejs'
import { getPatients } from '../api/client.js'
import { useTheme } from '../contexts/ThemeContext.jsx'

const gridVariants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.14, delayChildren: 0.08 } },
}
const cardVariants = {
  hidden: { opacity: 0, y: 40, scale: 0.96 },
  visible: { opacity: 1, y: 0, scale: 1, transition: { type: 'spring', stiffness: 260, damping: 24, mass: 0.9 } },
}

/* ─── Background layers ─────────────────────────────────────────────────── */
const DARK_GRADIENTS = `
  radial-gradient(ellipse 65% 55% at 12% 12%, rgba(45,110,220,.22) 0%, transparent 55%),
  radial-gradient(ellipse 55% 50% at 88% 88%, rgba(0,150,140,.16) 0%, transparent 55%),
  radial-gradient(ellipse 45% 40% at 50% 45%, rgba(20,60,150,.10) 0%, transparent 55%),
  linear-gradient(155deg, #040c1a 0%, #081525 48%, #050e1f 100%)
`
const LIGHT_GRADIENTS = `
  radial-gradient(ellipse 65% 55% at 10% 10%, rgba(45,110,220,.10) 0%, transparent 55%),
  radial-gradient(ellipse 55% 50% at 90% 90%, rgba(0,150,140,.07) 0%, transparent 55%),
  linear-gradient(155deg, #f0f6ff 0%, #e8f2fb 50%, #edf5fd 100%)
`
const DARK_DOTS = `url("data:image/svg+xml,%3Csvg width='36' height='36' viewBox='0 0 36 36' xmlns='http://www.w3.org/2000/svg'%3E%3Ccircle cx='18' cy='18' r='1.2' fill='rgba(80,140,255,0.14)'/%3E%3C/svg%3E")`
const LIGHT_DOTS = `url("data:image/svg+xml,%3Csvg width='36' height='36' viewBox='0 0 36 36' xmlns='http://www.w3.org/2000/svg'%3E%3Ccircle cx='18' cy='18' r='1.2' fill='rgba(45,100,200,0.09)'/%3E%3C/svg%3E")`

/* ─── Per-theme token helpers ───────────────────────────────────────────── */
function t(isDark, dark, light) { return isDark ? dark : light }

/* --- Live clock ----------------------------------------------------------- */
function useClock() {
  const [now, setNow] = React.useState(new Date())
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(id)
  }, [])
  return now
}

/* --- ECG waveform for nav ------------------------------------------------- */
function NavEcg() {
  return (
    <svg viewBox="0 0 64 18" width="64" height="18" aria-hidden="true" style={{ overflow: 'visible' }}>
      <path
        d="M0,9 L13,9 L17,8 L21,2 L25,16 L29,9 L33,10 L37,9 L64,9"
        fill="none" stroke="#e53535" strokeWidth="1.5"
        strokeLinecap="round" strokeLinejoin="round"
        style={{ strokeDasharray: '36 100', strokeDashoffset: 0, animation: 'ecgScroll 1.8s linear infinite', opacity: 0.85 }}
      />
    </svg>
  )
}

/* --- Top navigation ------------------------------------------------------- */
function TopNav({ isDark, toggleTheme }) {
  const now = useClock()
  const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  const dateStr = now.toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric' })

  const bg      = t(isDark, 'rgba(4,12,26,0.93)', 'rgba(255,255,255,0.93)')
  const borderC = t(isDark, 'rgba(255,255,255,0.08)', 'rgba(0,0,0,0.09)')
  const nameCol = t(isDark, 'rgba(255,255,255,0.97)', 'var(--t1)')
  const subCol  = t(isDark, 'rgba(255,255,255,0.38)', 'var(--t4)')
  const divCol  = t(isDark, 'rgba(255,255,255,0.10)', 'rgba(0,0,0,0.10)')
  const timeCol = t(isDark, 'rgba(255,255,255,0.58)', 'var(--t3)')

  return (
    <nav className="sticky top-0 z-50" style={{
      background: bg,
      backdropFilter: 'blur(28px) saturate(180%)',
      WebkitBackdropFilter: 'blur(28px) saturate(180%)',
      borderBottom: `1px solid ${borderC}`,
      boxShadow: isDark ? '0 4px 32px rgba(0,0,0,0.38)' : '0 4px 24px rgba(0,0,0,0.08)',
      position: 'sticky', top: 0, zIndex: 50,
    }}>
      {/* Top shimmer accent strip */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: 2,
        background: 'linear-gradient(90deg, transparent 0%, #2d82dc 22%, #0891b2 50%, #22d3ee 72%, transparent 100%)',
        opacity: isDark ? 0.72 : 0.48,
        zIndex: 1,
      }} />

      <div style={{ height: 72, display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 28px' }}>

        {/* Left: brand mark */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 14, flexShrink: 0 }}>
          <div style={{
            position: 'relative', width: 46, height: 46, borderRadius: 14,
            background: 'linear-gradient(135deg, #2d82dc 0%, #0891b2 100%)',
            boxShadow: '0 0 0 1px rgba(255,255,255,0.18) inset, 0 6px 24px rgba(45,130,220,0.52), 0 2px 8px rgba(14,165,201,0.30)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
          }}>
            <svg width="22" height="22" viewBox="0 0 22 22" fill="none" aria-hidden="true">
              <path d="M1,11 L6,11 L8,6 L10.5,17 L13,11 L16,11 L18,8 L19.5,11 L21,11"
                stroke="white" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            <div style={{ position: 'absolute', top: 5, left: 7, right: 14, height: 4, borderRadius: 4, background: 'rgba(255,255,255,0.28)', filter: 'blur(1px)' }} />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <span style={{ fontFamily: 'var(--font-display)', fontSize: '20px', fontWeight: 800, letterSpacing: '-0.04em', lineHeight: 1, color: nameCol }}>
              ClinIQ
            </span>
            <span style={{ fontFamily: 'var(--font-display)', fontSize: '9px', fontWeight: 600, letterSpacing: '0.14em', textTransform: 'uppercase', color: subCol, lineHeight: 1 }}>
              ICU Intelligence
            </span>
          </div>
        </div>

        {/* Center: LIVE chip */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10,
          padding: '9px 20px', borderRadius: 14,
          background: 'rgba(220,38,38,0.10)',
          border: '1px solid rgba(220,38,38,0.28)',
          boxShadow: '0 0 20px rgba(220,38,38,0.12), inset 0 1px 0 rgba(255,255,255,0.04)',
        }}>
          <span style={{ position: 'relative', display: 'flex', width: 9, height: 9, flexShrink: 0 }}>
            <span className="animate-ping" style={{ position: 'absolute', inset: 0, borderRadius: '50%', background: '#e53535', opacity: 0.50, display: 'block' }} />
            <span style={{ position: 'relative', width: 9, height: 9, borderRadius: '50%', background: '#e53535', boxShadow: '0 0 8px #e53535', display: 'block' }} />
          </span>
          <NavEcg />
          <span style={{ fontFamily: 'var(--font-display)', fontSize: '11px', fontWeight: 900, letterSpacing: '0.18em', color: '#e53535' }}>LIVE</span>
        </div>

        {/* Right: clock + theme toggle */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexShrink: 0 }}>
          <div className="hidden lg:flex flex-col items-end" style={{ gap: 2 }}>
            <span style={{ fontFamily: 'var(--mono)', fontSize: '13px', fontWeight: 600, letterSpacing: '-0.02em', color: timeCol }}>{timeStr}</span>
            <span style={{ fontFamily: 'var(--font-display)', fontSize: '9px', fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', color: subCol }}>{dateStr}</span>
          </div>
          <div className="hidden lg:block" style={{ width: 1, height: 28, background: divCol }} />
          <button
            onClick={toggleTheme}
            aria-label={`Switch to ${isDark ? 'light' : 'dark'} mode`}
            style={{
              display: 'flex', alignItems: 'center', gap: 8,
              padding: '8px 16px', borderRadius: 10,
              background: isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.05)',
              border: `1px solid ${isDark ? 'rgba(255,255,255,0.14)' : 'rgba(0,0,0,0.12)'}`,
              color: isDark ? 'rgba(255,255,255,0.75)' : 'var(--t2)',
              cursor: 'pointer', fontFamily: 'var(--font-display)',
              fontSize: '12px', fontWeight: 700, letterSpacing: '0.02em',
              transition: 'all 0.18s ease',
            }}
          >
            {isDark ? (
              <svg width="14" height="14" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
                <circle cx="10" cy="10" r="4"/>
                <line x1="10" y1="1.5" x2="10" y2="3.5"/>
                <line x1="10" y1="16.5" x2="10" y2="18.5"/>
                <line x1="1.5" y1="10" x2="3.5" y2="10"/>
                <line x1="16.5" y1="10" x2="18.5" y2="10"/>
                <line x1="4.2" y1="4.2" x2="5.6" y2="5.6"/>
                <line x1="14.4" y1="14.4" x2="15.8" y2="15.8"/>
              </svg>
            ) : (
              <svg width="14" height="14" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
                <path d="M17.5 12.5A8 8 0 0 1 7.5 2.5a7 7 0 1 0 10 10z"/>
              </svg>
            )}
            {isDark ? 'Light' : 'Dark'}
          </button>
        </div>
      </div>
    </nav>
  )
}

/* --- System status bar ---------------------------------------------------- */
function SystemStatusBar({ isDark, patientCount }) {
  const bg     = t(isDark, 'rgba(0,0,0,0.22)', 'rgba(0,0,0,0.03)')
  const borderC = t(isDark, '1px solid rgba(255,255,255,0.06)', '1px solid rgba(0,0,0,0.06)')
  const divCol = t(isDark, 'rgba(255,255,255,0.12)', 'rgba(0,0,0,0.12)')
  const sub    = t(isDark, 'rgba(255,255,255,0.48)', 'var(--t3)')
  const strong = t(isDark, 'rgba(255,255,255,0.82)', 'var(--t2)')

  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      height: 34, background: bg, borderBottom: borderC,
      fontFamily: 'var(--font-display)', gap: 0,
    }}>
      {/* Operational */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: isDark ? 'rgba(80,220,120,0.88)' : '#16a34a', fontSize: '11px', fontWeight: 600 }}>
        <span className="animate-pulse" style={{ width: 6, height: 6, borderRadius: '50%', background: '#22c55e', display: 'inline-block', boxShadow: '0 0 7px rgba(34,197,94,0.75)' }} />
        All Systems Operational
      </div>

      <div style={{ width: 1, height: 14, background: divCol, margin: '0 20px' }} />

      {/* Patient count */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: sub, fontSize: '11px', fontWeight: 600 }}>
        <span style={{ width: 6, height: 6, borderRadius: '50%', background: isDark ? 'rgba(255,255,255,0.30)' : 'rgba(0,0,0,0.22)', display: 'inline-block' }} />
        <span style={{ color: strong, fontWeight: 700 }}>{patientCount ?? 3}</span>&nbsp;Active Patients
      </div>

      <div style={{ width: 1, height: 14, background: divCol, margin: '0 20px' }} />

      {/* Ward */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: sub, fontSize: '11px', fontWeight: 600 }}>
        <span style={{ width: 6, height: 6, borderRadius: '50%', background: isDark ? 'rgba(255,255,255,0.30)' : 'rgba(0,0,0,0.22)', display: 'inline-block' }} />
        ICU Ward A
      </div>
    </div>
  )
}


/* ─── ECG hero line ─────────────────────────────────────────────────────── */
function EcgHeroLine({ isDark }) {
  const trackColor = t(isDark, 'rgba(255,255,255,0.08)', 'rgba(45,130,220,0.12)')
  const lineColor  = t(isDark, 'rgba(45,130,220,0.90)', 'rgba(45,130,220,0.85)')
  const glowFilter = isDark
    ? 'drop-shadow(0 0 6px rgba(45,130,220,0.70)) drop-shadow(0 0 16px rgba(45,130,220,0.30))'
    : 'drop-shadow(0 0 4px rgba(45,130,220,0.45))'
  const path = "M0,28 L55,28 L68,25 L80,7 L90,49 L100,28 L110,31 L124,28 L180,28 L193,25 L205,7 L215,49 L225,28 L235,31 L249,28 L330,28 L343,25 L355,7 L365,49 L375,28 L385,31 L399,28 L480,28 L493,25 L505,7 L515,49 L525,28 L535,31 L549,28 L640,28"

  return (
    <div style={{ width: '100%', maxWidth: 640, margin: '28px auto 0', height: 56, overflow: 'hidden' }}>
      <svg viewBox="0 0 640 56" width="100%" height="56" aria-hidden="true" preserveAspectRatio="none">
        <path d={path} fill="none" stroke={trackColor} strokeWidth="1.5" strokeLinecap="round" />
        <path d={path} fill="none" stroke={lineColor} strokeWidth="2.2" strokeLinecap="round"
          style={{ strokeDasharray: 750, strokeDashoffset: 750, animation: 'ecgDraw 2.8s ease-out 0.4s forwards', filter: glowFilter }} />
      </svg>
    </div>
  )
}

/* ─── Live ECG in patient card ──────────────────────────────────────────── */
function CardEcg({ color }) {
  const path = "M0,15 L18,15 L22,13 L27,3 L32,27 L37,15 L42,17 L48,15 L90,15 L94,13 L99,3 L104,27 L109,15 L114,17 L120,15 L170,15 L174,13 L179,3 L184,27 L189,15 L194,17 L200,15 L240,15"
  return (
    <div style={{ width: '100%', height: 30, overflow: 'hidden', opacity: 0.80 }}>
      <svg viewBox="0 0 240 30" width="100%" height="30" aria-hidden="true" preserveAspectRatio="none">
        <path d={path} fill="none" stroke="rgba(128,128,128,0.20)" strokeWidth="1.2" strokeLinecap="round" />
        <path d={path} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round"
          style={{ strokeDasharray: '80 160', strokeDashoffset: 0, animation: 'ecgScroll 2.2s linear infinite', filter: `drop-shadow(0 0 4px ${color})` }} />
      </svg>
    </div>
  )
}

/* ─── Patient card ──────────────────────────────────────────────────────── */
function PatientCard({ patient, isDark }) {
  const navigate = useNavigate()
  const prefersReduced = useReducedMotion()
  const c = patient.color

  const cardBg     = t(isDark, 'rgba(255,255,255,0.055)', 'rgba(255,255,255,0.95)')
  const cardBorder = t(isDark, `1px solid rgba(255,255,255,0.09)`, `1px solid ${c}30`)
  const cardShadow = t(isDark,
    `0 24px 64px rgba(0,0,0,0.50), 0 4px 20px rgba(0,0,0,0.30), inset 0 1px 0 rgba(255,255,255,0.10), 0 0 0 1px ${c}20`,
    `0 8px 32px rgba(0,0,0,0.08), 0 2px 8px rgba(0,0,0,0.04), 0 0 0 1px ${c}15`
  )
  const cardShadowHover = t(isDark,
    `0 36px 80px rgba(0,0,0,0.60), 0 8px 28px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.15), 0 0 0 2px ${c}40`,
    `0 16px 48px rgba(0,0,0,0.12), 0 4px 16px rgba(0,0,0,0.07), 0 0 0 2px ${c}30`
  )
  const nameCl   = t(isDark, 'rgba(255,255,255,0.95)', 'var(--t1)')
  const descCl   = t(isDark, 'rgba(255,255,255,0.55)', 'var(--t3)')
  const chipBg   = t(isDark, 'rgba(255,255,255,0.08)', 'var(--s2)')
  const chipBorder = t(isDark, '1px solid rgba(255,255,255,0.14)', '1px solid var(--b1)')
  const chipText = t(isDark, 'rgba(255,255,255,0.70)', 'var(--t2)')
  const chipSub  = t(isDark, 'rgba(255,255,255,0.40)', 'var(--t3)')
  const ecgBg    = t(isDark, 'rgba(0,0,0,0.30)', 'rgba(0,0,0,0.04)')
  const ecgBorderStyle = t(isDark, '1px solid rgba(255,255,255,0.07)', '1px solid rgba(0,0,0,0.06)')

  function go() { navigate(`/monitor/${patient.id}`) }

  return (
    <motion.div
      className="rounded-2xl overflow-hidden cursor-pointer flex flex-col"
      style={{
        background: cardBg,
        backdropFilter: isDark ? 'blur(28px) saturate(130%)' : 'none',
        WebkitBackdropFilter: isDark ? 'blur(28px) saturate(130%)' : 'none',
        border: cardBorder,
        boxShadow: cardShadow,
      }}
      onClick={go}
      role="button"
      aria-label={`Monitor ${patient.name}`}
      tabIndex={0}
      onKeyDown={e => e.key === 'Enter' && go()}
      whileHover={prefersReduced ? {} : {
        y: -8, scale: 1.016,
        transition: { type: 'spring', stiffness: 340, damping: 26 },
      }}
      whileTap={prefersReduced ? {} : {
        scale: 0.975, y: -3,
        transition: { duration: 0.08 },
      }}
      onMouseEnter={e => { e.currentTarget.style.boxShadow = cardShadowHover }}
      onMouseLeave={e => { e.currentTarget.style.boxShadow = cardShadow }}
    >
      {/* Colored top strip */}
      <div style={{ height: 4, background: `linear-gradient(to right, ${c}, ${c}60, transparent)`, flexShrink: 0 }} />

      <div className="flex flex-col flex-1 p-7">
        {/* Avatar + status */}
        <div className="flex items-center justify-between mb-6">
          <div className="relative">
            <div className="flex items-center justify-center rounded-2xl"
              style={{ width: 68, height: 68, background: `linear-gradient(135deg, ${c}35, ${c}15)`, border: `2px solid ${c}50`, color: c, fontFamily: 'var(--font-display)', fontSize: '1.8rem', fontWeight: 800, boxShadow: `0 8px 24px ${c}25, inset 0 1px 0 ${c}40`, letterSpacing: '-0.03em' }}>
              0{patient.id}
            </div>
            <span className="absolute -top-1.5 -right-1.5 flex h-5 w-5" aria-hidden="true">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-40" style={{ background: c }} />
              <span className="relative inline-flex h-5 w-5 rounded-full border-2" style={{ background: c, borderColor: isDark ? 'rgba(255,255,255,0.15)' : 'rgba(255,255,255,0.80)' }} />
            </span>
          </div>
          <div className="pill" style={{ background: `${c}22`, color: c, border: `1px solid ${c}45`, fontSize: '10px', boxShadow: `0 0 12px ${c}20`, fontFamily: 'var(--font-display)' }}>
            {patient.risk_level}
          </div>
        </div>

        {/* Name + description */}
        <h2 className="text-2xl font-bold mb-2 leading-tight" style={{ color: nameCl, fontFamily: 'var(--font-display)', letterSpacing: '-0.02em' }}>
          {patient.name}
        </h2>
        <p className="text-sm leading-relaxed mb-5" style={{ color: descCl }}>
          {patient.description}
        </p>

        {/* Info chips */}
        <div className="flex items-center gap-2 flex-wrap mb-5">
          <span className="text-xs font-semibold px-3 py-1 rounded-lg"
            style={{ background: chipBg, border: chipBorder, color: chipText, fontFamily: 'var(--font-display)' }}>
            <span style={{ color: chipSub, marginRight: 4, fontSize: '9px' }}>SOFA</span>
            {patient.sofa_range}
          </span>
          <span className="text-xs font-semibold px-3 py-1 rounded-lg"
            style={{ background: chipBg, border: chipBorder, color: chipText, fontFamily: 'var(--font-display)' }}>
            ICU Bay {patient.id}
          </span>
        </div>

        {/* Live ECG strip */}
        <div className="mb-5 rounded-xl overflow-hidden" style={{ background: ecgBg, border: ecgBorderStyle, padding: '8px 10px' }}>
          <CardEcg color={c} />
        </div>

        {/* CTA */}
        <motion.button
          onClick={e => { e.stopPropagation(); go() }}
          className="w-full py-3.5 rounded-xl text-sm font-bold tracking-wide flex items-center justify-center gap-2.5"
          style={{
            background: isDark ? `linear-gradient(135deg, ${c}28, ${c}12)` : `linear-gradient(135deg, ${c}18, ${c}0c)`,
            border: `1px solid ${c}45`,
            color: c,
            fontFamily: 'var(--font-display)',
            cursor: 'pointer',
            letterSpacing: '0.02em',
          }}
          whileHover={prefersReduced ? {} : { scale: 1.03 }}
          whileTap={prefersReduced ? {} : { scale: 0.96 }}
          transition={{ type: 'spring', stiffness: 420, damping: 28 }}
          onMouseEnter={e => {
            e.currentTarget.style.background = `linear-gradient(135deg, ${c}90, ${c}70)`
            e.currentTarget.style.color = '#ffffff'
            e.currentTarget.style.boxShadow = `0 6px 24px ${c}45`
          }}
          onMouseLeave={e => {
            e.currentTarget.style.background = isDark ? `linear-gradient(135deg, ${c}28, ${c}12)` : `linear-gradient(135deg, ${c}18, ${c}0c)`
            e.currentTarget.style.color = c
            e.currentTarget.style.boxShadow = 'none'
          }}
        >
          Open Monitoring Dashboard
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M3 8h10M9 4l4 4-4 4"/>
          </svg>
        </motion.button>
      </div>
    </motion.div>
  )
}

/* ─── Hero section with anime.js timeline ──────────────────────────────── */
function HeroSection({ isDark, heroText, heroSub, heroAccent }) {
  const iconRef     = useRef(null)
  const titleRef    = useRef(null)
  const badgeRef    = useRef(null)
  const subtitleRef = useRef(null)
  const ecgRef      = useRef(null)

  useEffect(() => {
    const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (prefersReduced) {
      ;[iconRef, titleRef, badgeRef, subtitleRef, ecgRef].forEach(r => {
        if (r.current) r.current.style.opacity = 1
      })
      return
    }

    ;[iconRef, titleRef, badgeRef, subtitleRef, ecgRef].forEach(r => {
      if (r.current) { r.current.style.opacity = '0' }
    })

    const tl = createTimeline({ ease: 'outExpo' })

    tl.add(iconRef.current, {
      opacity: [0, 1],
      scale: [0.55, 1],
      rotate: [-6, 0],
      duration: 700,
      delay: 100,
    })
    .add(titleRef.current, {
      opacity: [0, 1],
      translateY: [22, 0],
      duration: 580,
    }, '-=380')
    .add(badgeRef.current, {
      opacity: [0, 1],
      translateX: [-12, 0],
      duration: 420,
    }, '-=260')
    .add(ecgRef.current, {
      opacity: [0, 1],
      duration: 300,
    }, '-=180')
    .add(subtitleRef.current, {
      opacity: [0, 1],
      translateY: [10, 0],
      duration: 480,
    }, '-=80')

    return () => tl.pause()
  }, [])

  return (
    <div className="text-center mb-16">
      <div className="flex items-center justify-center gap-4 mb-2">
        <div
          ref={iconRef}
          className="flex items-center justify-center rounded-2xl"
          style={{
            width: 60, height: 60,
            background: 'linear-gradient(135deg, #2d82dc, #0891b2)',
            boxShadow: '0 12px 36px rgba(45,130,220,0.50), 0 4px 14px rgba(0,0,0,0.25)',
            flexShrink: 0,
          }}
        >
          <svg width="32" height="32" viewBox="0 0 32 32" fill="none" aria-hidden="true">
            <path d="M2,16 L8,16 L11,9 L14,23 L17,16 L20,16 L23,12 L25,16 L30,16"
              stroke="white" strokeWidth="2.8" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
        <div className="text-left">
          <h1
            ref={titleRef}
            className="font-bold leading-none"
            style={{ fontSize: 'clamp(2.8rem, 7vw, 4.5rem)', color: heroText, fontFamily: 'var(--font-display)', letterSpacing: '-0.04em' }}
          >
            ClinIQ
          </h1>
          <p
            ref={badgeRef}
            className="text-xs font-semibold mt-1"
            style={{ color: heroAccent, letterSpacing: '0.08em', fontFamily: 'var(--font-display)' }}
          >
            ICU Clinical Intelligence
          </p>
        </div>
      </div>

      <div ref={ecgRef}>
        <EcgHeroLine isDark={isDark} />
      </div>

      <p
        ref={subtitleRef}
        className="mt-6 text-base"
        style={{ color: heroSub }}
      >
        Select a patient to open their real-time monitoring dashboard
      </p>
    </div>
  )
}

/* ─── Skeleton ──────────────────────────────────────────────────────────── */
function SkeletonCard({ isDark }) {
  const bg     = t(isDark, 'rgba(255,255,255,0.04)', 'rgba(255,255,255,0.80)')
  const border = t(isDark, '1px solid rgba(255,255,255,0.07)', '1px solid rgba(0,0,0,0.06)')
  const el     = t(isDark, 'rgba(255,255,255,0.10)', 'rgba(0,0,0,0.07)')

  return (
    <div className="rounded-2xl overflow-hidden" style={{ background: bg, border, backdropFilter: isDark ? 'blur(20px)' : 'none' }}>
      <div className="h-1" style={{ background: el }} />
      <div className="p-7 space-y-5">
        <div className="flex justify-between items-center">
          <div style={{ width: 68, height: 68, borderRadius: 16, background: el }} />
          <div className="w-24 h-6 rounded-full" style={{ background: el }} />
        </div>
        <div>
          <div className="h-7 w-3/4 rounded-lg mb-2" style={{ background: el }} />
          <div className="h-4 w-full rounded mb-1" style={{ background: el }} />
          <div className="h-4 w-4/5 rounded" style={{ background: el }} />
        </div>
        <div className="flex gap-2">
          <div className="h-8 w-24 rounded-lg" style={{ background: el }} />
          <div className="h-8 w-20 rounded-lg" style={{ background: el }} />
        </div>
        <div className="h-10 rounded-xl" style={{ background: el }} />
        <div className="h-12 rounded-xl" style={{ background: el }} />
      </div>
    </div>
  )
}

/* ─── Page ──────────────────────────────────────────────────────────────── */
export default function SelectPatient() {
  const { theme, toggleTheme } = useTheme()
  const isDark = theme === 'dark'

  const { data: patients, isLoading, isError } = useQuery({
    queryKey: ['patients'],
    queryFn: () => getPatients().then(r => r.data),
  })

  const heroText  = t(isDark, 'rgba(255,255,255,0.97)', 'var(--t1)')
  const heroSub   = t(isDark, 'rgba(255,255,255,0.55)', 'var(--t3)')
  const heroAccent = t(isDark, 'rgba(45,130,220,0.90)', 'var(--blue-l)')
  const footerCl  = t(isDark, 'rgba(255,255,255,0.22)', 'var(--t4)')
  const errorBg   = t(isDark, 'rgba(176,32,48,0.18)', 'rgba(254,226,226,1)')
  const errorBorder = t(isDark, '1px solid rgba(220,50,70,0.40)', '1px solid rgba(220,50,70,0.30)')
  const errorText = t(isDark, '#ff8080', '#dc2626')

  return (
    <div className="min-h-screen" style={{ backgroundColor: isDark ? '#040c1a' : '#f0f6ff' }}>
      {/* Fixed: hero gradient */}
      <div style={{ position: 'fixed', inset: 0, backgroundImage: isDark ? DARK_GRADIENTS : LIGHT_GRADIENTS, backgroundSize: 'cover', pointerEvents: 'none', zIndex: 0 }} />
      {/* Fixed: dot grid */}
      <div style={{ position: 'fixed', inset: 0, backgroundImage: isDark ? DARK_DOTS : LIGHT_DOTS, backgroundSize: '36px 36px', pointerEvents: 'none', zIndex: 0 }} />
      {/* Fixed: floating ambient orbs */}
      <div style={{ position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 0, overflow: 'hidden' }}>
        {[
          { w: 500, h: 500, x: '5%', y: '10%', c: isDark ? 'rgba(45,130,220,0.08)' : 'rgba(45,130,220,0.06)', d: '18s', dd: '0s' },
          { w: 380, h: 380, x: '75%', y: '60%', c: isDark ? 'rgba(0,185,160,0.06)' : 'rgba(0,185,160,0.05)', d: '24s', dd: '6s' },
          { w: 280, h: 280, x: '55%', y: '5%',  c: isDark ? 'rgba(100,80,220,0.07)' : 'rgba(100,80,220,0.04)', d: '20s', dd: '12s' },
          { w: 200, h: 200, x: '15%', y: '70%', c: isDark ? 'rgba(220,80,100,0.05)' : 'rgba(220,80,100,0.04)', d: '16s', dd: '3s' },
        ].map((orb, i) => (
          <div key={i} style={{
            position: 'absolute',
            width: orb.w, height: orb.h,
            left: orb.x, top: orb.y,
            borderRadius: '50%',
            background: `radial-gradient(circle, ${orb.c} 0%, transparent 70%)`,
            animation: `floatY ${orb.d} ease-in-out ${orb.dd} infinite`,
            filter: 'blur(40px)',
          }} />
        ))}
      </div>

      <div style={{ position: 'relative', zIndex: 1 }}>
        <TopNav isDark={isDark} toggleTheme={toggleTheme} />
        <SystemStatusBar isDark={isDark} patientCount={patients?.length} />

        <div className="max-w-6xl mx-auto px-6 py-14 sm:py-20">

          {/* ── Hero ───────────────────────────────────────────────────── */}
          <HeroSection isDark={isDark} heroText={heroText} heroSub={heroSub} heroAccent={heroAccent} />

          {/* Error */}
          {isError && (
            <div className="rounded-xl px-5 py-4 mb-8 text-sm text-center"
              style={{ background: errorBg, border: errorBorder, color: errorText, fontFamily: 'var(--font-display)' }}>
              ⚠ Unable to load patients — ensure the API server is running on port 8000.
            </div>
          )}

          {/* ── Patient grid ───────────────────────────────────────────── */}
          <motion.div
            className="grid grid-cols-1 md:grid-cols-3 gap-6"
            role="list"
            aria-label="Patient list"
            variants={gridVariants}
            initial="hidden"
            animate="visible"
          >
            {isLoading
              ? Array.from({ length: 3 }, (_, i) => <SkeletonCard key={i} isDark={isDark} />)
              : patients?.map((p, i) => (
                  <motion.div
                    key={p.id}
                    variants={cardVariants}
                    data-aos="fade-up"
                    data-aos-delay={i * 90}
                    data-aos-duration="600"
                  >
                    <PatientCard patient={p} isDark={isDark} />
                  </motion.div>
                ))
            }
          </motion.div>

          {/* Footer */}
          <p className="text-center text-[11px] mt-16"
            style={{ color: footerCl, fontFamily: 'var(--font-display)', letterSpacing: '0.04em' }}>
            Research prototype · MIMIC-III dataset · Not for clinical use
          </p>
        </div>
      </div>
    </div>
  )
}
