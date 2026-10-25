import { useState, useEffect, useRef } from 'react'
import { animate } from 'animejs'


const BG = 'linear-gradient(165deg, #07111e 0%, #0c1828 55%, #091420 100%)'

function ArcGauge({ score, maxScore = 24, color, isRefreshing }) {
  const prefersReduced = typeof window !== 'undefined' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches
  const [ready, setReady] = useState(prefersReduced)

  useEffect(() => {
    if (prefersReduced) return
    const t = setTimeout(() => setReady(true), 180)
    return () => clearTimeout(t)
  }, [])

  const r = 52, cx = 66, cy = 66
  const C = 2 * Math.PI * r
  const trackArc = C * 0.75
  const pct = Math.min((score ?? 0) / maxScore, 1)
  const filled = ready ? pct * trackArc : 0

  /* Zone markers at 6, 9, 12 (significant SOFA thresholds) */
  const markers = [6, 9, 12].map(v => {
    const t = v / maxScore
    const angle = -225 + t * 270
    const rad = (angle * Math.PI) / 180
    const inner = 43, outer = 48
    return {
      x1: cx + inner * Math.cos(rad),
      y1: cy + inner * Math.sin(rad),
      x2: cx + outer * Math.cos(rad),
      y2: cy + outer * Math.sin(rad),
    }
  })

  return (
    <svg viewBox="0 0 132 132" width="168" height="168" aria-hidden="true">
      {/* Outer ambient glow ring for high risk */}
      {color !== '#0d8a52' && color !== '#18c872' && (
        <circle cx={cx} cy={cy} r={r + 16} fill="none"
          stroke={color} strokeWidth="1" opacity="0.07"
          style={{ filter: 'blur(6px)' }} />
      )}

      {/* Background track */}
      <circle
        cx={cx} cy={cy} r={r}
        fill="none"
        stroke="rgba(255,255,255,0.09)"
        strokeWidth="10"
        strokeLinecap="round"
        strokeDasharray={`${trackArc} ${C - trackArc}`}
        transform={`rotate(-225 ${cx} ${cy})`}
      />

      {/* Filled arc */}
      <circle
        cx={cx} cy={cy} r={r}
        fill="none"
        stroke={color}
        strokeWidth="10"
        strokeLinecap="round"
        strokeDasharray={`${filled} ${C - filled}`}
        transform={`rotate(-225 ${cx} ${cy})`}
        className={isRefreshing ? 'animate-pulse' : ''}
        style={{
          transition: 'stroke-dasharray 1.1s cubic-bezier(0.34,1.08,0.64,1), stroke 0.45s ease',
          filter: `drop-shadow(0 0 10px ${color}80) drop-shadow(0 0 24px ${color}30)`,
        }}
      />

      {/* SOFA threshold markers */}
      {markers.map((mk, i) => (
        <line key={i} x1={mk.x1} y1={mk.y1} x2={mk.x2} y2={mk.y2}
          stroke="rgba(255,255,255,0.18)" strokeWidth="1.2" strokeLinecap="round" />
      ))}
    </svg>
  )
}

export default function SOFAGauge({ predictData, isRefreshing }) {
  const { sofa_score, risk_level, risk_color, conformal_lo, conformal_hi } = predictData
  const scoreRef = useRef(null)
  const prevScore = useRef(0)

  const confPct = conformal_lo != null && conformal_hi != null && conformal_hi > conformal_lo
    ? Math.min(100, (((sofa_score ?? 0) - conformal_lo) / (conformal_hi - conformal_lo)) * 100)
    : 0

  /* Animate score number from previous value to new value */
  useEffect(() => {
    if (sofa_score == null || !scoreRef.current) return
    const from = prevScore.current
    const to = sofa_score
    prevScore.current = to

    const obj = { val: from }
    const anim = animate(obj, {
      val: to,
      duration: 1600,
      delay: 250,
      ease: 'outExpo',
      onUpdate() {
        if (scoreRef.current) {
          scoreRef.current.textContent = obj.val.toFixed(1)
        }
      },
    })
    return () => anim.pause()
  }, [sofa_score])

  return (
    <div
      className="flex flex-col items-center justify-center relative overflow-hidden"
      style={{
        background: BG,
        border: `1px solid rgba(255,255,255,0.07)`,
        borderRadius: 'var(--r-xl)',
        padding: '20px 20px 22px',
        boxShadow: `
          0 28px 72px rgba(0,0,0,0.42),
          0 8px 28px rgba(0,0,0,0.24),
          0 0 0 1px rgba(255,255,255,0.05),
          0 0 0 4px ${risk_color}10
        `,
        transition: 'box-shadow 0.38s var(--ease-out-smooth)',
      }}
    >
      {/* Ambient radial glow */}
      <div
        style={{
          position: 'absolute',
          top: '35%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: 150,
          height: 150,
          borderRadius: '50%',
          background: `radial-gradient(circle, ${risk_color}18 0%, transparent 65%)`,
          pointerEvents: 'none',
          transition: 'background 0.45s ease',
        }}
      />

      {/* Top accent line */}
      <div style={{
        position: 'absolute',
        top: 0, left: 0, right: 0,
        height: 2,
        background: `linear-gradient(to right, ${risk_color}, ${risk_color}50, transparent 75%)`,
        borderRadius: 'var(--r-xl) var(--r-xl) 0 0',
      }} />

      {/* Label */}
      <p
        className="self-start mb-3"
        style={{
          fontSize: '10px',
          fontWeight: 700,
          letterSpacing: '0.14em',
          textTransform: 'uppercase',
          color: 'rgba(255,255,255,0.45)',
          fontFamily: 'var(--font-display)',
        }}
      >
        Predicted SOFA
      </p>

      {/* Arc gauge + center */}
      <div className="relative flex items-center justify-center">
        <ArcGauge
          score={sofa_score ?? 0}
          maxScore={24}
          color={risk_color}
          isRefreshing={isRefreshing}
        />
        <div
          className="absolute inset-0 flex flex-col items-center justify-center"
          style={{ paddingBottom: 10 }}
        >
          <span
            ref={scoreRef}
            className="num font-bold leading-none"
            style={{
              fontSize: '3.2rem',
              color: risk_color,
              transition: 'color 0.45s',
              letterSpacing: '-0.05em',
              filter: `drop-shadow(0 0 14px ${risk_color}55)`,
            }}
          >
            {sofa_score?.toFixed(1)}
          </span>
          <span style={{ fontSize: '11px', color: 'rgba(255,255,255,0.35)', fontFamily: 'var(--font-display)', marginTop: 2 }}>
            out of 24
          </span>
        </div>
      </div>

      {/* Risk badge */}
      <div
        className="pill mt-1 mb-5"
        style={{
          background: `${risk_color}1e`,
          color: risk_color,
          border: `1px solid ${risk_color}45`,
          fontSize: '11px',
          fontFamily: 'var(--font-display)',
          fontWeight: 700,
          letterSpacing: '0.06em',
          boxShadow: `0 4px 16px ${risk_color}20`,
        }}
      >
        <span
          style={{
            width: 6, height: 6, borderRadius: '50%',
            background: risk_color,
            display: 'inline-block',
            boxShadow: `0 0 6px ${risk_color}`,
            flexShrink: 0,
          }}
        />
        {risk_level}
      </div>

      {/* Conformal interval */}
      <div className="w-full space-y-2">
        <div className="flex items-center justify-between">
          <p
            style={{ fontSize: '10px', color: 'rgba(255,255,255,0.35)', fontFamily: 'var(--font-display)', fontWeight: 500 }}
          >
            90% prediction interval
          </p>
          <span className="num text-[10px]" style={{ color: 'rgba(255,255,255,0.35)' }}>
            {conformal_lo?.toFixed(1)} – {conformal_hi?.toFixed(1)}
          </span>
        </div>
        <div
          className="rounded-full overflow-hidden"
          style={{ height: 5, background: 'rgba(255,255,255,0.08)' }}
        >
          <div
            className="h-full rounded-full"
            style={{
              width: `${confPct}%`,
              background: `linear-gradient(to right, ${risk_color}55, ${risk_color})`,
              transition: 'width 0.6s var(--ease)',
              boxShadow: `0 0 6px ${risk_color}50`,
            }}
          />
        </div>
      </div>
    </div>
  )
}
