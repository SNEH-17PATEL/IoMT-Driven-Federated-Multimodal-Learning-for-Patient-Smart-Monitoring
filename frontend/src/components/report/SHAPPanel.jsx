import { useEffect, useRef, useState } from 'react'
import { animate } from 'animejs'

export default function SHAPPanel({ factors }) {
  /* Keep the last non-empty factors so the panel never vanishes during a
     refresh where SHAP computation temporarily returns []. */
  const [displayFactors, setDisplayFactors] = useState(factors)
  useEffect(() => {
    if (factors?.length) setDisplayFactors(factors)
  }, [factors])

  const panelRef = useRef(null)
  const barsRef  = useRef([])
  const animated = useRef(false)

  useEffect(() => {
    /* Reset guard on every new batch so bars re-animate when vitals refresh */
    animated.current = false
    barsRef.current  = []

    const el = panelRef.current
    if (!el) return

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !animated.current) {
          animated.current = true
          barsRef.current.forEach((bar, i) => {
            if (!bar) return
            const target = bar.dataset.targetWidth
            animate(bar, {
              width: [0, target],
              duration: 900,
              delay: i * 110,
              ease: 'spring(1, 80, 14, 0)',
            })
          })
        }
      },
      { threshold: 0.2 }
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [displayFactors])

  if (!displayFactors?.length) return null

  return (
    <div ref={panelRef}
      className="rounded-2xl overflow-hidden mt-4"
      style={{
        background: 'linear-gradient(165deg, #07111e 0%, #0d1a2a 100%)',
        border: '1px solid rgba(255,255,255,0.07)',
        boxShadow: '0 24px 64px rgba(0,0,0,0.32), 0 4px 18px rgba(0,0,0,0.22), inset 0 1px 0 rgba(255,255,255,0.06)',
      }}
    >
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div
        style={{
          padding: '14px 20px 12px',
          borderBottom: '1px solid rgba(255,255,255,0.07)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div
            style={{
              width: 9, height: 9,
              borderRadius: '50%',
              background: 'var(--cyan-l)',
              boxShadow: '0 0 10px rgba(14,165,201,0.90), 0 0 24px rgba(14,165,201,0.40)',
            }}
          />
          <div>
            <p
              style={{
                color: 'rgba(255,255,255,0.92)',
                fontSize: '12.5px',
                fontFamily: 'var(--font-display)',
                fontWeight: 700,
                letterSpacing: '0.01em',
                margin: 0,
                lineHeight: 1.2,
              }}
            >
              Key Risk Drivers
            </p>
            <p style={{ color: 'rgba(255,255,255,0.35)', fontSize: '10px', fontFamily: 'var(--font-display)', margin: '2px 0 0', letterSpacing: '0.03em' }}>
              SHAP feature attribution
            </p>
          </div>
        </div>

        {/* Legend */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: '10px', color: 'rgba(255,255,255,0.40)', fontFamily: 'var(--font-display)' }}>
            <span style={{ width: 8, height: 8, borderRadius: 2, background: '#f97070', display: 'inline-block' }} />
            Raises risk
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: '10px', color: 'rgba(255,255,255,0.40)', fontFamily: 'var(--font-display)' }}>
            <span style={{ width: 8, height: 8, borderRadius: 2, background: '#4ade80', display: 'inline-block' }} />
            Lowers risk
          </span>
        </div>
      </div>

      {/* ── Factor rows ─────────────────────────────────────────────────── */}
      <div style={{ padding: '16px 20px 8px', display: 'flex', flexDirection: 'column', gap: 14 }}>
        {displayFactors.map((f, i) => {
          const isUp     = f.direction === 'up'
          const barColor = isUp ? '#f97070' : '#4ade80'
          const trackColor = isUp ? 'rgba(249,112,112,0.12)' : 'rgba(74,222,128,0.12)'
          const barPct   = Math.round((f.rel_impact ?? 0) * 100)
          const label    = f.label ?? f.feature
          const valueStr = f.unit
            ? `${f.value?.toFixed(1)} ${f.unit}`
            : f.value?.toFixed(2)

          return (
            <div key={i} style={{ display: 'flex', gap: 14, alignItems: 'center' }}>
              {/* Rank */}
              <span
                className="num"
                style={{
                  fontSize: '11px',
                  color: 'rgba(255,255,255,0.22)',
                  fontWeight: 600,
                  width: 14,
                  flexShrink: 0,
                  textAlign: 'right',
                }}
              >
                {i + 1}
              </span>

              {/* Main content */}
              <div style={{ flex: 1, minWidth: 0 }}>
                {/* Label row */}
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'baseline',
                    justifyContent: 'space-between',
                    marginBottom: 6,
                    gap: 8,
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'baseline', gap: 7, minWidth: 0 }}>
                    <span
                      style={{
                        fontSize: '13px',
                        fontWeight: 600,
                        color: 'rgba(255,255,255,0.88)',
                        fontFamily: 'var(--font-display)',
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                      }}
                    >
                      {label}
                    </span>
                    <span
                      className="num"
                      style={{
                        fontSize: '11px',
                        color: 'rgba(255,255,255,0.38)',
                        flexShrink: 0,
                      }}
                    >
                      {valueStr}
                    </span>
                  </div>

                  {/* Impact badge */}
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 3,
                      flexShrink: 0,
                      background: `${barColor}15`,
                      border: `1px solid ${barColor}30`,
                      borderRadius: 7,
                      padding: '2px 7px',
                    }}
                  >
                    <span
                      className="num"
                      style={{ fontSize: '12px', fontWeight: 700, color: barColor, letterSpacing: '-0.02em' }}
                    >
                      {f.impact > 0 ? '+' : ''}{f.impact?.toFixed(2)}
                    </span>
                    <span style={{ fontSize: '11px', color: barColor, fontWeight: 700 }}>
                      {isUp ? '↑' : '↓'}
                    </span>
                  </div>
                </div>

                {/* Bar track */}
                <div
                  style={{
                    height: 6,
                    background: 'rgba(255,255,255,0.06)',
                    borderRadius: 4,
                    overflow: 'hidden',
                    position: 'relative',
                  }}
                >
                  <div
                    ref={el => { barsRef.current[i] = el }}
                    data-target-width={`${barPct}%`}
                    style={{
                      width: 0,
                      height: '100%',
                      background: `linear-gradient(to right, ${barColor}50, ${barColor}dd)`,
                      borderRadius: 4,
                      boxShadow: `0 0 6px ${barColor}40`,
                    }}
                  />
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* ── Footer ──────────────────────────────────────────────────────── */}
      <div
        style={{
          padding: '10px 20px 14px',
          borderTop: '1px solid rgba(255,255,255,0.05)',
          marginTop: 8,
        }}
      >
        <p
          style={{
            fontSize: '10px',
            color: 'rgba(255,255,255,0.28)',
            fontFamily: 'var(--font-display)',
            letterSpacing: '0.03em',
          }}
        >
          Bar width = relative impact · ↑ increases predicted SOFA · ↓ reduces it
        </p>
      </div>
    </div>
  )
}
