const VERDICTS = {
  High:     { blurb: 'Validated across all quality checks' },
  Moderate: { blurb: 'Passed most validation checks'       },
  Low:      { blurb: 'Failed multiple validation checks'   },
}

function ScoreRing({ score, color }) {
  const pct    = Math.round((score ?? 0) * 100)
  const r      = 21
  const C      = 2 * Math.PI * r
  const offset = C - (pct / 100) * C

  return (
    <div className="relative flex-shrink-0" style={{ width: 60, height: 60 }}>
      <svg width="60" height="60" style={{ transform: 'rotate(-90deg)' }}>
        {/* Track */}
        <circle cx="30" cy="30" r={r} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="4.5" />
        {/* Filled arc */}
        <circle
          cx="30" cy="30" r={r}
          fill="none"
          stroke={color}
          strokeWidth="4.5"
          strokeLinecap="round"
          strokeDasharray={C}
          strokeDashoffset={offset}
          style={{
            transition: 'stroke-dashoffset 0.9s cubic-bezier(0.34,1.1,0.64,1), stroke 0.4s ease',
            filter: `drop-shadow(0 0 4px ${color}60)`,
          }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="num font-bold leading-none" style={{ color, fontSize: '14px', letterSpacing: '-0.03em' }}>{pct}</span>
        <span style={{ color: 'rgba(255,255,255,0.35)', fontSize: '9px', fontFamily: 'var(--font-display)' }}>%</span>
      </div>
    </div>
  )
}

export default function ReliabilityBadge({ reliability_score, reliability_label, reliability_color, caution }) {
  const verdict = VERDICTS[reliability_label] ?? VERDICTS.Moderate

  return (
    <div
      className="overflow-hidden"
      style={{
        borderRadius: 'var(--r-xl)',
        border: `1px solid ${reliability_color}20`,
        boxShadow: `var(--sh-sm), 0 0 0 1px ${reliability_color}06`,
      }}
    >
      {/* Dark header */}
      <div
        style={{
          background: 'linear-gradient(135deg, #07111e 0%, #0d1a2a 100%)',
          padding: '9px 16px',
          display: 'flex',
          alignItems: 'center',
          gap: 9,
          borderBottom: `1.5px solid ${reliability_color}45`,
        }}
      >
        <div style={{
          width: 7, height: 7, borderRadius: '50%', background: reliability_color, flexShrink: 0,
          boxShadow: `0 0 7px ${reliability_color}90, 0 0 14px ${reliability_color}35`,
        }} />
        <span style={{
          color: 'rgba(255,255,255,0.85)',
          fontSize: '12px',
          fontFamily: 'var(--font-display)',
          fontWeight: 600,
          letterSpacing: '0.01em',
        }}>
          AI Clinical Assessment
        </span>
      </div>

      {/* Body */}
      <div
        className="flex items-center gap-5 px-5 py-4"
        style={{ background: `linear-gradient(135deg, ${reliability_color}05 0%, var(--s1) 100%)` }}
      >
        <ScoreRing score={reliability_score} color={reliability_color} />

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span
              className="font-bold"
              style={{ color: reliability_color, fontFamily: 'var(--font-display)', fontSize: '14px', letterSpacing: '-0.01em' }}
            >
              {reliability_label} Reliability
            </span>
          </div>
          <p style={{ fontSize: '12px', color: 'var(--t3)', margin: 0, lineHeight: 1.5 }}>{verdict.blurb}</p>
          {caution && (
            <div
              className="flex items-center gap-1.5 mt-2"
              style={{ fontSize: '11px', fontWeight: 600, color: 'var(--warn-l)', fontFamily: 'var(--font-display)' }}
            >
              <svg width="11" height="11" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
                <polygon points="8,2 14.5,14 1.5,14"/>
                <line x1="8" y1="7" x2="8" y2="10"/>
                <circle cx="8" cy="12.5" r="0.5" fill="currentColor" stroke="none"/>
              </svg>
              Review carefully before acting on this assessment
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
