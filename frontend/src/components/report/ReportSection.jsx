/* ─── Section-specific SVG icons ───────────────────────────────────────── */
const ICONS = {
  'CURRENT CONDITION': () => (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M8 2a2 2 0 0 1 2 2v1h1a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h1V4a2 2 0 0 1 2-2z"/>
      <path d="M5.5 10h5M5.5 12h3"/>
    </svg>
  ),
  'PROBABLE CAUSE': () => (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="8" cy="7" r="4.5"/>
      <path d="M8 11.5v3"/>
      <path d="M6 15h4"/>
    </svg>
  ),
  'RISK FORECAST': () => (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polyline points="1.5,12.5 4.5,8 7.5,9.5 11,5 14.5,6"/>
      <line x1="1.5" y1="14" x2="14.5" y2="14"/>
    </svg>
  ),
  'IMMEDIATE ACTIONS': () => (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polygon points="8,2 14.5,14 1.5,14"/>
      <line x1="8" y1="7" x2="8" y2="10"/>
      <circle cx="8" cy="12" r="0.5" fill="currentColor" stroke="none"/>
    </svg>
  ),
}

/* Human-readable section names */
const DISPLAY_NAME = {
  'CURRENT CONDITION': 'Current Condition',
  'PROBABLE CAUSE':    'Probable Cause',
  'RISK FORECAST':     'Risk Forecast',
  'IMMEDIATE ACTIONS': 'Immediate Actions',
}

function renderBold(text) {
  const parts = text.split(/(\*\*[^*]+\*\*)/)
  return parts.map((part, i) =>
    part.startsWith('**') && part.endsWith('**')
      ? <strong key={i} style={{ color: 'var(--t1)', fontWeight: 700 }}>{part.slice(2, -2)}</strong>
      : part
  )
}

function renderLine(line, idx, accentColor) {
  const trimmed = line.trim()
  if (!trimmed) return null

  if (trimmed.endsWith(':') && !trimmed.startsWith('•') && trimmed.length < 60) {
    return (
      <div
        key={idx}
        className="flex items-center gap-2 mt-4 mb-2"
      >
        <span
          style={{
            fontSize: '10px',
            fontWeight: 800,
            textTransform: 'uppercase',
            letterSpacing: '0.12em',
            color: accentColor,
            fontFamily: 'var(--font-display)',
          }}
        >
          {trimmed.slice(0, -1)}
        </span>
        <div style={{ flex: 1, height: 1, background: `linear-gradient(to right, ${accentColor}30, transparent)` }} />
      </div>
    )
  }

  const text = /^[•\-\*]\s/.test(trimmed) ? trimmed.replace(/^[•\-\*]\s*/, '').trim() : trimmed

  if (text.includes('⚡')) {
    return (
      <div
        key={idx}
        className="flex gap-2.5 px-3 py-2 mb-2 rounded-lg text-xs"
        style={{
          borderLeft: `2.5px solid ${accentColor}`,
          background: `${accentColor}08`,
          color: 'var(--t1)',
        }}
      >
        <span className="flex-1 leading-relaxed">{renderBold(text)}</span>
      </div>
    )
  }

  return (
    <div key={idx} className="flex gap-2.5 text-xs mb-1.5 leading-relaxed" style={{ color: 'var(--t2)' }}>
      <span
        className="flex-shrink-0 mt-[5px] rounded-full"
        style={{ width: 3.5, height: 3.5, background: accentColor, opacity: 0.75 }}
      />
      <span className="flex-1">{renderBold(text)}</span>
    </div>
  )
}

export default function ReportSection({ name, content, color, subtitle }) {
  const lines      = (content ?? '').split('\n')
  const hasContent = lines.some(l => l.trim())
  const SectionIcon = ICONS[name]
  const displayName = DISPLAY_NAME[name] ?? name

  return (
    <div
      className="flex flex-col overflow-hidden"
      style={{
        background: 'var(--s1)',
        border: `1px solid ${color}1e`,
        borderRadius: 'var(--r-xl)',
        boxShadow: `var(--sh-sm), 0 0 0 1px ${color}06`,
        transition: 'box-shadow 0.28s var(--ease-out-smooth), transform 0.32s var(--ease-spring-soft)',
      }}
      onMouseEnter={e => {
        e.currentTarget.style.boxShadow = `var(--sh-md), 0 0 0 2px ${color}14`
        e.currentTarget.style.transform = 'translateY(-2px)'
      }}
      onMouseLeave={e => {
        e.currentTarget.style.boxShadow = `var(--sh-sm), 0 0 0 1px ${color}06`
        e.currentTarget.style.transform = 'translateY(0)'
      }}
    >
      {/* ── Header ──────────────────────────────────────────────────── */}
      <div
        style={{
          background: 'linear-gradient(135deg, #07111e 0%, #0d1a2a 100%)',
          padding: '10px 14px',
          display: 'flex',
          alignItems: 'center',
          gap: 9,
          borderBottom: `1.5px solid ${color}48`,
          flexShrink: 0,
        }}
      >
        {/* Glowing dot */}
        <div style={{
          width: 7, height: 7, borderRadius: '50%', background: color, flexShrink: 0,
          boxShadow: `0 0 7px ${color}90, 0 0 14px ${color}35`,
        }} />

        {/* Icon badge */}
        {SectionIcon && (
          <div
            style={{
              width: 24, height: 24, borderRadius: 6,
              background: `${color}1e`,
              border: `1px solid ${color}30`,
              color: color,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              flexShrink: 0,
            }}
          >
            <SectionIcon />
          </div>
        )}

        <div style={{ flex: 1, minWidth: 0 }}>
          <p
            style={{
              color: 'rgba(255,255,255,0.92)',
              fontSize: '12.5px',
              fontFamily: 'var(--font-display)',
              fontWeight: 600,
              letterSpacing: '-0.01em',
              margin: 0,
              lineHeight: 1.2,
            }}
          >
            {displayName}
          </p>
          {subtitle && (
            <p style={{
              color: 'rgba(255,255,255,0.38)',
              fontSize: '10px',
              fontFamily: 'var(--font-display)',
              margin: '1px 0 0',
              letterSpacing: '0.01em',
            }}>
              {subtitle}
            </p>
          )}
        </div>
      </div>

      {/* ── Content area ────────────────────────────────────────────── */}
      <div className="flex-1 p-4">
        {hasContent
          ? lines.map((line, idx) => renderLine(line, idx, color))
          : <p className="text-xs italic" style={{ color: 'var(--t3)' }}>No assessment available.</p>
        }
      </div>
    </div>
  )
}
