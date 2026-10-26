import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceArea,
} from 'recharts'

function CustomTooltip({ active, payload, label, unit, color }) {
  if (!active || !payload?.length) return null
  return (
    <div
      className="text-xs rounded-xl px-3.5 py-2.5"
      style={{
        background: '#07111e',
        border: `1px solid rgba(255,255,255,0.10)`,
        color: '#fff',
        boxShadow: `0 16px 40px rgba(0,0,0,0.50), 0 0 0 1px rgba(255,255,255,0.05)`,
        backdropFilter: 'blur(12px)',
      }}
    >
      <p className="mb-1" style={{ color: 'rgba(255,255,255,0.44)', fontSize: '10px', fontFamily: 'var(--font-display)', letterSpacing: '0.04em' }}>
        {label}
      </p>
      <p className="num font-bold" style={{ color, fontSize: '16px', letterSpacing: '-0.03em' }}>
        {payload[0].value != null ? Number(payload[0].value).toFixed(1) : '—'}
        <span className="ml-1.5 font-normal text-xs" style={{ color: 'rgba(255,255,255,0.44)', fontSize: '11px', letterSpacing: 0 }}>{unit}</span>
      </p>
    </div>
  )
}

function formatTime(timeStr) {
  if (!timeStr) return ''
  const parts = timeStr.split(' ')
  // Show HH:MM:SS for real-time readings (they share the same HH:MM minute)
  const timePart = parts.length === 2 ? parts[1] : timeStr
  return timePart.slice(0, 8) // HH:MM:SS
}

/* Latest value badge in the header */
function LatestBadge({ value, unit, color }) {
  if (value == null) return null
  return (
    <div style={{ display: 'flex', alignItems: 'baseline', gap: 3 }}>
      <span className="num" style={{ fontSize: '17px', fontWeight: 700, color, letterSpacing: '-0.04em', lineHeight: 1 }}>
        {Number.isInteger(value) ? value : value.toFixed(1)}
      </span>
      <span style={{ fontSize: '10px', color: 'rgba(255,255,255,0.40)', fontFamily: 'var(--font-display)', marginBottom: 1 }}>
        {unit}
      </span>
    </div>
  )
}

export default function TrendChart({ data, dataKey, label, unit, color, normalLo, normalHi }) {
  const values = data?.map(d => d[dataKey]).filter(v => v != null) ?? []
  const latest = values.at(-1)
  const dataMin = values.length ? Math.min(...values) : normalLo
  const dataMax = values.length ? Math.max(...values) : normalHi
  const pad  = Math.max((dataMax - dataMin) * 0.25, 1)
  const yMin = Math.floor(Math.min(dataMin, normalLo) - pad)
  const yMax = Math.ceil(Math.max(dataMax, normalHi) + pad)

  const formatted = data?.map(d => ({ ...d, _time: formatTime(d.time) })) ?? []
  const hasAbnormal      = values.some(v => v < normalLo || v > normalHi)
  const isAbnormalLatest = latest != null && (latest < normalLo || latest > normalHi)
  const latestIsHigh     = latest != null && latest > normalHi
  const latestIsLow      = latest != null && latest < normalLo

  return (
    <div
      className="rounded-2xl overflow-hidden"
      style={{
        background: 'var(--s1)',
        border: `1px solid ${hasAbnormal ? color + '30' : 'var(--b1)'}`,
        boxShadow: hasAbnormal
          ? `var(--sh-md), 0 0 0 3px ${color}0a`
          : 'var(--sh-sm)',
        transition: 'box-shadow 0.28s var(--ease-out-smooth)',
      }}
    >
      {/* ── Header ────────────────────────────────────────────────────── */}
      <div
        style={{
          background: 'linear-gradient(135deg, #07111e 0%, #0d1a2a 100%)',
          padding: '10px 14px 10px 14px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderBottom: `1.5px solid ${color}55`,
          gap: 8,
        }}
      >
        {/* Left: color indicator + label */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
          {/* Glowing color swatch */}
          <div
            style={{
              width: 10,
              height: 10,
              borderRadius: '50%',
              background: color,
              flexShrink: 0,
              boxShadow: `0 0 8px ${color}90, 0 0 18px ${color}40`,
            }}
          />
          <div style={{ minWidth: 0 }}>
            <span
              style={{
                color: 'rgba(255,255,255,0.94)',
                fontSize: '12.5px',
                fontFamily: 'var(--font-display)',
                fontWeight: 600,
                letterSpacing: '-0.01em',
                display: 'block',
                lineHeight: 1.15,
              }}
            >
              {label}
            </span>
            <span style={{ color: 'rgba(255,255,255,0.38)', fontSize: '10px', fontFamily: 'var(--font-display)', display: 'block' }}>
              {normalLo}–{normalHi} {unit}
            </span>
          </div>
        </div>

        {/* Right: latest value + abnormal badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
          {isAbnormalLatest && (
            <span
              style={{
                color: color,
                fontSize: '9px',
                fontFamily: 'var(--font-display)',
                fontWeight: 800,
                background: `${color}20`,
                padding: '2px 6px',
                borderRadius: 6,
                border: `1px solid ${color}38`,
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
              }}
            >
              {latestIsHigh ? '↑' : '↓'} abnormal
            </span>
          )}
          <LatestBadge value={latest} unit={unit} color={isAbnormalLatest ? color : 'rgba(255,255,255,0.80)'} />
        </div>
      </div>

      {/* ── Chart area ────────────────────────────────────────────────── */}
      <div style={{ padding: '6px 6px 8px 0', background: 'var(--s1)' }}>
        <ResponsiveContainer width="100%" height={144}>
          <LineChart data={formatted} margin={{ top: 10, right: 10, bottom: 0, left: 2 }}>
            <CartesianGrid stroke="var(--b1)" strokeDasharray="2 4" vertical={false} opacity={0.6} />

            <ReferenceArea
              y1={normalLo}
              y2={normalHi}
              fill={color}
              fillOpacity={0.07}
              strokeOpacity={0}
            />

            <XAxis
              dataKey="_time"
              tick={{ fontSize: 9.5, fill: 'var(--t4)', fontFamily: 'var(--mono)' }}
              tickLine={false}
              axisLine={false}
              interval="preserveStartEnd"
            />
            <YAxis
              domain={[yMin, yMax]}
              tick={{ fontSize: 9.5, fill: 'var(--t4)', fontFamily: 'var(--mono)' }}
              tickLine={false}
              axisLine={false}
              width={32}
            />
            <Tooltip
              content={<CustomTooltip unit={unit} color={color} />}
              cursor={{ stroke: color + '35', strokeWidth: 1, strokeDasharray: '3 3' }}
            />
            <Line
              type="monotone"
              dataKey={dataKey}
              stroke={color}
              strokeWidth={2.2}
              dot={false}
              activeDot={{ r: 5, fill: color, strokeWidth: 2, stroke: '#fff' }}
              isAnimationActive={false}
              style={{ filter: `drop-shadow(0 2px 5px ${color}45)` }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
