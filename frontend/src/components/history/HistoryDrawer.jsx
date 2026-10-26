import { useQuery } from '@tanstack/react-query'
import { AnimatePresence, motion } from 'framer-motion'
import { getHistory } from '../../api/client.js'

/* ── Design tokens ───────────────────────────────────────────────────────── */
const a = (c, p) => `color-mix(in srgb, ${c} ${p}%, transparent)`

/* Column definitions — order, label, width, normal ranges for coloring */
const COLS = [
  { key: 'Timestamp',  label: 'Timestamp',    w: 130, type: 'text' },
  { key: 'SOFA',       label: 'SOFA',          w: 64,  type: 'num', lo: 0,    hi: 9    },
  { key: 'Risk',       label: 'Risk',          w: 90,  type: 'risk' },
  { key: 'HR',         label: 'HR',            w: 56,  type: 'num', lo: 60,   hi: 100, unit: 'bpm' },
  { key: 'RR',         label: 'RR',            w: 50,  type: 'num', lo: 12,   hi: 20,  unit: '/m' },
  { key: 'SpO₂',       label: 'SpO₂',          w: 58,  type: 'num', lo: 95,   hi: 100, unit: '%' },
  { key: 'Temp (°C)',  label: 'Temp',          w: 62,  type: 'num', lo: 36.0, hi: 37.5,unit: '°C' },
  { key: 'SBP',        label: 'SBP',           w: 58,  type: 'num', lo: 90,   hi: 120, unit: 'mmHg' },
  { key: 'DBP',        label: 'DBP',           w: 58,  type: 'num', lo: 60,   hi: 80,  unit: 'mmHg' },
  { key: 'MAP',        label: 'MAP',           w: 58,  type: 'num', lo: 70,   hi: 100, unit: 'mmHg' },
  { key: 'GCS Eye',    label: 'GCS Eye',       w: 68,  type: 'num', lo: 3,    hi: 4    },
  { key: 'Stress',     label: 'Stress',        w: 60,  type: 'num', lo: 0,    hi: 3    },
]

/* ── Risk badge ─────────────────────────────────────────────────────────── */
const RISK_STYLES = {
  'Low Risk':      { color: 'var(--ok-l)',     bg: a('var(--ok-l)', 12)     },
  'Moderate Risk': { color: 'var(--warn-l)',   bg: a('var(--warn-l)', 12)   },
  'High Risk':     { color: 'var(--danger-l)', bg: a('var(--danger-l)', 12) },
  'Critical Risk': { color: 'var(--danger-l)', bg: a('var(--danger-l)', 18) },
}

function RiskBadge({ value }) {
  const s = RISK_STYLES[value] ?? { color: 'var(--t3)', bg: 'var(--s2)' }
  return (
    <span style={{
      fontSize: '10px', fontWeight: 700, padding: '2px 7px',
      borderRadius: 6, fontFamily: 'var(--font-display)',
      color: s.color, background: s.bg,
      border: `1px solid ${a(s.color, 22)}`,
      whiteSpace: 'nowrap',
    }}>
      {value ?? '—'}
    </span>
  )
}

/* ── Numeric cell with abnormal coloring ────────────────────────────────── */
function NumCell({ value, lo, hi }) {
  if (value == null) return <span style={{ color: 'var(--t4)' }}>—</span>
  const n = Number(value)
  const isHigh = hi != null && n > hi
  const isLow  = lo != null && n < lo
  const color = isHigh ? 'var(--danger-l)' : isLow ? 'var(--warn-l)' : 'var(--t2)'
  return (
    <span className="num" style={{ color, fontWeight: isHigh || isLow ? 700 : 500 }}>
      {Number.isInteger(n) ? n : n.toFixed(1)}
      {(isHigh || isLow) && (
        <sup style={{ fontSize: '8px', marginLeft: 1 }}>{isHigh ? '↑' : '↓'}</sup>
      )}
    </span>
  )
}

/* ── Header close button SVG ────────────────────────────────────────────── */
function CloseIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" aria-hidden="true">
      <path d="M2 2l12 12M14 2L2 14"/>
    </svg>
  )
}

/* ── History icon ────────────────────────────────────────────────────────── */
function HistoryIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 18 18" fill="none" stroke="currentColor"
      strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M2.5 9a6.5 6.5 0 1 1 1.7 4.4"/>
      <path d="M2.5 4.5V9H7"/>
      <path d="M9 6v3.5l2.5 1.5"/>
    </svg>
  )
}

/* ── Skeleton row ────────────────────────────────────────────────────────── */
function SkeletonRows() {
  return Array.from({ length: 6 }, (_, i) => (
    <tr key={i}>
      {COLS.map(c => (
        <td key={c.key} style={{ padding: '9px 10px' }}>
          <div className="skeleton" style={{ height: 12, borderRadius: 4, width: c.w * 0.6 }} />
        </td>
      ))}
    </tr>
  ))
}

/* ── Main drawer ─────────────────────────────────────────────────────────── */
export default function HistoryDrawer({ patientId, open, onClose }) {
  const { data: rows, isLoading, isError } = useQuery({
    queryKey: ['history', patientId],
    queryFn: () => getHistory(patientId, 50).then(r => r.data),
    enabled: open && patientId != null,
    staleTime: 30_000,
  })

  const totalCols = COLS.reduce((s, c) => s + c.w, 0)

  return (
    <AnimatePresence>
      {open && (
        <>
          {/* ── Backdrop ── */}
          <motion.div
            key="backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.22 }}
            onClick={onClose}
            style={{
              position: 'fixed', inset: 0, zIndex: 900,
              background: 'rgba(4, 10, 20, 0.65)',
              backdropFilter: 'blur(6px)',
              WebkitBackdropFilter: 'blur(6px)',
            }}
          />

          {/* ── Drawer panel ── */}
          <motion.div
            key="drawer"
            initial={{ x: '100%', opacity: 0.6 }}
            animate={{ x: 0,      opacity: 1 }}
            exit={{ x: '100%',    opacity: 0.4 }}
            transition={{ type: 'spring', stiffness: 340, damping: 34, mass: 1.0 }}
            style={{
              position: 'fixed', top: 0, right: 0, bottom: 0,
              width: 'min(82vw, 1020px)',
              zIndex: 910,
              display: 'flex', flexDirection: 'column',
              background: 'var(--s0)',
              borderLeft: '1px solid var(--b1)',
              boxShadow: '-12px 0 48px rgba(0,0,0,0.55), -2px 0 0 rgba(255,255,255,0.03)',
            }}
          >
            {/* ── Drawer header ── */}
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '18px 22px',
              background: 'linear-gradient(135deg, var(--s1) 0%, var(--s0) 100%)',
              borderBottom: '1px solid var(--b1)',
              flexShrink: 0,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                {/* Icon badge */}
                <div style={{
                  width: 38, height: 38, borderRadius: 11,
                  background: `linear-gradient(145deg, ${a('var(--cyan-l)', 16)}, ${a('var(--cyan-l)', 6)})`,
                  border: `1.5px solid ${a('var(--cyan-l)', 22)}`,
                  color: 'var(--cyan-l)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  animation: 'noteFloat 4.0s ease-in-out infinite',
                }}>
                  <HistoryIcon />
                </div>
                <div>
                  <span style={{
                    display: 'block',
                    fontSize: '15px', fontWeight: 700, color: 'var(--t1)',
                    fontFamily: 'var(--font-display)', letterSpacing: '-0.01em',
                  }}>
                    Patient History
                  </span>
                  <span style={{ fontSize: '11px', color: 'var(--t3)', fontFamily: 'var(--font-display)' }}>
                    {isLoading ? 'Loading…' : isError ? 'Failed to load' : `${rows?.length ?? 0} recordings · SOFA, vitals, CV features`}
                  </span>
                </div>
              </div>

              {/* Legend chips */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  {[
                    { color: 'var(--danger-l)', label: '↑ Above normal' },
                    { color: 'var(--warn-l)',   label: '↓ Below normal' },
                    { color: 'var(--t2)',        label: 'Normal' },
                  ].map(({ color, label }) => (
                    <span key={label} style={{
                      display: 'flex', alignItems: 'center', gap: 5,
                      fontSize: '10px', color: 'var(--t3)',
                      fontFamily: 'var(--font-display)',
                    }}>
                      <span style={{ width: 8, height: 8, borderRadius: '50%', background: color, display: 'inline-block', flexShrink: 0 }} />
                      {label}
                    </span>
                  ))}
                </div>

                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.92 }}
                  onClick={onClose}
                  title="Close history"
                  style={{
                    width: 34, height: 34, borderRadius: 9,
                    background: 'var(--s2)', border: '1px solid var(--b1)',
                    color: 'var(--t2)', cursor: 'pointer',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                  }}
                >
                  <CloseIcon />
                </motion.button>
              </div>
            </div>

            {/* ── Table area ── */}
            <div style={{ flex: 1, overflow: 'auto' }}>
              <table style={{
                width: '100%', minWidth: totalCols,
                borderCollapse: 'collapse',
                fontSize: '12px',
                fontFamily: 'var(--font-display)',
              }}>
                {/* Sticky header */}
                <thead>
                  <tr style={{
                    position: 'sticky', top: 0, zIndex: 2,
                    background: 'linear-gradient(180deg, var(--s1) 0%, var(--s0) 100%)',
                    borderBottom: `2px solid ${a('var(--cyan-l)', 25)}`,
                  }}>
                    {COLS.map(c => (
                      <th key={c.key} style={{
                        padding: '10px 10px',
                        textAlign: 'left',
                        fontSize: '10px', fontWeight: 700,
                        letterSpacing: '0.08em', textTransform: 'uppercase',
                        color: 'var(--t3)',
                        whiteSpace: 'nowrap',
                        minWidth: c.w,
                        borderRight: '1px solid var(--b1)',
                      }}>
                        {c.label}
                        {c.unit && (
                          <span style={{ color: 'var(--t4)', fontWeight: 500, marginLeft: 3 }}>
                            ({c.unit})
                          </span>
                        )}
                      </th>
                    ))}
                  </tr>
                </thead>

                <tbody>
                  {isLoading ? (
                    <SkeletonRows />
                  ) : isError ? (
                    <tr>
                      <td colSpan={COLS.length} style={{ padding: 32, textAlign: 'center', color: 'var(--danger-l)' }}>
                        Failed to load history. Make sure the backend is running.
                      </td>
                    </tr>
                  ) : !rows?.length ? (
                    <tr>
                      <td colSpan={COLS.length} style={{ padding: 32, textAlign: 'center', color: 'var(--t3)' }}>
                        No history recorded yet. Wait for the first 90-second poll to complete.
                      </td>
                    </tr>
                  ) : (
                    [...rows].reverse().map((row, idx) => {
                      const isEven = idx % 2 === 0
                      return (
                        <motion.tr
                          key={idx}
                          initial={{ opacity: 0, x: 12 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: idx * 0.018, duration: 0.22 }}
                          style={{
                            background: isEven ? 'transparent' : a('var(--cyan-l)', 2),
                            borderBottom: '1px solid var(--b1)',
                            transition: 'background 0.15s',
                          }}
                          onMouseEnter={e => e.currentTarget.style.background = a('var(--cyan-l)', 5)}
                          onMouseLeave={e => e.currentTarget.style.background = isEven ? 'transparent' : a('var(--cyan-l)', 2)}
                        >
                          {COLS.map(c => (
                            <td key={c.key} style={{
                              padding: '8px 10px',
                              borderRight: '1px solid var(--b1)',
                              whiteSpace: 'nowrap',
                              color: 'var(--t2)',
                              verticalAlign: 'middle',
                            }}>
                              {c.type === 'risk' ? (
                                <RiskBadge value={row[c.key]} />
                              ) : c.type === 'num' ? (
                                <NumCell value={row[c.key]} lo={c.lo} hi={c.hi} />
                              ) : (
                                <span style={{ color: 'var(--t3)' }}>{row[c.key] ?? '—'}</span>
                              )}
                            </td>
                          ))}
                        </motion.tr>
                      )
                    })
                  )}
                </tbody>
              </table>
            </div>

            {/* ── Footer ── */}
            <div style={{
              padding: '10px 22px',
              background: 'var(--s1)',
              borderTop: '1px solid var(--b1)',
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              flexShrink: 0,
            }}>
              <span style={{ fontSize: '10.5px', color: 'var(--t4)', fontFamily: 'var(--font-display)' }}>
                Showing up to 50 most recent readings · Newest at top
              </span>
              <span style={{ fontSize: '10.5px', color: 'var(--t4)', fontFamily: 'var(--font-display)' }}>
                Abnormal values are highlighted · Normal ranges per ICU CDSS thresholds
              </span>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}
