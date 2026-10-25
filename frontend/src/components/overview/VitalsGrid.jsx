import { motion } from 'framer-motion'
import VitalCard from './VitalCard.jsx'

const gridVariants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.07, delayChildren: 0.04 } },
}
const tileVariants = {
  hidden: { opacity: 0, scale: 0.93, y: 14 },
  visible: { opacity: 1, scale: 1, y: 0, transition: { type: 'spring', stiffness: 300, damping: 26 } },
}

/* ── SVG Icons ─────────────────────────────────────────────────────────────── */
const HRIcon = () => (
  <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" aria-hidden="true">
    <path d="M9 15C9 15 2 10 2 6a3.5 3.5 0 0 1 7-.5A3.5 3.5 0 0 1 16 6c0 4-7 9-7 9z"/>
  </svg>
)
const RRIcon = () => (
  <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
    <ellipse cx="5.5" cy="11.5" rx="2.8" ry="4"/>
    <ellipse cx="12.5" cy="11.5" rx="2.8" ry="4"/>
    <path d="M9 7.5V4.5" strokeLinecap="round"/>
    <path d="M7 7 Q9 5.5 11 7" strokeLinecap="round"/>
  </svg>
)
const SpO2Icon = () => (
  <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
    <path d="M9 2.5C9 2.5 3 8 3 11.5a6 6 0 0 0 12 0C15 8 9 2.5 9 2.5z" strokeLinejoin="round"/>
    <text x="9" y="13.5" textAnchor="middle" fontSize="4.5" stroke="none" fill="currentColor" fontWeight="700">O₂</text>
  </svg>
)
const TempIcon = () => (
  <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
    <rect x="7" y="2.5" width="4" height="9.5" rx="2"/>
    <circle cx="9" cy="14" r="2.5"/>
    <line x1="9" y1="12" x2="9" y2="6.5" strokeWidth="2" strokeLinecap="round"/>
    <line x1="11" y1="5" x2="12.5" y2="5" strokeLinecap="round"/>
    <line x1="11" y1="7" x2="12" y2="7" strokeLinecap="round"/>
  </svg>
)
const SBPIcon = () => (
  <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
    <path d="M9 2.5v11" strokeLinecap="round"/>
    <path d="M6.5 5.5 L9 2.5 L11.5 5.5" strokeLinejoin="round" strokeLinecap="round"/>
    <circle cx="9" cy="15.5" r="1.5" fill="currentColor" stroke="none"/>
  </svg>
)
const MAPIcon = () => (
  <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
    <circle cx="9" cy="9" r="6.5"/>
    <path d="M9 9 L13 5.5" strokeLinecap="round"/>
    <circle cx="9" cy="9" r="1.2" fill="currentColor" stroke="none"/>
    <line x1="2.5" y1="9" x2="4" y2="9" strokeLinecap="round"/>
    <line x1="14" y1="9" x2="15.5" y2="9" strokeLinecap="round"/>
    <line x1="9" y1="2.5" x2="9" y2="4" strokeLinecap="round"/>
  </svg>
)
const DBPIcon = () => (
  <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
    <path d="M9 2.5v11" strokeLinecap="round"/>
    <path d="M6.5 11 L9 13.5 L11.5 11" strokeLinejoin="round" strokeLinecap="round"/>
    <circle cx="9" cy="2.5" r="1.5" fill="currentColor" stroke="none"/>
  </svg>
)
const GCSEyeIcon = () => (
  <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
    <path d="M2 9C2 9 4.5 4.5 9 4.5S16 9 16 9 13.5 13.5 9 13.5 2 9 2 9z" strokeLinejoin="round"/>
    <circle cx="9" cy="9" r="2.4"/>
    <circle cx="9.7" cy="8.3" r="0.7" fill="currentColor" stroke="none"/>
  </svg>
)
const StressIcon = () => (
  <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
    <path d="M9 2.5a6.5 6.5 0 1 1 0 13 6.5 6.5 0 0 1 0-13z"/>
    <path d="M6 11.5c0 0 1-1.5 3-1.5s3 1.5 3 1.5" strokeLinecap="round"/>
    <circle cx="6.5" cy="7.5" r="0.9" fill="currentColor" stroke="none"/>
    <circle cx="11.5" cy="7.5" r="0.9" fill="currentColor" stroke="none"/>
  </svg>
)
const NoteIcon = () => (
  <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
    <rect x="3" y="2" width="12" height="14" rx="2"/>
    <line x1="6" y1="6" x2="12" y2="6" strokeLinecap="round"/>
    <line x1="6" y1="9" x2="12" y2="9" strokeLinecap="round"/>
    <line x1="6" y1="12" x2="10" y2="12" strokeLinecap="round"/>
  </svg>
)

/* ── Main vitals config ─────────────────────────────────────────────────────── */
const VITALS_CONFIG = [
  { key: 'HR',   label: 'Heart Rate',   Icon: HRIcon,   unit: 'bpm',  normalLo: 60,   normalHi: 100,  wide: true  },
  { key: 'SpO2', label: 'SpO₂',         Icon: SpO2Icon, unit: '%',    normalLo: 95,   normalHi: 100,  wide: true  },
  { key: 'RR',   label: 'Resp. Rate',   Icon: RRIcon,   unit: '/min', normalLo: 12,   normalHi: 20,   wide: false },
  { key: 'Temp', label: 'Temperature',  Icon: TempIcon, unit: '°C',   normalLo: 36.0, normalHi: 37.5, wide: false },
  { key: 'SBP',  label: 'Sys. BP',      Icon: SBPIcon,  unit: 'mmHg', normalLo: 90,   normalHi: 120,  wide: false },
  { key: 'MAP',  label: 'MAP',          Icon: MAPIcon,  unit: 'mmHg', normalLo: 70,   normalHi: 100,  wide: false },
]

/* ── Derive GCS / Stress status on frontend (not in vitals_status from API) ── */
function gcsStatus(val) {
  if (val == null) return 'normal'
  if (val <= 2) return 'low'
  return 'normal'
}
function stressStatus(val) {
  if (val == null) return 'normal'
  if (val >= 4) return 'high'
  return 'normal'
}

/* ── GCS Eye label map ──────────────────────────────────────────────────────── */
const GCS_LABELS = { 1: 'None', 2: 'To Pain', 3: 'To Voice', 4: 'Spontaneous' }

/* ── Color helper ───────────────────────────────────────────────────────────── */
const a = (c, p) => `color-mix(in srgb, ${c} ${p}%, transparent)`

const STATUS_COLORS = {
  normal: 'var(--ok-l)',
  high:   'var(--danger-l)',
  low:    'var(--warn-l)',
}

/* ── Stress Score wide card ─────────────────────────────────────────────────── */
function StressCard({ value }) {
  const st   = stressStatus(value)
  const color = STATUS_COLORS[st]
  const isAbnormal = st !== 'normal'
  const displayValue = value != null ? value : '—'
  const pct = value != null ? Math.min(100, (value / 10) * 100) : 0
  const stressLabel = value == null ? '—' : value <= 3 ? 'Low Stress' : value <= 6 ? 'Elevated' : 'High Stress'

  const dotColors = Array.from({ length: 10 }, (_, i) => {
    const idx = i + 1
    if (value == null || idx > value) return 'var(--b2)'
    if (idx <= 3) return 'var(--ok-l)'
    if (idx <= 6) return 'var(--warn-l)'
    return 'var(--danger-l)'
  })

  return (
    <motion.div
      whileHover={{ y: -3, scale: 1.008 }}
      whileTap={{ scale: 0.975 }}
      transition={{ type: 'spring', stiffness: 340, damping: 26, mass: 0.9 }}
      style={{
        height: '100%',
        background: isAbnormal
          ? `linear-gradient(145deg, ${a(color, 5)} 0%, var(--s1) 60%)`
          : 'linear-gradient(145deg, var(--s2) 0%, var(--s1) 60%)',
        border: `1px solid ${isAbnormal ? a(color, 22) : 'var(--b1)'}`,
        borderRadius: 'var(--r-xl)',
        padding: '18px 20px 16px',
        boxShadow: isAbnormal
          ? `var(--sh-sm), 0 0 0 3px ${a(color, 7)}, inset 0 1px 0 ${a(color, 8)}`
          : 'var(--sh-sm)',
        position: 'relative',
        overflow: 'hidden',
        cursor: 'default',
      }}
    >
      {/* Top band */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: 3,
        background: isAbnormal
          ? `linear-gradient(to right, ${color}, ${a(color, 40)}, transparent 80%)`
          : `linear-gradient(to right, var(--b1), transparent)`,
        borderRadius: 'var(--r-xl) var(--r-xl) 0 0',
      }} />

      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 40, height: 40, borderRadius: 12, flexShrink: 0,
            background: `linear-gradient(145deg, ${a(color, 12)}, ${a(color, 4)})`,
            border: `1.5px solid ${a(color, 16)}`,
            color, display: 'flex', alignItems: 'center', justifyContent: 'center',
            animation: 'stressSway 3.2s ease-in-out infinite',
          }}>
            <div style={{ width: 20, height: 20 }}><StressIcon /></div>
          </div>
          <div>
            <span style={{
              display: 'block',
              fontSize: '10px', fontWeight: 700, letterSpacing: '0.09em',
              textTransform: 'uppercase', color: 'var(--t3)',
              fontFamily: 'var(--font-display)',
            }}>
              Stress Score
            </span>
            <span style={{ fontSize: '10px', color: 'var(--t4)', fontFamily: 'var(--font-display)' }}>
              physiological distress
            </span>
          </div>
        </div>

        {/* Status badge */}
        <span style={{
          fontSize: '10px', fontWeight: 700, padding: '3px 10px',
          borderRadius: 999, fontFamily: 'var(--font-display)',
          color, background: a(color, 10), border: `1px solid ${a(color, 22)}`,
          letterSpacing: '0.04em',
        }}>
          {stressLabel}
        </span>
      </div>

      {/* Value + scale */}
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 16 }}>
        {/* Big number */}
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 4, flexShrink: 0 }}>
          <span className="num" style={{
            fontSize: '2.8rem', fontWeight: 800, letterSpacing: '-0.04em',
            color: isAbnormal ? color : 'var(--t1)', lineHeight: 1,
          }}>
            {displayValue}
          </span>
          <span style={{ fontSize: '12px', color: 'var(--t3)', fontWeight: 500 }}>/10</span>
        </div>

        {/* Dot scale */}
        <div style={{ flex: 1, paddingBottom: 4 }}>
          <div style={{ display: 'flex', gap: 4, alignItems: 'center', marginBottom: 6 }}>
            {dotColors.map((c, i) => (
              <div key={i} style={{
                flex: 1, height: 6, borderRadius: 3,
                background: c,
                transition: 'background 0.35s ease',
                opacity: value != null && i < value ? 1 : 0.25,
              }} />
            ))}
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ fontSize: '9px', color: 'var(--ok-l)', fontFamily: 'var(--font-display)', fontWeight: 600 }}>0 calm</span>
            <span style={{ fontSize: '9px', color: 'var(--warn-l)', fontFamily: 'var(--font-display)', fontWeight: 600 }}>5</span>
            <span style={{ fontSize: '9px', color: 'var(--danger-l)', fontFamily: 'var(--font-display)', fontWeight: 600 }}>10 severe</span>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

/* ── GCS Eye standard-sized card ─────────────────────────────────────────────── */
function GCSCard({ value }) {
  const st    = gcsStatus(value)
  const color = STATUS_COLORS[st]
  const isAbnormal = st !== 'normal'
  const displayValue = value != null ? value : '—'
  const gcsLabel = value != null ? (GCS_LABELS[value] ?? '—') : '—'

  return (
    <motion.div
      className="flex flex-col"
      whileHover={{ y: -4, scale: 1.012 }}
      whileTap={{ scale: 0.97 }}
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
        position: 'relative', overflow: 'hidden', cursor: 'default',
      }}
    >
      {/* Top band */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: 3,
        background: isAbnormal
          ? `linear-gradient(to right, ${color}, ${a(color, 40)})`
          : `linear-gradient(to right, var(--b1), transparent)`,
        borderRadius: 'var(--r-xl) var(--r-xl) 0 0',
      }} />

      {/* Icon + label */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14, marginTop: 2 }}>
        <div style={{
          width: 44, height: 44, borderRadius: 13, flexShrink: 0,
          background: `linear-gradient(145deg, ${a(color, 12)}, ${a(color, 4)})`,
          border: `1.5px solid ${a(color, 16)}`,
          color, display: 'flex', alignItems: 'center', justifyContent: 'center',
          animation: 'eyeBlink 4.5s ease-in-out infinite',
        }}>
          <div style={{ width: 21, height: 21 }}><GCSEyeIcon /></div>
        </div>
        <span style={{
          fontSize: '10px', fontWeight: 700, letterSpacing: '0.09em',
          textTransform: 'uppercase', color: 'var(--t3)',
          fontFamily: 'var(--font-display)', lineHeight: 1.35,
        }}>
          GCS Eye<br />Opening
        </span>
      </div>

      {/* Big value */}
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 5, marginBottom: 12 }}>
        <span className="num" style={{
          fontSize: '3.1rem', fontWeight: 800, letterSpacing: '-0.04em',
          color: isAbnormal ? color : 'var(--t1)', lineHeight: 1,
        }}>
          {displayValue}
        </span>
        <span style={{ fontSize: '12px', color: 'var(--t3)', fontWeight: 500 }}>/4</span>
      </div>

      {/* Response dots 1–4 */}
      <div style={{ display: 'flex', gap: 5, marginBottom: 10 }}>
        {[1, 2, 3, 4].map(n => (
          <div key={n} style={{
            flex: 1, height: 5, borderRadius: 3,
            background: value != null && n <= value
              ? (n <= 2 ? 'var(--danger-l)' : 'var(--ok-l)')
              : 'var(--b2)',
            transition: 'background 0.35s',
            opacity: value != null && n <= value ? 1 : 0.3,
          }} />
        ))}
      </div>

      {/* Status footer */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{
          fontSize: '10px', fontWeight: 700, color,
          fontFamily: 'var(--font-display)',
        }}>
          {gcsLabel}
        </span>
        <span style={{
          fontSize: '10px', padding: '2px 7px', borderRadius: 6,
          color: 'var(--t3)', background: 'var(--s2)', border: '1px solid var(--b1)',
        }}>
          E1–E4
        </span>
      </div>
    </motion.div>
  )
}

/* ── Clinical Note full-width card ──────────────────────────────────────────── */
function ClinicalNoteCard({ note }) {
  const noteColor = 'var(--cyan-l)'
  const hasNote = note && note.trim().length > 0 && note !== 'No clinical notes provided'
  const displayNote = hasNote ? note : 'No clinical notes on record for this reading.'

  return (
    <motion.div
      variants={tileVariants}
      style={{
        gridColumn: '1 / -1',
        background: `linear-gradient(145deg, ${a(noteColor, 4)} 0%, var(--s1) 50%, var(--s0) 100%)`,
        border: `1px solid ${a(noteColor, 18)}`,
        borderRadius: 'var(--r-xl)',
        padding: '18px 22px',
        boxShadow: `var(--sh-sm), inset 0 1px 0 ${a(noteColor, 8)}`,
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Top band */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: 3,
        background: `linear-gradient(to right, ${noteColor}, ${a(noteColor, 30)}, transparent 70%)`,
        borderRadius: 'var(--r-xl) var(--r-xl) 0 0',
      }} />

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10, flexShrink: 0,
            background: `linear-gradient(145deg, ${a(noteColor, 14)}, ${a(noteColor, 5)})`,
            border: `1.5px solid ${a(noteColor, 20)}`,
            color: noteColor, display: 'flex', alignItems: 'center', justifyContent: 'center',
            animation: 'noteFloat 4.2s ease-in-out infinite',
          }}>
            <div style={{ width: 18, height: 18 }}><NoteIcon /></div>
          </div>
          <div>
            <span style={{
              display: 'block',
              fontSize: '10.5px', fontWeight: 700, letterSpacing: '0.09em',
              textTransform: 'uppercase', color: 'var(--t2)',
              fontFamily: 'var(--font-display)',
            }}>
              Clinical Note
            </span>
            <span style={{ fontSize: '9.5px', color: 'var(--t4)', fontFamily: 'var(--font-display)' }}>
              nursing · physician · discharge summary
            </span>
          </div>
        </div>

        <span style={{
          fontSize: '9px', fontWeight: 700, padding: '3px 9px',
          borderRadius: 999, fontFamily: 'var(--font-display)',
          color: noteColor, background: a(noteColor, 10),
          border: `1px solid ${a(noteColor, 22)}`,
          letterSpacing: '0.05em', textTransform: 'uppercase', flexShrink: 0,
        }}>
          Static between readings
        </span>
      </div>

      {/* Note text */}
      <div style={{
        background: a(noteColor, 4),
        border: `1px solid ${a(noteColor, 12)}`,
        borderRadius: 10,
        padding: '12px 16px',
        position: 'relative',
      }}>
        {/* Left accent line */}
        <div style={{
          position: 'absolute', left: 0, top: 8, bottom: 8, width: 3,
          background: `linear-gradient(to bottom, ${noteColor}, ${a(noteColor, 30)})`,
          borderRadius: 2,
        }} />
        <p style={{
          margin: 0,
          fontSize: '12.5px',
          lineHeight: 1.65,
          color: 'var(--t2)',
          fontFamily: 'var(--font-display)',
          paddingLeft: 8,
          display: '-webkit-box',
          WebkitLineClamp: 3,
          WebkitBoxOrient: 'vertical',
          overflow: 'hidden',
          wordBreak: 'break-word',
        }}>
          {displayNote}
        </p>
      </div>
    </motion.div>
  )
}

/* ── Main grid ────────────────────────────────────────────────────────────────── */
/*
 * Bento layout (4-column grid):
 *   Row 1: [HR — span 2]           [SpO₂ — span 2]
 *   Row 2: [RR] [Temp] [SBP] [MAP]
 *   Row 3: [DBP] [GCS Eye] [Stress Score — span 2]
 *   Row 4: [Clinical Note — span 4]
 */
export default function VitalsGrid({ predictData }) {
  const { vitals = {}, vitals_status = {}, clinical_note } = predictData

  const gcsVal    = vitals.GCS_eye
  const stressVal = vitals.stress

  return (
    <motion.div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: '12px',
      }}
      variants={gridVariants}
      initial="hidden"
      animate="visible"
    >
      {/* ── Row 1 + 2: existing vitals ── */}
      {VITALS_CONFIG.map(({ key, label, Icon, unit, normalLo, normalHi, wide }, i) => (
        <motion.div
          key={key}
          style={{ gridColumn: wide ? 'span 2' : 'span 1' }}
          variants={tileVariants}
          data-aos="fade-up"
          data-aos-delay={i * 55}
          data-aos-duration="500"
        >
          <VitalCard
            Icon={Icon}
            label={label}
            value={vitals[key]}
            unit={unit}
            status={vitals_status[key] ?? 'normal'}
            normalLo={normalLo}
            normalHi={normalHi}
            wide={wide}
          />
        </motion.div>
      ))}

      {/* ── Row 3: DBP ── */}
      <motion.div
        style={{ gridColumn: 'span 1' }}
        variants={tileVariants}
        data-aos="fade-up"
        data-aos-delay={380}
        data-aos-duration="500"
      >
        <VitalCard
          Icon={DBPIcon}
          label="Dia. BP"
          value={vitals.DBP}
          unit="mmHg"
          status={vitals_status.DBP ?? 'normal'}
          normalLo={60}
          normalHi={80}
          wide={false}
        />
      </motion.div>

      {/* ── Row 3: GCS Eye ── */}
      <motion.div
        style={{ gridColumn: 'span 1' }}
        variants={tileVariants}
        data-aos="fade-up"
        data-aos-delay={440}
        data-aos-duration="500"
      >
        <GCSCard value={gcsVal} />
      </motion.div>

      {/* ── Row 3: Stress Score (wide) ── */}
      <motion.div
        style={{ gridColumn: 'span 2' }}
        variants={tileVariants}
        data-aos="fade-up"
        data-aos-delay={500}
        data-aos-duration="500"
      >
        <StressCard value={stressVal} />
      </motion.div>

      {/* ── Row 4: Clinical Note (full width) ── */}
      <motion.div
        style={{ gridColumn: '1 / -1' }}
        variants={tileVariants}
        data-aos="fade-up"
        data-aos-delay={560}
        data-aos-duration="500"
      >
        <ClinicalNoteCard note={clinical_note} />
      </motion.div>
    </motion.div>
  )
}
