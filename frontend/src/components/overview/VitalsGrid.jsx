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

/*
 * Bento layout:
 *   [  HR — wide (span 2)  ]  [  SpO2 — wide (span 2)  ]
 *   [ RR ]  [ Temp ]  [ SBP ]  [ MAP ]
 *
 * On smaller screens the grid collapses to 2-col, wide cards remain span 2.
 */
const VITALS_CONFIG = [
  { key: 'HR',   label: 'Heart Rate',   Icon: HRIcon,   unit: 'bpm',  normalLo: 60,   normalHi: 100,  wide: true  },
  { key: 'SpO2', label: 'SpO₂',         Icon: SpO2Icon, unit: '%',    normalLo: 95,   normalHi: 100,  wide: true  },
  { key: 'RR',   label: 'Resp. Rate',   Icon: RRIcon,   unit: '/min', normalLo: 12,   normalHi: 20,   wide: false },
  { key: 'Temp', label: 'Temperature',  Icon: TempIcon, unit: '°C',   normalLo: 36.0, normalHi: 37.5, wide: false },
  { key: 'SBP',  label: 'Sys. BP',      Icon: SBPIcon,  unit: 'mmHg', normalLo: 90,   normalHi: 120,  wide: false },
  { key: 'MAP',  label: 'MAP',          Icon: MAPIcon,  unit: 'mmHg', normalLo: 70,   normalHi: 100,  wide: false },
]

export default function VitalsGrid({ predictData }) {
  const { vitals = {}, vitals_status = {} } = predictData

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
    </motion.div>
  )
}
