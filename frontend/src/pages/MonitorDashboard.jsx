import { useState, useEffect } from 'react'
import { useParams, Navigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import AOS from 'aos'
import { getPatient } from '../api/client.js'
import { useMonitor } from '../hooks/useMonitor.js'
import HeaderBar from '../components/layout/HeaderBar.jsx'
import AlertBanner from '../components/layout/AlertBanner.jsx'
import SOFAGauge from '../components/overview/SOFAGauge.jsx'
import VitalsGrid from '../components/overview/VitalsGrid.jsx'
import TrendsSection from '../components/trends/TrendsSection.jsx'
import ReportToggle from '../components/report/ReportToggle.jsx'
import ReportContainer from '../components/report/ReportContainer.jsx'
import HistoryDrawer from '../components/history/HistoryDrawer.jsx'
import HistoryFAB from '../components/history/HistoryFAB.jsx'
import CVMonitorFAB from '../components/cv/CVMonitorFAB.jsx'
import CVMonitorModal from '../components/cv/CVMonitorModal.jsx'
import SOFAValidationDrawer from '../components/validation/SOFAValidationDrawer.jsx'
import LLMValidationDrawer from '../components/validation/LLMValidationDrawer.jsx'
import SOFAValidationFAB from '../components/validation/SOFAValidationFAB.jsx'
import LLMValidationFAB from '../components/validation/LLMValidationFAB.jsx'

/* ─────────────────────────────────────────────────────────────────────────── */
/* Clean SVG icons — no emojis, clinical precision                            */
/* ─────────────────────────────────────────────────────────────────────────── */
const StatusIcon = () => (
  <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <rect x="1" y="1" width="6" height="6" rx="1.5"/>
    <rect x="9" y="1" width="6" height="6" rx="1.5"/>
    <rect x="1" y="9" width="6" height="6" rx="1.5"/>
    <rect x="9" y="9" width="6" height="6" rx="1.5"/>
  </svg>
)
const TrendsIcon = () => (
  <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <polyline points="1,12 5,7 8,9 12,4 15,5"/>
    <line x1="1" y1="14" x2="15" y2="14"/>
  </svg>
)
const ReportIcon = () => (
  <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <circle cx="8" cy="8" r="6.5"/>
    <path d="M8 5v3.5l2.5 1.5"/>
  </svg>
)

/* ─────────────────────────────────────────────────────────────────────────── */
/* Section heading — architectural: weight and size over all-caps decoration  */
/* ─────────────────────────────────────────────────────────────────────────── */
function SectionHeading({ children, accent, sub, Icon }) {
  const ac = accent ?? 'var(--blue-l)'
  return (
    <div className="flex items-center gap-3.5 mb-7">
      {/* Accent bar */}
      <div style={{
        width: 3, height: 28, borderRadius: 2, flexShrink: 0,
        background: `linear-gradient(to bottom, ${ac}, color-mix(in srgb, ${ac} 15%, transparent))`,
        boxShadow: `0 0 10px ${ac}50`,
      }} />

      {/* Icon badge */}
      {Icon && (
        <div style={{
          width: 30, height: 30, borderRadius: 8, flexShrink: 0,
          background: `color-mix(in srgb, ${ac} 12%, transparent)`,
          border: `1px solid color-mix(in srgb, ${ac} 25%, transparent)`,
          color: ac, display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <Icon />
        </div>
      )}

      <div>
        <h2 style={{
          fontSize: '16px', fontWeight: 700, letterSpacing: '-0.02em',
          color: 'var(--t1)', fontFamily: 'var(--font-display)', margin: 0, lineHeight: 1.2,
        }}>
          {children}
        </h2>
        {sub && (
          <span style={{ fontSize: '11px', color: 'var(--t3)', fontWeight: 400, fontFamily: 'var(--font-display)' }}>
            {sub}
          </span>
        )}
      </div>
    </div>
  )
}

function Divider() {
  return (
    <div className="my-12 flex items-center gap-4">
      <div style={{ flex: 1, height: 1, background: 'linear-gradient(to right, transparent, var(--b2) 60%, var(--b1))' }} />
      <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
        <div style={{ width: 3, height: 3, borderRadius: '50%', background: 'var(--b3)', flexShrink: 0 }} />
        <div style={{ width: 5, height: 5, borderRadius: '50%', background: 'var(--b2)', flexShrink: 0 }} />
        <div style={{ width: 3, height: 3, borderRadius: '50%', background: 'var(--b3)', flexShrink: 0 }} />
      </div>
      <div style={{ flex: 1, height: 1, background: 'linear-gradient(to left, transparent, var(--b2) 60%, var(--b1))' }} />
    </div>
  )
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Patient context strip — first thing after the alert banner                 */
/* ─────────────────────────────────────────────────────────────────────────── */
function Stat({ label, value, riskColor, unit }) {
  return (
    <div className="flex flex-col justify-center px-5 py-4 flex-shrink-0">
      <span style={{
        fontSize: '9.5px', color: 'var(--t4)', fontFamily: 'var(--font-display)',
        fontWeight: 700, letterSpacing: '0.10em', textTransform: 'uppercase', marginBottom: 4,
      }}>
        {label}
      </span>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 3 }}>
        <span className="num" style={{ fontWeight: 800, fontSize: '16px', color: riskColor ?? 'var(--t1)', letterSpacing: '-0.03em' }}>
          {value}
        </span>
        {unit && <span style={{ fontSize: '11px', color: 'var(--t3)' }}>{unit}</span>}
      </div>
    </div>
  )
}

function PatientContextStrip({ patient, predictData, isRefreshing }) {
  if (!patient || !predictData) return null

  const riskColor = predictData.risk_color
  const sofa      = predictData.sofa_score
  const risk      = predictData.risk_level
  const conf      = predictData.confidence
  const confPct   = conf != null ? `${Math.round(conf * 100)}%` : '—'
  const isHighRisk = risk !== 'Low Risk'

  return (
    <div
      className="flex flex-wrap items-stretch mb-8 overflow-hidden"
      style={{
        borderRadius: 'var(--r-xl)',
        background: 'var(--s1)',
        border: '1px solid var(--b1)',
        borderLeft: `3px solid ${riskColor}`,
        boxShadow: `var(--sh-md), 0 0 0 1px ${riskColor}08`,
        opacity: isRefreshing ? 0.78 : 1,
        transition: 'opacity 0.38s ease',
      }}
    >
      {/* Patient identity */}
      <div className="flex items-center gap-3.5 px-5 py-4 flex-1 min-w-0">
        <div style={{
          width: 42, height: 42, borderRadius: 11, flexShrink: 0,
          background: `${riskColor}18`,
          border: `1.5px solid ${riskColor}45`,
          color: riskColor, fontFamily: 'var(--font-display)', fontWeight: 800,
          fontSize: '13px', display: 'flex', alignItems: 'center', justifyContent: 'center',
          letterSpacing: '-0.02em',
          boxShadow: `0 4px 14px ${riskColor}15`,
        }}>
          0{patient.id}
        </div>
        <div className="min-w-0">
          <p
            style={{ fontWeight: 700, fontSize: '15px', color: 'var(--t1)', margin: 0, lineHeight: 1.2, fontFamily: 'var(--font-display)', letterSpacing: '-0.01em' }}
            className="truncate"
          >
            {patient.name}
          </p>
          <p style={{ fontSize: '11.5px', color: 'var(--t3)', margin: '3px 0 0', lineHeight: 1.4 }} className="truncate">
            {patient.description}
          </p>
        </div>
      </div>

      {/* Dividers + stat columns */}
      <div style={{ width: 1, background: 'var(--b1)', flexShrink: 0, alignSelf: 'stretch' }} />
      <div className="flex items-center">
        <div style={{ display: 'flex', alignItems: 'center', gap: 3, flexShrink: 0, padding: '0 20px' }}>
          <span style={{
            width: 7, height: 7, borderRadius: '50%', background: riskColor, display: 'inline-block', flexShrink: 0,
            boxShadow: `0 0 6px ${riskColor}`,
            animation: isHighRisk ? 'pulse 1.6s ease-in-out infinite' : 'none',
          }} />
          <div>
            <span style={{ display: 'block', fontSize: '9.5px', color: 'var(--t4)', fontFamily: 'var(--font-display)', fontWeight: 700, letterSpacing: '0.10em', textTransform: 'uppercase', marginBottom: 2 }}>
              Risk Level
            </span>
            <span style={{ fontWeight: 700, fontSize: '13px', color: riskColor, fontFamily: 'var(--font-display)', letterSpacing: '-0.01em' }}>
              {risk}
            </span>
          </div>
        </div>
      </div>

      <div style={{ width: 1, background: 'var(--b1)', flexShrink: 0, alignSelf: 'stretch' }} />
      <Stat label="SOFA Score" value={sofa?.toFixed(1) ?? '—'} unit="/24" />

      <div style={{ width: 1, background: 'var(--b1)', flexShrink: 0, alignSelf: 'stretch' }} />
      <Stat label="AI Confidence" value={confPct} />
    </div>
  )
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Loading skeleton                                                           */
/* ─────────────────────────────────────────────────────────────────────────── */
function LoadingSkeleton() {
  return (
    <div className="space-y-10 animate-fade-in">
      <div className="h-16 rounded-2xl skeleton" />
      <div>
        <div className="h-3 w-36 mb-6 skeleton" />
        <div className="grid grid-cols-1 lg:grid-cols-[300px_1fr] gap-5">
          <div className="rounded-2xl h-64 skeleton" />
          <div className="grid grid-cols-4 gap-3">
            <div className="col-span-2 rounded-2xl h-28 skeleton" />
            <div className="col-span-2 rounded-2xl h-28 skeleton" />
            <div className="rounded-2xl h-32 skeleton" />
            <div className="rounded-2xl h-32 skeleton" />
            <div className="rounded-2xl h-32 skeleton" />
            <div className="rounded-2xl h-32 skeleton" />
          </div>
        </div>
      </div>
      <div>
        <div className="h-3 w-48 mb-6 skeleton" />
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {Array.from({ length: 6 }, (_, i) => <div key={i} className="rounded-2xl h-52 skeleton" />)}
        </div>
      </div>
      <div>
        <div className="h-3 w-40 mb-6 skeleton" />
        <div className="h-20 rounded-2xl skeleton mb-4" />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Array.from({ length: 4 }, (_, i) => <div key={i} className="rounded-2xl h-52 skeleton" />)}
        </div>
      </div>
    </div>
  )
}

function RefreshErrorBanner({ message }) {
  return (
    <div
      className="flex items-center gap-3 rounded-xl px-4 py-3 mb-6 text-sm"
      style={{ background: 'var(--warn-bg)', border: '1px solid var(--warn)50', color: 'var(--warn-l)', fontFamily: 'var(--font-display)' }}
    >
      ⚠ {message}
    </div>
  )
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Page                                                                       */
/* ─────────────────────────────────────────────────────────────────────────── */
export default function MonitorDashboard() {
  const { patientId } = useParams()
  const id = parseInt(patientId, 10)

  if (![1, 2, 3].includes(id)) return <Navigate to="/" replace />

  const { data: patient } = useQuery({
    queryKey: ['patient', id],
    queryFn: () => getPatient(id).then(r => r.data),
  })

  const { predictData, isRefreshing, lastUpdated, countdown, error } = useMonitor(id)
  const [careMode, setCareMode] = useState('hospital')
  const [historyOpen, setHistoryOpen] = useState(false)
  const [cvOpen, setCvOpen] = useState(false)
  const [sofaValidOpen, setSofaValidOpen] = useState(false)
  const [llmValidOpen, setLlmValidOpen] = useState(false)
  const [llmReportData, setLlmReportData] = useState(null)
  const [cvLiveData, setCvLiveData] = useState(null) // { stress, gcs_eye } from CV monitor

  /* Refresh AOS whenever new data renders so scroll-reveal picks up late-mounted elements */
  useEffect(() => {
    if (predictData) {
      setTimeout(() => AOS.refresh(), 120)
    }
  }, [predictData])

  return (
    <div className="ambient-page min-h-screen" style={{ background: 'var(--base)' }}>
      <HeaderBar
        patient={patient}
        predictData={predictData}
        isRefreshing={isRefreshing}
        lastUpdated={lastUpdated}
        countdown={countdown}
        error={error}
      />

      <AnimatePresence>
        {predictData && <AlertBanner key="alert" predictData={predictData} />}
      </AnimatePresence>

      <main className="max-w-7xl mx-auto px-5 py-8">
        {error && <RefreshErrorBanner message={error} />}

        {/* ── Section 1: Current Status — needs predictData ── */}
        {!predictData ? (
          <LoadingSkeleton />
        ) : (
          <>
            {/* ── Patient context strip ── */}
            <motion.div
              initial={{ opacity: 0, y: -16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ type: 'spring', stiffness: 320, damping: 28, delay: 0.05 }}
            >
              <PatientContextStrip
                patient={patient}
                predictData={predictData}
                isRefreshing={isRefreshing}
              />
            </motion.div>

            <motion.section
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ type: 'spring', stiffness: 260, damping: 26, delay: 0.12 }}
            >
              <SectionHeading accent="var(--blue-l)" Icon={StatusIcon}>Current Status</SectionHeading>
              <div
                className="grid grid-cols-1 lg:grid-cols-[300px_1fr] gap-5"
                style={{ opacity: isRefreshing ? 0.72 : 1, transition: 'opacity 0.38s' }}
              >
                <SOFAGauge predictData={predictData} isRefreshing={isRefreshing} />
                <VitalsGrid predictData={predictData} isRefreshing={isRefreshing} />
              </div>
            </motion.section>
          </>
        )}

        <Divider />

        {/* ── Section 2: Vital Trends — independent, fires immediately ── */}
        <section>
          <SectionHeading accent="var(--teal-l)" sub="last 20 readings" Icon={TrendsIcon}>Vital Trends</SectionHeading>
          <TrendsSection patientId={id} />
        </section>

        <Divider />

        {/* ── Section 3: AI Clinical Report — needs predictData ── */}
        {predictData && (
          <motion.section
            initial={{ opacity: 0, y: 28 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.12 }}
            transition={{ type: 'spring', stiffness: 240, damping: 26, delay: 0.04 }}
          >
            <div className="flex items-center justify-between mb-7">
              <div className="flex items-center gap-3.5">
                <div style={{
                  width: 3, height: 28, borderRadius: 2, flexShrink: 0,
                  background: 'linear-gradient(to bottom, var(--cyan-l), color-mix(in srgb, var(--cyan-l) 15%, transparent))',
                  boxShadow: '0 0 10px color-mix(in srgb, var(--cyan-l) 50%, transparent)',
                }} />
                <div style={{
                  width: 30, height: 30, borderRadius: 8, flexShrink: 0,
                  background: 'color-mix(in srgb, var(--cyan-l) 12%, transparent)',
                  border: '1px solid color-mix(in srgb, var(--cyan-l) 25%, transparent)',
                  color: 'var(--cyan-l)', display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                  <ReportIcon />
                </div>
                <h2 style={{ fontSize: '16px', fontWeight: 700, letterSpacing: '-0.02em', color: 'var(--t1)', fontFamily: 'var(--font-display)', margin: 0, lineHeight: 1.2 }}>
                  AI Clinical Report
                </h2>
              </div>
              <ReportToggle careMode={careMode} setCareMode={setCareMode} />
            </div>

            <ReportContainer
              predictData={predictData}
              careMode={careMode}
              onReportData={setLlmReportData}
            />

            <p
              className="text-center text-xs mt-10 pb-4"
              style={{ color: 'var(--t3)', fontFamily: 'var(--font-display)' }}
            >
              Research prototype · All clinical decisions require qualified healthcare professionals
            </p>
          </motion.section>
        )}
      </main>

      {/* ── FAB stack (right rail, bottom → top: History / CV / LLM / SOFA) ── */}
      {predictData && (
        <>
          <HistoryFAB
            onClick={() => setHistoryOpen(true)}
            recordCount={0}
          />
          <CVMonitorFAB
            onClick={() => setCvOpen(v => !v)}
            isActive={cvOpen}
          />
          <LLMValidationFAB
            onClick={() => setLlmValidOpen(true)}
            hasData={!!llmReportData}
          />
          <SOFAValidationFAB
            onClick={() => setSofaValidOpen(true)}
          />
        </>
      )}

      {/* ── CV Monitor Modal ── */}
      <CVMonitorModal open={cvOpen} onClose={() => setCvOpen(false)} onCVResults={setCvLiveData} />

      {/* ── History drawer ── */}
      <HistoryDrawer
        patientId={id ? Number(id) : null}
        open={historyOpen}
        onClose={() => setHistoryOpen(false)}
      />

      {/* ── SOFA Validation drawer ── */}
      <SOFAValidationDrawer
        open={sofaValidOpen}
        onClose={() => setSofaValidOpen(false)}
        predictData={predictData}
      />

      {/* ── LLM Validation drawer ── */}
      <LLMValidationDrawer
        open={llmValidOpen}
        onClose={() => setLlmValidOpen(false)}
        reportData={llmReportData}
        careMode={careMode}
      />
    </div>
  )
}
