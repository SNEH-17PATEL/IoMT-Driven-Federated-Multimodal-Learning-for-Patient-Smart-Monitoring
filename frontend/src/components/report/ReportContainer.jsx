import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { generateReport } from '../../api/client.js'
import ReliabilityBadge from './ReliabilityBadge.jsx'
import ReportSection from './ReportSection.jsx'
import SHAPPanel from './SHAPPanel.jsx'

/* Build a stable key from the values that actually matter for the report.
   LLM is only re-called when this key changes OR when careMode changes. */
function reportKey(predictData) {
  if (!predictData) return null
  const v = predictData.vitals ?? {}
  return [
    predictData.patient_id,
    (predictData.sofa_score ?? 0).toFixed(1),
    Math.round(v.HR  ?? 0),
    Math.round(v.SpO2 ?? 0),
    Math.round(v.SBP  ?? 0),
    Math.round(v.GCS  ?? 0),
  ].join('|')
}

const reportGridVariants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.10, delayChildren: 0.06 } },
}
const reportCardVariants = {
  hidden: { opacity: 0, y: 22, scale: 0.97 },
  visible: { opacity: 1, y: 0, scale: 1, transition: { type: 'spring', stiffness: 280, damping: 26 } },
}

const SECTIONS = [
  { key: 'CURRENT CONDITION', subtitle: 'Patient status right now',        color: 'var(--blue-l)'   },
  { key: 'PROBABLE CAUSE',    subtitle: 'Likely contributing factors',      color: 'var(--warn-l)'   },
  { key: 'RISK FORECAST',     subtitle: 'Expected trajectory if untreated', color: '#9b59b6'         },
  { key: 'IMMEDIATE ACTIONS', subtitle: 'Recommended next steps',           color: 'var(--danger-l)' },
]


function ReportSkeleton() {
  return (
    <div className="space-y-4">
      <div className="h-16 rounded-xl skeleton" />
      <p className="text-xs text-center" style={{ color: 'var(--t3)' }}>
        Generating AI assessment — this may take ~5s…
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {[0, 1, 2, 3].map(i => (
          <div key={i} className="h-48 rounded-xl skeleton" />
        ))}
      </div>
      <div className="h-44 rounded-xl skeleton" />
    </div>
  )
}

function ReportErrorState({ message }) {
  const isRateLimit = message && (
    message.includes('429') || message.includes('rate_limit') || message.includes('Rate limit')
  )
  return (
    <div style={{
      borderRadius: 'var(--r-xl)',
      background: 'var(--s1)',
      border: '1px solid var(--b1)',
      padding: '32px 24px',
      textAlign: 'center',
      fontFamily: 'var(--font-display)',
    }}>
      <div style={{ fontSize: '26px', marginBottom: 10, opacity: 0.5 }}>⚠</div>
      <p style={{ fontWeight: 700, color: 'var(--t1)', fontSize: '14px', marginBottom: 8 }}>
        AI Assessment Unavailable
      </p>
      <p style={{ fontSize: '12px', color: 'var(--t3)', maxWidth: 460, margin: '0 auto', lineHeight: 1.6 }}>
        {isRateLimit
          ? 'The AI model\'s daily token quota has been reached. The report will regenerate automatically on the next reading cycle once the quota resets (usually within 24 h).'
          : 'The AI service returned an error. It will retry automatically on the next reading.'}
      </p>
    </div>
  )
}

function StaleReportNotice({ message }) {
  return (
    <div
      className="flex items-center gap-2 rounded-lg px-4 py-2 mb-4 text-xs"
      style={{
        background: 'var(--warn-bg)',
        border: '1px solid var(--warn)',
        color: 'var(--warn-l)',
        borderColor: `color-mix(in srgb, var(--warn) 50%, transparent)`,
      }}
    >
      <span>⚠</span>
      <span>{message}</span>
    </div>
  )
}

export default function ReportContainer({ predictData, careMode, onReportData }) {
  const [reportData,     setReportData]     = useState(null)
  const [prevReportData, setPrevReportData] = useState(null)
  const [isLoading,      setIsLoading]      = useState(false)
  const [error,          setError]          = useState(null)

  /* Track the last key + careMode that actually triggered an LLM call.
     New vitals every 90s produce a new predictData object — we only want
     to call the LLM when the clinical values (SOFA, HR, SpO2, SBP, GCS)
     genuinely change, or when the user switches care mode. */
  const lastKeyRef      = useRef(null)
  const lastCareModeRef = useRef(null)
  // Each actual API call gets a unique ID. Old responses are ignored only
  // when a *newer* call has started — not just because predictData re-fired
  // with the same clinical values (which would set isCurrent=false via cleanup
  // even though no new request was made, permanently burying the response).
  const reqIdRef = useRef(0)

  useEffect(() => {
    if (!predictData) return

    const key           = reportKey(predictData)
    const modeChanged   = careMode !== lastCareModeRef.current
    const valuesChanged = key !== lastKeyRef.current

    // Skip LLM call if neither clinical values nor care mode changed
    if (!modeChanged && !valuesChanged) return

    lastKeyRef.current      = key
    lastCareModeRef.current = careMode

    // Stamp this request; any older in-flight response will see a stale ID
    const myId = ++reqIdRef.current

    setIsLoading(true)
    setError(null)

    generateReport(predictData.patient_id, careMode, predictData)
      .then(r => {
        if (reqIdRef.current !== myId) return
        setReportData(r.data)
        setPrevReportData(r.data)
        setIsLoading(false)
        onReportData?.(r.data)
      })
      .catch(() => {
        if (reqIdRef.current !== myId) return
        setError('Report generation failed. Showing previous assessment.')
        setIsLoading(false)
      })
  }, [predictData, careMode])

  if (isLoading && !prevReportData) return <ReportSkeleton />

  if (error && !prevReportData) {
    return (
      <div
        className="rounded-xl p-6 text-sm"
        style={{
          background: 'var(--danger-bg)',
          border: '1px solid var(--danger)',
          color: 'var(--danger-l)',
          borderRadius: 'var(--r-lg)',
        }}
      >
        ⚠ {error}
      </div>
    )
  }

  const display = reportData ?? prevReportData
  if (!display) return null

  const { sections, reliability_score, reliability_label, reliability_color, caution, error: reportError } = display
  const hasNoSections = !sections || !Object.values(sections).some(v => v)

  return (
    <div className="space-y-4">
      {isLoading && (
        <StaleReportNotice message="Updating report for new reading…" />
      )}
      {error && (
        <StaleReportNotice message={error} />
      )}

      {/* API returned an error and there's no content to show */}
      {reportError && hasNoSections ? (
        <ReportErrorState message={reportError} />
      ) : (
        <div style={{ opacity: isLoading ? 0.75 : 1, transition: 'opacity 0.3s' }}>
          {reliability_score != null && (
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ type: 'spring', stiffness: 300, damping: 28 }}
            >
              <ReliabilityBadge
                reliability_score={reliability_score}
                reliability_label={reliability_label}
                reliability_color={reliability_color}
                caution={caution}
              />
            </motion.div>
          )}

          <motion.div
            className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4"
            variants={reportGridVariants}
            initial="hidden"
            animate="visible"
          >
            {SECTIONS.map(({ key, subtitle, color }, i) => (
              <motion.div
                key={key}
                variants={reportCardVariants}
                data-aos={i % 2 === 0 ? 'fade-right' : 'fade-left'}
                data-aos-delay={i * 80}
                data-aos-duration="600"
              >
                <ReportSection
                  name={key}
                  content={sections?.[key] ?? ''}
                  color={color}
                  subtitle={subtitle}
                />
              </motion.div>
            ))}
          </motion.div>

          <SHAPPanel factors={predictData?.shap_top5} />
        </div>
      )}
    </div>
  )
}
