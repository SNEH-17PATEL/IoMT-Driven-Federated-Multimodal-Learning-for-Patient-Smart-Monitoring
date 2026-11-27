import { useState, useEffect, useLayoutEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { generateReport } from '../../api/client.js'
import ReliabilityBadge from './ReliabilityBadge.jsx'
import ReportSection from './ReportSection.jsx'
import SHAPPanel from './SHAPPanel.jsx'

const CARE_MODES = ['hospital', 'homecare']

/* Identifies one reading: every value the report prompt quotes.
   Keep in sync with _cache_key() in backend/services/llm_service.py. */
function reportKey(predictData) {
  if (!predictData) return null
  const v = predictData.vitals ?? {}
  return [
    predictData.patient_id,
    (predictData.sofa_score ?? 0).toFixed(1),
    Math.round(v.HR      ?? 0),
    Math.round(v.RR      ?? 0),
    Math.round(v.SpO2    ?? 0),
    Math.round(v.SBP     ?? 0),
    Math.round(v.DBP     ?? 0),
    Math.round(v.MAP     ?? 0),
    (v.Temp ?? 0).toFixed(1),
    Math.round(v.GCS_eye ?? 0),
    Math.round(v.stress  ?? 0),
  ].join('|')
}

const hasSections = (d) => !!d?.sections && Object.values(d.sections).some(Boolean)

function errorMessage(err) {
  const status = err?.response?.status
  const detail = err?.response?.data?.detail ?? err?.message ?? null
  const base = status === 401 ? 'Invalid API key'
    : status === 429 ? 'Rate limit reached — resets in ~24h'
    : status === 503 ? 'AI service temporarily unavailable'
    : detail ?? 'Report generation failed'
  return `${base}. Showing previous assessment.`
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
        Generating AI assessment — this may take ~30s…
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

  /* Reports for the current reading, one per care mode:
       "<reportKey>|<careMode>" → { promise, data }
     `promise` is shared, so a toggle (or StrictMode's double effect) while a
     request is in flight joins it instead of starting another. Only successful
     reports get `data`; failures are dropped so the next reading retries. */
  const cacheRef         = useRef(new Map())
  const lastPatientIdRef = useRef(null)
  const boxRef           = useRef(null)
  const lastHeightRef    = useRef(0)

  function getEntry(pd, mode, key) {
    const ck    = `${key}|${mode}`
    const cache = cacheRef.current
    let entry   = cache.get(ck)
    if (!entry) {
      entry = { data: null }
      entry.promise = generateReport(pd.patient_id, mode).then(
        r => {
          const d = r.data
          if (cache.get(ck) === entry) {
            if (hasSections(d)) entry.data = d
            else cache.delete(ck)
          }
          return d
        },
        err => {
          if (cache.get(ck) === entry) cache.delete(ck)
          throw err
        },
      )
      entry.promise.catch(() => {})  // prefetches may fail with nobody awaiting
      cache.set(ck, entry)
    }
    return entry
  }

  // H2: when patient changes, immediately clear the displayed report so
  // the previous patient's AI assessment is never shown under the new patient's header.
  useEffect(() => {
    if (!predictData) return
    const pid = predictData.patient_id
    if (lastPatientIdRef.current !== null && lastPatientIdRef.current !== pid) {
      setReportData(null)
      setPrevReportData(null)
      lastHeightRef.current = 0
    }
    lastPatientIdRef.current = pid
  }, [predictData?.patient_id])  // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!predictData) return
    const key = reportKey(predictData)

    // A new reading empties the cache: reports for older readings quote stale
    // vitals and must never be shown again.
    for (const ck of [...cacheRef.current.keys()]) {
      if (!ck.startsWith(`${key}|`)) cacheRef.current.delete(ck)
    }

    // Once this mode's report is ready, generate the other mode in the
    // background so switching Hospital ↔ Home Care is instant.
    const prefetchOther = () => {
      for (const m of CARE_MODES) if (m !== careMode) getEntry(predictData, m, key)
    }

    const entry = getEntry(predictData, careMode, key)
    if (entry.data) {
      setReportData(entry.data)
      setPrevReportData(entry.data)
      setError(null)
      setIsLoading(false)
      prefetchOther()
      return
    }

    let cancelled = false
    setError(null)
    setIsLoading(true)
    entry.promise.then(
      d => {
        if (cancelled) return
        setReportData(d)
        // A soft failure (sections: null) keeps the last good report visible.
        if (hasSections(d)) {
          setPrevReportData(d)
          prefetchOther()
        }
        setIsLoading(false)
      },
      err => {
        if (cancelled) return
        setError(errorMessage(err))
        setIsLoading(false)
      },
    )
    // Ignore this response if the mode or reading changes before it arrives
    return () => { cancelled = true }
  }, [predictData, careMode])  // eslint-disable-line react-hooks/exhaustive-deps

  // Read the cache during render so a cached mode switch paints the new report
  // on the same frame. Waiting for the effect would paint one blank frame, the
  // page would shrink, and the browser would drop the reader's scroll position.
  const cachedNow = predictData
    ? cacheRef.current.get(`${reportKey(predictData)}|${careMode}`)?.data ?? null
    : null
  // A reply with no sections (a soft failure) must not replace the last good report of the
  // SAME care mode. Prefer a reply that has content; otherwise keep the last good one; only
  // when there is nothing better show the failed reply (which renders the "unavailable" screen).
  const goodNow  = hasSections(reportData) ? reportData : null
  const lastGood = prevReportData && (!prevReportData.care_mode || prevReportData.care_mode === careMode)
    ? prevReportData : null
  const display = cachedNow ?? goodNow ?? lastGood ?? reportData ?? prevReportData
  // Never show (or publish) one care mode's report under the other mode's tab
  const wrongMode = !!display?.care_mode && display.care_mode !== careMode
  const shown = wrongMode ? null : display
  const loading = isLoading && !cachedNow

  // Remember the report's height; while the other mode is still generating,
  // the loading state keeps that height so the page doesn't shrink and jump.
  useLayoutEffect(() => {
    if (shown && boxRef.current) lastHeightRef.current = boxRef.current.offsetHeight
  })

  // Publish whatever report is on screen (or null) so the header strip and
  // validation drawer always show the same score as the visible report.
  useEffect(() => {
    onReportData?.(shown)
  }, [shown])  // eslint-disable-line react-hooks/exhaustive-deps

  if (!predictData) return null

  const boxStyle = shown ? undefined : { minHeight: lastHeightRef.current || undefined }
  const box = (children) => <div ref={boxRef} style={boxStyle}>{children}</div>

  if (loading && (!prevReportData || wrongMode)) return box(<ReportSkeleton />)

  if (error && (!prevReportData || wrongMode)) {
    return box(
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

  if (!shown) return null

  const { sections, reliability_score, reliability_label, reliability_color, caution, error: reportError } = shown
  const hasNoSections = !sections || !Object.values(sections).some(v => v)

  return box(
    <div className="space-y-4">
      {loading && (
        <StaleReportNotice message="Updating report for new reading…" />
      )}
      {error && (
        <StaleReportNotice message={error} />
      )}

      {/* API returned an error and there's no content to show */}
      {reportError && hasNoSections ? (
        <ReportErrorState message={reportError} />
      ) : (
        <div style={{ opacity: loading ? 0.75 : 1, transition: 'opacity 0.3s' }}>
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

          <SHAPPanel factors={predictData?.shap_top5} patientId={predictData?.patient_id} />
        </div>
      )}
    </div>
  )
}
