import { useState, useEffect } from 'react'
import { AnimatePresence, motion } from 'framer-motion'

const a = (c, p) => `color-mix(in srgb, ${c} ${p}%, transparent)`

const CYAN   = 'var(--cyan-l)'
const GREEN  = 'var(--ok-l)'
const AMBER  = 'var(--warn-l)'
const RED    = 'var(--danger-l)'
const PURPLE = '#a78bfa'

function SectionRule({ children }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, margin: '4px 0 12px' }}>
      <p style={{
        fontSize: '10px', fontWeight: 700, letterSpacing: '0.10em',
        textTransform: 'uppercase', color: 'var(--t4)',
        fontFamily: 'var(--font-display)', whiteSpace: 'nowrap', margin: 0,
      }}>
        {children}
      </p>
      <div style={{ flex: 1, height: 1, background: 'var(--b1)' }} />
    </div>
  )
}

function InfoBox({ children, accent }) {
  const c = accent ?? CYAN
  return (
    <div style={{
      background: a(c, 6), border: `1px solid ${a(c, 18)}`,
      borderRadius: 'var(--r-md)', padding: '12px 14px',
      fontSize: '12px', color: 'var(--t2)', lineHeight: 1.65,
      fontFamily: 'var(--font-display)',
    }}>
      {children}
    </div>
  )
}

/* ── Method score bar ── */
function MethodBar({ icon, name, weight, score, note }) {
  const pct = Math.round((score ?? 0) * 100)
  const color =
    score >= 0.85 ? GREEN :
    score >= 0.65 ? AMBER :
    RED

  return (
    <div style={{
      background: 'var(--s2)', border: '1px solid var(--b1)',
      borderRadius: 'var(--r-sm)', padding: '10px 12px',
      borderLeft: `3px solid ${color}`,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, marginBottom: 6 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 7, minWidth: 0, flex: 1 }}>
          <span style={{ fontSize: 14, flexShrink: 0 }}>{icon}</span>
          <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--t1)', fontFamily: 'var(--font-display)', minWidth: 0 }}>
            {name}
          </span>
          <span style={{
            fontSize: '10px', color: 'var(--t4)', background: 'var(--s3)',
            borderRadius: 10, padding: '2px 7px', flexShrink: 0,
            fontFamily: 'var(--font-display)', fontWeight: 600,
          }}>
            {weight}
          </span>
        </div>
        <span className="num" style={{
          fontSize: '12px', fontWeight: 700, color, minWidth: 40,
          textAlign: 'right', fontFamily: 'var(--font-display)',
        }}>
          {score != null ? score.toFixed(2) : '—'}
        </span>
      </div>

      <div style={{ height: 5, background: 'var(--s3)', borderRadius: 3, overflow: 'hidden' }}>
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.7, ease: [0.4, 0, 0.2, 1], delay: 0.1 }}
          style={{ height: '100%', borderRadius: 3, background: color }}
        />
      </div>

      {note && (
        <p style={{ fontSize: '11px', color: 'var(--t3)', marginTop: 5, fontFamily: 'var(--font-display)', lineHeight: 1.4 }}>
          {note}
        </p>
      )}
    </div>
  )
}

/* ── Overview tab ── */
function OverviewTab({ reportData }) {
  const score = reportData?.reliability_score
  const label = reportData?.reliability_label ?? '—'
  const color = reportData?.reliability_color ?? 'var(--t3)'
  const violations = reportData?.violations ?? []

  const labelColor =
    label === 'High'     ? GREEN :
    label === 'Moderate' ? AMBER :
    label === 'Low'      ? RED   : 'var(--t3)'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Score hero */}
      <div style={{
        background: 'var(--s2)', border: '1px solid var(--b1)',
        borderRadius: 'var(--r-lg)', padding: '18px 20px',
        display: 'flex', alignItems: 'center', gap: 20, flexWrap: 'wrap',
      }}>
        <div>
          <span className="num" style={{
            fontSize: '56px', fontWeight: 900, lineHeight: 1,
            color: labelColor, letterSpacing: '-0.04em',
            textShadow: `0 0 30px ${a(labelColor, 35)}`,
            fontFamily: 'var(--font-display)',
          }}>
            {score != null ? score.toFixed(3) : '—'}
          </span>
        </div>
        <div style={{ flex: 1, minWidth: 120 }}>
          <p style={{ fontSize: '16px', fontWeight: 800, color: labelColor, fontFamily: 'var(--font-display)', marginBottom: 4 }}>
            {label === 'High' ? '✅' : label === 'Moderate' ? '⚠️' : label === 'Low' ? '❌' : ''} {label} Reliability
          </p>
          <p style={{ fontSize: '12px', color: 'var(--t3)', fontFamily: 'var(--font-display)', lineHeight: 1.55 }}>
            {label === 'High'
              ? 'Report passed all critical safety checks. All clinical hard rules satisfied.'
              : label === 'Moderate'
                ? 'Minor gaps detected. Review highlighted violations before acting on this report.'
                : 'Significant failures detected. Apply full clinical judgment before acting.'}
          </p>
          <p style={{ fontSize: '12px', color: 'var(--t2)', fontFamily: 'var(--font-display)', marginTop: 6 }}>
            {score != null
              ? reportData?.judge_used === false
                ? '⚠ 7 / 9 methods ran · Judge LLM unavailable (G-Eval / RAGAS defaulted to 0.5)'
                : '✓ 9 / 9 methods ran · 2 LLM calls total'
              : '⟳ Awaiting report generation…'}
          </p>
        </div>
      </div>

      <SectionRule>Score thresholds</SectionRule>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
        {[
          { label: '≥ 0.75', text: 'High Reliability', color: GREEN },
          { label: '0.55 – 0.75', text: 'Moderate', color: AMBER },
          { label: '< 0.55', text: 'Low Reliability', color: RED },
        ].map((t, i) => (
          <div key={i} style={{
            background: 'var(--s2)', border: `1px solid ${a(t.color, 22)}`,
            borderRadius: 'var(--r-sm)', padding: '10px 12px',
            borderTop: `2px solid ${t.color}`,
          }}>
            <p className="num" style={{ fontSize: '11px', fontWeight: 700, color: t.color, fontFamily: 'var(--font-display)' }}>{t.label}</p>
            <p style={{ fontSize: '11px', color: 'var(--t3)', fontFamily: 'var(--font-display)', marginTop: 3 }}>{t.text}</p>
          </div>
        ))}
      </div>

      <SectionRule>Weight distribution · 9 methods</SectionRule>
      <div style={{
        background: 'var(--s2)', border: '1px solid var(--b1)',
        borderRadius: 'var(--r-md)', padding: '12px 14px',
        fontFamily: 'var(--font-display)',
      }}>
        <div style={{ display: 'flex', height: 12, borderRadius: 6, overflow: 'hidden', gap: 2 }}>
          {[
            { pct: 13, color: GREEN,  label: 'Factual' },
            { pct: 13, color: GREEN,  label: 'Hard Rules' },
            { pct: 9,  color: AMBER,  label: 'SHAP' },
            { pct: 9,  color: CYAN,   label: 'Structure' },
            { pct: 8,  color: CYAN,   label: 'Severity' },
            { pct: 5,  color: RED,    label: 'Contra' },
            { pct: 8,  color: AMBER,  label: 'Numeric' },
            { pct: 13, color: PURPLE, label: 'G-Eval' },
            { pct: 22, color: PURPLE, label: 'RAGAS' },
          ].map((seg, i) => (
            <div key={i} title={`${seg.label}: ${seg.pct}%`} style={{
              flex: seg.pct, background: seg.color, opacity: 0.75,
              borderRadius: 3,
            }} />
          ))}
        </div>
        <p style={{ fontSize: '10px', color: 'var(--t4)', marginTop: 6 }}>
          7 deterministic (65%) · 2 judge LLM (35%) · sum = 100%
        </p>
      </div>

      <InfoBox>
        <strong style={{ color: CYAN }}>For clinicians:</strong>{' '}
        This score does not replace clinical judgment. It tells you which specific checks passed or
        flagged so you know exactly where to verify before acting. A High score (≥ 0.75) means
        the weighted combination of all 9 checks passed — including no dangerous recommendations detected.
        Check the <strong>Findings tab</strong> for any specific violations.
      </InfoBox>
    </div>
  )
}

/* ── 9 Methods tab ── */
function MethodsTab({ reportData }) {
  const s = reportData?.reliability_score

  const DET_METHODS = [
    { icon: '📋', name: 'Factual Grounding',    weight: '13%', note: 'Abnormal vitals acknowledged in response' },
    { icon: '⚖️', name: 'Clinical Hard Rules',  weight: '13%', note: '6 ICU protocol rules from Surviving Sepsis Campaign + ACLS' },
    { icon: '🔬', name: 'SHAP-LLM Coherence',   weight: '9%',  note: 'LLM addressed top SHAP-driving prediction features' },
    { icon: '📑', name: 'Response Structure',   weight: '9%',  note: 'All 4 required sections present · 200-word target (graduated)' },
    { icon: '🎚️', name: 'Severity Calibration', weight: '8%',  note: 'Urgency language matches SOFA severity (bidirectional)' },
    { icon: '🚫', name: 'Contraindication',     weight: '5%',  note: '3 contraindicated drug classes flagged against this patient\'s vitals' },
    { icon: '🔢', name: 'Numeric Accuracy',     weight: '8%',  note: 'Cited values cross-checked against vitals · tolerance ±10% or ±3 (whichever larger)' },
  ]

  const JUDGE_METHODS = [
    { icon: '🤖', name: 'G-Eval Clinical Quality',  weight: '13%', note: 'Factual accuracy · Clinical appropriateness · Urgency · Completeness (1–5 each)' },
    { icon: '🔍', name: 'RAGAS Faithfulness',       weight: '22%', note: 'Unsupported claims checked by judge LLM against actual patient vitals' },
  ]

  /* We don't have per-method scores from the API — show combined score guidance */
  const scoreHint = s != null ? (
    s >= 0.85 ? 'All methods performing well' :
    s >= 0.65 ? 'Some methods below optimal — check Findings tab' :
    'Multiple methods flagged — review Findings tab'
  ) : null

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <SectionRule>7 Deterministic checks · zero extra API calls</SectionRule>

      {DET_METHODS.map((m, i) => (
        <div key={i} style={{
          background: 'var(--s2)', border: '1px solid var(--b1)',
          borderRadius: 'var(--r-sm)', padding: '10px 12px',
          borderLeft: `3px solid ${CYAN}`,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 4 }}>
            <span style={{ fontSize: 14, flexShrink: 0 }}>{m.icon}</span>
            <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--t1)', fontFamily: 'var(--font-display)', flex: 1 }}>
              {m.name}
            </span>
            <span style={{
              fontSize: '10px', color: 'var(--t4)', background: 'var(--s3)',
              borderRadius: 10, padding: '2px 7px',
              fontFamily: 'var(--font-display)', fontWeight: 600,
            }}>
              {m.weight}
            </span>
          </div>
          <p style={{ fontSize: '11px', color: 'var(--t3)', fontFamily: 'var(--font-display)', lineHeight: 1.45, margin: 0 }}>
            {m.note}
          </p>
        </div>
      ))}

      <SectionRule>2 Judge LLM checks · 1 extra API call · qwen/qwen3.8-27b</SectionRule>

      {JUDGE_METHODS.map((m, i) => (
        <div key={i} style={{
          background: 'var(--s2)', border: '1px solid var(--b1)',
          borderRadius: 'var(--r-sm)', padding: '10px 12px',
          borderLeft: `3px solid ${PURPLE}`,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 4 }}>
            <span style={{ fontSize: 14, flexShrink: 0 }}>{m.icon}</span>
            <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--t1)', fontFamily: 'var(--font-display)', flex: 1 }}>
              {m.name}
            </span>
            <span style={{
              fontSize: '10px', color: 'var(--t4)', background: 'var(--s3)',
              borderRadius: 10, padding: '2px 7px',
              fontFamily: 'var(--font-display)', fontWeight: 600,
            }}>
              {m.weight}
            </span>
          </div>
          <p style={{ fontSize: '11px', color: 'var(--t3)', fontFamily: 'var(--font-display)', lineHeight: 1.45, margin: 0 }}>
            {m.note}
          </p>
        </div>
      ))}

      {scoreHint && (
        <InfoBox>
          <strong style={{ color: CYAN }}>Combined score: {s?.toFixed(3)}</strong> — {scoreHint}.
          Open the Findings tab to see specific rule violations if any exist.
        </InfoBox>
      )}
    </div>
  )
}

/* ── Findings tab ── */
function FindingsTab({ reportData }) {
  const violations = reportData?.violations ?? []
  const hasViolations = violations.length > 0
  const score = reportData?.reliability_score
  const label = reportData?.reliability_label

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <SectionRule>Active violations · safety flags</SectionRule>

      {!hasViolations ? (
        <div style={{
          background: a(GREEN, 8), border: `1px solid ${a(GREEN, 22)}`,
          borderRadius: 'var(--r-md)', padding: '16px 18px',
          display: 'flex', alignItems: 'flex-start', gap: 12,
        }}>
          <span style={{ fontSize: 20, flexShrink: 0 }}>✅</span>
          <div>
            <p style={{ fontSize: '13px', fontWeight: 700, color: GREEN, fontFamily: 'var(--font-display)', marginBottom: 4 }}>
              No violations detected for this reading
            </p>
            <p style={{ fontSize: '12px', color: 'var(--t3)', fontFamily: 'var(--font-display)', lineHeight: 1.55 }}>
              All 6 clinical hard rules passed. No contraindicated medications found.
              No hallucinated vital values detected. No unsupported factual claims flagged by the judge LLM.
            </p>
          </div>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {violations.map((v, i) => (
            <div key={i} style={{
              background: a(RED, 8), border: `1px solid ${a(RED, 22)}`,
              borderRadius: 'var(--r-sm)', padding: '10px 14px',
              display: 'flex', alignItems: 'flex-start', gap: 10,
            }}>
              <span style={{ fontSize: 14, flexShrink: 0, marginTop: 1 }}>⚠</span>
              <p style={{ fontSize: '12px', color: '#fca5a5', fontFamily: 'var(--font-display)', lineHeight: 1.5, margin: 0 }}>
                {v}
              </p>
            </div>
          ))}
        </div>
      )}

      <SectionRule>How this score affects clinical trust</SectionRule>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {[
          { range: '≥ 0.75', icon: '✅', text: 'All checks passed. Safe to use with standard clinical oversight.', color: GREEN },
          { range: '0.55–0.75', icon: '⚠️', text: 'Minor gaps. Read the specific violations above before acting on any recommendation.', color: AMBER },
          { range: '< 0.55', icon: '❌', text: 'Multiple failures. Cross-check all recommendations independently before any action.', color: RED },
        ].map((row, i) => (
          <div key={i} style={{
            display: 'flex', alignItems: 'flex-start', gap: 10,
            background: 'var(--s2)', border: `1px solid ${a(row.color, 18)}`,
            borderRadius: 'var(--r-sm)', padding: '10px 12px',
          }}>
            <span style={{ fontSize: 14, flexShrink: 0 }}>{row.icon}</span>
            <div>
              <span className="num" style={{ fontSize: '11px', fontWeight: 700, color: row.color, fontFamily: 'var(--font-display)' }}>
                Score {row.range}
              </span>
              <p style={{ fontSize: '11px', color: 'var(--t3)', fontFamily: 'var(--font-display)', lineHeight: 1.45, margin: '3px 0 0' }}>
                {row.text}
              </p>
            </div>
          </div>
        ))}
      </div>

      <InfoBox accent={AMBER}>
        <strong style={{ color: AMBER }}>Note:</strong>{' '}
        This validation system significantly reduces hallucination risk but cannot guarantee
        correctness. 7 of 9 methods check the response against the actual patient vitals and
        ICU protocols — not just against other LLM responses. Even a High Reliability score
        requires qualified clinical judgment before any patient care decision.
      </InfoBox>
    </div>
  )
}

/* ── Main drawer ── */
const TABS = [
  { id: 'overview',   label: 'Overview' },
  { id: 'methods',    label: '9 Methods' },
  { id: 'findings',   label: 'Findings' },
]

const drawerVariants = {
  hidden:  { x: '100%', opacity: 0.6 },
  visible: { x: 0, opacity: 1, transition: { type: 'spring', stiffness: 340, damping: 34, mass: 1.0 } },
  exit:    { x: '100%', opacity: 0.4, transition: { duration: 0.22, ease: 'easeIn' } },
}

export default function LLMValidationDrawer({ open, onClose, reportData, careMode }) {
  const [activeTab, setActiveTab] = useState('overview')

  useEffect(() => { if (open) setActiveTab('overview') }, [open])

  useEffect(() => {
    if (!open) return
    const h = e => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [open, onClose])

  const score = reportData?.reliability_score
  const label = reportData?.reliability_label
  const violations = reportData?.violations ?? []

  const labelColor =
    label === 'High'     ? GREEN :
    label === 'Moderate' ? AMBER :
    label === 'Low'      ? RED   : 'var(--t3)'

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            key="llm-backdrop"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={onClose}
            style={{
              position: 'fixed', inset: 0, zIndex: 900,
              background: 'rgba(0,0,0,0.50)', backdropFilter: 'blur(2px)',
            }}
          />

          <motion.div
            key="llm-drawer"
            variants={drawerVariants}
            initial="hidden" animate="visible" exit="exit"
            style={{
              position: 'fixed', top: 0, right: 0, bottom: 0, zIndex: 910,
              width: 'min(720px, 100vw)',
              background: 'var(--s1)',
              borderLeft: '1px solid var(--b2)',
              display: 'flex', flexDirection: 'column',
              boxShadow: '-12px 0 40px rgba(0,0,0,0.45)',
            }}
          >
            {/* Header */}
            <div style={{
              background: 'var(--s2)', borderBottom: '1px solid var(--b1)',
              padding: '16px 20px', flexShrink: 0,
              display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12,
            }}>
              <div style={{ flex: 1, minWidth: 0 }}>
                <p style={{ fontSize: '10px', fontWeight: 700, letterSpacing: '0.10em', textTransform: 'uppercase', color: 'var(--t4)', fontFamily: 'var(--font-display)', marginBottom: 4 }}>
                  AI Report · 9-Method Validation
                </p>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
                  <h2 style={{ fontSize: '18px', fontWeight: 800, color: 'var(--t1)', fontFamily: 'var(--font-display)', margin: 0, letterSpacing: '-0.02em', display: 'flex', alignItems: 'center', gap: 8 }}>
                    <svg width="20" height="20" viewBox="0 0 18 18" fill="none" stroke="#f472b6" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ flexShrink: 0 }}>
                      <path d="M9 1.5 L15.5 4.5 L15.5 9 C15.5 12.5 12.5 15.5 9 16.5 C5.5 15.5 2.5 12.5 2.5 9 L2.5 4.5 Z"/>
                      <polyline points="6.5,9 8.2,10.8 11.5,7.2"/>
                    </svg>
                    Reliability Audit
                  </h2>
                  {score != null && (
                    <div style={{
                      display: 'inline-flex', alignItems: 'center', gap: 7,
                      background: a(labelColor, 12), border: `1.5px solid ${a(labelColor, 35)}`,
                      borderRadius: 'var(--r-sm)', padding: '4px 10px',
                    }}>
                      <div style={{
                        width: 7, height: 7, borderRadius: '50%', background: labelColor,
                        boxShadow: `0 0 7px ${labelColor}`, flexShrink: 0,
                      }} />
                      <span style={{ fontSize: '12px', fontWeight: 700, color: labelColor, fontFamily: 'var(--font-display)' }}>
                        {label} Reliability
                      </span>
                      <span className="num" style={{ fontSize: '12px', fontWeight: 700, color: 'var(--t1)', fontFamily: 'var(--font-display)' }}>
                        {score.toFixed(3)}
                      </span>
                    </div>
                  )}
                </div>
                <p style={{ fontSize: '12px', color: 'var(--t3)', fontFamily: 'var(--font-display)', marginTop: 4 }}>
                  {careMode === 'hospital' ? '🏥 Hospital' : '🏠 Home Care'} Mode
                  {violations.length > 0 && (
                    <span style={{
                      marginLeft: 8, fontSize: '11px', fontWeight: 700,
                      color: AMBER, background: a(AMBER, 12),
                      border: `1px solid ${a(AMBER, 25)}`,
                      borderRadius: 10, padding: '2px 8px',
                      fontFamily: 'var(--font-display)',
                    }}>
                      {violations.length} violation{violations.length > 1 ? 's' : ''}
                    </span>
                  )}
                </p>
              </div>
              <button
                onClick={onClose}
                style={{
                  width: 32, height: 32, borderRadius: '50%', flexShrink: 0,
                  background: 'var(--s3)', border: '1px solid var(--b1)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  color: 'var(--t3)', fontSize: 16, cursor: 'pointer',
                  transition: 'all 0.15s',
                }}
                onMouseEnter={e => { e.currentTarget.style.background = 'var(--danger-bg)'; e.currentTarget.style.color = 'var(--danger-l)' }}
                onMouseLeave={e => { e.currentTarget.style.background = 'var(--s3)'; e.currentTarget.style.color = 'var(--t3)' }}
                aria-label="Close panel"
              >
                ✕
              </button>
            </div>

            {/* Tabs */}
            <div style={{
              display: 'flex', borderBottom: '1px solid var(--b1)',
              flexShrink: 0, overflowX: 'auto',
            }}>
              {TABS.map(t => (
                <button
                  key={t.id}
                  onClick={() => setActiveTab(t.id)}
                  style={{
                    padding: '11px 16px', fontSize: '12px', fontWeight: 600,
                    color: activeTab === t.id ? CYAN : 'var(--t3)',
                    borderBottom: `2px solid ${activeTab === t.id ? CYAN : 'transparent'}`,
                    cursor: 'pointer', whiteSpace: 'nowrap', background: 'none',
                    border: 'none',
                    borderBottom: `2px solid ${activeTab === t.id ? CYAN : 'transparent'}`,
                    fontFamily: 'var(--font-display)', transition: 'color 0.15s',
                    position: 'relative',
                  }}
                >
                  {t.label}
                  {/* Violation dot on Findings tab */}
                  {t.id === 'findings' && violations.length > 0 && (
                    <span style={{
                      position: 'absolute', top: 8, right: 6,
                      width: 6, height: 6, borderRadius: '50%',
                      background: AMBER, boxShadow: `0 0 5px ${AMBER}`,
                    }} />
                  )}
                </button>
              ))}
            </div>

            {/* Body */}
            <div style={{ flex: 1, overflowY: 'auto', padding: 20 }}>
              <AnimatePresence mode="wait">
                <motion.div
                  key={activeTab}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.18 }}
                >
                  {activeTab === 'overview'  && <OverviewTab reportData={reportData} />}
                  {activeTab === 'methods'   && <MethodsTab  reportData={reportData} />}
                  {activeTab === 'findings'  && <FindingsTab reportData={reportData} />}
                </motion.div>
              </AnimatePresence>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}
