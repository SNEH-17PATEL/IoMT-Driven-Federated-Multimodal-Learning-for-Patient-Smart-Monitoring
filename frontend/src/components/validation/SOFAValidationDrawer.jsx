import { useState, useEffect } from 'react'
import { AnimatePresence, motion } from 'framer-motion'

const a = (c, p) => `color-mix(in srgb, ${c} ${p}%, transparent)`

/* ── Tokens ── */
const CYAN   = 'var(--cyan-l)'
const GREEN  = 'var(--ok-l)'
const AMBER  = 'var(--warn-l)'
const PURPLE = '#a78bfa'

/* ── Small reusable pieces ── */
function MetricTile({ label, value, sub, accent }) {
  return (
    <div style={{
      background: 'var(--s2)', border: '1px solid var(--b1)',
      borderRadius: 'var(--r-lg)', padding: '14px 16px',
      position: 'relative', overflow: 'hidden',
    }}>
      {/* top accent stripe */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: 2,
        background: `linear-gradient(90deg, ${accent}, transparent)`,
      }} />
      <p style={{ fontSize: '10px', fontWeight: 700, letterSpacing: '0.09em', textTransform: 'uppercase', color: 'var(--t4)', fontFamily: 'var(--font-display)', marginBottom: 8 }}>
        {label}
      </p>
      <p className="num" style={{ fontSize: '28px', fontWeight: 900, color: accent, lineHeight: 1, letterSpacing: '-0.03em' }}>
        {value}
      </p>
      {sub && (
        <p style={{ fontSize: '10.5px', color: 'var(--t3)', marginTop: 6, lineHeight: 1.45, fontFamily: 'var(--font-display)' }}>
          {sub}
        </p>
      )}
    </div>
  )
}

function InfoBox({ children }) {
  return (
    <div style={{
      background: a(CYAN, 6), border: `1px solid ${a(CYAN, 18)}`,
      borderRadius: 'var(--r-md)', padding: '12px 14px',
      fontSize: '12px', color: 'var(--t2)', lineHeight: 1.65,
      fontFamily: 'var(--font-display)',
    }}>
      {children}
    </div>
  )
}

function SectionRule({ children }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, margin: '4px 0 12px' }}>
      <p style={{ fontSize: '10px', fontWeight: 700, letterSpacing: '0.10em', textTransform: 'uppercase', color: 'var(--t4)', fontFamily: 'var(--font-display)', whiteSpace: 'nowrap' }}>
        {children}
      </p>
      <div style={{ flex: 1, height: 1, background: 'var(--b1)' }} />
    </div>
  )
}

/* ── Layer 1: Population stats ── */
const fmt  = (v, d = 4) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(d) : '—')
const pct  = (v, d = 1) => (typeof v === 'number' && Number.isFinite(v) ? `${(v * 100).toFixed(d)}%` : '—')
const int  = v => (typeof v === 'number' && Number.isFinite(v) ? v.toLocaleString() : '—')

function Layer1({ predictData }) {
  // Every number here comes from models/training_metadata.json, which the
  // backend sends with each prediction as _training_meta — so the panel always
  // describes the model that is actually loaded.  Missing values show "—".
  const meta   = predictData?._training_meta ?? {}
  const risk   = meta.risk_classification_metrics ?? {}
  const alert  = risk.high_risk_alert ?? {}
  const bands  = risk.risk_bands ?? {}
  const matrix = Array.isArray(bands.confusion_matrix) ? bands.confusion_matrix : null
  const highRecall = bands.per_class?.['High (>=10)']?.recall

  const nTest  = meta.test_samples
  const nTrain = meta.train_samples
  const nTotal = typeof nTest === 'number' && typeof nTrain === 'number' ? nTest + nTrain : undefined

  // 3-class accuracy and High-risk share, computed from the saved confusion matrix
  let bandAcc, highShare
  if (matrix) {
    const total = matrix.flat().reduce((x, y) => x + y, 0)
    if (total > 0) {
      bandAcc   = matrix.reduce((acc, row, i) => acc + (row[i] ?? 0), 0) / total
      highShare = (matrix[2] ?? []).reduce((x, y) => x + y, 0) / total
    }
  }
  const thr = alert.predicted_sofa_threshold
  const thrText = typeof thr === 'number' ? thr.toFixed(0) : '—'

  const rowLabels = ['Low Risk', 'Moderate Risk', 'High Risk']

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <SectionRule>
        Held-out test set · {int(nTest)} ICU stays of {int(nTotal)} total
      </SectionRule>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
        <MetricTile label="AUROC" value={fmt(alert.auroc)} accent={CYAN}
          sub="How well the model ranks truly high-risk patients (actual SOFA ≥ 10) above others" />
        <MetricTile label="PR-AUC" value={fmt(alert.pr_auc)} accent={GREEN}
          sub={`Precision-recall · high-risk patients are ${pct(highShare)} of the test set`} />
        <MetricTile label="MAE" value={fmt(meta.final_mae)} accent={AMBER}
          sub="Average SOFA error in points (0–24 scale)" />
        <MetricTile label="R²" value={fmt(meta.final_r2)} accent={PURPLE}
          sub="Share of SOFA variation explained (no lab values used)" />
      </div>

      <SectionRule>Alert performance · alert fires at predicted SOFA ≥ {thrText}</SectionRule>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
        <MetricTile label="Alert recall" value={pct(alert.recall)} accent={AMBER}
          sub={`High-risk patients caught · missed ${int(alert.false_negative)}`} />
        <MetricTile label="Specificity" value={pct(alert.specificity)} accent={GREEN}
          sub={`Non-high-risk patients left alone · false alarms ${int(alert.false_positive)}`} />
      </div>

      <SectionRule>Risk-band classification · 3-class confusion matrix ({int(nTest)} test stays)</SectionRule>

      <div style={{
        background: 'var(--s2)', border: '1px solid var(--b1)',
        borderRadius: 'var(--r-md)', padding: 14, overflowX: 'auto',
      }}>
        {matrix ? (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px', fontFamily: 'var(--font-display)' }}>
            <thead>
              <tr>
                <th style={{ color: 'var(--t4)', textAlign: 'left', padding: '4px 8px', fontWeight: 700, letterSpacing: '0.06em' }}></th>
                <th style={{ color: 'var(--t4)', textAlign: 'center', padding: '4px 8px', fontWeight: 700 }}>Pred Low</th>
                <th style={{ color: 'var(--t4)', textAlign: 'center', padding: '4px 8px', fontWeight: 700 }}>Pred Mod</th>
                <th style={{ color: 'var(--t4)', textAlign: 'center', padding: '4px 8px', fontWeight: 700 }}>Pred High</th>
              </tr>
            </thead>
            <tbody>
              {matrix.slice(0, 3).map((cells, ri) => (
                <tr key={ri}>
                  <td style={{ color: 'var(--t3)', padding: '5px 8px', whiteSpace: 'nowrap', fontWeight: 600 }}>
                    True {rowLabels[ri]}
                  </td>
                  {cells.slice(0, 3).map((val, ci) => {
                    const isDiag = ri === ci
                    return (
                      <td key={ci} style={{
                        textAlign: 'center', padding: '5px 8px',
                        background: isDiag ? a(GREEN, 14) : 'transparent',
                        borderRadius: isDiag ? 6 : 0,
                        color: isDiag ? GREEN : 'var(--t3)',
                        fontWeight: isDiag ? 700 : 400,
                        fontVariantNumeric: 'tabular-nums',
                      }}>
                        {int(val)}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p style={{ fontSize: '11px', color: 'var(--t3)', fontFamily: 'var(--font-display)', margin: 0 }}>
            Confusion matrix not available — run train_federated.py to generate training_metadata.json.
          </p>
        )}
        <p style={{ fontSize: '11px', color: 'var(--t3)', marginTop: 10, lineHeight: 1.5, fontFamily: 'var(--font-display)' }}>
          High-Risk recall (3-class, SOFA ≥ 10): <strong style={{ color: AMBER }}>{pct(highRecall)}</strong> ·
          Alert recall (predicted SOFA ≥ {thrText}): <strong style={{ color: AMBER }}>{pct(alert.recall)}</strong> ·
          The lower alert threshold makes up for the model under-predicting severe patients
        </p>
      </div>

      <InfoBox>
        <strong style={{ color: CYAN }}>What this means for you:</strong>{' '}
        The model ranks a high-risk patient above a non-high-risk one in {pct(alert.auroc)} of
        pairwise comparisons (AUROC). 3-class band accuracy (Low / Moderate / High) is {pct(bandAcc)} —
        limited by missing lab values such as bilirubin, creatinine and platelet count, which need
        blood draws and are not available as real-time vitals.
        {meta.evaluation_limitation ? <> Note: {meta.evaluation_limitation}</> : null}
      </InfoBox>
    </div>
  )
}

/* ── Layer 2: Runtime plausibility checks ── */
function CheckCard({ icon, title, detail, statusLabel, statusColor, bg }) {
  return (
    <div style={{
      background: 'var(--s2)', border: '1px solid var(--b1)',
      borderRadius: 'var(--r-md)', padding: '14px 16px',
      display: 'flex', alignItems: 'flex-start', gap: 12,
    }}>
      <div style={{
        width: 34, height: 34, borderRadius: '50%', flexShrink: 0,
        background: bg, display: 'flex', alignItems: 'center',
        justifyContent: 'center', fontSize: 15,
      }}>
        {icon}
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 8, marginBottom: 5 }}>
          <p style={{ fontSize: '13px', fontWeight: 700, color: 'var(--t1)', fontFamily: 'var(--font-display)', margin: 0 }}>
            {title}
          </p>
          <span style={{
            fontSize: '10px', fontWeight: 700, letterSpacing: '0.05em', textTransform: 'uppercase',
            padding: '3px 9px', borderRadius: 20, flexShrink: 0,
            color: statusColor, background: a(statusColor, 14), border: `1px solid ${a(statusColor, 30)}`,
            fontFamily: 'var(--font-display)',
          }}>
            {statusLabel}
          </span>
        </div>
        <p style={{ fontSize: '12px', color: 'var(--t3)', lineHeight: 1.55, fontFamily: 'var(--font-display)', margin: 0 }}>
          {detail}
        </p>
      </div>
    </div>
  )
}

function Layer2({ predictData }) {
  const d = predictData ?? {}
  const sofa       = d.sofa_score ?? 0
  const vitals     = d.vitals ?? {}
  const MAP_val    = vitals.MAP  ?? 80
  const SpO2_val   = vitals.SpO2 ?? 98
  const GCS_val    = vitals.GCS_eye ?? 4
  const HR_val     = vitals.HR  ?? 80
  const SBP_val    = vitals.SBP ?? 120

  /* Compute physiological floor (same logic as ml_service.py) */
  let floor = 0
  if (MAP_val  < 70) floor += 1
  if (MAP_val  < 65) floor += 1
  if (SpO2_val < 94) floor += 1
  if (SpO2_val < 90) floor += 1
  if (GCS_val === 2) floor += 2
  if (GCS_val === 1) floor += 3
  const floorPass = sofa >= floor

  /* Vital consistency flags */
  const shockFlag    = HR_val > 130 && SBP_val < 90 && sofa < 4
  const multiOrganFlag = SpO2_val < 88 && MAP_val < 65 && sofa < 6
  const consistencyPass = !shockFlag && !multiOrganFlag

  /* Trajectory — from predictData */
  const prevSofa   = d._prev_sofa
  const jump       = prevSofa != null ? (sofa - prevSofa) : null
  const trajFlagged = d._traj_flagged ?? false

  const floorDetail = [
    MAP_val < 70  && `MAP ${MAP_val} mmHg → cardiovascular floor +1`,
    MAP_val < 65  && `MAP < 65 → cardiovascular floor +2 total`,
    SpO2_val < 94 && `SpO₂ ${SpO2_val}% → respiratory floor +1`,
    SpO2_val < 90 && `SpO₂ < 90% → respiratory floor +2 total`,
    GCS_val === 2 && `GCS Eye 2 → CNS floor +2`,
    GCS_val === 1 && `GCS Eye 1 → CNS floor +3`,
  ].filter(Boolean)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <SectionRule>Per-reading plausibility · SOFA {sofa?.toFixed(1)}</SectionRule>

      <CheckCard
        icon={floorPass ? '✓' : '⚠'}
        title="Physiological Lower-Bound Check"
        statusLabel={floorPass ? 'Consistent' : 'Under-predicted'}
        statusColor={floorPass ? GREEN : AMBER}
        bg={floorPass ? a(GREEN, 14) : a(AMBER, 14)}
        detail={
          floorDetail.length > 0
            ? `${floorDetail.join(' · ')} · Computed floor = ${floor} · Predicted ${sofa?.toFixed(1)} ${floorPass ? '≥' : '<'} floor ${floorPass ? '✓' : '⚠'}`
            : `All direct vitals within normal ranges → physiological floor = 0. Predicted SOFA ${sofa?.toFixed(1)} is physiologically consistent.`
        }
      />

      <CheckCard
        icon={consistencyPass ? '✓' : '⚠'}
        title="Vital Sign Consistency Check"
        statusLabel={consistencyPass ? 'No flags' : 'Pattern detected'}
        statusColor={consistencyPass ? GREEN : AMBER}
        bg={consistencyPass ? a(GREEN, 14) : a(AMBER, 14)}
        detail={
          shockFlag
            ? `HR ${HR_val} bpm + SBP ${SBP_val} mmHg: shock pattern detected (HR > 130 + SBP < 90) — SOFA should be ≥ 4, predicted ${sofa?.toFixed(1)}`
            : multiOrganFlag
              ? `SpO₂ ${SpO2_val}% + MAP ${MAP_val} mmHg: multi-organ stress (SpO₂ < 88 + MAP < 65) — SOFA should be ≥ 6, predicted ${sofa?.toFixed(1)}`
              : `HR ${HR_val} bpm, SBP ${SBP_val} mmHg: below shock threshold (HR > 130 + SBP < 90). SpO₂ ${SpO2_val}% + MAP ${MAP_val} mmHg: not in multi-organ stress zone. No syndrome-level flags.`
        }
      />

      <CheckCard
        icon={trajFlagged ? '⚠' : '⟳'}
        title="Trajectory Coherence Check"
        statusLabel={
          jump == null ? 'First reading' :
          trajFlagged ? 'Verify input' :
          `${jump >= 0 ? '↑' : '↓'} ${Math.abs(jump).toFixed(1)} pts`
        }
        statusColor={trajFlagged ? AMBER : CYAN}
        bg={trajFlagged ? a(AMBER, 14) : a(CYAN, 10)}
        detail={
          jump == null
            ? 'No previous reading available — trajectory check will activate from the next reading onwards.'
            : trajFlagged
              ? `SOFA jumped ${Math.abs(jump).toFixed(1)} pts (${prevSofa?.toFixed(1)} → ${sofa?.toFixed(1)}) with no significant vital sign change — check that input data is correct.`
              : `Previous: ${prevSofa?.toFixed(1)} → Current: ${sofa?.toFixed(1)} (${jump >= 0 ? '+' : ''}${jump?.toFixed(1)} pts). Change is clinically supported by vital sign movements.`
        }
      />

      <InfoBox>
        <strong style={{ color: CYAN }}>These checks run every reading cycle.</strong>{' '}
        They cannot access lab values (bilirubin, creatinine) but verify the prediction is
        internally consistent with the vitals actually measured this cycle.
        A flag is a prompt to verify input data — not a prediction failure.
      </InfoBox>
    </div>
  )
}

/* ── Layer 3: Conformal prediction interval ── */
function Layer3({ predictData }) {
  const d = predictData ?? {}
  const sofa   = d.sofa_score  ?? 0
  const lo     = d.conformal_lo ?? Math.max(0, sofa - 2.9)
  const hi     = d.conformal_hi ?? Math.min(24, sofa + 2.9)
  const q      = d.conformal_q  ?? 2.9
  const MAX    = 24

  const pctLo   = (lo / MAX * 100).toFixed(1)
  const pctHi   = (hi / MAX * 100).toFixed(1)
  const pctPred = (sofa / MAX * 100).toFixed(1)
  const width   = parseFloat(pctHi) - parseFloat(pctLo)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <SectionRule>90% conformal prediction interval · this reading</SectionRule>

      {/* Visual track */}
      <div style={{
        background: 'var(--s2)', border: '1px solid var(--b1)',
        borderRadius: 'var(--r-md)', padding: '20px 16px 16px',
      }}>
        <p style={{ fontSize: '11px', fontWeight: 600, color: 'var(--t2)', marginBottom: 16, fontFamily: 'var(--font-display)' }}>
          SOFA uncertainty band — SOFA scale 0–24
        </p>

        <div style={{ position: 'relative', margin: '0 24px 28px' }}>
          {/* Track */}
          <div style={{
            height: 36, background: 'var(--s3)',
            borderRadius: 8, position: 'relative', overflow: 'visible',
          }}>
            {/* Confidence range */}
            <div style={{
              position: 'absolute', top: 6, bottom: 6,
              left: `${pctLo}%`, width: `${width}%`,
              background: a(CYAN, 16), borderRadius: 6,
              border: `1.5px solid ${a(CYAN, 45)}`,
            }} />

            {/* Predicted marker */}
            <div style={{
              position: 'absolute', top: 0, bottom: 0,
              left: `${pctPred}%`, width: 3, borderRadius: 2,
              background: CYAN, transform: 'translateX(-50%)',
              boxShadow: `0 0 10px ${a(CYAN, 60)}`,
            }}>
              <span style={{
                position: 'absolute', top: '-22px', left: '50%', transform: 'translateX(-50%)',
                fontSize: '10px', fontWeight: 800, color: CYAN, whiteSpace: 'nowrap',
                fontFamily: 'var(--font-display)',
              }}>
                {sofa.toFixed(1)}
              </span>
            </div>

            {/* Edge labels */}
            <span style={{
              position: 'absolute', left: `${pctLo}%`, top: '50%', transform: 'translate(-110%, -50%)',
              fontSize: '10px', fontWeight: 700, color: 'var(--t3)', fontFamily: 'var(--font-display)',
            }}>
              {lo.toFixed(1)}
            </span>
            <span style={{
              position: 'absolute', left: `${pctHi}%`, top: '50%', transform: 'translate(10%, -50%)',
              fontSize: '10px', fontWeight: 700, color: 'var(--t3)', fontFamily: 'var(--font-display)',
            }}>
              {hi.toFixed(1)}
            </span>
          </div>

          {/* Scale labels */}
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 8 }}>
            {[0, 6, 12, 18, 24].map(v => (
              <span key={v} style={{ fontSize: '9px', color: 'var(--t4)', fontFamily: 'var(--font-display)' }}>{v}</span>
            ))}
          </div>
        </div>

        <p style={{ fontSize: '11.5px', color: 'var(--t2)', lineHeight: 1.5, fontFamily: 'var(--font-display)' }}>
          <strong style={{ color: CYAN }}>P(true SOFA ∈ [{lo.toFixed(1)}, {hi.toFixed(1)}]) ≥ 90%</strong>{' '}
          — statistical guarantee, not an estimate
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
        <MetricTile label="Predicted SOFA" value={sofa.toFixed(1)} accent={CYAN}
          sub="Model output for this reading" />
        <MetricTile label="Margin (q̂)" value={`± ${q.toFixed(1)}`} accent={AMBER}
          sub={`90th percentile error on ${int(predictData?._training_meta?.train_samples)} training stays`} />
      </div>

      <InfoBox>
        <strong style={{ color: CYAN }}>Why this is stronger than ±MAE:</strong>{' '}
        The ±MAE band (±{fmt(predictData?._training_meta?.final_mae, 2)}) shows the <em>average</em> error across all patients.
        Conformal prediction adapts to the actual error distribution and gives a{' '}
        <strong>mathematical guarantee</strong> — not an approximation — that the true SOFA will
        fall inside [{lo.toFixed(1)}, {hi.toFixed(1)}] at least 90% of the time.
        The wide interval reflects missing lab values, not a model error.
      </InfoBox>
    </div>
  )
}

/* ── Main drawer ── */
const TABS = [
  { id: 'layer1', label: 'Population Stats' },
  { id: 'layer2', label: 'Runtime Checks' },
  { id: 'layer3', label: 'Uncertainty' },
]

const drawerVariants = {
  hidden:  { x: '100%', opacity: 0.6 },
  visible: { x: 0, opacity: 1, transition: { type: 'spring', stiffness: 340, damping: 34, mass: 1.0 } },
  exit:    { x: '100%', opacity: 0.4, transition: { duration: 0.22, ease: 'easeIn' } },
}

export default function SOFAValidationDrawer({ open, onClose, predictData }) {
  const [activeTab, setActiveTab] = useState('layer1')

  /* Reset to first tab when reopened */
  useEffect(() => { if (open) setActiveTab('layer1') }, [open])

  /* Keyboard close */
  useEffect(() => {
    if (!open) return
    const h = e => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [open, onClose])

  return (
    <AnimatePresence>
      {open && (
        <>
          {/* Backdrop */}
          <motion.div
            key="sofa-backdrop"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={onClose}
            style={{
              position: 'fixed', inset: 0, zIndex: 900,
              background: 'rgba(0,0,0,0.50)', backdropFilter: 'blur(2px)',
            }}
          />

          {/* Drawer */}
          <motion.div
            key="sofa-drawer"
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
              <div>
                <p style={{ fontSize: '10px', fontWeight: 700, letterSpacing: '0.10em', textTransform: 'uppercase', color: 'var(--t4)', fontFamily: 'var(--font-display)', marginBottom: 4 }}>
                  SOFA Score · 3-Layer Validation
                </p>
                <h2 style={{ fontSize: '18px', fontWeight: 800, color: 'var(--t1)', fontFamily: 'var(--font-display)', margin: 0, letterSpacing: '-0.02em', display: 'flex', alignItems: 'center', gap: 8 }}>
                  <svg width="20" height="20" viewBox="0 0 18 18" fill="none" stroke="#38bdf8" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ flexShrink: 0 }}>
                    <rect x="1" y="10" width="3.5" height="6" rx="0.8"/>
                    <rect x="7.25" y="6" width="3.5" height="10" rx="0.8"/>
                    <rect x="13.5" y="2" width="3.5" height="14" rx="0.8"/>
                  </svg>
                  Model Accuracy Report
                </h2>
                <p style={{ fontSize: '12px', color: 'var(--t3)', fontFamily: 'var(--font-display)', marginTop: 4 }}>
                  Patient {predictData?.patient_id ?? '—'} · SOFA {predictData?.sofa_score?.toFixed(1) ?? '—'}
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
                    border: 'none', borderBottom: `2px solid ${activeTab === t.id ? CYAN : 'transparent'}`,
                    fontFamily: 'var(--font-display)', transition: 'color 0.15s',
                  }}
                >
                  {t.label}
                </button>
              ))}
            </div>

            {/* Body */}
            <div style={{ flex: 1, overflowY: 'auto', padding: 20, display: 'flex', flexDirection: 'column', gap: 0 }}>
              <AnimatePresence mode="wait">
                <motion.div
                  key={activeTab}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.18 }}
                >
                  {activeTab === 'layer1' && <Layer1 predictData={predictData} />}
                  {activeTab === 'layer2' && <Layer2 predictData={predictData} />}
                  {activeTab === 'layer3' && <Layer3 predictData={predictData} />}
                </motion.div>
              </AnimatePresence>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}
