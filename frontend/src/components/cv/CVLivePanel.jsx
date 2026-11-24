/*
 * CVLivePanel — vertical "model input → output" panel shown to the right of the
 * camera feed inside the CV monitor window.
 *
 * Purpose: prove that the camera values really go into the SOFA model.
 *   • "Live from camera"   = what the camera is measuring right now (cvLiveData)
 *   • "Used in last SOFA"  = the values the backend actually used (predictData.vitals)
 * When the two match, the badge shows "In model ✓". When they differ, the camera
 * value is waiting for the next 90-second prediction.
 */

const a = (c, p) => `color-mix(in srgb, ${c} ${p}%, transparent)`

const GREEN = '#4ade80'
const AMBER = '#fbbf24'
const RED   = '#f87171'
const PURPLE = '#a78bfa'

const GCS_LABELS = { 1: 'No Response', 2: 'To Pain', 3: 'To Voice', 4: 'Spontaneous' }

const stressColor = s => (s < 3.5 ? GREEN : s < 6.5 ? AMBER : RED)
const stressLabel = s => (s < 3.5 ? 'LOW' : s < 6.5 ? 'MODERATE' : 'HIGH')
const gcsColor    = g => (g >= 4 ? GREEN : g === 3 ? AMBER : RED)
const statusColor = st => (st === 'normal' ? GREEN : st === 'low' || st === 'high' ? AMBER : 'var(--t3)')

const VITAL_ROWS = [
  ['HR',   'HR',   'bpm'],
  ['RR',   'RR',   '/min'],
  ['SpO2', 'SpO₂', '%'],
  ['Temp', 'Temp', '°C'],
  ['SBP',  'SBP',  'mmHg'],
  ['DBP',  'DBP',  'mmHg'],
  ['MAP',  'MAP',  'mmHg'],
]

const label = {
  fontSize: 10, fontWeight: 700, color: 'var(--t4)', letterSpacing: '0.10em',
  textTransform: 'uppercase', fontFamily: 'var(--font-display)',
}

function Section({ title, children, accent }) {
  return (
    <div style={{
      background: 'var(--s1)',
      border: `1px solid ${accent ? a(accent, 22) : 'var(--b1)'}`,
      borderRadius: 12, padding: '12px 14px', marginBottom: 12,
    }}>
      <p style={{ ...label, marginBottom: 8 }}>{title}</p>
      {children}
    </div>
  )
}

function Badge({ ok, text }) {
  const c = ok ? GREEN : AMBER
  return (
    <span style={{
      fontSize: 9.5, fontWeight: 800, letterSpacing: '0.06em', fontFamily: 'var(--font-display)',
      color: c, background: a(c, 14), border: `1px solid ${a(c, 30)}`,
      borderRadius: 20, padding: '2px 8px', whiteSpace: 'nowrap',
    }}>
      {text}
    </span>
  )
}

/* One camera input: live value vs the value the last SOFA prediction used. */
function CvInputRow({ name, live, used, fmt, color }) {
  const hasLive = live != null
  const inModel = hasLive && used != null && Number(live) === Number(used)
  return (
    <div style={{ padding: '8px 0', borderTop: '1px solid var(--b1)' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
        <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--t1)', fontFamily: 'var(--font-display)' }}>{name}</span>
        {!hasLive
          ? <Badge ok={false} text="WAITING FOR CAMERA" />
          : <Badge ok={inModel} text={inModel ? 'IN MODEL ✓' : 'SENDS NEXT CYCLE'} />}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        <div>
          <p style={{ fontSize: 9.5, color: 'var(--t4)', fontFamily: 'var(--font-display)', marginBottom: 2 }}>Live from camera</p>
          <span className="num" style={{ fontSize: 20, fontWeight: 800, color: hasLive ? color(live) : 'var(--t4)' }}>
            {hasLive ? fmt(live) : '—'}
          </span>
        </div>
        <div>
          <p style={{ fontSize: 9.5, color: 'var(--t4)', fontFamily: 'var(--font-display)', marginBottom: 2 }}>Used in last SOFA</p>
          <span className="num" style={{ fontSize: 20, fontWeight: 800, color: used != null ? color(used) : 'var(--t4)' }}>
            {used != null ? fmt(used) : '—'}
          </span>
        </div>
      </div>
    </div>
  )
}

export default function CVLivePanel({ predictData, cvLiveData, isRefreshing, countdown }) {
  if (!predictData) {
    return (
      <aside style={{ ...panelBox, alignItems: 'center', justifyContent: 'center', textAlign: 'center' }}>
        <p style={{ fontSize: 12, color: 'var(--t3)', fontFamily: 'var(--font-display)' }}>
          Waiting for the first SOFA prediction…
        </p>
      </aside>
    )
  }

  const { sofa_score, risk_level, risk_color, is_alert, vitals = {}, vitals_status = {},
          clinical_note, conformal_lo, conformal_hi } = predictData
  const riskC = risk_color || 'var(--t2)'
  const usedGcs    = vitals.GCS_eye
  const usedStress = vitals.stress
  const hasNote = clinical_note && clinical_note.trim() && clinical_note !== 'No clinical notes provided'
  // The dashboard sets cvLiveData only while the camera is calibrated and sees a face.
  const usingCamera = cvLiveData != null

  return (
    <aside style={panelBox} aria-label="Live model input and output">
      <p style={{ ...label, color: PURPLE, marginBottom: 10 }}>Live Model Input → Output</p>

      {/* ── OUTPUT: SOFA ── */}
      <Section title="Predicted SOFA (output)" accent={riskC}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
          <span className="num" style={{ fontSize: 38, fontWeight: 800, color: riskC, lineHeight: 1, letterSpacing: '-0.04em' }}>
            {Number(sofa_score).toFixed(1)}
          </span>
          <span style={{ fontSize: 12, color: 'var(--t3)' }}>/ 24</span>
          <span style={{ marginLeft: 'auto', fontSize: 11, fontWeight: 800, color: riskC, fontFamily: 'var(--font-display)' }}>
            {String(risk_level || '').toUpperCase()}
          </span>
        </div>
        {conformal_lo != null && (
          <p style={{ fontSize: 10.5, color: 'var(--t3)', fontFamily: 'var(--font-display)', marginTop: 4 }}>
            90% range {conformal_lo} – {conformal_hi}
            {is_alert && <span style={{ color: RED, fontWeight: 800 }}>  ·  ALERT</span>}
          </p>
        )}
      </Section>

      {/* ── TIMER: when it ends, the next prediction reads the inputs below ── */}
      <Section title="Next SOFA prediction" accent={usingCamera ? PURPLE : undefined}>
        <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 8 }}>
          <span className="num" style={{ fontSize: 30, fontWeight: 800, color: isRefreshing ? AMBER : 'var(--t1)', lineHeight: 1 }}>
            {isRefreshing ? '…' : `${Math.floor(countdown / 60)}:${String(countdown % 60).padStart(2, '0')}`}
          </span>
          <span style={{ fontSize: 10.5, color: 'var(--t3)', fontFamily: 'var(--font-display)' }}>
            {isRefreshing ? 'Predicting now' : 'until prediction'}
          </span>
        </div>
        <div style={{ height: 6, borderRadius: 3, background: 'var(--s2)', marginBottom: 10, overflow: 'hidden' }}>
          <div style={{
            height: '100%', borderRadius: 3,
            width: `${Math.min(100, Math.max(0, ((90 - countdown) / 89) * 100))}%`,
            background: usingCamera ? PURPLE : 'var(--t3)',
            transition: 'width 1s linear',
          }} />
        </div>
        <p style={{ fontSize: 11, fontWeight: 700, fontFamily: 'var(--font-display)', color: usingCamera ? PURPLE : 'var(--t3)' }}>
          {usingCamera
            ? 'At 0 → CAMERA values go into the model'
            : 'At 0 → DATABASE values go into the model'}
        </p>
        <p style={{ fontSize: 10, color: 'var(--t4)', fontFamily: 'var(--font-display)', marginTop: 3, lineHeight: 1.5 }}>
          {usingCamera
            ? 'Camera is calibrated and sees a face. Stress and GCS eye are read at the moment the timer ends.'
            : 'No live camera reading yet (calibrating, no face, or camera closed), so the stored patient-record values are used.'}
        </p>
      </Section>

      {/* ── INPUTS 1 & 2: camera features ── */}
      <Section title="Input 1 & 2 — camera features" accent={PURPLE}>
        <CvInputRow
          name="Stress score (0–10)"
          live={cvLiveData?.stress}
          used={usedStress}
          fmt={v => Number(v).toFixed(1)}
          color={stressColor}
        />
        <CvInputRow
          name="GCS eye opening (1–4)"
          live={cvLiveData?.gcs_eye}
          used={usedGcs}
          fmt={v => `E${v}`}
          color={gcsColor}
        />
        <p style={{ fontSize: 10, color: 'var(--t4)', fontFamily: 'var(--font-display)', marginTop: 6, lineHeight: 1.5 }}>
          {cvLiveData
            ? `Camera says ${stressLabel(cvLiveData.stress)} stress, ${GCS_LABELS[cvLiveData.gcs_eye] ?? '—'} eyes. The model reads it on the next prediction.`
            : 'Until the camera is calibrated, the stored values from the patient record are used.'}
        </p>
      </Section>

      {/* ── INPUT 3: clinical note ── */}
      <Section title="Input 3 — clinical note (text → 90 TF-IDF features)">
        <p style={{
          fontSize: 11.5, lineHeight: 1.55, fontFamily: 'var(--font-display)',
          color: hasNote ? 'var(--t2)' : 'var(--t4)', maxHeight: 110, overflowY: 'auto', whiteSpace: 'pre-wrap',
        }}>
          {hasNote ? clinical_note : 'No clinical notes on record for this reading.'}
        </p>
      </Section>

      {/* ── Other inputs: vitals ── */}
      <Section title="Other inputs — vitals (this reading)">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px 12px' }}>
          {VITAL_ROWS.map(([key, name, unit]) => (
            <div key={key} style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between' }}>
              <span style={{ fontSize: 10.5, color: 'var(--t3)', fontFamily: 'var(--font-display)' }}>{name}</span>
              <span className="num" style={{ fontSize: 13, fontWeight: 700, color: statusColor(vitals_status[key]) }}>
                {vitals[key] ?? '—'}
                <span style={{ fontSize: 9, color: 'var(--t4)', fontWeight: 400, marginLeft: 2 }}>{unit}</span>
              </span>
            </div>
          ))}
        </div>
      </Section>
    </aside>
  )
}

const panelBox = {
  flex: '1 1 300px',
  minWidth: 280,
  maxWidth: 380,
  display: 'flex',
  flexDirection: 'column',
  alignSelf: 'flex-start',
  background: 'var(--s0)',
  borderLeft: '1px solid var(--b1)',
  paddingLeft: 16,
}
