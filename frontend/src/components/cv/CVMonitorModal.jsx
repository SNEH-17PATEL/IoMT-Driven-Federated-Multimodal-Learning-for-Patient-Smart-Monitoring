import { useEffect, useRef, useState, useCallback } from 'react'
import { AnimatePresence, motion } from 'framer-motion'

/* ── Design tokens ─────────────────────────────────────────────────────────── */
const PURPLE = '#a78bfa'
const a = (c, p) => `color-mix(in srgb, ${c} ${p}%, transparent)`

/* ── MediaPipe landmark indices (same as cv_monitor.py) ───────────────────── */
const L_EYE = [33, 160, 158, 133, 153, 144]
const R_EYE = [362, 385, 387, 263, 373, 380]
const L_BROW_INNER = 107
const R_BROW_INNER = 336
const NOSE_TIP = 1

/* ── Constants (mirrored from cv_monitor.py) ───────────────────────────────── */
const EMA_ALPHA    = 0.15
const PSPI_MAX     = 16.0
const BLINK_CLOSED = 0.55
const MOTION_SAT   = 0.020
const FACIAL_W     = 0.80
const SMOOTH_WIN   = 20

const GCS_WIN_S        = 30.0
const GCS_CALIB_N      = 90
const GCS_CALIB_BLINK  = 0.30
const GCS_CALIB_EAR_MIN = 0.15
const GCS_CALIB_EAR_MAX = 0.50
const GCS_OPEN_DEFAULT  = 0.22
const GCS_PART_DEFAULT  = 0.12
const GCS_OPEN_FRAC     = 0.70
const GCS_ANY_FRAC      = 0.30
const GCS_FLICKER_S     = 0.10
const GCS_MIN_OBS_S     = 10.0
const CALIB_FACE_N      = 90

/* ── Algorithm helpers (ported 1:1 from cv_monitor.py) ─────────────────────── */
function dist2D(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y)
}

function calcEAR(landmarks, idxs, W, H) {
  const p = idxs.map(i => ({ x: landmarks[i].x * W, y: landmarks[i].y * H }))
  const num = dist2D(p[1], p[5]) + dist2D(p[2], p[4])
  const den = 2.0 * dist2D(p[0], p[3]) + 1e-6
  return num / den
}

function pspiFromBlendshapes(bs) {
  const pair = name =>
    ((bs[`${name}Left`] ?? 0) + (bs[`${name}Right`] ?? 0)) / 2

  const au4  = pair('browDown')      * 5.0
  const au6  = pair('cheekSquint')   * 5.0
  const au7  = pair('eyeSquint')     * 5.0
  const au9  = pair('noseSneer')     * 5.0
  const au10 = pair('mouthUpperUp')  * 5.0
  const au43 = pair('eyeBlink') > BLINK_CLOSED ? 1.0 : 0.0

  const pspi = au4 + Math.max(au6, au7) + Math.max(au9, au10) + au43
  return pspi * (1.0 - pair('mouthSmile'))
}

function updateStress(pspi, motion, ema) {
  const facial    = Math.min(pspi / PSPI_MAX, 1.0)
  const agitation = Math.min(motion / MOTION_SAT, 1.0)
  const frameS    = 10.0 * (FACIAL_W * facial + (1 - FACIAL_W) * agitation)
  const newEma    = ema === null ? frameS : EMA_ALPHA * frameS + (1 - EMA_ALPHA) * ema
  return { display: Math.min(newEma, 10.0), ema: newEma }
}

function eyeState(ear, blinkScore, openTh, partialTh) {
  if (blinkScore !== null && blinkScore > BLINK_CLOSED) return 'closed'
  if (ear >= openTh)    return 'open'
  if (ear >= partialTh) return 'partial'
  return 'closed'
}

function gcsFromWindow(history) {
  if (!history.length) return { score: 4, label: 'Spontaneous' }
  const states    = history.map(h => h.state)
  const n         = states.length
  const openFrac  = states.filter(s => s === 'open').length / n
  const anyFrac   = states.filter(s => s !== 'closed').length / n

  if (openFrac >= GCS_OPEN_FRAC) return { score: 4, label: 'Spontaneous' }
  if (anyFrac  >= GCS_ANY_FRAC)  return { score: 3, label: 'To Voice' }

  const obsSecs  = history.length >= 2 ? history[history.length - 1].ts - history[0].ts : 0
  const frameSec = n > 1 ? obsSecs / (n - 1) : 1 / 30
  let run = 0, longest = 0
  for (const s of states) {
    run     = s !== 'closed' ? run + 1 : 0
    longest = Math.max(longest, run)
  }
  if (longest * frameSec >= GCS_FLICKER_S) return { score: 2, label: 'To Pain' }
  return { score: 1, label: 'No Response' }
}

function stressColor(s) {
  if (s < 3.5) return '#4ade80'   // green
  if (s < 6.5) return '#fbbf24'   // amber
  return '#f87171'                  // red
}
function stressLabel(s) {
  if (s < 3.5) return 'LOW'
  if (s < 6.5) return 'MODERATE'
  return 'HIGH'
}
function gcsColor(g) {
  // E1=RED, E2=RED, E3=AMBER, E4=GREEN — matches cv_monitor.py _gcs_col()
  return [null, '#f87171', '#f87171', '#fbbf24', '#4ade80'][g]
}

/* ── SVG Icons ─────────────────────────────────────────────────────────────── */
function EyeIcon({ size = 16 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 18 18" fill="none" stroke="currentColor"
      strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M1 9C1 9 4 3 9 3s8 6 8 6-3 6-8 6S1 9 1 9z"/>
      <circle cx="9" cy="9" r="2.5"/>
    </svg>
  )
}

function CloseIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" aria-hidden="true">
      <path d="M2 2l12 12M14 2L2 14"/>
    </svg>
  )
}

/* ── Score Card subcomponents ───────────────────────────────────────────────── */
function MiniBar({ value, color, label, pct }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
      <span style={{ fontSize: '10px', color: 'var(--t4)', fontFamily: 'var(--font-display)', width: 80, flexShrink: 0 }}>
        {label}
      </span>
      <div style={{ flex: 1, height: 6, borderRadius: 3, background: 'var(--s2)' }}>
        <div style={{
          height: '100%', borderRadius: 3,
          width: `${Math.round(pct * 100)}%`,
          background: color,
          transition: 'width 0.25s ease, background 0.3s ease',
        }} />
      </div>
      <span style={{ fontSize: '10px', color, fontFamily: 'var(--font-display)', width: 28, textAlign: 'right', flexShrink: 0 }}>
        {Math.round(pct * 100)}%
      </span>
    </div>
  )
}

/* ── Loading / Error / Permission states ───────────────────────────────────── */
function StatusPanel({ status, errorMsg, onRetry }) {
  if (status === 'loading-model') return (
    <div style={{ textAlign: 'center', padding: '40px 24px' }}>
      <div style={{
        width: 48, height: 48, borderRadius: '50%', margin: '0 auto 16px',
        border: `3px solid ${a(PURPLE, 20)}`,
        borderTopColor: PURPLE,
        animation: 'mapSpin 0.9s linear infinite',
      }} />
      <p style={{ color: 'var(--t1)', fontSize: 14, fontWeight: 600, fontFamily: 'var(--font-display)', marginBottom: 6 }}>
        Loading AI Model
      </p>
      <p style={{ color: 'var(--t4)', fontSize: 12, fontFamily: 'var(--font-display)' }}>
        Downloading MediaPipe FaceLandmarker (~12 MB) — first launch only
      </p>
    </div>
  )

  if (status === 'loading-camera') return (
    <div style={{ textAlign: 'center', padding: '40px 24px' }}>
      <div style={{
        width: 48, height: 48, borderRadius: '50%', margin: '0 auto 16px',
        border: `3px solid ${a(PURPLE, 20)}`,
        borderTopColor: PURPLE,
        animation: 'mapSpin 0.9s linear infinite',
      }} />
      <p style={{ color: 'var(--t1)', fontSize: 14, fontWeight: 600, fontFamily: 'var(--font-display)', marginBottom: 6 }}>
        Starting Camera
      </p>
      <p style={{ color: 'var(--t4)', fontSize: 12, fontFamily: 'var(--font-display)' }}>
        Please allow camera access when prompted
      </p>
    </div>
  )

  if (status === 'error') return (
    <div style={{ textAlign: 'center', padding: '40px 24px' }}>
      <div style={{ fontSize: 36, marginBottom: 12, opacity: 0.6 }}>⚠</div>
      <p style={{ color: 'var(--t1)', fontSize: 14, fontWeight: 600, fontFamily: 'var(--font-display)', marginBottom: 8 }}>
        CV Monitor Unavailable
      </p>
      <p style={{ color: 'var(--t3)', fontSize: 12, maxWidth: 340, margin: '0 auto 16px', fontFamily: 'var(--font-display)' }}>
        {errorMsg || 'Could not start the camera or load the face detection model.'}
      </p>
      <button
        onClick={onRetry}
        style={{
          padding: '8px 18px', borderRadius: 8, cursor: 'pointer',
          background: a(PURPLE, 15), border: `1px solid ${a(PURPLE, 30)}`,
          color: PURPLE, fontSize: 12, fontWeight: 700, fontFamily: 'var(--font-display)',
        }}
      >
        Retry
      </button>
    </div>
  )

  return null
}

/* ══════════════════════════════════════════════════════════════════════════════
   MAIN MODAL
══════════════════════════════════════════════════════════════════════════════ */
export default function CVMonitorModal({ open, onClose, onCVResults }) {
  const videoRef   = useRef(null)
  const canvasRef  = useRef(null)
  const flRef      = useRef(null)       // FaceLandmarker instance
  const streamRef  = useRef(null)       // MediaStream
  const rafRef     = useRef(null)       // requestAnimationFrame handle
  const stateRef   = useRef(null)       // mutable detection state

  const [status,     setStatus]     = useState('idle')     // idle|loading-model|loading-camera|running|error
  const [errorMsg,   setErrorMsg]   = useState('')
  const [initKey,    setInitKey]    = useState(0)           // bump to retry

  /* ── Rendered (React-state) metrics ── */
  const [stress,       setStress]       = useState(0)
  const [stressLabel_, setStressLabel_] = useState('LOW')
  const [stressColor_, setStressColor_] = useState('#4ade80')
  const [stressPeak,   setStressPeak]   = useState(0)
  const [comps,        setComps]        = useState([0, 0, 0])
  const [gcsScore,     setGcsScore]     = useState(4)
  const [gcsLabel_,    setGcsLabel_]    = useState('Spontaneous')
  const [gcsColor_,    setGcsColor_]    = useState('#4ade80')
  const [gcsReady,     setGcsReady]     = useState(false)
  const [gcsProvis,    setGcsProvis]    = useState(true)
  const [earVal,       setEarVal]       = useState(0)
  const [gcsTh,        setGcsTh]        = useState([GCS_OPEN_DEFAULT, GCS_PART_DEFAULT])
  const [detected,     setDetected]     = useState(false)
  const [calibPct,     setCalibPct]     = useState(0)

  /* ── Canvas drawing ──────────────────────────────────────────────────────── */
  const drawLandmarks = useCallback((landmarks) => {
    const canvas = canvasRef.current
    const video  = videoRef.current
    if (!canvas || !video) return
    const ctx = canvas.getContext('2d')
    const W   = canvas.width
    const H   = canvas.height

    ctx.clearRect(0, 0, W, H)

    // Eye contours
    ctx.strokeStyle = 'rgba(60, 220, 200, 0.85)'
    ctx.lineWidth   = 1.5
    for (const eyeIdxs of [L_EYE, R_EYE]) {
      ctx.beginPath()
      eyeIdxs.forEach((idx, i) => {
        const x = landmarks[idx].x * W
        const y = landmarks[idx].y * H
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
      })
      ctx.closePath()
      ctx.stroke()
    }
    // Key dots
    const dotIdxs = new Set([...L_EYE, ...R_EYE, L_BROW_INNER, R_BROW_INNER])
    ctx.fillStyle = 'rgba(60, 220, 200, 0.9)'
    for (const idx of dotIdxs) {
      ctx.beginPath()
      ctx.arc(landmarks[idx].x * W, landmarks[idx].y * H, 2, 0, Math.PI * 2)
      ctx.fill()
    }
  }, [])

  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
  }, [])

  /* ── Detection loop ──────────────────────────────────────────────────────── */
  const runLoop = useCallback(() => {
    const video = videoRef.current
    const fl    = flRef.current
    const st    = stateRef.current
    if (!video || !fl || !st || video.readyState < 2) {
      rafRef.current = requestAnimationFrame(runLoop)
      return
    }

    const W = video.videoWidth  || 640
    const H = video.videoHeight || 480
    if (canvasRef.current) {
      if (canvasRef.current.width  !== W) canvasRef.current.width  = W
      if (canvasRef.current.height !== H) canvasRef.current.height = H
    }

    const result = fl.detectForVideo(video, Date.now())
    const det    = result.faceLandmarks && result.faceLandmarks.length > 0

    setDetected(det)

    if (det) {
      const lm = result.faceLandmarks[0]
      st.faceFrames++

      // Blendshapes
      const bs = {}
      if (result.faceBlendshapes && result.faceBlendshapes[0]) {
        for (const b of result.faceBlendshapes[0].categories) {
          bs[b.categoryName] = b.score
        }
      }

      const blinkV = bs.eyeBlinkLeft != null
        ? (bs.eyeBlinkLeft + (bs.eyeBlinkRight ?? 0)) / 2
        : null

      // EAR
      const earL = calcEAR(lm, L_EYE, W, H)
      const earR = calcEAR(lm, R_EYE, W, H)
      const earV = (earL + earR) / 2

      st.earQueue.push(earV)
      if (st.earQueue.length > SMOOTH_WIN) st.earQueue.shift()
      const earSm = st.earQueue.reduce((s, v) => s + v, 0) / st.earQueue.length

      // GCS calibration
      if (!st.gcsCalibDone) {
        if (blinkV === null || blinkV < GCS_CALIB_BLINK) st.gcsCalibEar.push(earV)
        if (st.gcsCalibEar.length >= GCS_CALIB_N) {
          const sorted = [...st.gcsCalibEar].sort((a, b) => a - b)
          const mid    = Math.floor(sorted.length / 2)
          // np.median: average of two middle elements for even-length arrays
          const baseline = sorted.length % 2 === 0
            ? (sorted[mid - 1] + sorted[mid]) / 2
            : sorted[mid]
          if (baseline >= GCS_CALIB_EAR_MIN && baseline <= GCS_CALIB_EAR_MAX) {
            st.gcsOpenTh    = baseline * 0.70
            st.gcsPartialTh = baseline * 0.40
          }
          st.gcsCalibDone = true
          setGcsReady(true)
          setGcsTh([st.gcsOpenTh, st.gcsPartialTh])
        }
      }

      // Head motion agitation
      const nose = lm[NOSE_TIP]
      st.noseTrack.push({ x: nose.x, y: nose.y })
      if (st.noseTrack.length > 30) st.noseTrack.shift()
      let motion = 0
      if (st.noseTrack.length >= 2) {
        let sum = 0
        for (let i = 1; i < st.noseTrack.length; i++) {
          sum += Math.hypot(
            st.noseTrack[i].x - st.noseTrack[i - 1].x,
            st.noseTrack[i].y - st.noseTrack[i - 1].y,
          )
        }
        motion = sum / (st.noseTrack.length - 1)
      }

      // PSPI stress
      const pspi = pspiFromBlendshapes(bs)
      const { display: stressV, ema: newEma } = updateStress(pspi, motion, st.stressEma)
      st.stressEma = newEma

      const sRound  = Math.round(stressV * 10) / 10
      st.stressPeak = Math.max(st.stressPeak, sRound)   // track rounded value, same as Python
      setStress(sRound)
      setStressLabel_(stressLabel(sRound))
      setStressColor_(stressColor(sRound))
      setStressPeak(st.stressPeak)

      // PSPI components
      const pairFn = name =>
        ((bs[`${name}Left`] ?? 0) + (bs[`${name}Right`] ?? 0)) / 2
      setComps([
        pairFn('browDown'),
        Math.max(pairFn('cheekSquint'), pairFn('eyeSquint')),
        Math.max(pairFn('noseSneer'),   pairFn('mouthUpperUp')),
      ])

      // GCS window
      const nowS = Date.now() / 1000
      const est  = eyeState(earSm, blinkV, st.gcsOpenTh, st.gcsPartialTh)
      st.gcsHistory.push({ ts: nowS, state: est })
      while (st.gcsHistory.length > 0 && nowS - st.gcsHistory[0].ts > GCS_WIN_S) {
        st.gcsHistory.shift()
      }
      const { score, label } = gcsFromWindow(st.gcsHistory)
      setGcsScore(score)
      setGcsLabel_(label)
      setGcsColor_(gcsColor(score))
      onCVResults?.({ stress: sRound, gcs_eye: score })

      const obsSec = st.gcsHistory.length >= 2
        ? st.gcsHistory[st.gcsHistory.length - 1].ts - st.gcsHistory[0].ts
        : 0
      setGcsProvis(obsSec < GCS_MIN_OBS_S)
      setEarVal(Math.round(earSm * 1000) / 1000)

      drawLandmarks(lm)
    } else {
      clearCanvas()
    }

    // Calibration progress
    const cpct = Math.min(st.faceFrames / CALIB_FACE_N, 1.0)
    setCalibPct(cpct)

    rafRef.current = requestAnimationFrame(runLoop)
  }, [drawLandmarks, clearCanvas])

  /* ── Escape key closes modal ─────────────────────────────────────────────── */
  useEffect(() => {
    if (!open) return
    const handler = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [open, onClose])

  /* ── Initialise / teardown on open ──────────────────────────────────────── */
  useEffect(() => {
    if (!open) return

    let cancelled = false

    stateRef.current = {
      faceFrames:   0,
      earQueue:     [],
      stressEma:    null,
      stressPeak:   0,
      noseTrack:    [],
      gcsHistory:   [],
      gcsCalibEar:  [],
      gcsCalibDone: false,
      gcsOpenTh:    GCS_OPEN_DEFAULT,
      gcsPartialTh: GCS_PART_DEFAULT,
    }

    // Reset display state
    setStress(0); setStressLabel_('LOW'); setStressColor_('#4ade80')
    setStressPeak(0); setComps([0, 0, 0]); setGcsScore(4)
    setGcsLabel_('Spontaneous'); setGcsColor_('#4ade80')
    setGcsReady(false); setGcsProvis(true); setEarVal(0)
    setGcsTh([GCS_OPEN_DEFAULT, GCS_PART_DEFAULT]); setDetected(false)
    setCalibPct(0); setErrorMsg('')

    async function init() {
      try {
        setStatus('loading-model')
        const { FaceLandmarker, FilesetResolver } = await import('@mediapipe/tasks-vision')
        if (cancelled) return

        const resolver = await FilesetResolver.forVisionTasks('/mediapipe/wasm')
        if (cancelled) return

        flRef.current = await FaceLandmarker.createFromOptions(resolver, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task',
            delegate: 'GPU',
          },
          outputFaceBlendshapes: true,
          runningMode:           'VIDEO',
          numFaces:              1,
          minFaceDetectionConfidence: 0.50,
          minFacePresenceConfidence:  0.50,
          minTrackingConfidence:      0.50,
        })
        if (cancelled) { flRef.current.close(); return }

        setStatus('loading-camera')
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
        })
        if (cancelled) { stream.getTracks().forEach(t => t.stop()); return }

        streamRef.current = stream
        videoRef.current.srcObject = stream
        await videoRef.current.play()
        if (cancelled) return

        setStatus('running')
        rafRef.current = requestAnimationFrame(runLoop)

      } catch (err) {
        if (!cancelled) {
          setStatus('error')
          const msg = err.name === 'NotAllowedError'
            ? 'Camera permission denied. Please allow camera access in your browser.'
            : err.name === 'NotFoundError'
            ? 'No camera found. Please connect a webcam and try again.'
            : (err.message?.toLowerCase().includes('fetch') || err.message?.toLowerCase().includes('network'))
            ? 'Could not download face detection model — internet connection required for first use.'
            : err.message || 'Unknown error starting the CV monitor.'
          setErrorMsg(msg)
        }
      }
    }

    init()

    return () => {
      cancelled = true
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop())
      if (flRef.current)   { try { flRef.current.close() } catch (_) {} }
      flRef.current    = null
      streamRef.current = null
      stateRef.current  = null
    }
  }, [open, initKey, runLoop])

  const handleRetry = () => {
    setInitKey(k => k + 1)
    setStatus('idle')
  }

  const isRunning  = status === 'running'
  const showStatus = !isRunning

  return (
    <AnimatePresence>
      {open && (
        <>
          {/* ── Backdrop ── */}
          <motion.div
            key="cv-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.22 }}
            onClick={onClose}
            style={{
              position: 'fixed', inset: 0, zIndex: 920,
              background: 'rgba(4, 10, 20, 0.72)',
              backdropFilter: 'blur(8px)',
              WebkitBackdropFilter: 'blur(8px)',
            }}
          />

          {/* ── Modal panel ── */}
          {/* Outer div handles CSS centering; inner motion.div handles Framer transform */}
          <div style={{
            position: 'fixed',
            top: '50%', left: '50%',
            transform: 'translate(-50%, -50%)',
            width: 'min(94vw, 980px)',
            maxHeight: '92vh',
            zIndex: 930,
          }}>
          <motion.div
            key="cv-modal"
            initial={{ opacity: 0, scale: 0.96, y: 24 }}
            animate={{ opacity: 1, scale: 1,    y: 0 }}
            exit={{ opacity: 0,   scale: 0.96,  y: 16 }}
            transition={{ type: 'spring', stiffness: 340, damping: 30 }}
            style={{
              width: '100%',
              maxHeight: '92vh',
              display: 'flex',
              flexDirection: 'column',
              background: 'var(--s0)',
              borderRadius: 'var(--r-xl)',
              border: `1px solid ${a(PURPLE, 18)}`,
              boxShadow: `0 24px 80px rgba(0,0,0,0.65), 0 0 0 1px ${a(PURPLE, 8)}, 0 0 60px ${a(PURPLE, 8)}`,
              overflow: 'hidden',
            }}
          >
            {/* ══ Header ════════════════════════════════════════════════════════ */}
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '16px 22px',
              background: `linear-gradient(135deg, var(--s1) 0%, var(--s0) 100%)`,
              borderBottom: `1px solid ${a(PURPLE, 18)}`,
              flexShrink: 0,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                {/* Icon badge */}
                <div style={{
                  width: 38, height: 38, borderRadius: 11,
                  background: `linear-gradient(145deg, ${a(PURPLE, 18)}, ${a(PURPLE, 6)})`,
                  border: `1.5px solid ${a(PURPLE, 25)}`,
                  color: PURPLE,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  animation: isRunning ? 'eyeBlink 2.5s ease-in-out infinite' : 'none',
                }}>
                  <EyeIcon size={16} />
                </div>
                <div>
                  <span style={{
                    display: 'block',
                    fontSize: '15px', fontWeight: 700, color: 'var(--t1)',
                    fontFamily: 'var(--font-display)', letterSpacing: '-0.01em',
                  }}>
                    Live CV Assessment
                  </span>
                  <span style={{ fontSize: '11px', color: 'var(--t3)', fontFamily: 'var(--font-display)' }}>
                    Stress Score &amp; GCS Eye Score via computer vision · PSPI formula
                  </span>
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                {/* Live indicator */}
                {isRunning && (
                  <div style={{
                    display: 'flex', alignItems: 'center', gap: 6,
                    padding: '4px 10px', borderRadius: 20,
                    background: a('#ef4444', 12),
                    border: `1px solid ${a('#ef4444', 25)}`,
                    fontSize: '11px', fontWeight: 700,
                    color: '#f87171', fontFamily: 'var(--font-display)',
                  }}>
                    <span style={{
                      width: 6, height: 6, borderRadius: '50%',
                      background: '#ef4444', display: 'inline-block',
                      animation: 'pulse 1.6s ease-in-out infinite',
                    }} />
                    LIVE
                  </div>
                )}

                {/* Close */}
                <motion.button
                  whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.92 }}
                  onClick={onClose}
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

            {/* ══ Content ═══════════════════════════════════════════════════════ */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '20px' }}>

              {/* ── Video + Canvas stack ── */}
              <div style={{
                position: 'relative',
                width: '100%', borderRadius: 14,
                overflow: 'hidden',
                background: '#07111e',
                border: `1px solid ${a(PURPLE, 15)}`,
                marginBottom: 16,
                aspectRatio: '16/9',
                display: showStatus && !isRunning ? 'flex' : 'block',
                alignItems: 'center', justifyContent: 'center',
                minHeight: showStatus ? 200 : undefined,
              }}>
                {/* Status overlay (loading / error) */}
                {showStatus && (
                  <div style={{
                    position: isRunning ? 'absolute' : 'static',
                    inset: 0, zIndex: 2,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                  }}>
                    <StatusPanel status={status} errorMsg={errorMsg} onRetry={handleRetry} />
                  </div>
                )}

                {/* Video element — always mounted so ref is ready */}
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  style={{
                    display: isRunning ? 'block' : 'none',
                    width: '100%', height: '100%',
                    objectFit: 'cover',
                    transform: 'scaleX(-1)',  // mirror
                  }}
                />

                {/* Canvas overlay for landmark drawing */}
                <canvas
                  ref={canvasRef}
                  style={{
                    display: isRunning ? 'block' : 'none',
                    position: 'absolute', top: 0, left: 0,
                    width: '100%', height: '100%',
                    transform: 'scaleX(-1)',  // mirror to match video
                    pointerEvents: 'none',
                  }}
                />

                {/* Calibration bar overlay */}
                {isRunning && calibPct < 1.0 && (
                  <div style={{
                    position: 'absolute', bottom: 0, left: 0, right: 0,
                    padding: '12px 16px',
                    background: 'linear-gradient(to top, rgba(4,10,20,0.92), transparent)',
                    zIndex: 3,
                  }}>
                    <p style={{ color: '#fbbf24', fontSize: 11, fontWeight: 700, fontFamily: 'var(--font-display)', marginBottom: 6 }}>
                      Calibrating personal eye baseline — keep eyes open and expression neutral
                    </p>
                    <div style={{ height: 5, borderRadius: 3, background: a('#fbbf24', 20) }}>
                      <div style={{
                        height: '100%', borderRadius: 3,
                        width: `${Math.round(calibPct * 100)}%`,
                        background: '#fbbf24',
                        transition: 'width 0.18s ease',
                      }} />
                    </div>
                    <p style={{ color: '#fbbf24', fontSize: 10, fontFamily: 'var(--font-display)', marginTop: 4 }}>
                      {Math.round(calibPct * 100)}% complete
                    </p>
                  </div>
                )}

                {/* No-face warning */}
                {isRunning && !detected && calibPct >= 1.0 && (
                  <div style={{
                    position: 'absolute', bottom: 16, left: '50%', transform: 'translateX(-50%)',
                    padding: '6px 14px', borderRadius: 20, zIndex: 3,
                    background: a('#f87171', 15), border: `1px solid ${a('#f87171', 30)}`,
                    color: '#f87171', fontSize: 11, fontWeight: 600, fontFamily: 'var(--font-display)',
                    whiteSpace: 'nowrap',
                  }}>
                    No face detected — centre your face in the camera
                  </div>
                )}
              </div>

              {/* ── Score cards row ── */}
              {isRunning && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>

                  {/* ── STRESS SCORE card ── */}
                  <div style={{
                    background: 'var(--s1)',
                    border: `1px solid ${a(stressColor_, 22)}`,
                    borderRadius: 14,
                    padding: '16px 18px',
                    boxShadow: `0 0 20px ${a(stressColor_, 8)}`,
                    transition: 'border-color 0.4s, box-shadow 0.4s',
                  }}>
                    <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 10 }}>
                      <div>
                        <p style={{ fontSize: 10, fontWeight: 700, color: 'var(--t4)', letterSpacing: '0.10em', textTransform: 'uppercase', fontFamily: 'var(--font-display)', marginBottom: 4 }}>
                          Stress Score  <span style={{ color: 'var(--t4)', fontWeight: 400 }}>(PSPI)</span>
                        </p>
                        <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
                          <span className="num" style={{ fontSize: 36, fontWeight: 800, color: stressColor_, letterSpacing: '-0.04em', lineHeight: 1, transition: 'color 0.3s' }}>
                            {stress.toFixed(1)}
                          </span>
                          <span style={{ fontSize: 13, color: 'var(--t3)' }}>/ 10</span>
                        </div>
                        <span style={{
                          display: 'inline-block', marginTop: 4,
                          fontSize: 11, fontWeight: 800, letterSpacing: '0.10em',
                          color: stressColor_, fontFamily: 'var(--font-display)',
                        }}>
                          {stressLabel_}
                        </span>
                      </div>
                      <div style={{ textAlign: 'right' }}>
                        <p style={{ fontSize: 10, color: 'var(--t4)', fontFamily: 'var(--font-display)', marginBottom: 3 }}>Session peak</p>
                        <span className="num" style={{ fontSize: 16, fontWeight: 700, color: stressColor(stressPeak) }}>
                          {stressPeak.toFixed(1)}
                        </span>
                      </div>
                    </div>

                    {/* Overall bar */}
                    <div style={{ height: 7, borderRadius: 4, background: 'var(--s2)', marginBottom: 14 }}>
                      <div style={{
                        height: '100%', borderRadius: 4,
                        width: `${(stress / 10) * 100}%`,
                        background: stressColor_,
                        transition: 'width 0.25s ease, background 0.3s ease',
                        boxShadow: `0 0 8px ${a(stressColor_, 50)}`,
                      }} />
                    </div>

                    {/* Component bars */}
                    <MiniBar label="Brow Furrow"  pct={comps[0]} color={comps[0] > 0.65 ? '#f87171' : comps[0] > 0.35 ? '#fbbf24' : '#4ade80'} />
                    <MiniBar label="Eye Tension"  pct={comps[1]} color={comps[1] > 0.65 ? '#f87171' : comps[1] > 0.35 ? '#fbbf24' : '#4ade80'} />
                    <MiniBar label="Nose / Lip"   pct={comps[2]} color={comps[2] > 0.65 ? '#f87171' : comps[2] > 0.35 ? '#fbbf24' : '#4ade80'} />
                  </div>

                  {/* ── GCS EYE SCORE card ── */}
                  <div style={{
                    background: 'var(--s1)',
                    border: `1px solid ${a(gcsColor_, 22)}`,
                    borderRadius: 14,
                    padding: '16px 18px',
                    boxShadow: `0 0 20px ${a(gcsColor_, 8)}`,
                    transition: 'border-color 0.4s, box-shadow 0.4s',
                  }}>
                    <p style={{ fontSize: 10, fontWeight: 700, color: 'var(--t4)', letterSpacing: '0.10em', textTransform: 'uppercase', fontFamily: 'var(--font-display)', marginBottom: 8 }}>
                      GCS Eye Score  <span style={{ fontWeight: 400 }}>(Rolling 30s)</span>
                    </p>

                    <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, marginBottom: 4 }}>
                      <span className="num" style={{ fontSize: 36, fontWeight: 800, color: gcsColor_, letterSpacing: '-0.04em', lineHeight: 1, transition: 'color 0.3s' }}>
                        E{gcsScore}
                      </span>
                      <span style={{ fontSize: 15, fontWeight: 700, color: gcsColor_, fontFamily: 'var(--font-display)' }}>
                        {gcsLabel_}
                      </span>
                    </div>

                    {gcsProvis && (
                      <p style={{ fontSize: 10, color: '#fbbf24', fontFamily: 'var(--font-display)', marginBottom: 8 }}>
                        Provisional — observing (&lt; 10s window)
                      </p>
                    )}

                    {/* Dot scale E1–E4 */}
                    <div style={{ display: 'flex', gap: 10, marginBottom: 14, marginTop: gcsProvis ? 0 : 8 }}>
                      {[1, 2, 3, 4].map(n => (
                        <div key={n} style={{
                          width: 40, height: 40, borderRadius: '50%',
                          background: n === gcsScore ? gcsColor_ : a(gcsColor_, 12),
                          border: `2px solid ${n === gcsScore ? gcsColor_ : a(gcsColor_, 25)}`,
                          display: 'flex', alignItems: 'center', justifyContent: 'center',
                          fontSize: 11, fontWeight: 700, fontFamily: 'var(--font-display)',
                          color: n === gcsScore ? '#07111e' : 'var(--t4)',
                          transition: 'all 0.3s ease',
                          boxShadow: n === gcsScore ? `0 0 12px ${a(gcsColor_, 50)}` : 'none',
                        }}>
                          E{n}
                        </div>
                      ))}
                    </div>

                    {/* EAR metrics */}
                    <div style={{ fontSize: 11, color: 'var(--t3)', fontFamily: 'var(--font-display)', lineHeight: 1.7 }}>
                      <div>Eye Aspect Ratio (EAR):&ensp;
                        <span className="num" style={{ color: 'var(--t2)' }}>{earVal.toFixed(3)}</span>
                      </div>
                      <div>
                        Open≥<span className="num">{gcsTh[0].toFixed(3)}</span>&ensp;
                        Partial≥<span className="num">{gcsTh[1].toFixed(3)}</span>
                        &ensp;
                        <span style={{ color: gcsReady ? '#4ade80' : '#fbbf24', fontWeight: 700 }}>
                          ({gcsReady ? 'personal' : 'default'})
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* ══ Footer ═════════════════════════════════════════════════════════ */}
            <div style={{
              padding: '10px 22px',
              background: 'var(--s1)',
              borderTop: `1px solid ${a(PURPLE, 12)}`,
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              flexShrink: 0,
            }}>
              <span style={{ fontSize: '10.5px', color: 'var(--t4)', fontFamily: 'var(--font-display)' }}>
                ⚠ Supplementary visual assessment — not a clinical diagnosis
              </span>
              <span style={{ fontSize: '10.5px', color: 'var(--t4)', fontFamily: 'var(--font-display)' }}>
                PSPI formula · 30s GCS window · Live CV data supplements clinical context; SOFA prediction uses dataset values
              </span>
            </div>
          </motion.div>
          </div>
        </>
      )}
    </AnimatePresence>
  )
}
