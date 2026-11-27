/*
 * CVMonitorModal — the full start-up and detection flow, driven by fakes:
 *
 *   • @mediapipe/tasks-vision  -> a landmarker whose detectForVideo() returns scripted frames
 *   • navigator.mediaDevices   -> a fake camera stream
 *   • <video>                  -> readyState / currentTime / play() controlled by the test
 *   • requestAnimationFrame    -> run manually, one frame per call to nextFrame()
 *   • Date.now / performance   -> a fake clock the test advances
 *
 * Nothing touches a real camera or downloads the face model.
 */
import { render, screen, fireEvent, act, waitFor } from '@testing-library/react'
import CVMonitorModal from '@/components/cv/CVMonitorModal.jsx'
import * as mp from '@mediapipe/tasks-vision'

// ── Fakes ─────────────────────────────────────────────────────────────────────

/** 478 landmarks; eyes shaped so the eye-aspect-ratio is `ear` (0.30 = wide open). */
function faceLandmarks(ear = 0.30, noseX = 0.5) {
  const lm = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5 }))
  const setEye = (idx, cx) => {
    // p0 / p3 are the corners (width 0.10), p1 p5 and p2 p4 are the lid pairs
    const h = ear * 0.10 * 1.0
    lm[idx[0]] = { x: cx - 0.05, y: 0.4 }
    lm[idx[3]] = { x: cx + 0.05, y: 0.4 }
    lm[idx[1]] = { x: cx - 0.02, y: 0.4 - h / 2 }; lm[idx[5]] = { x: cx - 0.02, y: 0.4 + h / 2 }
    lm[idx[2]] = { x: cx + 0.02, y: 0.4 - h / 2 }; lm[idx[4]] = { x: cx + 0.02, y: 0.4 + h / 2 }
  }
  setEye([33, 160, 158, 133, 153, 144], 0.35)
  setEye([362, 385, 387, 263, 373, 380], 0.65)
  lm[1] = { x: noseX, y: 0.55 }
  return lm
}

const blend = (o = {}) => ({
  categories: Object.entries(o).map(([categoryName, score]) => ({ categoryName, score })),
})
const frame = ({ ear = 0.30, blendshapes = { eyeBlinkLeft: 0.05, eyeBlinkRight: 0.05 }, noseX = 0.5 } = {}) =>
  ({ faceLandmarks: [faceLandmarks(ear, noseX)], faceBlendshapes: [blend(blendshapes)] })
const NO_FACE = { faceLandmarks: [], faceBlendshapes: [] }

let landmarker, stream, trackStop, rafQueue, clockMs, videoTime

function setupEnvironment({ cameraError = null, modelError = null } = {}) {
  clockMs = 1_700_000_000_000
  videoTime = 0
  rafQueue = []
  trackStop = vi.fn()
  stream = { getTracks: () => [{ stop: trackStop }] }
  landmarker = { detectForVideo: vi.fn(() => frame()), close: vi.fn() }

  mp.FilesetResolver.forVisionTasks.mockReset().mockResolvedValue({})
  mp.FaceLandmarker.createFromOptions.mockReset()
  if (modelError) mp.FaceLandmarker.createFromOptions.mockRejectedValue(modelError)
  else mp.FaceLandmarker.createFromOptions.mockResolvedValue(landmarker)

  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: { getUserMedia: cameraError ? vi.fn().mockRejectedValue(cameraError) : vi.fn().mockResolvedValue(stream) },
  })

  vi.spyOn(Date, 'now').mockImplementation(() => clockMs)
  vi.spyOn(performance, 'now').mockImplementation(() => clockMs)
  vi.stubGlobal('requestAnimationFrame', cb => { rafQueue.push(cb); return rafQueue.length })
  vi.stubGlobal('cancelAnimationFrame', () => { rafQueue = [] })

  // jsdom has no media playback or canvas drawing
  HTMLMediaElement.prototype.play = vi.fn().mockResolvedValue(undefined)
  Object.defineProperty(HTMLMediaElement.prototype, 'readyState', { configurable: true, get: () => 4 })
  Object.defineProperty(HTMLMediaElement.prototype, 'currentTime', { configurable: true, get: () => videoTime })
  HTMLCanvasElement.prototype.getContext = vi.fn(() => new Proxy({}, { get: () => () => {} }))
}

/** Run one animation frame: advance the video clock, then fire the queued rAF callback. */
async function nextFrame(deltaMs = 33) {
  clockMs += deltaMs
  videoTime += deltaMs / 1000
  const cb = rafQueue.shift()
  if (cb) await act(async () => { cb() })
}
async function frames(n, deltaMs = 33) { for (let i = 0; i < n; i++) await nextFrame(deltaMs) }

async function openAndStart(props = {}) {
  const onCVResults = props.onCVResults ?? vi.fn()
  const utils = render(<CVMonitorModal open onClose={props.onClose ?? (() => {})} onCVResults={onCVResults} {...props} />)
  await waitFor(() => expect(screen.getByText(/Calibrating personal eye baseline/)).toBeInTheDocument())
  return { ...utils, onCVResults }
}

beforeEach(() => setupEnvironment())
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals() })

// ── Start-up ──────────────────────────────────────────────────────────────────

describe('start-up', () => {
  it('loads the model, then the camera, then runs', async () => {
    await openAndStart()
    expect(mp.FilesetResolver.forVisionTasks).toHaveBeenCalledWith('/mediapipe/wasm')
    const [, opts] = mp.FaceLandmarker.createFromOptions.mock.calls[0]
    expect(opts.runningMode).toBe('VIDEO')
    expect(opts.numFaces).toBe(1)
    expect(opts.outputFaceBlendshapes).toBe(true)
    expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledTimes(1)
    expect(screen.getByText('LIVE')).toBeInTheDocument()
  })

  it('shows the model-loading message first', () => {
    mp.FaceLandmarker.createFromOptions.mockReturnValue(new Promise(() => {}))
    render(<CVMonitorModal open onClose={() => {}} onCVResults={() => {}} />)
    return waitFor(() => expect(screen.getByText('Loading AI Model')).toBeInTheDocument())
  })

  it('shows the camera message while waiting for permission', async () => {
    navigator.mediaDevices.getUserMedia = vi.fn(() => new Promise(() => {}))
    render(<CVMonitorModal open onClose={() => {}} onCVResults={() => {}} />)
    expect(await screen.findByText('Starting Camera')).toBeInTheDocument()
  })

  it.each([
    ['NotAllowedError', /Camera permission denied/],
    ['NotFoundError', /No camera found/],
  ])('maps %s to a friendly message', async (name, text) => {
    setupEnvironment({ cameraError: Object.assign(new Error('x'), { name }) })
    render(<CVMonitorModal open onClose={() => {}} onCVResults={() => {}} />)
    expect(await screen.findByText(text)).toBeInTheDocument()
    expect(screen.getByText('CV Monitor Unavailable')).toBeInTheDocument()
  })

  it.each(['Failed to fetch', 'Network request failed'])('explains a model download failure ("%s")', async (msg) => {
    setupEnvironment({ modelError: new Error(msg) })
    render(<CVMonitorModal open onClose={() => {}} onCVResults={() => {}} />)
    expect(await screen.findByText(/internet connection required for first use/)).toBeInTheDocument()
  })

  it('passes through any other error message', async () => {
    setupEnvironment({ modelError: new Error('GPU delegate exploded') })
    render(<CVMonitorModal open onClose={() => {}} onCVResults={() => {}} />)
    expect(await screen.findByText('GPU delegate exploded')).toBeInTheDocument()
  })

  it('uses a generic message for an error with no text', async () => {
    setupEnvironment({ modelError: {} })
    render(<CVMonitorModal open onClose={() => {}} onCVResults={() => {}} />)
    expect(await screen.findByText(/Unknown error starting the CV monitor/)).toBeInTheDocument()
  })

  it('Retry starts the whole sequence again', async () => {
    setupEnvironment({ cameraError: Object.assign(new Error('x'), { name: 'NotAllowedError' }) })
    render(<CVMonitorModal open onClose={() => {}} onCVResults={() => {}} />)
    await screen.findByText(/Camera permission denied/)
    const before = mp.FaceLandmarker.createFromOptions.mock.calls.length
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    await waitFor(() => expect(mp.FaceLandmarker.createFromOptions.mock.calls.length).toBe(before + 1))
  })
})

// ── Close / cleanup ───────────────────────────────────────────────────────────

describe('closing', () => {
  it('Escape and the ✕ button both call onClose', async () => {
    const onClose = vi.fn()
    await openAndStart({ onClose })
    fireEvent.keyDown(window, { key: 'Escape' })
    fireEvent.click(screen.getAllByRole('button').find(b => b.querySelector('svg') && !b.textContent))
    expect(onClose).toHaveBeenCalledTimes(2)
  })

  it('the backdrop closes it too', async () => {
    const onClose = vi.fn()
    const { container } = await openAndStart({ onClose })
    fireEvent.click(container.ownerDocument.querySelector('div[style*="rgba(4, 10, 20"]'))
    expect(onClose).toHaveBeenCalled()
  })

  it('stops the camera, closes the model and clears live readings on unmount', async () => {
    const { unmount, onCVResults } = await openAndStart()
    onCVResults.mockClear()
    unmount()
    expect(trackStop).toHaveBeenCalled()
    expect(landmarker.close).toHaveBeenCalled()
    expect(onCVResults).toHaveBeenCalledWith(null)
  })

  it('tolerates the model refusing to close', async () => {
    const { unmount } = await openAndStart()
    landmarker.close.mockImplementation(() => { throw new Error('already closed') })
    expect(() => unmount()).not.toThrow()
  })

  it('does not start the camera if closed while the model is still loading', async () => {
    let release
    mp.FaceLandmarker.createFromOptions.mockReturnValue(new Promise(r => { release = () => r(landmarker) }))
    const { rerender } = render(<CVMonitorModal open onClose={() => {}} onCVResults={() => {}} />)
    await screen.findByText('Loading AI Model')
    rerender(<CVMonitorModal open={false} onClose={() => {}} onCVResults={() => {}} />)
    await act(async () => { release() })
    expect(navigator.mediaDevices.getUserMedia).not.toHaveBeenCalled()
    expect(landmarker.close).toHaveBeenCalled()
  })

  it('releases a camera stream that arrives after the monitor was closed', async () => {
    let grant
    navigator.mediaDevices.getUserMedia = vi.fn(() => new Promise(r => { grant = () => r(stream) }))
    const { rerender } = render(<CVMonitorModal open onClose={() => {}} onCVResults={() => {}} />)
    await screen.findByText('Starting Camera')
    rerender(<CVMonitorModal open={false} onClose={() => {}} onCVResults={() => {}} />)
    await act(async () => { grant() })
    expect(trackStop).toHaveBeenCalled()
  })
})

// ── Calibration and readings ──────────────────────────────────────────────────

describe('calibration', () => {
  it('shows progress while a face is being calibrated', async () => {
    await openAndStart()
    await frames(20)
    expect(screen.getByText(/% complete/)).toBeInTheDocument()
    expect(screen.getByText(/Stress Score and GCS Eye Score will appear once calibration is complete/)).toBeInTheDocument()
    expect(screen.getByText(/Hold still and look at the camera/)).toBeInTheDocument()
  })

  it('asks the user to centre their face when none is detected', async () => {
    landmarker.detectForVideo.mockReturnValue(NO_FACE)
    await openAndStart()
    await frames(5)
    expect(screen.getByText(/Centre your face in the camera to begin calibrating/)).toBeInTheDocument()
  })

  it('does not send anything to the dashboard during calibration', async () => {
    const { onCVResults } = await openAndStart()
    onCVResults.mockClear()
    await frames(60)
    expect(onCVResults).not.toHaveBeenCalled()
  })

  it('skips frames that carry the same video timestamp', async () => {
    await openAndStart()
    await nextFrame()
    const calls = landmarker.detectForVideo.mock.calls.length
    clockMs += 10                                            // time passes, video does not advance
    const cb = rafQueue.shift(); await act(async () => { cb() })
    expect(landmarker.detectForVideo.mock.calls.length).toBe(calls)
  })

  it('survives a frame where detection throws', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    await openAndStart()
    landmarker.detectForVideo.mockImplementationOnce(() => { throw new Error('bad frame') })
    await frames(2)
    expect(warn).toHaveBeenCalled()
    await frames(3)                                          // loop is still alive
    expect(landmarker.detectForVideo.mock.calls.length).toBeGreaterThan(2)
  })

  it('treats a camera that is not ready as "wait for the next frame"', async () => {
    await openAndStart()
    Object.defineProperty(HTMLMediaElement.prototype, 'readyState', { configurable: true, get: () => 1 })
    const before = landmarker.detectForVideo.mock.calls.length
    await frames(3)
    expect(landmarker.detectForVideo.mock.calls.length).toBe(before)
  })
})

async function calibrate(extra = {}) {
  // 90 clearly-open, non-blinking frames complete both the face and the eye calibration
  landmarker.detectForVideo.mockReturnValue(frame(extra))
  await frames(95)
}

describe('after calibration', () => {
  it('shows the stress and GCS cards', async () => {
    await openAndStart()
    await calibrate()
    // each card title is split across a <span>, so match on the element's full text
    const titled = (re) => screen.getAllByText((_, el) => el?.tagName === 'P' && re.test(el.textContent)).length
    expect(titled(/^Stress Score\s*\(PSPI\)$/)).toBe(1)
    expect(titled(/^GCS Eye Score\s*\(Rolling 30s\)$/)).toBe(1)
    expect(screen.getByText('Brow Furrow')).toBeInTheDocument()
    expect(screen.getByText(/\(personal\)/)).toBeInTheDocument()
  })

  it('sends nothing until the 10 s observation window is full', async () => {
    const { onCVResults } = await openAndStart()
    onCVResults.mockClear()
    await calibrate()
    await frames(60)                                         // ~2 s more
    expect(onCVResults).not.toHaveBeenCalled()
    expect(screen.getByText(/Provisional — observing/)).toBeInTheDocument()
  })

  it('then sends a reading with stress and GCS eye', async () => {
    const { onCVResults } = await openAndStart()
    onCVResults.mockClear()
    await calibrate()
    await frames(330)                                        // > 10 s of video
    const real = onCVResults.mock.calls.map(c => c[0]).filter(Boolean)
    expect(real.length).toBeGreaterThan(0)
    expect(real.at(-1)).toEqual({ stress: expect.any(Number), gcs_eye: 4 })
  })

  it('open eyes give E4 "Spontaneous"', async () => {
    await openAndStart()
    await calibrate()
    await frames(330)
    expect(screen.getAllByText('E4').length).toBeGreaterThan(0)       // big readout + the dot scale
    expect(screen.getAllByText('Spontaneous').length).toBeGreaterThan(0)
  })

  it('closed eyes give E1 "No Response" and that is what is sent', async () => {
    const { onCVResults } = await openAndStart()
    await calibrate()
    onCVResults.mockClear()
    landmarker.detectForVideo.mockReturnValue(frame({
      ear: 0.05, blendshapes: { eyeBlinkLeft: 0.9, eyeBlinkRight: 0.9 },
    }))
    // 0.5 s per frame: 90 frames = 45 s of video, enough to push every open frame out of the 30 s window
    await frames(90, 500)
    const last = onCVResults.mock.calls.map(c => c[0]).filter(Boolean).at(-1)
    expect(last.gcs_eye).toBe(1)
    expect(screen.getAllByText('E1').length).toBeGreaterThan(0)
    expect(screen.getAllByText('No Response').length).toBeGreaterThan(0)
  }, 20000)

  it('a strained face raises the stress score and the component bars', async () => {
    const { onCVResults } = await openAndStart()
    await calibrate()
    onCVResults.mockClear()
    landmarker.detectForVideo.mockReturnValue(frame({
      blendshapes: { eyeBlinkLeft: 0.05, eyeBlinkRight: 0.05, browDownLeft: 1, browDownRight: 1,
        cheekSquintLeft: 1, cheekSquintRight: 1, noseSneerLeft: 1, noseSneerRight: 1 },
    }))
    await frames(400)
    const last = onCVResults.mock.calls.map(c => c[0]).filter(Boolean).at(-1)
    expect(last.stress).toBeGreaterThan(3.5)
    expect(screen.getAllByText(/MODERATE|HIGH/).length).toBeGreaterThan(0)
  })

  it('does not resend an unchanged reading', async () => {
    const { onCVResults } = await openAndStart()
    await calibrate()
    await frames(330)
    const n = onCVResults.mock.calls.filter(c => c[0]).length
    await frames(150)                                        // another ~5 s of identical frames
    expect(onCVResults.mock.calls.filter(c => c[0]).length - n).toBeLessThanOrEqual(2)
  })

  it('falls back to default thresholds when the eyes never open enough to calibrate', async () => {
    await openAndStart()
    landmarker.detectForVideo.mockReturnValue(frame({ ear: 0.05, blendshapes: { eyeBlinkLeft: 0.9, eyeBlinkRight: 0.9 } }))
    await frames(320)                                        // reaches the 300-face-frame cap
    expect(screen.getByText(/\(default\)/)).toBeInTheDocument()
  })
})

describe('face lost', () => {
  it('shows the no-face warning once calibrated', async () => {
    await openAndStart()
    await calibrate()
    landmarker.detectForVideo.mockReturnValue(NO_FACE)
    await frames(3)
    expect(screen.getByText(/No face detected — centre your face in the camera/)).toBeInTheDocument()
  })

  it('stops reporting readings after 3 s without a face', async () => {
    const { onCVResults } = await openAndStart()
    await calibrate()
    await frames(330)
    expect(onCVResults.mock.calls.some(c => c[0])).toBe(true)
    onCVResults.mockClear()
    landmarker.detectForVideo.mockReturnValue(NO_FACE)
    await frames(60, 100)                                    // 6 s of no face
    expect(onCVResults).toHaveBeenCalledWith(null)
    expect(screen.getByText(/Provisional — observing/)).toBeInTheDocument()
  })

  it('does not clear readings that were never sent', async () => {
    const { onCVResults } = await openAndStart()
    landmarker.detectForVideo.mockReturnValue(NO_FACE)
    onCVResults.mockClear()
    await frames(5, 100)
    await frames(60, 100)
    expect(onCVResults).not.toHaveBeenCalledWith(null)
  })
})

// ── Right-hand live panel ─────────────────────────────────────────────────────

describe('live panel wiring', () => {
  const predictData = {
    sofa_score: 4.2, risk_level: 'Low Risk', risk_color: '#28a745', is_alert: false,
    vitals: { HR: 80, RR: 16, SpO2: 98, Temp: 37, SBP: 120, DBP: 80, MAP: 90, GCS_eye: 4, stress: 1 },
    vitals_status: {}, clinical_note: 'stable',
  }

  it('renders the panel next to the camera and passes the props through', async () => {
    await openAndStart({ predictData, cvLiveData: { stress: 1, gcs_eye: 4 }, countdown: 42, isRefreshing: false })
    expect(screen.getByLabelText('Live model input and output')).toBeInTheDocument()
    expect(screen.getByText('4.2')).toBeInTheDocument()
    expect(screen.getByText('0:42')).toBeInTheDocument()
    expect(screen.getAllByText('IN MODEL ✓')).toHaveLength(2)
  })

  it('shows the waiting message when no prediction exists yet', async () => {
    await openAndStart()
    expect(screen.getByText(/Waiting for the first SOFA prediction/)).toBeInTheDocument()
  })
})
