import {
  calcEAR, pspiFromBlendshapes, updateStress, eyeState, gcsFromWindow, stressLabel,
} from '@/components/cv/CVMonitorModal.jsx'

const pair = (name, v) => ({ [`${name}Left`]: v, [`${name}Right`]: v })

describe('PSPI (Prkachin–Solomon) from blendshapes', () => {
  it('is 0 for a neutral face', () => {
    expect(pspiFromBlendshapes({})).toBe(0)
  })
  it('sums AU4 + max(AU6,AU7) + max(AU9,AU10) + AU43, each AU 0–5', () => {
    const bs = { ...pair('browDown', 0.4), ...pair('cheekSquint', 0.2), ...pair('eyeSquint', 0.6),
                 ...pair('noseSneer', 0.1), ...pair('mouthUpperUp', 0.3), ...pair('eyeBlink', 0.9) }
    // 2.0 + max(1,3) + max(0.5,1.5) + 1 = 7.5
    expect(pspiFromBlendshapes(bs)).toBeCloseTo(7.5)
  })
  it('reaches the 16-point maximum', () => {
    const bs = { ...pair('browDown', 1), ...pair('eyeSquint', 1), ...pair('noseSneer', 1), ...pair('eyeBlink', 1) }
    expect(pspiFromBlendshapes(bs)).toBeCloseTo(16)
  })
  it('a full smile cancels the score', () => {
    const bs = { ...pair('browDown', 1), ...pair('eyeSquint', 1), ...pair('mouthSmile', 1) }
    expect(pspiFromBlendshapes(bs)).toBeCloseTo(0)
  })
})

describe('stress score (0–10)', () => {
  it('first frame seeds the EMA directly', () => {
    expect(updateStress(16, 0, null).display).toBeCloseTo(8)   // 10 × 0.8 facial
  })
  it('maximal pain and agitation give 10', () => {
    expect(updateStress(16, 1, null).display).toBeCloseTo(10)
  })
  it('smooths with α = 0.15', () => {
    expect(updateStress(16, 0, 0).display).toBeCloseTo(1.2)    // 0.15 × 8
  })
  it('bands: LOW < 3.5 ≤ MODERATE < 6.5 ≤ HIGH', () => {
    expect(stressLabel(3.4)).toBe('LOW')
    expect(stressLabel(3.5)).toBe('MODERATE')
    expect(stressLabel(6.4)).toBe('MODERATE')
    expect(stressLabel(6.5)).toBe('HIGH')
  })
})

describe('eye aspect ratio', () => {
  it('is height / width for an open eye', () => {
    const lm = []
    // p0..p5 at the six eye indices: width 1, vertical gaps 0.3
    const pts = [[0, 0], [0.3, -0.15], [0.7, -0.15], [1, 0], [0.7, 0.15], [0.3, 0.15]]
    ;[33, 160, 158, 133, 153, 144].forEach((idx, i) => { lm[idx] = { x: pts[i][0], y: pts[i][1] } })
    expect(calcEAR(lm, [33, 160, 158, 133, 153, 144], 1, 1)).toBeCloseTo(0.3, 3)
  })
})

describe('per-frame eye state', () => {
  it('blink blendshape overrides EAR', () => {
    expect(eyeState(0.4, 0.9, 0.22, 0.12)).toBe('closed')
  })
  it('classifies by thresholds', () => {
    expect(eyeState(0.25, 0.1, 0.22, 0.12)).toBe('open')
    expect(eyeState(0.15, 0.1, 0.22, 0.12)).toBe('partial')
    expect(eyeState(0.05, 0.1, 0.22, 0.12)).toBe('closed')
  })
})

describe('GCS eye score from the 30 s window', () => {
  const win = (states, fps = 30, browDown = 0) =>
    states.map((state, i) => ({ ts: i / fps, state, browDown }))
  const repeat = (s, n) => Array(n).fill(s)
  // 100 frames: `open` frames then `closed` frames, repeated `times`
  const flicker = (times) =>
    Array.from({ length: times }, () => [...repeat('open', 6), ...repeat('closed', 24)]).flat()

  it('E4 when eyes are completely open (blinks allowed)', () => {
    expect(gcsFromWindow(win(repeat('open', 100))).score).toBe(4)
    expect(gcsFromWindow(win([...repeat('open', 80), ...repeat('closed', 20)])).score).toBe(4)
  })
  it('E3 when eyes are completely closed with a calm face', () => {
    expect(gcsFromWindow(win(repeat('closed', 100), 30, 0.05)).score).toBe(3)
  })
  it('E2 when the eyes flicker open repeatedly', () => {
    expect(gcsFromWindow(win(flicker(4), 30, 0.05)).score).toBe(2)
  })
  it('E1 when the eyes are closed and the eyebrows are narrowed', () => {
    expect(gcsFromWindow(win(repeat('closed', 100), 30, 0.5)).score).toBe(1)
  })
  it('E1 appears right after an earlier flicker once the eyes shut with narrowed brows', () => {
    const frames = [...win(flicker(3), 30, 0.05), ...win(repeat('closed', 150), 30, 0.2).map(f => ({ ...f, ts: f.ts + 3 }))]
    expect(gcsFromWindow(frames).score).toBe(1)
  })
  it('a slight brow lowering (0.15) while eyes are shut is enough for E1', () => {
    expect(gcsFromWindow(win(repeat('closed', 100), 30, 0.15)).score).toBe(1)
  })
  it('one or two openings are not a flicker', () => {
    expect(gcsFromWindow(win(flicker(2), 30, 0.05)).score).toBe(3)
  })
  it('single-frame twitches are not openings', () => {
    const frames = repeat('closed', 100)
    ;[20, 50, 80].forEach(i => { frames[i] = 'open' })
    expect(gcsFromWindow(win(frames, 30, 0.5)).score).toBe(1)
  })
})
