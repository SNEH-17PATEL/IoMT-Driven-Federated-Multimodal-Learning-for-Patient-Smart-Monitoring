/*
 * CVLivePanel — the vertical "model input -> output" panel shown beside the camera feed.
 * It exists to prove that camera values reach the model, so most tests check the
 * "live from camera" vs "used in last SOFA" comparison and the timer.
 */
import { render, screen, within } from '@testing-library/react'
import CVLivePanel from '@/components/cv/CVLivePanel.jsx'

const predict = (over = {}) => ({
  sofa_score: 6.4, risk_level: 'Moderate Risk', risk_color: '#e6a817', is_alert: false,
  conformal_lo: 3.1, conformal_hi: 9.7,
  vitals: { HR: 92, RR: 18, SpO2: 96, Temp: 37.2, SBP: 118, DBP: 74, MAP: 88, GCS_eye: 4, stress: 2 },
  vitals_status: { HR: 'normal', RR: 'normal', SpO2: 'normal', Temp: 'normal', SBP: 'normal', DBP: 'normal', MAP: 'high' },
  clinical_note: 'septic shock, intubated, on vasopressor',
  ...over,
})
const panel = (props = {}) =>
  render(<CVLivePanel predictData={predict()} cvLiveData={null} isRefreshing={false} countdown={90} {...props} />)

describe('before the first prediction', () => {
  it('shows a waiting message and no scores', () => {
    panel({ predictData: null })
    expect(screen.getByText(/Waiting for the first SOFA prediction/)).toBeInTheDocument()
    expect(screen.queryByText(/Predicted SOFA/)).not.toBeInTheDocument()
  })
})

describe('output — predicted SOFA', () => {
  it('shows score, risk level and the 90% range', () => {
    panel()
    expect(screen.getByText('6.4')).toBeInTheDocument()
    expect(screen.getByText('MODERATE RISK')).toBeInTheDocument()
    expect(screen.getByText(/90% range 3\.1 – 9\.7/)).toBeInTheDocument()
  })

  it('shows ALERT when the alert flag is set', () => {
    panel({ predictData: predict({ is_alert: true, sofa_score: 8.2 }) })
    expect(screen.getByText(/ALERT/)).toBeInTheDocument()
  })

  it('shows no ALERT when the flag is off', () => {
    panel()
    expect(screen.queryByText(/ALERT/)).not.toBeInTheDocument()
  })

  it('omits the range line when the backend sent no interval', () => {
    panel({ predictData: predict({ conformal_lo: undefined, conformal_hi: undefined }) })
    expect(screen.queryByText(/90% range/)).not.toBeInTheDocument()
  })
})

describe('timer and the source of the next prediction', () => {
  it('formats the countdown as m:ss', () => {
    panel({ countdown: 90 })
    expect(screen.getByText('1:30')).toBeInTheDocument()
  })

  it('pads seconds', () => {
    panel({ countdown: 5 })
    expect(screen.getByText('0:05')).toBeInTheDocument()
  })

  it('shows a refreshing state instead of the countdown', () => {
    panel({ isRefreshing: true })
    expect(screen.getByText('Predicting now')).toBeInTheDocument()
    expect(screen.queryByText('1:30')).not.toBeInTheDocument()
  })

  it('says DATABASE values will be used when the camera has no live reading', () => {
    panel({ cvLiveData: null })
    expect(screen.getByText(/At 0 → DATABASE values go into the model/)).toBeInTheDocument()
    expect(screen.getByText(/No live camera reading yet/)).toBeInTheDocument()
  })

  it('says CAMERA values will be used once a live reading exists', () => {
    panel({ cvLiveData: { stress: 5, gcs_eye: 3 } })
    expect(screen.getByText(/At 0 → CAMERA values go into the model/)).toBeInTheDocument()
    expect(screen.getByText(/Camera is calibrated and sees a face/)).toBeInTheDocument()
  })

  // The timer runs 90 -> 1 (it holds at 1 while predicting), so the bar spans 89 steps: (90 - c) / 89.
  it.each([[90, 0], [45, 51], [1, 100]])('progress bar at countdown %i is about %i%%', (countdown, pct) => {
    const { container } = panel({ countdown })
    const bar = container.querySelector('div[style*="transition: width 1s linear"]')
    expect(Math.round(parseFloat(bar.style.width))).toBe(pct)
  })

  it('clamps the progress bar for out-of-range countdowns', () => {
    const { container, rerender } = panel({ countdown: 500 })
    const width = () => parseFloat(container.querySelector('div[style*="width 1s linear"]').style.width)
    expect(width()).toBe(0)
    rerender(<CVLivePanel predictData={predict()} cvLiveData={null} isRefreshing={false} countdown={-20} />)
    expect(width()).toBe(100)
  })
})

describe('inputs 1 & 2 — camera features', () => {
  const rowFor = (title) => screen.getByText(title).closest('div[style*="border-top"]')

  it('shows WAITING FOR CAMERA and dashes when there is no live data', () => {
    panel()
    expect(screen.getAllByText('WAITING FOR CAMERA')).toHaveLength(2)
    const row = rowFor('Stress score (0–10)')
    expect(within(row).getByText('—')).toBeInTheDocument()       // nothing live
    expect(within(row).getByText('2.0')).toBeInTheDocument()     // value the last SOFA used (dataset)
  })

  it('shows IN MODEL ✓ when live values equal what the last SOFA used', () => {
    panel({ cvLiveData: { stress: 2, gcs_eye: 4 } })
    expect(screen.getAllByText('IN MODEL ✓')).toHaveLength(2)
  })

  it('shows SENDS NEXT CYCLE when the camera differs from the last SOFA input', () => {
    panel({ cvLiveData: { stress: 7, gcs_eye: 2 } })
    expect(screen.getAllByText('SENDS NEXT CYCLE')).toHaveLength(2)
  })

  it('decides each feature independently', () => {
    panel({ cvLiveData: { stress: 2, gcs_eye: 1 } })          // stress matches, gcs does not
    expect(screen.getAllByText('IN MODEL ✓')).toHaveLength(1)
    expect(screen.getAllByText('SENDS NEXT CYCLE')).toHaveLength(1)
  })

  it('formats GCS as E<n> and stress to one decimal', () => {
    panel({ cvLiveData: { stress: 7.25, gcs_eye: 3 } })
    expect(screen.getAllByText('E3').length).toBeGreaterThan(0)
    expect(screen.getByText('7.3')).toBeInTheDocument()
  })

  it('describes the live reading in words', () => {
    panel({ cvLiveData: { stress: 8, gcs_eye: 1 } })
    expect(screen.getByText(/HIGH stress, No Response eyes/)).toBeInTheDocument()
    panel({ cvLiveData: { stress: 5, gcs_eye: 3 } })
    expect(screen.getByText(/MODERATE stress, To Voice eyes/)).toBeInTheDocument()
  })

  it.each([[1, 'No Response'], [2, 'To Pain'], [3, 'To Voice'], [4, 'Spontaneous']])(
    'GCS %i is described as %s', (n, word) => {
      panel({ cvLiveData: { stress: 1, gcs_eye: n } })
      expect(screen.getByText(new RegExp(`${word} eyes`))).toBeInTheDocument()
    })

  it('explains the fallback when the camera is off', () => {
    panel()
    expect(screen.getByText(/stored values from the patient record are used/)).toBeInTheDocument()
  })

  it('shows dashes when the last SOFA has no camera fields', () => {
    panel({ predictData: predict({ vitals: { HR: 80 } }) })
    expect(screen.getAllByText('—').length).toBeGreaterThanOrEqual(4)
  })
})

describe('input 3 — clinical note', () => {
  it('shows the note text', () => {
    panel()
    expect(screen.getByText('septic shock, intubated, on vasopressor')).toBeInTheDocument()
    expect(screen.getByText(/90 TF-IDF features/)).toBeInTheDocument()
  })

  it.each([undefined, '', '   ', 'No clinical notes provided'])('shows the empty message for %j', (note) => {
    panel({ predictData: predict({ clinical_note: note }) })
    expect(screen.getByText('No clinical notes on record for this reading.')).toBeInTheDocument()
  })
})

describe('other inputs — vitals', () => {
  it('lists the seven vitals with units', () => {
    panel()
    for (const n of ['HR', 'RR', 'SpO₂', 'Temp', 'SBP', 'DBP', 'MAP'])
      expect(screen.getByText(n)).toBeInTheDocument()
    expect(screen.getByText('92')).toBeInTheDocument()
    expect(screen.getByText('37.2')).toBeInTheDocument()
  })

  it('shows a dash for a missing vital', () => {
    panel({ predictData: predict({ vitals: { HR: 92 } }) })
    expect(screen.getAllByText('—').length).toBeGreaterThan(0)
  })
})
