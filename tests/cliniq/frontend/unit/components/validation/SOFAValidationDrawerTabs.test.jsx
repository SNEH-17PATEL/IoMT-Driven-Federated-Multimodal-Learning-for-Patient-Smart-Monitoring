/*
 * Tabs 2 and 3 of the SOFA validation drawer (Runtime Checks and Uncertainty), plus the
 * drawer chrome (tab switching, Escape key, backdrop). Tab 1 (Population Stats) is covered in
 * SOFAValidationDrawer.test.jsx.
 */
import { render, screen, fireEvent } from '@testing-library/react'
import SOFAValidationDrawer from '@/components/validation/SOFAValidationDrawer.jsx'

const base = {
  patient_id: 1,
  sofa_score: 3.0,
  vitals: { HR: 80, SBP: 120, SpO2: 98, MAP: 90, GCS_eye: 4 },
  conformal_lo: 0.0, conformal_hi: 6.3, conformal_q: 3.28,
  _training_meta: { train_samples: 41798, final_mae: 1.8006 },
}

const open = (overrides = {}, props = {}) => {
  const onClose = props.onClose ?? vi.fn()
  const utils = render(
    <SOFAValidationDrawer open onClose={onClose} predictData={{ ...base, ...overrides }} {...props} />
  )
  return { ...utils, onClose }
}
const goTo = (name) => fireEvent.click(screen.getByRole('button', { name }))

describe('drawer chrome', () => {
  it('shows patient and SOFA in the header', () => {
    open()
    expect(screen.getByText(/Patient 1 · SOFA 3\.0/)).toBeInTheDocument()
  })

  it('shows dashes when there is no prediction yet', () => {
    render(<SOFAValidationDrawer open onClose={() => {}} predictData={null} />)
    expect(screen.getByText(/Patient — · SOFA —/)).toBeInTheDocument()
  })

  it('starts on Population Stats', () => {
    open()
    expect(screen.getByText(/Held-out test set/)).toBeInTheDocument()
  })

  it('has exactly three tabs', () => {
    open()
    for (const n of ['Population Stats', 'Runtime Checks', 'Uncertainty'])
      expect(screen.getByRole('button', { name: n })).toBeInTheDocument()
  })

  it('closes on Escape', () => {
    const { onClose } = open()
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('ignores other keys', () => {
    const { onClose } = open()
    fireEvent.keyDown(window, { key: 'Enter' })
    expect(onClose).not.toHaveBeenCalled()
  })

  it('does not listen for Escape while closed', () => {
    const onClose = vi.fn()
    render(<SOFAValidationDrawer open={false} onClose={onClose} predictData={base} />)
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(onClose).not.toHaveBeenCalled()
  })

  it('closes from the ✕ button', () => {
    const { onClose } = open()
    fireEvent.click(screen.getByLabelText('Close panel'))
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('returns to the first tab when reopened', () => {
    const onClose = vi.fn()
    const { rerender } = render(<SOFAValidationDrawer open onClose={onClose} predictData={base} />)
    goTo('Uncertainty')
    expect(screen.getByText(/90% conformal prediction interval/)).toBeInTheDocument()
    rerender(<SOFAValidationDrawer open={false} onClose={onClose} predictData={base} />)
    rerender(<SOFAValidationDrawer open onClose={onClose} predictData={base} />)
    expect(screen.getByText(/Held-out test set/)).toBeInTheDocument()
  })
})

describe('Runtime Checks tab — lower-bound check', () => {
  it('passes with normal vitals and explains the floor is 0', () => {
    open()
    goTo('Runtime Checks')
    expect(screen.getByText('Consistent')).toBeInTheDocument()
    expect(screen.getByText(/physiological floor = 0/)).toBeInTheDocument()
  })

  it('lists every floor contribution for badly abnormal vitals', () => {
    open({ vitals: { ...base.vitals, MAP: 60, SpO2: 85, GCS_eye: 1 }, sofa_score: 9 })
    goTo('Runtime Checks')
    const t = document.body.textContent
    expect(t).toMatch(/MAP 60 mmHg → cardiovascular floor \+1/)
    expect(t).toMatch(/MAP < 65 → cardiovascular floor \+2 total/)
    expect(t).toMatch(/SpO₂ 85% → respiratory floor \+1/)
    expect(t).toMatch(/SpO₂ < 90% → respiratory floor \+2 total/)
    expect(t).toMatch(/GCS Eye 1 → CNS floor \+3/)
    expect(t).toMatch(/Computed floor = 7/)
    expect(screen.getByText('Consistent')).toBeInTheDocument()   // 9 >= 7
  })

  it('shows GCS eye 2 contribution', () => {
    open({ vitals: { ...base.vitals, GCS_eye: 2 }, sofa_score: 3 })
    goTo('Runtime Checks')
    expect(document.body.textContent).toMatch(/GCS Eye 2 → CNS floor \+2/)
  })

  it('flags a prediction that is below the physiological floor', () => {
    open({ vitals: { ...base.vitals, MAP: 60, SpO2: 85 }, sofa_score: 1.0 })
    goTo('Runtime Checks')
    expect(screen.getByText('Under-predicted')).toBeInTheDocument()
    expect(document.body.textContent).toMatch(/Predicted 1\.0 < floor/)
  })

  it('falls back to safe defaults when vitals are missing', () => {
    open({ vitals: undefined })
    goTo('Runtime Checks')
    expect(screen.getByText('Consistent')).toBeInTheDocument()
  })
})

describe('Runtime Checks tab — vital consistency', () => {
  it('reports no flags for normal vitals', () => {
    open()
    goTo('Runtime Checks')
    expect(screen.getByText('No flags')).toBeInTheDocument()
    expect(document.body.textContent).toMatch(/below shock threshold/)
  })

  it('flags the shock pattern (HR > 130, SBP < 90, SOFA < 4)', () => {
    open({ vitals: { ...base.vitals, HR: 140, SBP: 80 }, sofa_score: 2 })
    goTo('Runtime Checks')
    expect(screen.getByText('Pattern detected')).toBeInTheDocument()
    expect(document.body.textContent).toMatch(/shock pattern detected/)
  })

  it('flags multi-organ stress (SpO2 < 88, MAP < 65, SOFA < 6)', () => {
    open({ vitals: { ...base.vitals, SpO2: 85, MAP: 60 }, sofa_score: 5 })
    goTo('Runtime Checks')
    expect(screen.getByText('Pattern detected')).toBeInTheDocument()
    expect(document.body.textContent).toMatch(/multi-organ stress/)
  })

  it('does not flag when SOFA is already high enough', () => {
    open({ vitals: { ...base.vitals, HR: 140, SBP: 80 }, sofa_score: 8 })
    goTo('Runtime Checks')
    expect(screen.getByText('No flags')).toBeInTheDocument()
  })
})

describe('Runtime Checks tab — trajectory', () => {
  it('says "First reading" when there is no previous prediction', () => {
    open({ _prev_sofa: undefined })
    goTo('Runtime Checks')
    expect(screen.getByText('First reading')).toBeInTheDocument()
    expect(document.body.textContent).toMatch(/trajectory check will activate/)
  })

  it('shows an upward change', () => {
    open({ _prev_sofa: 2.0, sofa_score: 3.0, _traj_flagged: false })
    goTo('Runtime Checks')
    expect(screen.getByText('↑ 1.0 pts')).toBeInTheDocument()
    expect(document.body.textContent).toMatch(/Previous: 2\.0 → Current: 3\.0 \(\+1\.0 pts\)/)
  })

  it('shows a downward change', () => {
    open({ _prev_sofa: 5.0, sofa_score: 3.0, _traj_flagged: false })
    goTo('Runtime Checks')
    expect(screen.getByText('↓ 2.0 pts')).toBeInTheDocument()
    expect(document.body.textContent).toMatch(/\(-2\.0 pts\)/)
  })

  it('asks the user to verify input when the backend flagged a jump', () => {
    open({ _prev_sofa: 2.0, sofa_score: 9.0, _traj_flagged: true })
    goTo('Runtime Checks')
    expect(screen.getByText('Verify input')).toBeInTheDocument()
    expect(document.body.textContent).toMatch(/SOFA jumped 7\.0 pts \(2\.0 → 9\.0\)/)
  })
})

describe('Uncertainty tab', () => {
  it('shows the interval, predicted SOFA and margin', () => {
    open()
    goTo('Uncertainty')
    expect(document.body.textContent).toMatch(/P\(true SOFA ∈ \[0\.0, 6\.3\]\) ≥ 90%/)
    expect(screen.getByText('± 3.3')).toBeInTheDocument()
    expect(screen.getByText('Predicted SOFA')).toBeInTheDocument()
    expect(document.body.textContent).toMatch(/41,798 training stays/)
  })

  it('compares against the MAE band using the real MAE', () => {
    open()
    goTo('Uncertainty')
    expect(document.body.textContent).toMatch(/±MAE band \(±1\.80\)/)
  })

  it('draws the scale 0 to 24', () => {
    open()
    goTo('Uncertainty')
    for (const n of ['0', '6', '12', '18', '24'])
      expect(screen.getAllByText(n).length).toBeGreaterThan(0)
  })

  it('falls back to ±2.9 around the prediction when the backend sent no interval', () => {
    open({ conformal_lo: undefined, conformal_hi: undefined, conformal_q: undefined, sofa_score: 10 })
    goTo('Uncertainty')
    expect(document.body.textContent).toMatch(/\[7\.1, 12\.9\]/)
    expect(screen.getByText('± 2.9')).toBeInTheDocument()
  })

  it('clamps the fallback interval to 0 and 24', () => {
    open({ conformal_lo: undefined, conformal_hi: undefined, sofa_score: 1 })
    goTo('Uncertainty')
    expect(document.body.textContent).toMatch(/\[0\.0, 3\.9\]/)
    cleanup_()
    open({ conformal_lo: undefined, conformal_hi: undefined, sofa_score: 23 })
    goTo('Uncertainty')
    expect(document.body.textContent).toMatch(/\[20\.1, 24\.0\]/)
  })

  it('handles a missing prediction', () => {
    render(<SOFAValidationDrawer open onClose={() => {}} predictData={null} />)
    goTo('Uncertainty')
    expect(screen.getByText('Predicted SOFA')).toBeInTheDocument()
  })
})

function cleanup_() {
  document.body.innerHTML = ''
}
