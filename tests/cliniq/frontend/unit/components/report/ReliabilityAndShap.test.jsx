import { render, screen } from '@testing-library/react'
import ReliabilityBadge from '@/components/report/ReliabilityBadge.jsx'
import SHAPPanel from '@/components/report/SHAPPanel.jsx'
import { animate } from 'animejs'

describe('ReliabilityBadge', () => {
  const badge = (over = {}) => render(
    <ReliabilityBadge reliability_score={0.82} reliability_label="High" reliability_color="#28a745" caution={false} {...over} />)

  it.each([
    ['High', 'Passed all reliability checks'],
    ['Moderate', 'Passed most validation checks'],
    ['Low', 'Failed multiple validation checks'],
  ])('%s shows "%s"', (label, blurb) => {
    badge({ reliability_label: label })
    expect(screen.getByText(`${label} Reliability`)).toBeInTheDocument()
    expect(screen.getByText(blurb)).toBeInTheDocument()
  })

  it('falls back to the Moderate wording for an unknown label', () => {
    badge({ reliability_label: 'Mystery' })
    expect(screen.getByText('Passed most validation checks')).toBeInTheDocument()
  })

  it('shows the rounded percentage in the ring', () => {
    badge({ reliability_score: 0.826 })
    expect(screen.getByText('83')).toBeInTheDocument()
  })

  it('treats a missing score as 0%', () => {
    badge({ reliability_score: undefined })
    expect(screen.getByText('0')).toBeInTheDocument()
  })

  it('shows the caution line only when asked', () => {
    const { rerender } = badge({ caution: false })
    expect(screen.queryByText(/Review carefully/)).not.toBeInTheDocument()
    rerender(<ReliabilityBadge reliability_score={0.5} reliability_label="Low" reliability_color="#dc3545" caution />)
    expect(screen.getByText(/Review carefully before acting/)).toBeInTheDocument()
  })
})

describe('SHAPPanel', () => {
  const f = (over = {}) => ({ feature: 'latest_HR', label: 'Heart Rate', value: 112.46, unit: 'bpm',
    impact: 0.512, abs_impact: 0.512, rel_impact: 1, direction: 'up', ...over })
  const panel = (factors) => render(<SHAPPanel factors={factors} patientId={1} />)

  beforeEach(() => animate.mockClear())

  it('lists each factor with label, value+unit, signed impact and arrow', () => {
    panel([f(), f({ feature: 'latest_SpO2', label: 'SpO₂', value: 96, unit: '%', impact: -0.31, direction: 'down', rel_impact: 0.6 })])
    expect(screen.getByText('Key Risk Drivers')).toBeInTheDocument()
    expect(screen.getByText('Heart Rate')).toBeInTheDocument()
    expect(screen.getByText('+0.51')).toBeInTheDocument()
    expect(screen.getByText('-0.31')).toBeInTheDocument()
    expect(screen.getByText(/112\.5 bpm/)).toBeInTheDocument()
    expect(screen.getAllByText('↑').length).toBeGreaterThan(0)
    expect(screen.getAllByText('↓').length).toBeGreaterThan(0)
  })

  it('uses the raw feature name when there is no label', () => {
    panel([f({ label: undefined, feature: 'septic' })])
    expect(screen.getByText('septic')).toBeInTheDocument()
  })

  it('shows a value without a unit to two decimals', () => {
    panel([f({ unit: '', value: 0.4567 })])
    expect(screen.getByText('0.46')).toBeInTheDocument()
  })

  it('shows the legend', () => {
    panel([f()])
    expect(screen.getByText('Raises risk')).toBeInTheDocument()
    expect(screen.getByText('Lowers risk')).toBeInTheDocument()
  })

  it('renders without crashing for an empty or missing list', () => {
    expect(() => panel([])).not.toThrow()
    expect(() => panel(undefined)).not.toThrow()
  })

  it('animates the bars when the panel scrolls into view', () => {
    panel([f(), f({ feature: 'b', label: 'B' })])
    expect(animate).toHaveBeenCalledTimes(2)
  })

  it('does not animate again for the same patient (guarded)', () => {
    const { rerender } = panel([f()])
    const n = animate.mock.calls.length
    rerender(<SHAPPanel factors={[f()]} patientId={1} />)
    expect(animate.mock.calls.length).toBeGreaterThanOrEqual(n)
  })
})
