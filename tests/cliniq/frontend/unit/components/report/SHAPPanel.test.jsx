import { render, screen } from '@testing-library/react'
import SHAPPanel from '@/components/report/SHAPPanel.jsx'

const factors = [
  { feature: 'Heart Rate',        direction: 'up',   rel_impact: 0.85 },
  { feature: 'SpO₂',             direction: 'down',  rel_impact: 0.72 },
  { feature: 'Systolic BP',      direction: 'down',  rel_impact: 0.60 },
  { feature: 'Respiratory Rate', direction: 'up',   rel_impact: 0.45 },
  { feature: 'Temperature',      direction: 'up',   rel_impact: 0.30 },
]

describe('SHAPPanel', () => {
  it('renders when factors are provided', () => {
    const { container } = render(<SHAPPanel factors={factors} />)
    expect(container.firstChild).not.toBeNull()
  })

  it('does not render when factors is empty', () => {
    const { container } = render(<SHAPPanel factors={[]} />)
    // Panel stays hidden or empty when no factors
    expect(container.textContent).not.toMatch(/Heart Rate/)
  })

  it('does not render when factors is null', () => {
    const { container } = render(<SHAPPanel factors={null} />)
    expect(container.textContent).not.toMatch(/Heart Rate/)
  })

  it('shows feature names', () => {
    render(<SHAPPanel factors={factors} />)
    expect(screen.getByText('Heart Rate')).toBeInTheDocument()
    expect(screen.getByText('SpO₂')).toBeInTheDocument()
  })

  it('preserves last factors when new empty array arrives', () => {
    const { rerender } = render(<SHAPPanel factors={factors} />)
    expect(screen.getByText('Heart Rate')).toBeInTheDocument()
    rerender(<SHAPPanel factors={[]} />)
    // Should still show previous factors
    expect(screen.getByText('Heart Rate')).toBeInTheDocument()
  })

  it('renders up to 5 factors', () => {
    render(<SHAPPanel factors={factors} />)
    const featureTexts = ['Heart Rate', 'SpO₂', 'Systolic BP', 'Respiratory Rate', 'Temperature']
    featureTexts.forEach(f => expect(screen.getByText(f)).toBeInTheDocument())
  })

  it('renders ↑ direction indicator text for "up" factors', () => {
    render(<SHAPPanel factors={factors} />)
    // Heart Rate, Respiratory Rate, Temperature have direction='up' → rendered as '↑' text
    expect(screen.getAllByText('↑').length).toBeGreaterThan(0)
  })

  it('renders ↓ direction indicator text for "down" factors', () => {
    render(<SHAPPanel factors={factors} />)
    // SpO₂ and Systolic BP have direction='down' → rendered as '↓' text
    expect(screen.getAllByText('↓').length).toBeGreaterThan(0)
  })

  it('renders impact bar elements with data-target-width attribute', () => {
    const { container } = render(<SHAPPanel factors={factors} />)
    // Each factor row has a bar div with data-target-width set to the computed %
    const bars = container.querySelectorAll('[data-target-width]')
    expect(bars.length).toBeGreaterThan(0)
  })
})
