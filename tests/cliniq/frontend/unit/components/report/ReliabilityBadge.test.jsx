import { render, screen } from '@testing-library/react'
import ReliabilityBadge from '@/components/report/ReliabilityBadge.jsx'

describe('ReliabilityBadge', () => {
  it('renders the percentage score', () => {
    render(<ReliabilityBadge reliability_score={0.85} reliability_label="High" reliability_color="#28a745" caution={null} />)
    expect(screen.getByText('85')).toBeInTheDocument()
  })

  it('renders "High" label when reliability_label is High', () => {
    render(<ReliabilityBadge reliability_score={0.85} reliability_label="High" reliability_color="#28a745" caution={null} />)
    expect(screen.getByText(/High/i)).toBeInTheDocument()
  })

  it('renders "Moderate" label', () => {
    render(<ReliabilityBadge reliability_score={0.60} reliability_label="Moderate" reliability_color="#e6a817" caution={null} />)
    expect(screen.getByText(/Moderate/i)).toBeInTheDocument()
  })

  it('renders "Low" label', () => {
    render(<ReliabilityBadge reliability_score={0.30} reliability_label="Low" reliability_color="#dc3545" caution={null} />)
    expect(screen.getByText(/Low/i)).toBeInTheDocument()
  })

  it('shows caution block when caution prop is truthy', () => {
    // Component shows a static caution message when caution prop is truthy,
    // not the prop value itself
    render(<ReliabilityBadge reliability_score={0.55} reliability_label="Moderate" reliability_color="#e6a817" caution="Vitals out of training range." />)
    expect(screen.getByText(/Review carefully/i)).toBeInTheDocument()
  })

  it('does not show caution block when caution is null', () => {
    render(<ReliabilityBadge reliability_score={0.88} reliability_label="High" reliability_color="#28a745" caution={null} />)
    expect(screen.queryByText(/caution/i)).toBeNull()
  })

  it('renders score ring SVG', () => {
    const { container } = render(<ReliabilityBadge reliability_score={0.70} reliability_label="High" reliability_color="#28a745" caution={null} />)
    expect(container.querySelector('svg')).toBeInTheDocument()
  })

  it('shows AI Clinical Assessment header', () => {
    render(<ReliabilityBadge reliability_score={0.80} reliability_label="High" reliability_color="#28a745" caution={null} />)
    expect(screen.getByText(/AI Clinical Assessment/i)).toBeInTheDocument()
  })
})
