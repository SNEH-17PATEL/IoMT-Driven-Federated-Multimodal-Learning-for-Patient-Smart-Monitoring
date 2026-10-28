import { render, screen } from '@testing-library/react'
import VitalCard from '@/components/overview/VitalCard.jsx'

// VitalCard requires an Icon component (SVG or function returning SVG)
const HeartIcon = () => <svg aria-hidden="true" />

const defaultProps = {
  Icon: HeartIcon,
  label: 'Heart Rate',
  value: 72,
  unit: 'bpm',
  status: 'normal',
  normalLo: 60,
  normalHi: 100,
}

describe('VitalCard', () => {
  it('renders vital label', () => {
    render(<VitalCard {...defaultProps} />)
    expect(screen.getByText('Heart Rate')).toBeInTheDocument()
  })

  it('renders vital value', () => {
    render(<VitalCard {...defaultProps} />)
    expect(screen.getByText('72')).toBeInTheDocument()
  })

  it('renders unit string', () => {
    render(<VitalCard {...defaultProps} />)
    expect(screen.getByText('bpm')).toBeInTheDocument()
  })

  it('renders without crash when value is null', () => {
    render(<VitalCard {...defaultProps} value={null} />)
    expect(screen.getByText('Heart Rate')).toBeInTheDocument()
  })

  it('renders without crash for SpO2 label', () => {
    render(<VitalCard {...defaultProps} label="SpO₂" unit="%" value={97} />)
    expect(screen.getByText('97')).toBeInTheDocument()
  })

  it('renders with high status', () => {
    render(<VitalCard {...defaultProps} status="high" value={130} />)
    expect(screen.getByText('Heart Rate')).toBeInTheDocument()
  })

  it('renders with low status', () => {
    render(<VitalCard {...defaultProps} status="low" value={45} />)
    expect(screen.getByText('Heart Rate')).toBeInTheDocument()
  })

  it('renders trend section with value present', () => {
    const { container } = render(<VitalCard {...defaultProps} />)
    expect(container.firstChild).not.toBeNull()
  })

  it('shows "Normal" status label for status="normal"', () => {
    render(<VitalCard {...defaultProps} status="normal" />)
    expect(screen.getByText('Normal')).toBeInTheDocument()
  })

  it('shows "Above Normal" status label for status="high"', () => {
    render(<VitalCard {...defaultProps} status="high" value={130} />)
    expect(screen.getByText('Above Normal')).toBeInTheDocument()
  })

  it('shows "Below Normal" status label for status="low"', () => {
    render(<VitalCard {...defaultProps} status="low" value={45} />)
    expect(screen.getByText('Below Normal')).toBeInTheDocument()
  })

  it('renders "—" display value when value is null', () => {
    render(<VitalCard {...defaultProps} value={null} />)
    expect(screen.getByText('—')).toBeInTheDocument()
  })

  it('renders up-arrow SVG path for status="high"', () => {
    const { container } = render(<VitalCard {...defaultProps} status="high" value={130} />)
    // TrendArrow for high renders path d starting with M6 10V2 (up arrow)
    const upPath = container.querySelector('path[d*="M6 10V2"]')
    expect(upPath).toBeTruthy()
  })

  it('renders down-arrow SVG path for status="low"', () => {
    const { container } = render(<VitalCard {...defaultProps} status="low" value={45} />)
    // TrendArrow for low renders path d starting with M6 2v8 (down arrow)
    const downPath = container.querySelector('path[d*="M6 2v8"]')
    expect(downPath).toBeTruthy()
  })
})
