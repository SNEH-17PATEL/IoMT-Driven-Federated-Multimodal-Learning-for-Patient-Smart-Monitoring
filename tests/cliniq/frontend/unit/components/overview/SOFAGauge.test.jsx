import { render, screen } from '@testing-library/react'
import SOFAGauge from '@/components/overview/SOFAGauge.jsx'

// Component expects { predictData: {...}, isRefreshing }
const defaultPredictData = {
  sofa_score: 6.5,
  risk_level: 'Moderate Risk',
  risk_color: '#e6a817',
  conformal_lo: 4.0,
  conformal_hi: 9.0,
  is_alert: false,
}
const defaultProps = { predictData: defaultPredictData, isRefreshing: false }

describe('SOFAGauge', () => {
  it('renders without crashing', () => {
    render(<SOFAGauge {...defaultProps} />)
    expect(document.body).toBeTruthy()
  })

  it('displays SOFA score', () => {
    render(<SOFAGauge {...defaultProps} />)
    expect(screen.getByText(/6\.5|6,5/)).toBeInTheDocument()
  })

  it('displays risk level label', () => {
    render(<SOFAGauge {...defaultProps} />)
    expect(screen.getByText(/Moderate Risk/i)).toBeInTheDocument()
  })

  it('shows SVG arc element', () => {
    const { container } = render(<SOFAGauge {...defaultProps} />)
    expect(container.querySelector('svg')).toBeInTheDocument()
  })

  it('handles score=0 without crash', () => {
    render(<SOFAGauge predictData={{ ...defaultPredictData, sofa_score: 0, risk_level: 'Low Risk', risk_color: '#28a745' }} isRefreshing={false} />)
    expect(document.body).toBeTruthy()
  })

  it('handles score=24 (max) without crash', () => {
    render(<SOFAGauge predictData={{ ...defaultPredictData, sofa_score: 24, risk_level: 'High Risk', risk_color: '#dc3545' }} isRefreshing={false} />)
    expect(screen.getByText(/High Risk/i)).toBeInTheDocument()
  })

  it('shows conformal interval values', () => {
    render(<SOFAGauge {...defaultProps} />)
    expect(screen.getByText(/4\.0|4,0/)).toBeInTheDocument()
    expect(screen.getByText(/9\.0|9,0/)).toBeInTheDocument()
  })

  it('renders 3 SVG zone marker lines for SOFA thresholds 6, 9, 12', () => {
    const { container } = render(<SOFAGauge {...defaultProps} />)
    // ArcGauge renders <line> elements at threshold values 6, 9, 12
    const lines = container.querySelectorAll('svg line')
    expect(lines.length).toBe(3)
  })
})
