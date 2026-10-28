import { render, screen } from '@testing-library/react'
import AlertBanner from '@/components/layout/AlertBanner.jsx'

const mkData = (risk_level, sofa_score = 8.0) => ({ risk_level, sofa_score })

describe('AlertBanner', () => {
  it('returns null when predictData is null', () => {
    const { container } = render(<AlertBanner predictData={null} />)
    expect(container.firstChild).toBeNull()
  })

  it('returns null when risk_level is Low Risk', () => {
    const { container } = render(<AlertBanner predictData={mkData('Low Risk', 2.5)} />)
    expect(container.firstChild).toBeNull()
  })

  it('renders an alert for Moderate Risk', () => {
    render(<AlertBanner predictData={mkData('Moderate Risk', 6.5)} />)
    expect(screen.getByRole('alert')).toBeInTheDocument()
  })

  it('renders an alert for High Risk', () => {
    render(<AlertBanner predictData={mkData('High Risk', 12.0)} />)
    expect(screen.getByRole('alert')).toBeInTheDocument()
  })

  it('shows SOFA score in the alert message', () => {
    render(<AlertBanner predictData={mkData('High Risk', 12.3)} />)
    expect(screen.getByText(/12\.3/)).toBeInTheDocument()
  })

  it('shows ELEVATED RISK label for Moderate Risk', () => {
    render(<AlertBanner predictData={mkData('Moderate Risk', 7.0)} />)
    expect(screen.getByText(/ELEVATED RISK/i)).toBeInTheDocument()
  })

  it('shows CRITICAL ALERT label for High Risk', () => {
    render(<AlertBanner predictData={mkData('High Risk', 14.0)} />)
    expect(screen.getByText(/CRITICAL ALERT/i)).toBeInTheDocument()
  })

  it('has aria-live attribute on alert', () => {
    render(<AlertBanner predictData={mkData('High Risk', 10.0)} />)
    const el = screen.getByRole('alert')
    expect(el).toHaveAttribute('aria-live', 'assertive')
  })

  it('moderate risk uses polite aria-live', () => {
    render(<AlertBanner predictData={mkData('Moderate Risk', 6.0)} />)
    expect(screen.getByRole('alert')).toHaveAttribute('aria-live', 'polite')
  })
})
