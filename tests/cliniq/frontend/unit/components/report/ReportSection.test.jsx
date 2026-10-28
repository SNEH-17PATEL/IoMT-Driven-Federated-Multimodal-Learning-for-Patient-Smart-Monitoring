import { render, screen } from '@testing-library/react'
import ReportSection from '@/components/report/ReportSection.jsx'

// Component maps 'CURRENT CONDITION' → 'Current Condition' via DISPLAY_NAME
describe('ReportSection', () => {
  it('renders section name as title-case display name', () => {
    render(<ReportSection name="CURRENT CONDITION" content="Patient is stable." subtitle="Status" color="#60a5fa" />)
    expect(screen.getByText('Current Condition')).toBeInTheDocument()
  })

  it('renders content text', () => {
    render(<ReportSection name="RISK FORECAST" content="Risk is moderate." subtitle="Outlook" color="#e6a817" />)
    expect(screen.getByText('Risk is moderate.')).toBeInTheDocument()
  })

  it('renders subtitle', () => {
    render(<ReportSection name="PROBABLE CAUSE" content="Sepsis suspected." subtitle="Contributing factors" color="#9b59b6" />)
    expect(screen.getByText('Contributing factors')).toBeInTheDocument()
  })

  it('renders without crash when content is empty', () => {
    render(<ReportSection name="IMMEDIATE ACTIONS" content="" subtitle="Next steps" color="#dc3545" />)
    expect(screen.getByText('Immediate Actions')).toBeInTheDocument()
  })

  it('renders with no color prop', () => {
    render(<ReportSection name="TEST" content="Content." subtitle="Sub" />)
    // Unknown name falls back to itself via ?? name
    expect(screen.getByText('TEST')).toBeInTheDocument()
  })
})
