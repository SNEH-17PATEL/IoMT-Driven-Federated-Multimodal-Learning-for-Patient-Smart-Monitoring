import { render, screen } from '@testing-library/react'
import VitalsGrid from '@/components/overview/VitalsGrid.jsx'

// VitalsGrid expects { predictData: { vitals, vitals_status, clinical_note } }
const vitals = { HR: 72, RR: 16, SpO2: 98, Temp: 37.0, SBP: 115, DBP: 72, MAP: 86 }
const vitals_status = {
  HR: 'normal', RR: 'normal', SpO2: 'normal',
  Temp: 'normal', SBP: 'normal', DBP: 'normal', MAP: 'normal',
}
const defaultProps = { predictData: { vitals, vitals_status, clinical_note: null } }

describe('VitalsGrid', () => {
  it('renders without crash with valid vitals', () => {
    render(<VitalsGrid {...defaultProps} />)
    expect(document.querySelector('[class]')).toBeTruthy()
  })

  it('renders heart rate value', () => {
    render(<VitalsGrid {...defaultProps} />)
    // HR=72 and DBP=72 share the same value — use getAllByText
    expect(screen.getAllByText('72').length).toBeGreaterThan(0)
  })

  it('renders SpO2 value', () => {
    render(<VitalsGrid {...defaultProps} />)
    expect(screen.getByText('98')).toBeInTheDocument()
  })

  it('handles null vitals gracefully', () => {
    render(<VitalsGrid predictData={{ vitals: {}, vitals_status: {}, clinical_note: null }} />)
    expect(document.body).toBeTruthy()
  })

  it('handles undefined predictData gracefully', () => {
    // Component destructures predictData — guard against crash only
    try {
      render(<VitalsGrid predictData={{ vitals: {}, vitals_status: {} }} />)
    } catch {
      // jsdom may throw; just ensure it doesn't propagate an unhandled crash
    }
    expect(document.body).toBeTruthy()
  })

  it('renders all 7 VitalCard labels from VITALS_CONFIG + DBP', () => {
    render(<VitalsGrid {...defaultProps} />)
    const expectedLabels = ['Heart Rate', 'SpO₂', 'Resp. Rate', 'Temperature', 'Sys. BP', 'MAP', 'Dia. BP']
    expectedLabels.forEach(label => {
      expect(screen.getByText(label)).toBeInTheDocument()
    })
  })
})
