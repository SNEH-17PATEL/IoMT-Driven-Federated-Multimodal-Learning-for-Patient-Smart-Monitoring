import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { ThemeProvider } from '@/contexts/ThemeContext.jsx'
import HeaderBar from '@/components/layout/HeaderBar.jsx'

const patient = { id: 1, name: 'Patient 1 — Low Risk', short: 'Low Risk' }
const predictData = { risk_color: '#28a745', risk_level: 'Low Risk', sofa_score: 3.0 }

const renderHeader = (props = {}) =>
  render(
    <ThemeProvider>
      <MemoryRouter>
        <HeaderBar
          patient={patient}
          predictData={predictData}
          isRefreshing={false}
          lastUpdated={null}
          countdown={75}
          error={null}
          {...props}
        />
      </MemoryRouter>
    </ThemeProvider>
  )

describe('HeaderBar', () => {
  it('renders without crashing', () => {
    renderHeader()
    expect(document.querySelector('header')).toBeInTheDocument()
  })

  it('shows LIVE indicator', () => {
    renderHeader()
    expect(screen.getByText('LIVE')).toBeInTheDocument()
  })

  it('shows countdown in seconds', () => {
    renderHeader({ countdown: 42 })
    expect(screen.getByText('42s')).toBeInTheDocument()
  })

  it('shows "···" while refreshing', () => {
    renderHeader({ isRefreshing: true })
    expect(screen.getByText('···')).toBeInTheDocument()
  })

  it('shows patient name in patient pill', () => {
    renderHeader()
    expect(screen.getByText('Patient 1 — Low Risk')).toBeInTheDocument()
  })

  it('shows Exit button', () => {
    renderHeader()
    expect(screen.getByText('Exit')).toBeInTheDocument()
  })

  it('shows ClinIQ brand mark', () => {
    renderHeader()
    expect(screen.getByText('ClinIQ')).toBeInTheDocument()
  })

  it('shows formatted lastUpdated time when provided', () => {
    const date = new Date('2026-01-01T14:30:00')
    renderHeader({ lastUpdated: date })
    expect(screen.getByText('Updated')).toBeInTheDocument()
  })

  it('shows connection lost state on error', () => {
    renderHeader({ error: 'Failed', isRefreshing: false })
    expect(screen.getByText(/Connection lost/i)).toBeInTheDocument()
  })

  it('shows All Systems Operational when no error', () => {
    renderHeader()
    expect(screen.getByText(/All Systems Operational/i)).toBeInTheDocument()
  })

  it('does not show timestamp section when lastUpdated is null', () => {
    renderHeader({ lastUpdated: null })
    // The "Updated" label is only rendered when lastUpdated is truthy
    expect(screen.queryByText('Updated')).toBeNull()
  })

  it('renders ThemeToggle as an accessible button', () => {
    renderHeader()
    // ThemeToggle renders a button with aria-label mentioning "Switch to"
    expect(screen.getByRole('button', { name: /Switch to/i })).toBeInTheDocument()
  })
})
