import { render, screen, fireEvent } from '@testing-library/react'
import { MemoryRouter, Routes, Route } from 'react-router-dom'
import { ThemeProvider } from '@/contexts/ThemeContext.jsx'
import HeaderBar from '@/components/layout/HeaderBar.jsx'

const PATIENT = { id: 1, name: 'Patient 1 — Low Risk' }
const PD = { risk_color: '#dc3545', risk_level: 'High Risk' }

const mount = (props = {}) => render(
  <ThemeProvider>
    <MemoryRouter initialEntries={['/monitor/1']}>
      <Routes>
        <Route path="/monitor/1" element={
          <HeaderBar patient={PATIENT} predictData={PD} isRefreshing={false}
                     lastUpdated={new Date('2026-11-24T10:20:30')} countdown={42} error={null} {...props} />} />
        <Route path="/" element={<div data-testid="home" />} />
      </Routes>
    </MemoryRouter>
  </ThemeProvider>
)

describe('live chip', () => {
  it('shows LIVE and the countdown', () => {
    mount()
    expect(screen.getByText('LIVE')).toBeInTheDocument()
    expect(screen.getByText('42s')).toBeInTheDocument()
  })

  it('shows dots instead of the countdown while refreshing', () => {
    mount({ isRefreshing: true })
    expect(screen.getByText('···')).toBeInTheDocument()
    expect(screen.queryByText('42s')).not.toBeInTheDocument()
  })

  it('shows "Connection lost" when there is an error and no refresh in flight', () => {
    mount({ error: 'Refresh failed' })
    expect(screen.getByText('Connection lost')).toBeInTheDocument()
    expect(screen.queryByText('LIVE')).not.toBeInTheDocument()
  })

  it('keeps showing LIVE while a retry is in flight, even with an error', () => {
    mount({ error: 'Refresh failed', isRefreshing: true })
    expect(screen.getByText('LIVE')).toBeInTheDocument()
    expect(screen.queryByText('Connection lost')).not.toBeInTheDocument()
  })
})

describe('patient pill', () => {
  it('shows the patient name, initial and risk badge', () => {
    mount()
    expect(screen.getByText('Patient 1 — Low Risk')).toBeInTheDocument()
    expect(screen.getByText('P')).toBeInTheDocument()
    expect(screen.getByText('High Risk')).toBeInTheDocument()
  })

  it('uses an explicit initial when given', () => {
    mount({ patient: { ...PATIENT, initial: 'Z' } })
    expect(screen.getByText('Z')).toBeInTheDocument()
  })

  it('falls back to "P" when the name is empty', () => {
    mount({ patient: { id: 1, name: '' } })
    expect(screen.getByText('P')).toBeInTheDocument()
  })

  it('shows age and gender when present', () => {
    mount({ patient: { ...PATIENT, age: 64, gender: 'F' } })
    expect(screen.getByText('64y · F')).toBeInTheDocument()
  })

  it('shows age without gender', () => {
    mount({ patient: { ...PATIENT, age: 64 } })
    expect(screen.getByText('64y ·')).toBeInTheDocument()
  })

  it('has no pill before the patient loads', () => {
    mount({ patient: undefined })
    expect(screen.queryByText('Patient 1 — Low Risk')).not.toBeInTheDocument()
  })

  it('has no risk badge before a prediction exists', () => {
    mount({ predictData: undefined })
    expect(screen.queryByText('High Risk')).not.toBeInTheDocument()
  })
})

describe('right side', () => {
  it('shows the last-updated time when known', () => {
    mount()
    expect(screen.getByText('Updated')).toBeInTheDocument()
  })

  it('hides it before the first update', () => {
    mount({ lastUpdated: null })
    expect(screen.queryByText('Updated')).not.toBeInTheDocument()
  })

  it('Exit goes back to the patient list', () => {
    mount()
    fireEvent.click(screen.getByRole('button', { name: /Exit/ }))
    expect(screen.getByTestId('home')).toBeInTheDocument()
  })

  it('Exit highlights on hover and restores on leave', () => {
    mount()
    const btn = screen.getByRole('button', { name: /Exit/ })
    fireEvent.mouseEnter(btn)
    expect(btn.style.borderColor).toBe('var(--blue-l)')
    expect(btn.style.boxShadow).not.toBe('none')
    fireEvent.mouseLeave(btn)
    expect(btn.style.borderColor).toBe('var(--b2)')
    expect(btn.style.boxShadow).toBe('none')
  })
})

describe('accent strip', () => {
  const strip = (c) => c.querySelector('header > div[style*="height: 2px"]')

  it('uses the risk colour and a stronger glow for a non-low risk', () => {
    const { container } = mount()
    expect(strip(container).style.opacity).toBe('0.9')
    expect(container.querySelector('header').style.borderBottom).toMatch(/rgba\(220, 53, 69/)   // risk colour, as the browser reports it
  })

  it('stays neutral for Low Risk', () => {
    const { container } = mount({ predictData: { risk_color: '#28a745', risk_level: 'Low Risk' } })
    expect(strip(container).style.opacity).toBe('0.55')
    expect(container.querySelector('header').style.borderBottom).toContain('var(--b1)')
  })

  it('stays neutral with no prediction yet', () => {
    const { container } = mount({ predictData: undefined })
    expect(strip(container).style.opacity).toBe('0.55')
  })
})
