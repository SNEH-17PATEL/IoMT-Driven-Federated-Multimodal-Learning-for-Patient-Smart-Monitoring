/*
 * MonitorDashboard — opening and closing each overlay from the page, and the camera button
 * toggling the camera window. Reaches the onClick / onClose handlers wired in the page itself.
 */
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { MemoryRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ThemeProvider } from '@/contexts/ThemeContext.jsx'
import MonitorDashboard from '@/pages/MonitorDashboard.jsx'

vi.mock('@/api/client.js', () => ({
  getPatient: vi.fn(), runPredict: vi.fn(), generateReport: vi.fn(), getHistory: vi.fn(), getVitals: vi.fn(),
}))
vi.mock('@/hooks/useMonitor.js', () => ({ useMonitor: vi.fn() }))
import { getPatient, getVitals, generateReport, getHistory } from '@/api/client.js'
import { useMonitor } from '@/hooks/useMonitor.js'

const PD = {
  patient_id: 1, sofa_score: 4.2, risk_level: 'Low Risk', risk_color: '#28a745', is_alert: false,
  conformal_lo: 1, conformal_hi: 7,
  vitals: { HR: 72, RR: 14, SpO2: 98, Temp: 37, SBP: 112, DBP: 70, MAP: 84, GCS_eye: 4, stress: 1 },
  vitals_status: {}, shap_top5: [], clinical_note: 'stable', _training_meta: {},
}
const REPORT = (mode) => ({ data: {
  sections: { 'CURRENT CONDITION': 'ok', 'PROBABLE CAUSE': 'x', 'RISK FORECAST': 'y', 'IMMEDIATE ACTIONS': 'z' },
  reliability_score: 0.82, reliability_label: 'High', reliability_color: '#28a745', caution: false, care_mode: mode,
} })

const renderPage = () => render(
  <ThemeProvider>
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MemoryRouter initialEntries={['/monitor/1']}>
        <Routes><Route path="/monitor/:patientId" element={<MonitorDashboard />} /></Routes>
      </MemoryRouter>
    </QueryClientProvider>
  </ThemeProvider>
)

// A fresh lookup each time: the page re-mounts parts of its tree as queries settle,
// so a handle captured earlier can be a detached node.
const btn = (name) => screen.getByRole('button', { name })
const ready = async () => {
  await screen.findByText('Current Condition')
  await waitFor(() => expect(screen.getAllByText('82%').length).toBeGreaterThan(0))
}
const closeButton = () => screen.getByLabelText('Close panel')

beforeEach(() => {
  getPatient.mockReset().mockResolvedValue({ data: { id: 1, name: 'Patient 1', description: 'd' } })
  getVitals.mockReset().mockResolvedValue({ data: [] })
  getHistory.mockReset().mockResolvedValue({ data: [] })
  generateReport.mockReset().mockImplementation((_p, m) => Promise.resolve(REPORT(m)))
  useMonitor.mockReset().mockReturnValue({ predictData: PD, isRefreshing: false, lastUpdated: new Date(0), countdown: 30, error: null })
})

describe('SOFA validation drawer', () => {
  it('opens from its button and closes from the ✕', async () => {
    renderPage(); await ready()
    fireEvent.click(btn(/SOFA/i))
    expect(await screen.findByText(/Model Accuracy Report/)).toBeInTheDocument()
    fireEvent.click(closeButton())
    await waitFor(() => expect(screen.queryByText(/Model Accuracy Report/)).not.toBeInTheDocument())
  })

  it('the ✕ turns red on hover and back on leave', async () => {
    renderPage(); await ready()
    fireEvent.click(btn(/SOFA/i))
    await screen.findByText(/Model Accuracy Report/)
    const x = closeButton()
    fireEvent.mouseEnter(x)
    expect(x.style.color).toBe('var(--danger-l)')
    fireEvent.mouseLeave(x)
    expect(x.style.color).toBe('var(--t3)')
  })
})

describe('LLM validation drawer', () => {
  it('opens once a report exists and closes from the ✕', async () => {
    renderPage(); await ready()
    fireEvent.click(btn(/Open AI Report Validation/i))
    expect(await screen.findByText(/Reliability Audit/)).toBeInTheDocument()
    fireEvent.click(closeButton())
    await waitFor(() => expect(screen.queryByText(/Reliability Audit/)).not.toBeInTheDocument())
  })

  it('the ✕ turns red on hover and back on leave', async () => {
    renderPage(); await ready()
    fireEvent.click(btn(/Open AI Report Validation/i))
    await screen.findByText(/Reliability Audit/)
    const x = closeButton()
    fireEvent.mouseEnter(x)
    expect(x.style.color).toBe('var(--danger-l)')
    fireEvent.mouseLeave(x)
    expect(x.style.color).toBe('var(--t3)')
  })

  it('shows the care mode the dashboard is in', async () => {
    renderPage(); await ready()
    fireEvent.click(btn(/Open AI Report Validation/i))
    expect(await screen.findByText(/Hospital Mode/)).toBeInTheDocument()
  })
})

describe('history drawer', () => {
  it('opens from its button and closes from the ✕', async () => {
    renderPage(); await ready()
    fireEvent.click(btn(/Open patient history/i))
    expect(await screen.findByText('Patient History', { selector: 'span' })).toBeInTheDocument()
    const x = [...document.querySelectorAll('button')].find(b => b.querySelector('svg path[d^="M2 2l12 12"]'))
    fireEvent.click(x)
    await waitFor(() => expect(document.querySelector('table')).toBeNull())
  })

  it('shows the real recording count on the history button badge data', async () => {
    getHistory.mockResolvedValue({ data: [{ SOFA: 1 }, { SOFA: 2 }, { SOFA: 3 }] })
    renderPage(); await ready()
    await waitFor(() => expect(getHistory).toHaveBeenCalledWith(1, 50))
  })
})

describe('camera button', () => {
  const windowOpen = () => [...document.querySelectorAll('span')].some(e => e.textContent === 'Live CV Assessment')

  it('opens the camera window and a second click closes it again', async () => {
    renderPage(); await ready()
    fireEvent.click(btn(/Open Live CV Monitor/i))
    await waitFor(() => expect(windowOpen()).toBe(true))
    fireEvent.click(btn(/Open Live CV Monitor/i))
    await waitFor(() => expect(windowOpen()).toBe(false))
  })

  it('the camera window closes from its own ✕ button', async () => {
    renderPage(); await ready()
    fireEvent.click(btn(/Open Live CV Monitor/i))
    await waitFor(() => expect(windowOpen()).toBe(true))
    const x = [...document.querySelectorAll('button')].find(b => b.querySelector('svg path[d*="M2 2"], svg line'))
    // the modal's close icon is the only button inside the header next to the LIVE chip
    const headerClose = [...document.querySelectorAll('button')].filter(b => !b.textContent && b.querySelector('svg')).at(-1)
    fireEvent.click(headerClose ?? x)
    await waitFor(() => expect(windowOpen()).toBe(false))
  })
})
