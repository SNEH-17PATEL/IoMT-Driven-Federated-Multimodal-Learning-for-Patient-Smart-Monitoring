/*
 * MonitorDashboard page — the states the main integration test does not reach:
 * first load (skeleton), refresh error, refreshing dim state, missing patient / report data,
 * the patient-context strip, switching care mode, and the live camera hand-off.
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

const PATIENT = { id: 2, name: 'Patient 2 — Moderate', description: 'Pneumonia, day 3', risk_level: 'Moderate Risk' }
const PREDICT = (over = {}) => ({
  patient_id: 2, sofa_score: 6.4, risk_level: 'Moderate Risk', risk_color: '#e6a817', risk_icon: '🟡',
  is_alert: false, conformal_lo: 3, conformal_hi: 9.7,
  vitals: { HR: 92, RR: 18, SpO2: 96, Temp: 37.2, SBP: 118, DBP: 74, MAP: 88, GCS_eye: 4, stress: 2 },
  vitals_status: { HR: 'normal', RR: 'normal', SpO2: 'normal', Temp: 'normal', SBP: 'normal', DBP: 'normal', MAP: 'normal' },
  shap_top5: [], clinical_note: 'stable', ...over,
})
const REPORT = {
  sections: { 'CURRENT CONDITION': 'ok', 'PROBABLE CAUSE': 'x', 'RISK FORECAST': 'y', 'IMMEDIATE ACTIONS': 'z' },
  reliability_score: 0.82, reliability_label: 'High', reliability_color: '#28a745', caution: null, judge_used: true,
  care_mode: 'hospital',
}

// Build the hook result ONCE per call. The real hook returns the same predictData object between
// renders; returning a fresh object every render would make effects that depend on it re-run forever.
const monitor = (over = {}) => {
  const value = { predictData: PREDICT(), isRefreshing: false, lastUpdated: new Date(), countdown: 60, error: null, ...over }
  useMonitor.mockReturnValue(value)
}

const renderPage = (id = '2') => render(
  <ThemeProvider>
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MemoryRouter initialEntries={[`/monitor/${id}`]}>
        <Routes>
          <Route path="/monitor/:patientId" element={<MonitorDashboard />} />
          <Route path="/" element={<div data-testid="home" />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  </ThemeProvider>
)

beforeEach(() => {
  getPatient.mockReset().mockResolvedValue({ data: PATIENT })
  getVitals.mockReset().mockResolvedValue({ data: [] })
  getHistory.mockReset().mockResolvedValue({ data: [] })
  generateReport.mockReset().mockImplementation((_p, mode) => Promise.resolve({ data: { ...REPORT, care_mode: mode } }))
  monitor()
})

describe('first load', () => {
  it('shows the loading skeleton and no readings before the first prediction', () => {
    monitor({ predictData: null })
    const { container } = renderPage()
    expect(container.querySelectorAll('.skeleton').length).toBeGreaterThan(10)
    expect(screen.queryByText('Current Status')).not.toBeInTheDocument()
    expect(screen.queryByText('AI Clinical Report')).not.toBeInTheDocument()
  })

  it('hides the floating buttons until there is a prediction', () => {
    monitor({ predictData: null })
    renderPage()
    expect(screen.queryByRole('button', { name: /Open patient history/i })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Open Live CV Monitor/i })).not.toBeInTheDocument()
  })

  it('still shows the vital-trends section while the prediction is loading', () => {
    monitor({ predictData: null })
    renderPage()
    expect(screen.getByText('Vital Trends')).toBeInTheDocument()
  })
})

describe('refresh states', () => {
  it('shows the refresh error banner and keeps the last reading', () => {
    monitor({ error: 'Refresh failed (boom). Retrying in 90s.' })
    renderPage()
    expect(screen.getByText(/Refresh failed \(boom\)/)).toBeInTheDocument()
    expect(screen.getByText('Current Status')).toBeInTheDocument()
  })

  it('dims the status area while refreshing', () => {
    monitor({ isRefreshing: true })
    const { container } = renderPage()
    const grid = container.querySelector('div.grid[style*="opacity"]')
    expect(grid.style.opacity).toBe('0.72')
  })

  it('does not dim it when idle', () => {
    const { container } = renderPage()
    expect(container.querySelector('div.grid[style*="opacity"]').style.opacity).toBe('1')
  })
})

describe('patient context strip', () => {
  it('shows patient identity, risk, SOFA and a dash for reliability before the report loads', async () => {
    generateReport.mockReturnValue(new Promise(() => {}))
    renderPage()
    expect((await screen.findAllByText('Patient 2 — Moderate')).length).toBeGreaterThan(0)
    expect(screen.getByText('Pneumonia, day 3')).toBeInTheDocument()
    expect(screen.getByText('02')).toBeInTheDocument()
    expect(screen.getAllByText('Moderate Risk').length).toBeGreaterThan(0)
    expect(screen.getByText('AI Reliability').nextSibling).toHaveTextContent('—')
  })

  it('shows the report reliability as a percentage once the report arrives', async () => {
    renderPage()
    await waitFor(() => expect(screen.getAllByText('82%').length).toBeGreaterThan(0))
  })

  it('is not drawn until the patient record has loaded', async () => {
    getPatient.mockReturnValue(new Promise(() => {}))
    renderPage()
    await screen.findByText('Current Status')
    expect(screen.queryByText('AI Reliability')).not.toBeInTheDocument()
  })

  it('shows a dash instead of a score when the backend gave none', async () => {
    monitor({ predictData: PREDICT({ sofa_score: undefined }) })
    renderPage()
    await screen.findByText('SOFA Score')
    expect(screen.getByText('SOFA Score').nextSibling).toHaveTextContent('—')
  })

  it('treats Low Risk as calm (no pulsing risk dot)', async () => {
    monitor({ predictData: PREDICT({ risk_level: 'Low Risk', risk_color: '#28a745' }) })
    const { container } = renderPage()
    await screen.findAllByText('Patient 2 — Moderate')
    const dot = [...container.querySelectorAll('span')].find(s => s.style.borderRadius === '50%' && s.style.width === '8px')
    expect(dot.style.animation).toBe('none')
  })

  it('pulses the risk dot for anything above Low Risk', async () => {
    const { container } = renderPage()
    await screen.findAllByText('Patient 2 — Moderate')
    const dot = [...container.querySelectorAll('span')].find(s => s.style.borderRadius === '50%' && s.style.width === '8px')
    expect(dot.style.animation).toContain('pulse')
  })
})

describe('care mode', () => {
  it('starts in hospital mode and switches to home care', async () => {
    renderPage()
    await waitFor(() => expect(generateReport).toHaveBeenCalledWith(2, 'hospital'))
    fireEvent.click(screen.getByRole('button', { name: /Home/i }))
    await waitFor(() => expect(generateReport).toHaveBeenCalledWith(2, 'homecare'))
  })
})

describe('invalid patient', () => {
  it.each(['0', '4', 'abc', '-1'])('redirects "%s" to the selection page', (id) => {
    renderPage(id)
    expect(screen.getByTestId('home')).toBeInTheDocument()
  })

  it('does not ask the server for an invalid patient', () => {
    renderPage('9')
    expect(getPatient).not.toHaveBeenCalled()
  })
})

describe('camera hand-off', () => {
  it('passes null camera data to the monitor hook until the camera reports', () => {
    renderPage()
    expect(useMonitor).toHaveBeenLastCalledWith(2, null)
  })

  it('opens the camera window with the live panel', async () => {
    renderPage()
    // The page re-mounts parts of its tree while the report and patient queries settle, which
    // detaches any button handle captured earlier. Wait until the report is on screen, then look
    // the button up again at click time so the click lands on the live element.
    await screen.findByText('Current Condition')
    await waitFor(() => expect(screen.getAllByText('82%').length).toBeGreaterThan(0))
    const fab = () => screen.getByRole('button', { name: /Open Live CV Monitor/i })
    await waitFor(() => expect(document.body.contains(fab())).toBe(true))
    fireEvent.click(fab())
    // the camera window opened: its title is on screen and its right-hand panel was handed the page's data
    await waitFor(() => expect([...document.querySelectorAll('span')].some(e => e.textContent === 'Live CV Assessment')).toBe(true))
    const panel = await screen.findByLabelText('Live model input and output', {}, { timeout: 3000 })
    expect(panel).toHaveTextContent('6.4')
    expect(panel).toHaveTextContent('1:00')                   // countdown passed down from the monitor hook
  })
})
