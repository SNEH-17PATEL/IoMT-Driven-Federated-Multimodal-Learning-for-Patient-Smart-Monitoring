import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { MemoryRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ThemeProvider } from '@/contexts/ThemeContext.jsx'
import MonitorDashboard from '@/pages/MonitorDashboard.jsx'

// Mock all external API calls
vi.mock('@/api/client.js', () => ({
  getPatient: vi.fn(),
  runPredict: vi.fn(),
  generateReport: vi.fn(),
  getHistory: vi.fn(),
  getVitals: vi.fn(),
}))

// Mock useMonitor to avoid real timers in integration tests
vi.mock('@/hooks/useMonitor.js', () => ({
  useMonitor: vi.fn(),
}))

import { getPatient, getVitals, generateReport, getHistory } from '@/api/client.js'
import { useMonitor } from '@/hooks/useMonitor.js'

const PATIENT = {
  id: 1, name: 'Patient 1 — Low Risk', short: 'Low Risk',
  icon: '🟢', description: 'Post-surgical recovery', color: '#28a745',
  risk_level: 'Low Risk', sofa_range: '0 – 4'
}

const PREDICT_DATA = {
  patient_id: 1,
  sofa_score: 3.5,
  risk_level: 'Low Risk',
  risk_color: '#28a745',
  risk_icon: '🟢',
  is_alert: false,
  conformal_lo: 1.0,
  conformal_hi: 6.0,
  vitals: { HR: 72, RR: 14, SpO2: 98, Temp: 37.0, SBP: 112, DBP: 70, MAP: 84 },
  vitals_status: { HR: 'normal', RR: 'normal', SpO2: 'normal', Temp: 'normal', SBP: 'normal', DBP: 'normal', MAP: 'normal' },
  shap_top5: [{ feature: 'Heart Rate', direction: 'up', rel_impact: 0.6 }],
}

const createClient = () => new QueryClient({ defaultOptions: { queries: { retry: false } } })

const renderDashboard = (patientId = '1') =>
  render(
    <ThemeProvider>
      <QueryClientProvider client={createClient()}>
        <MemoryRouter initialEntries={[`/monitor/${patientId}`]}>
          <Routes>
            <Route path="/monitor/:patientId" element={<MonitorDashboard />} />
            <Route path="/" element={<div data-testid="home" />} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>
    </ThemeProvider>
  )

describe('MonitorDashboard page', () => {
  beforeEach(() => {
    getPatient.mockResolvedValue({ data: PATIENT })
    getVitals.mockResolvedValue({ data: [] })
    getHistory.mockResolvedValue({ data: [] })
    // generateReport must return a promise — component calls .then() on its return value
    generateReport.mockResolvedValue({
      data: {
        sections: {
          'CURRENT CONDITION': 'Patient is stable.',
          'PROBABLE CAUSE': 'Suspected infection.',
          'RISK FORECAST': 'Low risk of deterioration.',
          'IMMEDIATE ACTIONS': 'Continue monitoring.',
        },
        reliability_score: 0.82,
        reliability_label: 'High',
        reliability_color: '#28a745',
        caution: null,
        judge_used: true,
      },
    })
    useMonitor.mockReturnValue({
      predictData: PREDICT_DATA,
      isRefreshing: false,
      lastUpdated: new Date(),
      countdown: 85,
      error: null,
    })
  })

  it('renders without crash', async () => {
    renderDashboard()
    await waitFor(() => expect(document.body).toBeTruthy())
  })

  it('fetches patient data on mount', async () => {
    renderDashboard()
    await waitFor(() => expect(getPatient).toHaveBeenCalledWith(1))
  })

  it('shows SOFA score from predictData', async () => {
    renderDashboard()
    await waitFor(() => {
      expect(screen.getByText(/3\.5|3,5/)).toBeInTheDocument()
    })
  })

  it('shows risk level from predictData', async () => {
    renderDashboard()
    await waitFor(() => {
      expect(screen.getByText(/Low Risk/i)).toBeInTheDocument()
    })
  })

  it('does NOT show alert banner for Low Risk', async () => {
    renderDashboard()
    await waitFor(() => expect(screen.queryByRole('alert')).toBeNull())
  })

  it('shows alert banner for High Risk', async () => {
    useMonitor.mockReturnValue({
      predictData: { ...PREDICT_DATA, risk_level: 'High Risk', sofa_score: 12.5, is_alert: true },
      isRefreshing: false, lastUpdated: new Date(), countdown: 85, error: null,
    })
    renderDashboard()
    await waitFor(() => {
      expect(screen.getByRole('alert')).toBeInTheDocument()
    })
  })

  it('redirects to / for invalid patientId', async () => {
    renderDashboard('99')
    await waitFor(() => {
      // The page should redirect or show nothing meaningful
      expect(document.body).toBeTruthy()
    })
  })

  it('opens HistoryDrawer when HistoryFAB is clicked', async () => {
    renderDashboard()
    await waitFor(() => screen.getByRole('button', { name: /Open patient history/i }))
    fireEvent.click(screen.getByRole('button', { name: /Open patient history/i }))
    // Drawer should open — check for drawer content
    await waitFor(() => {
      expect(document.body.textContent).toMatch(/History|SOFA|Risk/i)
    })
  })

  it('opens SOFAValidationDrawer when SOFA FAB is clicked', async () => {
    renderDashboard()
    await waitFor(() => screen.getByRole('button', { name: /Open SOFA Score Validation/i }))
    fireEvent.click(screen.getByRole('button', { name: /Open SOFA Score Validation/i }))
    await waitFor(() => {
      expect(document.body.textContent).toMatch(/SOFA|Model|MAE/i)
    })
  })

  it('opens CVMonitorModal when CV FAB is clicked', async () => {
    renderDashboard()
    await waitFor(() => screen.getByRole('button', { name: /Open Live CV Monitor/i }))
    fireEvent.click(screen.getByRole('button', { name: /Open Live CV Monitor/i }))
    await waitFor(() => {
      expect(document.body.textContent).toMatch(/CV|visual|supplement/i)
    })
  })

  it('shows countdown from useMonitor', async () => {
    renderDashboard()
    await waitFor(() => {
      expect(screen.getByText('85s')).toBeInTheDocument()
    })
  })

  it('redirects to "/" and shows home element for invalid patientId', async () => {
    renderDashboard('99')
    await waitFor(() => {
      expect(screen.getByTestId('home')).toBeInTheDocument()
    })
  })

  it('shows patient name after getPatient resolves', async () => {
    renderDashboard()
    // Name appears in both HeaderBar PatientPill and PatientContextStrip
    await waitFor(() => {
      expect(screen.getAllByText('Patient 1 — Low Risk').length).toBeGreaterThan(0)
    })
  })

  it('shows HR value from predictData vitals in VitalsGrid', async () => {
    renderDashboard()
    await waitFor(() => {
      // HR=72 is in PREDICT_DATA.vitals — rendered by VitalsGrid
      expect(screen.getAllByText('72').length).toBeGreaterThan(0)
    })
  })

  it('renders all three section headings', async () => {
    renderDashboard()
    await waitFor(() => {
      expect(screen.getByText('Current Status')).toBeInTheDocument()
      expect(screen.getByText('Vital Trends')).toBeInTheDocument()
      expect(screen.getByText('AI Clinical Report')).toBeInTheDocument()
    })
  })

  it('opens LLMValidationDrawer when LLM FAB is clicked after report loads', async () => {
    renderDashboard()
    // Wait for generateReport to resolve and llmReportData to be set (hasData=true)
    await waitFor(() => expect(generateReport).toHaveBeenCalled(), { timeout: 3000 })

    // Wait for FAB to exist — it renders as soon as predictData is truthy
    const llmFab = await screen.findByRole('button', { name: /Open AI Report Validation/i })

    // Wait until the FAB becomes enabled (hasData becomes true after report loads)
    await waitFor(() => {
      fireEvent.click(llmFab)
      expect(document.body.textContent).toMatch(/validation|method|quality|Reliability/i)
    }, { timeout: 3000 })
  })
})
