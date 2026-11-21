import { render, screen, waitFor, act } from '@testing-library/react'
import ReportContainer from '@/components/report/ReportContainer.jsx'

// Mock the API call
vi.mock('@/api/client.js', () => ({
  generateReport: vi.fn(),
}))

import { generateReport } from '@/api/client.js'

// The component also prefetches the other care mode in the background, so
// count calls per mode rather than in total.
const callsFor = (mode) => generateReport.mock.calls.filter(c => c[1] === mode).length

const predictData = {
  patient_id: 1,
  sofa_score: 5.5,
  risk_level: 'Moderate Risk',
  vitals: { HR: 80, SpO2: 95, SBP: 100, GCS_eye: 3, stress: 4.0 },
  shap_top5: [],
}

const reportResponse = {
  sections: {
    'CURRENT CONDITION': 'Patient is haemodynamically compromised.',
    'PROBABLE CAUSE': 'Suspected sepsis secondary to pneumonia.',
    'RISK FORECAST': 'SOFA likely to rise without intervention.',
    'IMMEDIATE ACTIONS': 'Administer IV antibiotics and fluids.',
  },
  reliability_score: 0.78,
  reliability_label: 'High',
  reliability_color: '#28a745',
  caution: null,
  judge_used: true,
}

describe('ReportContainer', () => {
  beforeEach(() => {
    generateReport.mockReset()
  })

  it('renders skeleton while loading (no previous data)', () => {
    generateReport.mockReturnValue(new Promise(() => {})) // never resolves
    render(<ReportContainer predictData={predictData} careMode="hospital" onReportData={() => {}} />)
    expect(screen.getByText(/Generating AI assessment/i)).toBeInTheDocument()
  })

  it('calls generateReport when predictData is provided', async () => {
    generateReport.mockResolvedValue({ data: reportResponse })
    render(<ReportContainer predictData={predictData} careMode="hospital" onReportData={() => {}} />)
    await waitFor(() => expect(generateReport).toHaveBeenCalledTimes(1))
  })

  it('shows report content after successful load', async () => {
    generateReport.mockResolvedValue({ data: reportResponse })
    render(<ReportContainer predictData={predictData} careMode="hospital" onReportData={() => {}} />)
    // ReportSection maps 'CURRENT CONDITION' → 'Current Condition' in the DOM
    await waitFor(() => {
      expect(screen.getByText(/Current Condition/i)).toBeInTheDocument()
    })
  })

  it('shows all 4 report sections', async () => {
    generateReport.mockResolvedValue({ data: reportResponse })
    render(<ReportContainer predictData={predictData} careMode="hospital" onReportData={() => {}} />)
    await waitFor(() => {
      expect(screen.getByText(/Current Condition/i)).toBeInTheDocument()
      expect(screen.getByText(/Probable Cause/i)).toBeInTheDocument()
      expect(screen.getByText(/Risk Forecast/i)).toBeInTheDocument()
      expect(screen.getByText(/Immediate Actions/i)).toBeInTheDocument()
    })
  })

  it('shows 401 error message correctly', async () => {
    const err = { response: { status: 401, data: {} }, message: 'Unauthorized' }
    generateReport.mockRejectedValue(err)
    render(<ReportContainer predictData={predictData} careMode="hospital" onReportData={() => {}} />)
    await waitFor(() => {
      expect(screen.getByText(/Invalid API key/i)).toBeInTheDocument()
    })
  })

  it('shows 429 rate limit message correctly', async () => {
    const err = { response: { status: 429, data: {} }, message: 'Too Many Requests' }
    generateReport.mockRejectedValue(err)
    render(<ReportContainer predictData={predictData} careMode="hospital" onReportData={() => {}} />)
    await waitFor(() => {
      expect(screen.getByText(/Rate limit reached/i)).toBeInTheDocument()
    })
  })

  it('shows 503 unavailable message correctly', async () => {
    const err = { response: { status: 503, data: {} }, message: 'Service Unavailable' }
    generateReport.mockRejectedValue(err)
    render(<ReportContainer predictData={predictData} careMode="hospital" onReportData={() => {}} />)
    await waitFor(() => {
      expect(screen.getByText(/temporarily unavailable/i)).toBeInTheDocument()
    })
  })

  it('does not call generateReport again when non-clinical values change', async () => {
    generateReport.mockResolvedValue({ data: reportResponse })
    const { rerender } = render(
      <ReportContainer predictData={predictData} careMode="hospital" onReportData={() => {}} />
    )
    await waitFor(() => expect(callsFor('hospital')).toBe(1))

    // Re-render with same clinical values (same key) — should not re-call
    rerender(<ReportContainer predictData={{ ...predictData }} careMode="hospital" onReportData={() => {}} />)
    await waitFor(() => expect(callsFor('hospital')).toBe(1))
  })

  it('requests a report for each care mode', async () => {
    generateReport.mockResolvedValue({ data: reportResponse })
    const { rerender } = render(
      <ReportContainer predictData={predictData} careMode="hospital" onReportData={() => {}} />
    )
    await waitFor(() => expect(callsFor('hospital')).toBe(1))

    rerender(<ReportContainer predictData={predictData} careMode="homecare" onReportData={() => {}} />)
    await waitFor(() => expect(callsFor('homecare')).toBe(1))
  })

  it('prefetches the other care mode once the current report has loaded', async () => {
    generateReport.mockResolvedValue({ data: reportResponse })
    render(<ReportContainer predictData={predictData} careMode="hospital" onReportData={() => {}} />)
    await waitFor(() => expect(callsFor('homecare')).toBe(1))
    expect(callsFor('hospital')).toBe(1)
  })

  it('switches care mode instantly from cache, with no extra request or skeleton', async () => {
    const hospital = { ...reportResponse, care_mode: 'hospital', reliability_score: 0.64 }
    const homecare = { ...reportResponse, care_mode: 'homecare', reliability_score: 0.81,
      sections: { ...reportResponse.sections, 'CURRENT CONDITION': 'Resting comfortably at home.' } }
    generateReport.mockImplementation((_, mode) =>
      Promise.resolve({ data: mode === 'homecare' ? homecare : hospital }))
    const onReportData = vi.fn()

    const { rerender } = render(
      <ReportContainer predictData={predictData} careMode="hospital" onReportData={onReportData} />
    )
    await waitFor(() => expect(callsFor('homecare')).toBe(1))  // prefetch done
    await act(async () => {})

    rerender(<ReportContainer predictData={predictData} careMode="homecare" onReportData={onReportData} />)
    // Shown on the very next render — no loading skeleton, no new request
    expect(screen.getByText(/Resting comfortably at home/i)).toBeInTheDocument()
    expect(screen.queryByText(/Generating AI assessment/i)).not.toBeInTheDocument()
    expect(onReportData).toHaveBeenLastCalledWith(homecare)

    rerender(<ReportContainer predictData={predictData} careMode="hospital" onReportData={onReportData} />)
    expect(screen.getByText(/haemodynamically compromised/i)).toBeInTheDocument()
    expect(onReportData).toHaveBeenLastCalledWith(hospital)
    expect(generateReport).toHaveBeenCalledTimes(2)
  })

  it('never shows one care mode\'s report under the other tab while loading', async () => {
    const hospital = { ...reportResponse, care_mode: 'hospital' }
    generateReport.mockImplementation((_, mode) =>
      mode === 'hospital' ? Promise.resolve({ data: hospital }) : new Promise(() => {}))
    const onReportData = vi.fn()
    const { rerender } = render(
      <ReportContainer predictData={predictData} careMode="hospital" onReportData={onReportData} />
    )
    await waitFor(() => expect(screen.getByText(/haemodynamically compromised/i)).toBeInTheDocument())

    rerender(<ReportContainer predictData={predictData} careMode="homecare" onReportData={onReportData} />)
    expect(screen.queryByText(/haemodynamically compromised/i)).not.toBeInTheDocument()
    expect(screen.getByText(/Generating AI assessment/i)).toBeInTheDocument()
    expect(onReportData).toHaveBeenLastCalledWith(null)
  })

  it('a new reading empties the cache so both modes are regenerated', async () => {
    generateReport.mockResolvedValue({ data: reportResponse })
    const { rerender } = render(
      <ReportContainer predictData={predictData} careMode="hospital" onReportData={() => {}} />
    )
    await waitFor(() => expect(callsFor('homecare')).toBe(1))

    const next = { ...predictData, vitals: { ...predictData.vitals, HR: 120 } }
    rerender(<ReportContainer predictData={next} careMode="hospital" onReportData={() => {}} />)
    await waitFor(() => expect(callsFor('homecare')).toBe(2))
    expect(callsFor('hospital')).toBe(2)
  })

  it('does not cache a failed report — the same reading retries', async () => {
    generateReport.mockRejectedValueOnce({ response: { status: 503, data: {} } })
    generateReport.mockResolvedValue({ data: reportResponse })
    const { rerender } = render(
      <ReportContainer predictData={predictData} careMode="hospital" onReportData={() => {}} />
    )
    await waitFor(() => expect(screen.getByText(/temporarily unavailable/i)).toBeInTheDocument())

    rerender(<ReportContainer predictData={predictData} careMode="homecare" onReportData={() => {}} />)
    rerender(<ReportContainer predictData={predictData} careMode="hospital" onReportData={() => {}} />)
    await waitFor(() => expect(callsFor('hospital')).toBe(2))
  })

  it('calls generateReport again when GCS_eye changes', async () => {
    generateReport.mockResolvedValue({ data: reportResponse })
    const { rerender } = render(
      <ReportContainer predictData={predictData} careMode="hospital" onReportData={() => {}} />
    )
    await waitFor(() => expect(callsFor('hospital')).toBe(1))

    const newData = { ...predictData, vitals: { ...predictData.vitals, GCS_eye: 1 } }
    rerender(<ReportContainer predictData={newData} careMode="hospital" onReportData={() => {}} />)
    await waitFor(() => expect(callsFor('hospital')).toBe(2))
  })

  it('renders nothing when predictData is null', () => {
    const { container } = render(
      <ReportContainer predictData={null} careMode="hospital" onReportData={() => {}} />
    )
    expect(container.firstChild).toBeNull()
  })

  it('calls onReportData callback with report data after load', async () => {
    const onReportData = vi.fn()
    generateReport.mockResolvedValue({ data: reportResponse })
    render(<ReportContainer predictData={predictData} careMode="hospital" onReportData={onReportData} />)
    await waitFor(() => expect(onReportData).toHaveBeenCalledWith(reportResponse))
  })

  it('shows stale notice while refreshing with existing data', async () => {
    generateReport.mockResolvedValue({ data: reportResponse })
    const { rerender } = render(
      <ReportContainer predictData={predictData} careMode="hospital" onReportData={() => {}} />
    )
    // Wait for first report to load
    await waitFor(() => expect(screen.getByText(/Current Condition/i)).toBeInTheDocument())

    // Next fetch never resolves — simulates a slow refresh
    generateReport.mockReturnValue(new Promise(() => {}))
    // Change sofa_score and HR so reportKey differs → triggers new fetch
    const newData = { ...predictData, sofa_score: 8.0, vitals: { ...predictData.vitals, HR: 85 } }
    rerender(<ReportContainer predictData={newData} careMode="hospital" onReportData={() => {}} />)

    await waitFor(() => {
      expect(screen.getByText(/Updating report/i)).toBeInTheDocument()
    })
  })

  it('shows stale notice when API fails after existing data is loaded', async () => {
    generateReport.mockResolvedValue({ data: reportResponse })
    const { rerender } = render(
      <ReportContainer predictData={predictData} careMode="hospital" onReportData={() => {}} />
    )
    // Wait for first report
    await waitFor(() => expect(screen.getByText(/Current Condition/i)).toBeInTheDocument())

    // Subsequent fetch fails
    generateReport.mockRejectedValue({ response: { status: 503, data: {} }, message: 'Error' })
    const newData = { ...predictData, sofa_score: 9.0, vitals: { ...predictData.vitals, HR: 90 } }
    rerender(<ReportContainer predictData={newData} careMode="hospital" onReportData={() => {}} />)

    // Previous content still visible, error/stale notice appears
    await waitFor(() => {
      expect(screen.getByText(/Current Condition/i)).toBeInTheDocument()
    })
  })

  it('calls generateReport again when sofa_score changes', async () => {
    generateReport.mockResolvedValue({ data: reportResponse })
    const { rerender } = render(
      <ReportContainer predictData={predictData} careMode="hospital" onReportData={() => {}} />
    )
    await waitFor(() => expect(callsFor('hospital')).toBe(1))

    const newData = { ...predictData, sofa_score: 10.5 }
    rerender(<ReportContainer predictData={newData} careMode="hospital" onReportData={() => {}} />)
    await waitFor(() => expect(callsFor('hospital')).toBe(2))
  })

  it('calls generateReport again when stress changes', async () => {
    generateReport.mockResolvedValue({ data: reportResponse })
    const { rerender } = render(
      <ReportContainer predictData={predictData} careMode="hospital" onReportData={() => {}} />
    )
    await waitFor(() => expect(callsFor('hospital')).toBe(1))

    const newData = { ...predictData, vitals: { ...predictData.vitals, stress: 8.0 } }
    rerender(<ReportContainer predictData={newData} careMode="hospital" onReportData={() => {}} />)
    await waitFor(() => expect(callsFor('hospital')).toBe(2))
  })

  it('does not call generateReport when only non-reportKey fields change', async () => {
    generateReport.mockResolvedValue({ data: reportResponse })
    const { rerender } = render(
      <ReportContainer predictData={predictData} careMode="hospital" onReportData={() => {}} />
    )
    await waitFor(() => expect(callsFor('hospital')).toBe(1))

    // risk_level and row_idx are not part of reportKey — same clinical values
    const newData = { ...predictData, risk_level: 'High Risk', row_idx: 999 }
    rerender(<ReportContainer predictData={newData} careMode="hospital" onReportData={() => {}} />)
    await new Promise(r => setTimeout(r, 80))
    expect(callsFor('hospital')).toBe(1)
  })
})
