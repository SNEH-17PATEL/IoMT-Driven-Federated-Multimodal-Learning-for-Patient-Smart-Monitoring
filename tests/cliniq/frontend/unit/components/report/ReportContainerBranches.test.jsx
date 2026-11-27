/*
 * Branches of ReportContainer that the main test file does not reach:
 * patient switch, soft failures (HTTP 200 with no sections), the report-level error screen,
 * rate-limit wording, the fallback error message, and the mismatched-care-mode guard.
 */
import { render, screen, waitFor, act } from '@testing-library/react'
import ReportContainer from '@/components/report/ReportContainer.jsx'

vi.mock('@/api/client.js', () => ({ generateReport: vi.fn() }))
import { generateReport } from '@/api/client.js'

const pd = (over = {}) => ({
  patient_id: 1, sofa_score: 5.5, risk_level: 'Moderate Risk',
  vitals: { HR: 80, RR: 16, SpO2: 95, Temp: 37, SBP: 100, DBP: 60, MAP: 75, GCS_eye: 3, stress: 4 },
  shap_top5: [], ...over,
})

const good = (mode = 'hospital', text = 'Patient is compromised.') => ({
  data: {
    sections: {
      'CURRENT CONDITION': text, 'PROBABLE CAUSE': 'Sepsis.',
      'RISK FORECAST': 'Rising.', 'IMMEDIATE ACTIONS': 'Fluids.',
    },
    reliability_score: 0.8, reliability_label: 'High', reliability_color: '#28a745',
    caution: false, care_mode: mode,
  },
})

describe('ReportContainer — more branches', () => {
  beforeEach(() => { generateReport.mockReset() })

  it('clears the previous patient\'s report immediately when the patient changes', async () => {
    generateReport.mockImplementation((pid, mode) =>
      pid === 1 ? Promise.resolve(good(mode, 'Patient ONE assessment.')) : new Promise(() => {}))
    const { rerender } = render(<ReportContainer predictData={pd({ patient_id: 1 })} careMode="hospital" />)
    await screen.findByText(/Patient ONE assessment/)
    rerender(<ReportContainer predictData={pd({ patient_id: 2 })} careMode="hospital" />)
    expect(screen.queryByText(/Patient ONE assessment/)).not.toBeInTheDocument()
    expect(await screen.findByText(/Generating AI assessment/)).toBeInTheDocument()
  })

  it('publishes null to the parent when the patient changes', async () => {
    generateReport.mockImplementation((pid, mode) =>
      pid === 1 ? Promise.resolve(good(mode)) : new Promise(() => {}))
    const onReportData = vi.fn()
    const { rerender } = render(<ReportContainer predictData={pd({ patient_id: 1 })} careMode="hospital" onReportData={onReportData} />)
    await waitFor(() => expect(onReportData).toHaveBeenCalledWith(expect.objectContaining({ care_mode: 'hospital' })))
    rerender(<ReportContainer predictData={pd({ patient_id: 3 })} careMode="hospital" onReportData={onReportData} />)
    await waitFor(() => expect(onReportData).toHaveBeenLastCalledWith(null))
  })

  it('shows the report-level error screen when the API returns an error and no sections', async () => {
    generateReport.mockResolvedValue({ data: { sections: null, error: 'LLM unavailable — please retry.' } })
    render(<ReportContainer predictData={pd()} careMode="hospital" />)
    expect(await screen.findByText('AI Assessment Unavailable')).toBeInTheDocument()
    expect(screen.getByText(/will retry automatically on the next reading/i)).toBeInTheDocument()
  })

  it.each(['429 Too Many Requests', 'rate_limit_exceeded', 'Rate limit reached'])(
    'uses the quota wording when the error mentions "%s"', async (msg) => {
      generateReport.mockResolvedValue({ data: { sections: null, error: msg } })
      render(<ReportContainer predictData={pd()} careMode="hospital" />)
      expect(await screen.findByText(/daily token quota has been reached/i)).toBeInTheDocument()
    })

  it('keeps the last good report when the next reply is a soft failure (no sections)', async () => {
    generateReport.mockResolvedValue(good('hospital', 'First good report.'))
    const onReportData = vi.fn()
    const { rerender } = render(<ReportContainer predictData={pd()} careMode="hospital" onReportData={onReportData} />)
    await screen.findByText(/First good report/)

    generateReport.mockResolvedValue({ data: { sections: null, error: 'LLM unavailable', care_mode: 'hospital' } })
    const before = generateReport.mock.calls.length
    rerender(<ReportContainer predictData={pd({ sofa_score: 9.0, vitals: { ...pd().vitals, HR: 130 } })}
                              careMode="hospital" onReportData={onReportData} />)
    await waitFor(() => expect(generateReport.mock.calls.length).toBeGreaterThan(before))
    await new Promise(r => setTimeout(r, 100))
    // the earlier assessment stays on screen and no error screen replaced it
    expect(screen.getByText(/First good report/)).toBeInTheDocument()
    expect(screen.queryByText('AI Assessment Unavailable')).not.toBeInTheDocument()
    expect(onReportData).toHaveBeenLastCalledWith(expect.objectContaining({ sections: expect.any(Object) }))
  })

  it('shows the unavailable screen on a soft failure when there is no earlier report to keep', async () => {
    generateReport.mockResolvedValue({ data: { sections: null, error: 'LLM unavailable', care_mode: 'hospital' } })
    render(<ReportContainer predictData={pd()} careMode="hospital" />)
    expect(await screen.findByText('AI Assessment Unavailable')).toBeInTheDocument()
  })

  it('does not keep the OTHER care mode\'s last good report on a soft failure', async () => {
    generateReport.mockImplementation((_p, mode) =>
      mode === 'hospital' ? Promise.resolve(good('hospital', 'HOSPITAL TEXT'))
                          : Promise.resolve({ data: { sections: null, error: 'LLM unavailable', care_mode: 'homecare' } }))
    const { rerender } = render(<ReportContainer predictData={pd()} careMode="hospital" />)
    await screen.findByText(/HOSPITAL TEXT/)
    rerender(<ReportContainer predictData={pd()} careMode="homecare" />)
    expect(await screen.findByText('AI Assessment Unavailable')).toBeInTheDocument()
    expect(screen.queryByText(/HOSPITAL TEXT/)).not.toBeInTheDocument()
  })

  it('uses a generic message when a failure has no detail', async () => {
    generateReport.mockRejectedValue({})
    render(<ReportContainer predictData={pd()} careMode="hospital" />)
    expect(await screen.findByText(/Report generation failed\. Showing previous assessment\./)).toBeInTheDocument()
  })

  it('falls back to the error detail text when the status is not a known one', async () => {
    generateReport.mockRejectedValue({ response: { status: 500, data: { detail: 'GROQ_API_KEY not configured' } } })
    render(<ReportContainer predictData={pd()} careMode="hospital" />)
    expect(await screen.findByText(/GROQ_API_KEY not configured\. Showing previous assessment\./)).toBeInTheDocument()
  })

  it('uses the plain Error message when there is no response body', async () => {
    generateReport.mockRejectedValue(new Error('Network Error'))
    render(<ReportContainer predictData={pd()} careMode="hospital" />)
    expect(await screen.findByText(/Network Error\. Showing previous assessment\./)).toBeInTheDocument()
  })

  it('never shows a hospital report under the home-care tab', async () => {
    generateReport.mockImplementation((_p, mode) =>
      mode === 'hospital' ? Promise.resolve(good('hospital', 'HOSPITAL TEXT')) : new Promise(() => {}))
    const { rerender } = render(<ReportContainer predictData={pd()} careMode="hospital" />)
    await screen.findByText(/HOSPITAL TEXT/)
    rerender(<ReportContainer predictData={pd()} careMode="homecare" />)
    expect(screen.queryByText(/HOSPITAL TEXT/)).not.toBeInTheDocument()
  })

  it('shows the error box when a failure happens with nothing to fall back on', async () => {
    generateReport.mockRejectedValue({ response: { status: 401 } })
    render(<ReportContainer predictData={pd()} careMode="hospital" />)
    expect(await screen.findByText(/Invalid API key/)).toBeInTheDocument()
    expect(screen.queryByText('Current Condition')).not.toBeInTheDocument()
  })

  it('renders the reliability badge only when a score is present', async () => {
    const noScore = good()
    noScore.data.reliability_score = null
    generateReport.mockResolvedValue(noScore)
    render(<ReportContainer predictData={pd()} careMode="hospital" />)
    await screen.findByText(/Patient is compromised/)
    expect(screen.queryByText(/Reliability/)).not.toBeInTheDocument()
  })
})
