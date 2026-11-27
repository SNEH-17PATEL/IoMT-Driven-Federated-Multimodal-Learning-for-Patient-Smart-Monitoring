import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import TrendsSection from '@/components/trends/TrendsSection.jsx'

vi.mock('@/api/client.js', () => ({ getVitals: vi.fn() }))
import { getVitals } from '@/api/client.js'

const renderSection = (patientId = 1) =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <TrendsSection patientId={patientId} />
    </QueryClientProvider>
  )

const rows = [
  { time: '2026-01-01 12:00:00', HR: 72, RR: 16, SpO2: 98, Temp: 37, SBP: 118, DBP: 76, MAP: 90 },
  { time: '2026-01-01 12:01:30', HR: 75, RR: 17, SpO2: 97, Temp: 37.1, SBP: 120, DBP: 78, MAP: 92 },
]

describe('TrendsSection data states', () => {
  beforeEach(() => { getVitals.mockReset() })

  it('shows seven skeletons while loading', () => {
    getVitals.mockReturnValue(new Promise(() => {}))
    const { container } = renderSection()
    expect(container.querySelectorAll('.skeleton')).toHaveLength(7)
  })

  it('shows an error message when the vitals request fails', async () => {
    getVitals.mockRejectedValue(new Error('boom'))
    renderSection()
    expect(await screen.findByText(/Could not load trend data/)).toBeInTheDocument()
  })

  it('draws all seven vital charts once data arrives', async () => {
    getVitals.mockResolvedValue({ data: rows })
    renderSection()
    for (const label of ['Heart Rate', 'Resp. Rate', 'SpO₂', 'Temperature', 'Sys. BP', 'Dia. BP', 'MAP'])
      expect(await screen.findByText(label)).toBeInTheDocument()
  })

  it('asks for the last 20 readings of the right patient', async () => {
    getVitals.mockResolvedValue({ data: rows })
    renderSection(3)
    await waitFor(() => expect(getVitals).toHaveBeenCalledWith(3, 20))
  })

  it('draws charts with no data when the server returns nothing', async () => {
    getVitals.mockResolvedValue({ data: null })
    renderSection()
    expect(await screen.findByText('Heart Rate')).toBeInTheDocument()
  })
})
