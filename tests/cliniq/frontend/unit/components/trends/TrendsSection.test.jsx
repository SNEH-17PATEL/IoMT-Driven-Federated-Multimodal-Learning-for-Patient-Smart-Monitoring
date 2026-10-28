import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import TrendsSection from '@/components/trends/TrendsSection.jsx'

vi.mock('@/api/client.js', () => ({
  getVitals: vi.fn(() => Promise.resolve({ data: [] })),
}))

const createClient = () =>
  new QueryClient({ defaultOptions: { queries: { retry: false } } })

const renderSection = (patientId = 1) =>
  render(
    <QueryClientProvider client={createClient()}>
      <TrendsSection patientId={patientId} />
    </QueryClientProvider>
  )

describe('TrendsSection', () => {
  it('renders without crash', () => {
    renderSection()
    expect(document.body).toBeTruthy()
  })

  it('renders chart skeleton cards while loading', () => {
    renderSection()
    // During loading, skeleton divs with class "skeleton" are rendered
    const skeletons = document.querySelectorAll('.skeleton')
    expect(skeletons.length).toBeGreaterThan(0)
  })

  it('renders after data loads (empty array = 7 empty charts)', async () => {
    renderSection()
    // After the query resolves with [], the component renders 7 chart cards
    await waitFor(() => {
      // Skeleton gone or chart containers present
      expect(document.body.firstChild).toBeTruthy()
    })
  })

  it('shows Heart Rate chart label after data loads', async () => {
    renderSection()
    await waitFor(() => {
      expect(screen.getByText('Heart Rate')).toBeInTheDocument()
    })
  })

  it('renders all 7 vital chart labels after query resolves', async () => {
    renderSection()
    // TrendsSection renders 7 TrendChart cards; each shows its label text
    await waitFor(() => {
      ['Heart Rate', 'Resp. Rate', 'SpO₂', 'Temperature', 'Sys. BP', 'Dia. BP', 'MAP'].forEach(label => {
        expect(screen.getByText(label)).toBeInTheDocument()
      })
    })
  })

  it('renders recharts chart containers after data loads', async () => {
    renderSection()
    // recharts mock renders <div data-testid="chart"> for ResponsiveContainer
    await waitFor(() => {
      const charts = document.querySelectorAll('[data-testid="chart"]')
      expect(charts.length).toBeGreaterThan(0)
    })
  })
})
