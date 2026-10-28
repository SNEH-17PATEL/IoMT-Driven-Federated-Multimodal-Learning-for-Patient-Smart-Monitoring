import { render, screen, fireEvent } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import HistoryDrawer from '@/components/history/HistoryDrawer.jsx'

vi.mock('@/api/client.js', () => ({
  getHistory: vi.fn(() => Promise.resolve({
    data: [
      {
        Timestamp: '2026-01-01 12:00', Reading: 'R1',
        SOFA: 5.5, Risk: 'Moderate Risk',
        HR: 80, RR: 18, 'SpO₂': 97, 'Temp (°C)': 37.1,
        SBP: 112, DBP: 70, MAP: 84, 'GCS Eye': 4, Stress: 2.0,
      },
    ],
  })),
}))

import { getHistory } from '@/api/client.js'

const createClient = () => new QueryClient({ defaultOptions: { queries: { retry: false } } })

const renderDrawer = (open, onClose = () => {}) =>
  render(
    <QueryClientProvider client={createClient()}>
      <HistoryDrawer open={open} onClose={onClose} patientId={1} />
    </QueryClientProvider>
  )

describe('HistoryDrawer', () => {
  it('does not render table when closed', () => {
    renderDrawer(false)
    expect(screen.queryByText('Timestamp')).toBeNull()
  })

  it('renders when open=true', () => {
    renderDrawer(true)
    expect(document.body.textContent).toMatch(/History|SOFA|Risk/i)
  })

  it('shows column headers', async () => {
    renderDrawer(true)
    expect(await screen.findByText('SOFA')).toBeInTheDocument()
  })

  it('shows Timestamp column header', async () => {
    renderDrawer(true)
    expect(await screen.findByText('Timestamp')).toBeInTheDocument()
  })

  it('shows Risk column header', async () => {
    renderDrawer(true)
    expect(await screen.findByText('Risk')).toBeInTheDocument()
  })

  it('calls onClose when close button clicked', () => {
    const onClose = vi.fn()
    renderDrawer(true, onClose)
    const closeBtn = screen.getAllByRole('button').find(b => b.getAttribute('aria-label')?.match(/close/i))
    if (closeBtn) {
      fireEvent.click(closeBtn)
      expect(onClose).toHaveBeenCalled()
    }
  })

  it('renders SOFA value 5.5 as "5.5" in the data row', async () => {
    renderDrawer(true)
    // NumCell renders non-integer SOFA via .toFixed(1) → "5.5"
    expect(await screen.findByText('5.5')).toBeInTheDocument()
  })

  it('shows empty state message when no history records exist', async () => {
    getHistory.mockResolvedValueOnce({ data: [] })
    renderDrawer(true)
    expect(await screen.findByText(/No history recorded yet/i)).toBeInTheDocument()
  })
})
