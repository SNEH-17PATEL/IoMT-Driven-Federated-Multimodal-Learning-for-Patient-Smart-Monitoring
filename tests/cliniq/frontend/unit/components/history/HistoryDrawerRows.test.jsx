/*
 * HistoryDrawer — rows, cells, states and closing behaviour.
 */
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import HistoryDrawer from '@/components/history/HistoryDrawer.jsx'

vi.mock('@/api/client.js', () => ({ getHistory: vi.fn() }))
import { getHistory } from '@/api/client.js'

const row = (over = {}) => ({
  Timestamp: '2026-11-24 10:00', SOFA: 4.2, Risk: 'Low Risk', HR: 80, RR: 16, 'SpO₂': 98,
  'Temp (°C)': 37.0, SBP: 118, DBP: 76, MAP: 90, 'GCS Eye': 4, Stress: 1.0, ...over,
})

const open = (props = {}) => {
  const onClose = props.onClose ?? vi.fn()
  const utils = render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <HistoryDrawer open patientId={1} onClose={onClose} {...props} />
    </QueryClientProvider>
  )
  return { ...utils, onClose }
}

beforeEach(() => { getHistory.mockReset() })

describe('states', () => {
  it('shows skeleton rows while loading', () => {
    getHistory.mockReturnValue(new Promise(() => {}))
    open()
    expect(screen.getByText('Loading…')).toBeInTheDocument()
  })

  it('shows an error message and header when the request fails', async () => {
    getHistory.mockRejectedValue(new Error('down'))
    open()
    expect(await screen.findByText(/Failed to load history\. Make sure the backend is running\./)).toBeInTheDocument()
    expect(screen.getByText('Failed to load')).toBeInTheDocument()
  })

  it('shows the empty message when there is no history', async () => {
    getHistory.mockResolvedValue({ data: [] })
    open()
    expect(await screen.findByText(/No history recorded yet/)).toBeInTheDocument()
    expect(screen.getByText(/0 recordings/)).toBeInTheDocument()
  })

  it('treats a null body as empty', async () => {
    getHistory.mockResolvedValue({ data: null })
    open()
    expect(await screen.findByText(/No history recorded yet/)).toBeInTheDocument()
  })

  it('counts the recordings in the header', async () => {
    getHistory.mockResolvedValue({ data: [row(), row({ SOFA: 5 })] })
    open()
    expect(await screen.findByText(/2 recordings/)).toBeInTheDocument()
  })

  it('does not fetch while closed', () => {
    getHistory.mockResolvedValue({ data: [] })
    open({ open: false })
    expect(getHistory).not.toHaveBeenCalled()
  })

  it('asks for the 50 latest rows of the right patient', async () => {
    getHistory.mockResolvedValue({ data: [] })
    open({ patientId: 3 })
    await waitFor(() => expect(getHistory).toHaveBeenCalledWith(3, 50))
  })
})

describe('rows', () => {
  it('lists newest first', async () => {
    getHistory.mockResolvedValue({ data: [row({ Timestamp: 'OLD' }), row({ Timestamp: 'NEW' })] })
    open()
    await screen.findByText('NEW')
    const cells = screen.getAllByText(/^(OLD|NEW)$/).map(e => e.textContent)
    expect(cells).toEqual(['NEW', 'OLD'])
  })

  it.each([
    ['Low Risk'], ['Moderate Risk'], ['High Risk'], ['Critical Risk'], ['Something Else'],
  ])('shows a "%s" risk badge', async (risk) => {
    getHistory.mockResolvedValue({ data: [row({ Risk: risk })] })
    open()
    expect(await screen.findByText(risk)).toBeInTheDocument()
  })

  it('shows a dash for a missing risk or timestamp', async () => {
    getHistory.mockResolvedValue({ data: [row({ Risk: null, Timestamp: null })] })
    open()
    await screen.findByText(/1 recordings/)
    expect(screen.getAllByText('—').length).toBeGreaterThanOrEqual(2)
  })

  it('shows a dash for a missing number', async () => {
    getHistory.mockResolvedValue({ data: [row({ HR: null })] })
    open()
    await screen.findByText(/1 recordings/)
    expect(screen.getAllByText('—').length).toBeGreaterThan(0)
  })

  it('marks values above and below the normal range', async () => {
    getHistory.mockResolvedValue({ data: [row({ HR: 130, SpO2: undefined, 'SpO₂': 90 })] })
    const { container } = open()
    await screen.findByText(/1 recordings/)
    const sups = [...container.querySelectorAll('sup')].map(s => s.textContent)
    expect(sups).toContain('↑')                // HR 130 > 100
    expect(sups).toContain('↓')                // SpO₂ 90 < 95
  })

  it('does not mark values inside the range', async () => {
    getHistory.mockResolvedValue({ data: [row()] })
    const { container } = open()
    await screen.findByText(/1 recordings/)
    expect(container.querySelectorAll('sup')).toHaveLength(0)
  })

  it('shows whole numbers as integers and others with one decimal', async () => {
    getHistory.mockResolvedValue({ data: [row({ HR: 80, SOFA: 4.26 })] })
    open()
    await screen.findByText('4.3')
    expect(screen.getByText('80')).toBeInTheDocument()
  })

  it('highlights a row on hover and restores it on leave', async () => {
    getHistory.mockResolvedValue({ data: [row(), row()] })
    const { container } = open()
    await screen.findAllByText('Low Risk')
    const tr = container.querySelectorAll('tbody tr')[0]
    const before = tr.style.background
    fireEvent.mouseEnter(tr)
    expect(tr.style.background).not.toBe(before)
    fireEvent.mouseLeave(tr)
    expect(tr.style.background).toBe(before)
  })
})

describe('closing', () => {
  const backdrop = () => document.querySelector('div[style*="position: fixed"][style*="inset"]')

  it('closes from the backdrop', async () => {
    getHistory.mockResolvedValue({ data: [] })
    const { onClose } = open()
    await screen.findByText(/No history recorded yet/)
    fireEvent.click(backdrop())
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('closes from the ✕ button in the header', async () => {
    getHistory.mockResolvedValue({ data: [] })
    const { onClose } = open()
    await screen.findByText(/No history recorded yet/)
    const closeBtn = screen.getAllByRole('button').find(b => b.querySelector('svg path[d^="M2 2l12 12"]'))
    fireEvent.click(closeBtn)
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('closes on Escape, like the other drawers', async () => {
    getHistory.mockResolvedValue({ data: [] })
    const { onClose } = open()
    await screen.findByText(/No history recorded yet/)
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('ignores other keys', async () => {
    getHistory.mockResolvedValue({ data: [] })
    const { onClose } = open()
    await screen.findByText(/No history recorded yet/)
    fireEvent.keyDown(window, { key: 'Enter' })
    expect(onClose).not.toHaveBeenCalled()
  })

  it('does not listen for Escape while closed', async () => {
    getHistory.mockResolvedValue({ data: [] })
    const { onClose } = open({ open: false })
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(onClose).not.toHaveBeenCalled()
  })

  it('stops listening for Escape once unmounted', async () => {
    getHistory.mockResolvedValue({ data: [] })
    const { onClose, unmount } = open()
    await screen.findByText(/No history recorded yet/)
    unmount()
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(onClose).not.toHaveBeenCalled()
  })
})
