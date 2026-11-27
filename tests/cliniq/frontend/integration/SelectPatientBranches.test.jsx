/*
 * SelectPatient — theme, keyboard, hover, reduced-motion and clock behaviour that the main
 * integration test does not reach.
 */
import { render, screen, waitFor, fireEvent, act } from '@testing-library/react'
import { MemoryRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ThemeProvider } from '@/contexts/ThemeContext.jsx'
import SelectPatient from '@/pages/SelectPatient.jsx'

vi.mock('@/api/client.js', () => ({ getPatients: vi.fn() }))
import { getPatients } from '@/api/client.js'

const PATIENTS = [
  { id: 1, name: 'Patient 1 — Low Risk', description: 'Post-surgical', color: '#28a745', risk_level: 'Low Risk', sofa_range: '0 – 4' },
  { id: 2, name: 'Patient 2 — Moderate Risk', description: 'Pneumonia', color: '#e6a817', risk_level: 'Moderate Risk', sofa_range: '5 – 9' },
  { id: 3, name: 'Patient 3 — High Risk', description: 'Septic shock', color: '#dc3545', risk_level: 'High Risk', sofa_range: '10 – 24' },
]

const renderPage = () => render(
  <ThemeProvider>
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MemoryRouter initialEntries={['/']}>
        <Routes>
          <Route path="/" element={<SelectPatient />} />
          <Route path="/monitor/:patientId" element={<div data-testid="monitor-page" />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  </ThemeProvider>
)

const setReducedMotion = (reduced) => {
  window.matchMedia.mockImplementation(query => ({
    matches: reduced && query.includes('prefers-reduced-motion'),
    media: query, onchange: null,
    addListener: vi.fn(), removeListener: vi.fn(),
    addEventListener: vi.fn(), removeEventListener: vi.fn(), dispatchEvent: vi.fn(),
  }))
}

beforeEach(() => {
  getPatients.mockReset().mockResolvedValue({ data: PATIENTS })
  try { localStorage.clear() } catch { /* ignore */ }
  setReducedMotion(false)
})
afterEach(() => { vi.useRealTimers() })

describe('theme', () => {
  const toggle = () => screen.getByRole('button', { name: /Switch to (light|dark) mode/ })

  it('flips between dark and light and updates the button each time', async () => {
    renderPage()
    await screen.findByText('Patient 1 — Low Risk')
    const first = toggle().getAttribute('aria-label')
    fireEvent.click(toggle())
    expect(toggle().getAttribute('aria-label')).not.toBe(first)
    fireEvent.click(toggle())
    expect(toggle().getAttribute('aria-label')).toBe(first)
  })

  it('the page background colour follows the theme', async () => {
    const { container } = renderPage()
    await screen.findByText('Patient 1 — Low Risk')
    const bg = () => container.firstChild.style.backgroundColor
    const before = bg()
    fireEvent.click(toggle())
    expect(bg()).not.toBe(before)
    expect(['rgb(4, 12, 26)', 'rgb(240, 246, 255)']).toContain(bg())
  })
})

describe('patient card interaction', () => {
  const cardFor = (n) => screen.getByRole('button', { name: new RegExp(`Monitor Patient ${n}`) })

  it('Enter on a focused card opens that patient', async () => {
    renderPage()
    await screen.findByText('Patient 2 — Moderate Risk')
    fireEvent.keyDown(cardFor(2), { key: 'Enter' })
    expect(await screen.findByTestId('monitor-page')).toBeInTheDocument()
  })

  it('other keys do not navigate', async () => {
    renderPage()
    await screen.findByText('Patient 2 — Moderate Risk')
    fireEvent.keyDown(cardFor(2), { key: 'a' })
    fireEvent.keyDown(cardFor(2), { key: ' ' })
    expect(screen.queryByTestId('monitor-page')).not.toBeInTheDocument()
  })

  it('cards are keyboard-focusable', async () => {
    renderPage()
    await screen.findByText('Patient 1 — Low Risk')
    for (const n of [1, 2, 3]) expect(cardFor(n)).toHaveAttribute('tabindex', '0')
  })

  it('the "Open Monitoring Dashboard" button navigates without double-firing the card click', async () => {
    renderPage()
    await screen.findByText('Patient 3 — High Risk')
    const buttons = screen.getAllByRole('button', { name: /Open Monitoring Dashboard/ })
    expect(buttons).toHaveLength(3)
    fireEvent.click(buttons[2])
    expect(await screen.findByTestId('monitor-page')).toBeInTheDocument()
  })

  it('hovering a card raises its shadow and leaving restores it', async () => {
    renderPage()
    await screen.findByText('Patient 1 — Low Risk')
    const card = cardFor(1)
    const resting = card.style.boxShadow
    fireEvent.mouseEnter(card)
    expect(card.style.boxShadow).not.toBe(resting)
    fireEvent.mouseLeave(card)
    expect(card.style.boxShadow).toBe(resting)
  })

  it('hovering the button fills it and leaving restores the style', async () => {
    renderPage()
    await screen.findByText('Patient 1 — Low Risk')
    const btn = screen.getAllByRole('button', { name: /Open Monitoring Dashboard/ })[0]
    const resting = { color: btn.style.color, shadow: btn.style.boxShadow }
    fireEvent.mouseEnter(btn)
    expect(btn.style.color).toBe('rgb(255, 255, 255)')
    expect(btn.style.boxShadow).not.toBe('none')
    fireEvent.mouseLeave(btn)
    expect(btn.style.color).toBe(resting.color)
    expect(btn.style.boxShadow).toBe('none')
  })

  it('shows each patient\'s risk, SOFA range and bay', async () => {
    renderPage()
    await screen.findByText('Patient 1 — Low Risk')
    expect(screen.getByText('0 – 4')).toBeInTheDocument()
    expect(screen.getByText('10 – 24')).toBeInTheDocument()
    expect(screen.getByText('ICU Bay 2')).toBeInTheDocument()
    expect(screen.getByText('High Risk')).toBeInTheDocument()
  })
})

describe('status bar and clock', () => {
  it('shows the number of patients that loaded', async () => {
    renderPage()
    await screen.findByText('Patient 1 — Low Risk')
    expect(screen.getByText('Active Patients', { exact: false }).textContent).toMatch(/3/)
  })

  it('assumes 3 patients while loading', () => {
    getPatients.mockReturnValue(new Promise(() => {}))
    renderPage()
    expect(screen.getByText('Active Patients', { exact: false }).textContent).toMatch(/3/)
  })

  it('shows the real count when fewer patients come back', async () => {
    getPatients.mockResolvedValue({ data: PATIENTS.slice(0, 2) })
    renderPage()
    await screen.findByText('Patient 2 — Moderate Risk')
    expect(screen.getByText('Active Patients', { exact: false }).textContent).toMatch(/2/)
  })

  it('the clock ticks every second', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'Date'] })
    vi.setSystemTime(new Date('2026-11-24T10:00:00'))
    renderPage()
    const timeText = () => document.querySelector('nav span[style*="var(--mono)"]').textContent
    const before = timeText()
    await act(async () => { vi.advanceTimersByTime(3000) })
    expect(timeText()).not.toBe(before)
  })

  it('stops the clock when the page unmounts', () => {
    const clear = vi.spyOn(globalThis, 'clearInterval')
    const { unmount } = renderPage()
    unmount()
    expect(clear).toHaveBeenCalled()
  })

  it('shows the all-systems-operational banner and ward', async () => {
    renderPage()
    expect(screen.getByText(/All Systems Operational/)).toBeInTheDocument()
    expect(screen.getByText(/ICU Ward A/)).toBeInTheDocument()
  })
})

describe('reduced motion', () => {
  it('shows the hero straight away when the user prefers reduced motion', () => {
    setReducedMotion(true)
    const { container } = renderPage()
    const faded = [...container.querySelectorAll('[style*="opacity"]')]
    expect(faded.some(el => el.style.opacity === '1')).toBe(true)
  })

  it('plays the intro animation when motion is allowed', async () => {
    const { createTimeline } = await import('animejs')
    createTimeline.mockClear()
    renderPage()
    expect(createTimeline).toHaveBeenCalled()
  })

  it('skips the intro animation when motion is reduced', async () => {
    const { createTimeline } = await import('animejs')
    createTimeline.mockClear()
    setReducedMotion(true)
    renderPage()
    expect(createTimeline).not.toHaveBeenCalled()
  })
})

describe('error and loading', () => {
  it('shows skeleton cards, not patients, while loading', () => {
    getPatients.mockReturnValue(new Promise(() => {}))
    const { container } = renderPage()
    expect(screen.queryByRole('button', { name: /Monitor Patient/ })).not.toBeInTheDocument()
    expect(container.querySelectorAll('.rounded-2xl.overflow-hidden').length).toBeGreaterThanOrEqual(3)
  })

  it('shows the API error banner and no cards when the request fails', async () => {
    getPatients.mockRejectedValue(new Error('down'))
    renderPage()
    expect(await screen.findByText(/Unable to load patients/)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Monitor Patient/ })).not.toBeInTheDocument()
  })

  it('renders an empty list without crashing', async () => {
    getPatients.mockResolvedValue({ data: [] })
    renderPage()
    await waitFor(() => expect(screen.getByRole('list', { name: 'Patient list' })).toBeInTheDocument())
    expect(screen.queryByRole('button', { name: /Monitor Patient/ })).not.toBeInTheDocument()
  })
})
