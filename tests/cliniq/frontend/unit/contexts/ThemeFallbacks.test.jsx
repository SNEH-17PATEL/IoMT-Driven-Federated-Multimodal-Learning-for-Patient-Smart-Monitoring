/*
 * ThemeProvider start-up: which theme wins (saved choice > OS preference > light),
 * and what happens when storage is unavailable.
 */
import { render, screen, fireEvent } from '@testing-library/react'
import { ThemeProvider, useTheme } from '@/contexts/ThemeContext.jsx'

function Probe() {
  const { theme, toggleTheme } = useTheme()
  return <button onClick={toggleTheme}>{theme}</button>
}
const mount = () => render(<ThemeProvider><Probe /></ThemeProvider>)
const prefersDark = (dark) => window.matchMedia.mockImplementation(q => ({
  matches: dark && q.includes('prefers-color-scheme: dark'), media: q, onchange: null,
  addListener: vi.fn(), removeListener: vi.fn(), addEventListener: vi.fn(), removeEventListener: vi.fn(), dispatchEvent: vi.fn(),
}))

beforeEach(() => { localStorage.clear(); prefersDark(false) })
afterEach(() => { vi.restoreAllMocks() })

describe('initial theme', () => {
  it('uses the saved "dark" choice', () => {
    localStorage.setItem('cliniq-theme', 'dark')
    mount()
    expect(screen.getByRole('button')).toHaveTextContent('dark')
  })

  it('uses the saved "light" choice even when the OS prefers dark', () => {
    localStorage.setItem('cliniq-theme', 'light')
    prefersDark(true)
    mount()
    expect(screen.getByRole('button')).toHaveTextContent('light')
  })

  it('falls back to the OS preference when nothing is saved', () => {
    prefersDark(true)
    mount()
    expect(screen.getByRole('button')).toHaveTextContent('dark')
  })

  it('falls back to light when nothing is saved and the OS has no dark preference', () => {
    mount()
    expect(screen.getByRole('button')).toHaveTextContent('light')
  })

  it('ignores a garbage saved value', () => {
    localStorage.setItem('cliniq-theme', 'purple')
    prefersDark(true)
    mount()
    expect(screen.getByRole('button')).toHaveTextContent('dark')
  })
})

describe('persistence', () => {
  it('saves the choice and sets the data-theme attribute on every change', () => {
    mount()
    fireEvent.click(screen.getByRole('button'))
    expect(localStorage.getItem('cliniq-theme')).toBe('dark')
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark')
    fireEvent.click(screen.getByRole('button'))
    expect(localStorage.getItem('cliniq-theme')).toBe('light')
    expect(document.documentElement.getAttribute('data-theme')).toBe('light')
  })

  it('still works when reading storage throws', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('blocked') })
    prefersDark(true)
    expect(() => mount()).not.toThrow()
    expect(screen.getByRole('button')).toHaveTextContent('dark')
  })

  it('still works when writing storage throws', () => {
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('quota') })
    mount()
    expect(() => fireEvent.click(screen.getByRole('button'))).not.toThrow()
    expect(screen.getByRole('button')).toHaveTextContent('dark')
  })
})

describe('default context (no provider)', () => {
  it('gives a harmless light theme and a no-op toggle', () => {
    render(<Probe />)
    expect(screen.getByRole('button')).toHaveTextContent('light')
    expect(() => fireEvent.click(screen.getByRole('button'))).not.toThrow()
  })
})
