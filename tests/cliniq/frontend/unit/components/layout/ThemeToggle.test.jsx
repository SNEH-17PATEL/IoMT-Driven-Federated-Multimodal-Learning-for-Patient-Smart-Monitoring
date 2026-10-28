import { render, screen, fireEvent } from '@testing-library/react'
import { ThemeProvider } from '@/contexts/ThemeContext.jsx'
import ThemeToggle from '@/components/layout/ThemeToggle.jsx'

const renderWithTheme = (ui) => render(<ThemeProvider>{ui}</ThemeProvider>)

describe('ThemeToggle', () => {
  it('renders a button', () => {
    renderWithTheme(<ThemeToggle />)
    expect(screen.getByRole('button')).toBeInTheDocument()
  })

  it('button has an aria-label', () => {
    renderWithTheme(<ThemeToggle />)
    expect(screen.getByRole('button')).toHaveAttribute('aria-label')
  })

  it('shows text label when compact=false', () => {
    renderWithTheme(<ThemeToggle compact={false} />)
    const btn = screen.getByRole('button')
    // Either "Light" or "Dark" is shown based on current theme
    expect(btn.textContent).toMatch(/Light|Dark/)
  })

  it('hides text label in compact mode', () => {
    renderWithTheme(<ThemeToggle compact={true} />)
    const btn = screen.getByRole('button')
    expect(btn.textContent).not.toMatch(/Light|Dark/)
  })

  it('toggles theme label when clicked', () => {
    renderWithTheme(<ThemeToggle compact={false} />)
    const btn = screen.getByRole('button')
    const before = btn.textContent
    fireEvent.click(btn)
    const after = btn.textContent
    // Text should have changed (dark ↔ light label swap)
    expect(before).not.toBe(after)
  })

  it('shows sun icon (SVG with circle) in dark mode', () => {
    renderWithTheme(<ThemeToggle compact={false} />)
    const btn = screen.getByRole('button')
    // Toggle to dark mode if currently in light mode (shows "Dark" label when in light)
    if (btn.textContent.includes('Dark')) {
      fireEvent.click(btn)
    }
    // SunIcon renders a <circle> element; MoonIcon does not
    expect(btn.querySelector('svg circle')).toBeTruthy()
  })

  it('shows moon icon (SVG without circle) in light mode', () => {
    renderWithTheme(<ThemeToggle compact={false} />)
    const btn = screen.getByRole('button')
    // Toggle to light mode if currently in dark mode (shows "Light" label when in dark)
    if (btn.textContent.includes('Light')) {
      fireEvent.click(btn)
    }
    // MoonIcon has only a <path>, no <circle>
    expect(btn.querySelector('svg circle')).toBeNull()
  })
})
