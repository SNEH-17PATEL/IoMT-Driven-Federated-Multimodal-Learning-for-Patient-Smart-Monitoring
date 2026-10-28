import { render, screen, fireEvent, act } from '@testing-library/react'
import { ThemeProvider, useTheme } from '@/contexts/ThemeContext.jsx'

function ThemeConsumer() {
  const { theme, toggleTheme } = useTheme()
  return (
    <div>
      <span data-testid="theme">{theme}</span>
      <button onClick={toggleTheme}>Toggle</button>
    </div>
  )
}

describe('ThemeContext', () => {
  it('provides theme value', () => {
    render(<ThemeProvider><ThemeConsumer /></ThemeProvider>)
    const theme = screen.getByTestId('theme').textContent
    expect(['light', 'dark']).toContain(theme)
  })

  it('provides toggleTheme function', () => {
    render(<ThemeProvider><ThemeConsumer /></ThemeProvider>)
    expect(screen.getByRole('button')).toBeInTheDocument()
  })

  it('toggleTheme flips the theme', () => {
    render(<ThemeProvider><ThemeConsumer /></ThemeProvider>)
    const before = screen.getByTestId('theme').textContent
    fireEvent.click(screen.getByRole('button'))
    const after = screen.getByTestId('theme').textContent
    expect(before).not.toBe(after)
  })

  it('toggleTheme twice returns to original', () => {
    render(<ThemeProvider><ThemeConsumer /></ThemeProvider>)
    const original = screen.getByTestId('theme').textContent
    fireEvent.click(screen.getByRole('button'))
    fireEvent.click(screen.getByRole('button'))
    expect(screen.getByTestId('theme').textContent).toBe(original)
  })

  it('sets data-theme on document.documentElement', () => {
    render(<ThemeProvider><ThemeConsumer /></ThemeProvider>)
    const theme = screen.getByTestId('theme').textContent
    expect(document.documentElement.getAttribute('data-theme')).toBe(theme)
  })
})
