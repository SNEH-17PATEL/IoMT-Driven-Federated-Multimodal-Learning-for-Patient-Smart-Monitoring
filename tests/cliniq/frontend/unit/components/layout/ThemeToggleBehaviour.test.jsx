import { render, screen, fireEvent } from '@testing-library/react'
import { ThemeProvider } from '@/contexts/ThemeContext.jsx'
import ThemeToggle from '@/components/layout/ThemeToggle.jsx'

const setup = (props) => render(<ThemeProvider><ThemeToggle {...props} /></ThemeProvider>)

describe('ThemeToggle behaviour', () => {
  beforeEach(() => { try { localStorage.clear() } catch { /* ignore */ } })

  it('hover highlights the button and leaving restores it', () => {
    setup()
    const btn = screen.getByRole('button')
    fireEvent.mouseEnter(btn)
    expect(btn.style.borderColor).toBe('var(--blue-l)')
    expect(btn.style.color).toBe('var(--blue-l)')
    expect(btn.style.background).toBe('var(--s3)')
    fireEvent.mouseLeave(btn)
    expect(btn.style.borderColor).toBe('var(--b2)')
    expect(btn.style.color).toBe('var(--t2)')
    expect(btn.style.background).toBe('var(--s2)')
  })

  it('click flips the label and the aria-label', () => {
    setup()
    const btn = screen.getByRole('button')
    const before = btn.getAttribute('aria-label')
    fireEvent.click(btn)
    expect(btn.getAttribute('aria-label')).not.toBe(before)
    fireEvent.click(btn)
    expect(btn.getAttribute('aria-label')).toBe(before)
  })

  it('aria-label and title always agree', () => {
    setup()
    const btn = screen.getByRole('button')
    expect(btn.getAttribute('title')).toBe(btn.getAttribute('aria-label'))
    fireEvent.click(btn)
    expect(btn.getAttribute('title')).toBe(btn.getAttribute('aria-label'))
  })

  it('compact mode uses tighter padding', () => {
    const { rerender } = setup({ compact: false })
    const pad = () => screen.getByRole('button').style.padding
    const full = pad()
    rerender(<ThemeProvider><ThemeToggle compact /></ThemeProvider>)
    expect(pad()).not.toBe(full)
  })
})
