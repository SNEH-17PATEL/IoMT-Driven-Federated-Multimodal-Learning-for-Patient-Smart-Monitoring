import { render, screen, fireEvent } from '@testing-library/react'
import LLMValidationFAB from '@/components/validation/LLMValidationFAB.jsx'

describe('LLMValidationFAB', () => {
  it('renders button with correct aria-label', () => {
    render(<LLMValidationFAB onClick={() => {}} hasData={true} />)
    expect(screen.getByRole('button', { name: /Open AI Report Validation/i })).toBeInTheDocument()
  })

  it('calls onClick when hasData=true and clicked', () => {
    const onClick = vi.fn()
    render(<LLMValidationFAB onClick={onClick} hasData={true} />)
    fireEvent.click(screen.getByRole('button'))
    expect(onClick).toHaveBeenCalledTimes(1)
  })

  it('is visually disabled (opacity + pointer-events) when hasData=false', () => {
    render(<LLMValidationFAB onClick={() => {}} hasData={false} />)
    const btn = screen.getByRole('button')
    // Component sets inline style opacity:0.45 and pointerEvents:'none' when !hasData
    expect(btn.style.opacity).toBe('0.45')
  })

  it('shows full opacity when hasData=true', () => {
    render(<LLMValidationFAB onClick={() => {}} hasData={true} />)
    const btn = screen.getByRole('button')
    expect(btn.style.opacity).toBe('1')
  })
})
