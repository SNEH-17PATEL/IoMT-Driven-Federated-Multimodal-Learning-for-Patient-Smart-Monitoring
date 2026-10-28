import { render, screen, fireEvent } from '@testing-library/react'
import SOFAValidationFAB from '@/components/validation/SOFAValidationFAB.jsx'

describe('SOFAValidationFAB', () => {
  it('renders button with correct aria-label', () => {
    render(<SOFAValidationFAB onClick={() => {}} />)
    expect(screen.getByRole('button', { name: /Open SOFA Score Validation/i })).toBeInTheDocument()
  })

  it('calls onClick when clicked', () => {
    const onClick = vi.fn()
    render(<SOFAValidationFAB onClick={onClick} />)
    fireEvent.click(screen.getByRole('button'))
    expect(onClick).toHaveBeenCalledTimes(1)
  })

  it('renders without crash', () => {
    render(<SOFAValidationFAB onClick={() => {}} />)
    expect(document.body).toBeTruthy()
  })
})
