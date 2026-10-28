import { render, screen, fireEvent } from '@testing-library/react'
import HistoryFAB from '@/components/history/HistoryFAB.jsx'

describe('HistoryFAB', () => {
  it('renders button with correct aria-label', () => {
    render(<HistoryFAB onClick={() => {}} recordCount={0} />)
    expect(screen.getByRole('button', { name: /Open patient history/i })).toBeInTheDocument()
  })

  it('calls onClick when clicked', () => {
    const onClick = vi.fn()
    render(<HistoryFAB onClick={onClick} recordCount={5} />)
    fireEvent.click(screen.getByRole('button'))
    expect(onClick).toHaveBeenCalledTimes(1)
  })

  it('renders without crash when recordCount is 0', () => {
    render(<HistoryFAB onClick={() => {}} recordCount={0} />)
    expect(screen.getByRole('button')).toBeInTheDocument()
  })

  it('renders without crash when recordCount > 0', () => {
    render(<HistoryFAB onClick={() => {}} recordCount={12} />)
    expect(screen.getByRole('button')).toBeInTheDocument()
  })

  it('renders SVG icon inside the button regardless of recordCount', () => {
    // The count badge only shows on hover (framer-motion onHoverStart, untestable here)
    // We confirm the FAB button renders the clock SVG icon at initial (non-hovered) state
    const { container } = render(<HistoryFAB onClick={() => {}} recordCount={5} />)
    expect(container.querySelector('button svg')).toBeTruthy()
  })
})
