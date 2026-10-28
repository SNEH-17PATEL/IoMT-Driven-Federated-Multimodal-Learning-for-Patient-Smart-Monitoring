import { render, screen, fireEvent } from '@testing-library/react'
import CVMonitorFAB from '@/components/cv/CVMonitorFAB.jsx'

describe('CVMonitorFAB', () => {
  it('renders button with correct aria-label', () => {
    render(<CVMonitorFAB onClick={() => {}} isActive={false} />)
    expect(screen.getByRole('button', { name: /Open Live CV Monitor/i })).toBeInTheDocument()
  })

  it('calls onClick when clicked', () => {
    const onClick = vi.fn()
    render(<CVMonitorFAB onClick={onClick} isActive={false} />)
    fireEvent.click(screen.getByRole('button'))
    expect(onClick).toHaveBeenCalledTimes(1)
  })

  it('renders without crash when isActive=true', () => {
    render(<CVMonitorFAB onClick={() => {}} isActive={true} />)
    expect(screen.getByRole('button')).toBeInTheDocument()
  })

  it('isActive=true renders different HTML than isActive=false', () => {
    const { container: active } = render(<CVMonitorFAB onClick={() => {}} isActive={true} />)
    const { container: inactive } = render(<CVMonitorFAB onClick={() => {}} isActive={false} />)
    // Active state adds a live indicator dot and changes boxShadow — HTML differs
    expect(active.innerHTML).not.toBe(inactive.innerHTML)
  })
})
