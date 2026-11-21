import { render, screen } from '@testing-library/react'
import CVMonitorModal from '@/components/cv/CVMonitorModal.jsx'

// No camera or MediaPipe in jsdom: init stops at the model load, so these
// tests cover the static UI and the open/close contract.
describe('CVMonitorModal', () => {
  it('does not render when open=false', () => {
    const { container } = render(<CVMonitorModal open={false} onClose={() => {}} onCVResults={() => {}} />)
    expect(container.textContent).not.toMatch(/CV Monitor|Live CV|supplementary/i)
  })

  it('renders when open=true', () => {
    render(<CVMonitorModal open={true} onClose={() => {}} onCVResults={() => {}} />)
    expect(document.body.textContent).toMatch(/CV|visual|supplement/i)
  })

  it('shows supplementary label text', () => {
    render(<CVMonitorModal open={true} onClose={() => {}} onCVResults={() => {}} />)
    expect(screen.getByText(/[Ss]upplementary/i)).toBeInTheDocument()
  })

  it('states that live readings feed the next SOFA prediction', () => {
    render(<CVMonitorModal open={true} onClose={() => {}} onCVResults={() => {}} />)
    expect(screen.getByText(/Live readings replace dataset GCS Eye & Stress in the next SOFA prediction/i))
      .toBeInTheDocument()
  })

  it('does not show score cards before the camera is running', () => {
    render(<CVMonitorModal open={true} onClose={() => {}} onCVResults={() => {}} />)
    expect(screen.queryByText(/Stress Score\s+\(PSPI\)/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/Rolling 30s/i)).not.toBeInTheDocument()
  })

  it('clears live readings when the monitor closes', () => {
    const onCVResults = vi.fn()
    const { rerender } = render(<CVMonitorModal open={true} onClose={() => {}} onCVResults={onCVResults} />)
    onCVResults.mockClear()
    rerender(<CVMonitorModal open={false} onClose={() => {}} onCVResults={onCVResults} />)
    expect(onCVResults).toHaveBeenCalledWith(null)
  })
})
