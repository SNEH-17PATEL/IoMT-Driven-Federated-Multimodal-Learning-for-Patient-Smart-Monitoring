import { render, screen } from '@testing-library/react'
import CVMonitorModal from '@/components/cv/CVMonitorModal.jsx'

describe('CVMonitorModal', () => {
  it('does not render when open=false', () => {
    const { container } = render(<CVMonitorModal open={false} onClose={() => {}} onData={() => {}} />)
    expect(container.textContent).not.toMatch(/CV Monitor|Live CV|supplementary/i)
  })

  it('renders when open=true', () => {
    render(<CVMonitorModal open={true} onClose={() => {}} onData={() => {}} />)
    expect(document.body.textContent).toMatch(/CV|visual|supplement/i)
  })

  it('shows supplementary label text', () => {
    render(<CVMonitorModal open={true} onClose={() => {}} onData={() => {}} />)
    expect(screen.getByText(/[Ss]upplementary/i)).toBeInTheDocument()
  })

  it('shows SOFA prediction disclaimer', () => {
    render(<CVMonitorModal open={true} onClose={() => {}} onData={() => {}} />)
    expect(screen.getByText(/SOFA prediction uses dataset values/i)).toBeInTheDocument()
  })
})
