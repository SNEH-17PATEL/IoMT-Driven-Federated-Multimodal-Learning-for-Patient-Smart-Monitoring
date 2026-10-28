import { render, screen, fireEvent } from '@testing-library/react'
import ReportToggle from '@/components/report/ReportToggle.jsx'

describe('ReportToggle', () => {
  it('renders Hospital button', () => {
    render(<ReportToggle careMode="hospital" setCareMode={() => {}} />)
    expect(screen.getByText('Hospital')).toBeInTheDocument()
  })

  it('renders Home Care button', () => {
    render(<ReportToggle careMode="hospital" setCareMode={() => {}} />)
    expect(screen.getByText('Home Care')).toBeInTheDocument()
  })

  it('clicking Home Care calls setCareMode with "homecare"', () => {
    const setCareMode = vi.fn()
    render(<ReportToggle careMode="hospital" setCareMode={setCareMode} />)
    fireEvent.click(screen.getByText('Home Care'))
    expect(setCareMode).toHaveBeenCalledWith('homecare')
  })

  it('clicking Hospital calls setCareMode with "hospital"', () => {
    const setCareMode = vi.fn()
    render(<ReportToggle careMode="homecare" setCareMode={setCareMode} />)
    fireEvent.click(screen.getByText('Hospital'))
    expect(setCareMode).toHaveBeenCalledWith('hospital')
  })

  it('renders two buttons', () => {
    render(<ReportToggle careMode="hospital" setCareMode={() => {}} />)
    expect(screen.getAllByRole('button')).toHaveLength(2)
  })

  it('active button has gradient background, inactive has transparent', () => {
    render(<ReportToggle careMode="hospital" setCareMode={() => {}} />)
    const [hospitalBtn, homeCareBtn] = screen.getAllByRole('button')
    // Active (Hospital) button uses a CSS gradient; inactive uses transparent
    expect(hospitalBtn.style.background).toMatch(/gradient/i)
    expect(homeCareBtn.style.background).toBe('transparent')
  })

  it('clicking the already-active button still fires setCareMode', () => {
    // ReportToggle has no guard against redundant calls — always fires
    const setCareMode = vi.fn()
    render(<ReportToggle careMode="hospital" setCareMode={setCareMode} />)
    fireEvent.click(screen.getByText('Hospital'))
    expect(setCareMode).toHaveBeenCalledWith('hospital')
  })
})
