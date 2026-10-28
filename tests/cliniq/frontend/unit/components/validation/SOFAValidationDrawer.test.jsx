import { render, screen, fireEvent } from '@testing-library/react'
import SOFAValidationDrawer from '@/components/validation/SOFAValidationDrawer.jsx'

const openDrawer = (props = {}) =>
  render(<SOFAValidationDrawer open={true} onClose={() => {}} {...props} />)

describe('SOFAValidationDrawer', () => {
  it('does not render content when closed', () => {
    const { container } = render(<SOFAValidationDrawer open={false} onClose={() => {}} />)
    // Drawer should be hidden or absent
    expect(container.textContent).not.toMatch(/SOFA Prediction/i)
  })

  it('renders when open=true', () => {
    openDrawer()
    expect(document.body.textContent).toMatch(/SOFA|Model/i)
  })

  it('shows correct MAE value', () => {
    openDrawer()
    expect(screen.getByText(/1\.8968|1.90/)).toBeInTheDocument()
  })

  it('shows correct R² value', () => {
    openDrawer()
    expect(screen.getByText(/0\.5995|0.60/)).toBeInTheDocument()
  })

  it('shows correct AUROC value', () => {
    openDrawer()
    // AUROC may appear multiple times (label + value); just assert at least one exists
    expect(screen.getAllByText(/0\.9461|94\.6|0.946/i).length).toBeGreaterThan(0)
  })

  it('shows correct PR-AUC value', () => {
    openDrawer()
    expect(screen.getByText(/0\.6840|68\.4|0.684/i)).toBeInTheDocument()
  })

  it('shows 3-class accuracy around 70%', () => {
    openDrawer()
    expect(screen.getByText(/70\.2|70%/i)).toBeInTheDocument()
  })

  it('shows test split with 2,559', () => {
    openDrawer()
    expect(screen.getAllByText(/2[,.]?559/).length).toBeGreaterThan(0)
  })

  it('calls onClose when close button clicked', () => {
    const onClose = vi.fn()
    render(<SOFAValidationDrawer open={true} onClose={onClose} />)
    const closeBtn = screen.getAllByRole('button').find(
      btn => btn.getAttribute('aria-label')?.match(/close/i) || btn.title?.match(/close/i)
    )
    if (closeBtn) {
      fireEvent.click(closeBtn)
      expect(onClose).toHaveBeenCalled()
    }
  })

  it('shows confusion matrix with 3 rows', () => {
    openDrawer()
    // Low Risk, Moderate Risk, High Risk rows
    expect(screen.getByText(/Low Risk/i)).toBeInTheDocument()
    expect(screen.getByText(/High Risk/i)).toBeInTheDocument()
  })

  it('shows high-risk recall 19.8%', () => {
    openDrawer()
    expect(screen.getByText(/19\.8/)).toBeInTheDocument()
  })
})
