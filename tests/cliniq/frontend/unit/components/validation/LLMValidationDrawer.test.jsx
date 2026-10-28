import { render, screen, fireEvent } from '@testing-library/react'
import LLMValidationDrawer from '@/components/validation/LLMValidationDrawer.jsx'

const reportData = {
  sections: {
    'CURRENT CONDITION': 'Patient is stable.',
    'PROBABLE CAUSE': 'Suspected sepsis.',
    'RISK FORECAST': 'Rising SOFA.',
    'IMMEDIATE ACTIONS': 'Administer antibiotics.',
  },
  reliability_score: 0.82,
  reliability_label: 'High',
  reliability_color: '#28a745',
  caution: null,
  judge_used: true,
}

describe('LLMValidationDrawer', () => {
  it('does not render when open=false', () => {
    const { container } = render(
      <LLMValidationDrawer open={false} onClose={() => {}} reportData={reportData} />
    )
    expect(container.textContent).not.toMatch(/9 \/ 9|methods ran/i)
  })

  it('renders when open=true', () => {
    render(<LLMValidationDrawer open={true} onClose={() => {}} reportData={reportData} />)
    expect(document.body.textContent).toMatch(/validation|method|quality/i)
  })

  it('shows "9 / 9 methods ran" when judge_used=true', () => {
    render(<LLMValidationDrawer open={true} onClose={() => {}} reportData={{ ...reportData, judge_used: true }} />)
    expect(screen.getByText(/9 \/ 9 methods ran/i)).toBeInTheDocument()
  })

  it('shows "7 / 9 methods ran" when judge_used=false', () => {
    render(<LLMValidationDrawer open={true} onClose={() => {}} reportData={{ ...reportData, judge_used: false }} />)
    expect(screen.getByText(/7 \/ 9 methods ran/i)).toBeInTheDocument()
  })

  it('shows warning when judge_used=false', () => {
    render(<LLMValidationDrawer open={true} onClose={() => {}} reportData={{ ...reportData, judge_used: false }} />)
    expect(screen.getByText(/Judge LLM unavailable/i)).toBeInTheDocument()
  })

  it('renders without crash when reportData is null', () => {
    render(<LLMValidationDrawer open={true} onClose={() => {}} reportData={null} />)
    expect(document.body).toBeTruthy()
  })

  it('shows all 3 tab labels: Overview, 9 Methods, Findings', () => {
    render(<LLMValidationDrawer open={true} onClose={() => {}} reportData={reportData} />)
    expect(screen.getByText('Overview')).toBeInTheDocument()
    expect(screen.getByText('9 Methods')).toBeInTheDocument()
    expect(screen.getByText('Findings')).toBeInTheDocument()
  })

  it('calls onClose when the close button is clicked', () => {
    const onClose = vi.fn()
    render(<LLMValidationDrawer open={true} onClose={onClose} reportData={reportData} />)
    fireEvent.click(screen.getByRole('button', { name: /close panel/i }))
    expect(onClose).toHaveBeenCalledTimes(1)
  })
})
