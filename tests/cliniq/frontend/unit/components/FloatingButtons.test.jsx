/*
 * Click, state and layout behaviour of the four floating action buttons (History, Camera,
 * LLM validation, SOFA validation). The hover expand / collapse behaviour needs the REAL
 * framer-motion (the global test setup stubs it), so it lives in FloatingButtonsHover.test.jsx.
 */
import { render, screen, fireEvent } from '@testing-library/react'
import HistoryFAB from '@/components/history/HistoryFAB.jsx'
import CVMonitorFAB from '@/components/cv/CVMonitorFAB.jsx'
import LLMValidationFAB from '@/components/validation/LLMValidationFAB.jsx'
import SOFAValidationFAB from '@/components/validation/SOFAValidationFAB.jsx'

describe('floating action buttons — click', () => {
  it.each([
    ['History',    () => <HistoryFAB onClick={vi.fn()} recordCount={3} />],
    ['Camera',     () => <CVMonitorFAB onClick={vi.fn()} isActive={false} />],
    ['LLM',        () => <LLMValidationFAB onClick={vi.fn()} hasData />],
    ['SOFA',       () => <SOFAValidationFAB onClick={vi.fn()} />],
  ])('%s button renders one accessible button with an icon', (_name, make) => {
    const { container } = render(make())
    expect(screen.getAllByRole('button')).toHaveLength(1)
    expect(container.querySelector('button svg')).toBeTruthy()
  })

  it('CVMonitorFAB calls onClick', () => {
    const onClick = vi.fn()
    render(<CVMonitorFAB onClick={onClick} isActive={false} />)
    fireEvent.click(screen.getByRole('button'))
    expect(onClick).toHaveBeenCalledTimes(1)
  })

  it('SOFAValidationFAB calls onClick', () => {
    const onClick = vi.fn()
    render(<SOFAValidationFAB onClick={onClick} />)
    fireEvent.click(screen.getByRole('button', { name: /SOFA/i }))
    expect(onClick).toHaveBeenCalledTimes(1)
  })
})

describe('CVMonitorFAB states', () => {
  it('shows the live indicator dot only while the camera monitor is active', () => {
    const { container, rerender } = render(<CVMonitorFAB onClick={() => {}} isActive={false} />)
    const dot = () => container.querySelector('button span[style*="border-radius: 50%"]')
    expect(dot()).toBeNull()
    rerender(<CVMonitorFAB onClick={() => {}} isActive />)
    expect(dot()).not.toBeNull()
  })

  it('uses the faster blink animation while active', () => {
    const { container, rerender } = render(<CVMonitorFAB onClick={() => {}} isActive={false} />)
    const blink = () => container.querySelector('button > span[style*="eyeBlink"]').style.animation
    expect(blink()).toContain('4.5s')
    rerender(<CVMonitorFAB onClick={() => {}} isActive />)
    expect(blink()).toContain('2.0s')
  })

  it('is positioned second from the bottom (bottom: 88px)', () => {
    render(<CVMonitorFAB onClick={() => {}} isActive={false} />)
    expect(screen.getByRole('button').style.bottom).toBe('88px')
  })
})

describe('LLMValidationFAB states', () => {
  it('is dimmed and ignores pointer events when there is no report yet', () => {
    render(<LLMValidationFAB onClick={() => {}} hasData={false} />)
    const btn = screen.getByRole('button')
    expect(btn.style.opacity).toBe('0.45')
    expect(btn.style.pointerEvents).toBe('none')
  })

  it('is fully interactive once a report exists', () => {
    render(<LLMValidationFAB onClick={() => {}} hasData />)
    const btn = screen.getByRole('button')
    expect(btn.style.opacity).toBe('1')
    expect(btn.style.pointerEvents).toBe('auto')
  })
})

describe('stacking order on the right edge (bottom to top)', () => {
  it('History 28, Camera 88, LLM 148, SOFA 208', () => {
    const { container } = render(
      <>
        <HistoryFAB onClick={() => {}} recordCount={0} />
        <CVMonitorFAB onClick={() => {}} isActive={false} />
        <LLMValidationFAB onClick={() => {}} hasData />
        <SOFAValidationFAB onClick={() => {}} />
      </>
    )
    const bottoms = [...container.querySelectorAll('button')].map(b => b.style.bottom)
    expect(bottoms).toEqual(['28px', '88px', '148px', '208px'])
  })
})
