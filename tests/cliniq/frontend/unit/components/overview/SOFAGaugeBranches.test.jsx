import { render, screen } from '@testing-library/react'
import SOFAGauge from '@/components/overview/SOFAGauge.jsx'
import { animate } from 'animejs'

const data = (over = {}) => ({
  patient_id: 1, sofa_score: 6.4, risk_level: 'Moderate Risk', risk_color: '#e6a817',
  conformal_lo: 3.0, conformal_hi: 9.7, ...over,
})
const confWidth = (c) => c.querySelector('.h-full.rounded-full').style.width

beforeEach(() => { animate.mockClear() })

describe('conformal interval bar', () => {
  it('shows the range and positions the score inside it', () => {
    const { container } = render(<SOFAGauge predictData={data()} isRefreshing={false} />)
    expect(screen.getByText('3.0 – 9.7')).toBeInTheDocument()
    // (6.4 - 3) / (9.7 - 3) = 50.7%
    expect(parseFloat(confWidth(container))).toBeCloseTo(50.7, 0)
  })

  it('caps the bar at 100% when the score is above the interval', () => {
    const { container } = render(<SOFAGauge predictData={data({ sofa_score: 20 })} isRefreshing={false} />)
    expect(confWidth(container)).toBe('100%')
  })

  it.each([
    ['missing bounds', { conformal_lo: undefined, conformal_hi: undefined }],
    ['an empty interval', { conformal_lo: 5, conformal_hi: 5 }],
    ['an inverted interval', { conformal_lo: 9, conformal_hi: 3 }],
  ])('draws an empty bar for %s', (_n, over) => {
    const { container } = render(<SOFAGauge predictData={data(over)} isRefreshing={false} />)
    expect(confWidth(container)).toBe('0%')
  })

  it('treats a missing score as 0 inside the interval maths', () => {
    const { container } = render(<SOFAGauge predictData={data({ sofa_score: undefined, conformal_lo: 0, conformal_hi: 10 })} isRefreshing={false} />)
    expect(confWidth(container)).toBe('0%')
  })
})

describe('risk badge and colours', () => {
  it('shows the risk level', () => {
    render(<SOFAGauge predictData={data()} isRefreshing={false} />)
    expect(screen.getByText('Moderate Risk')).toBeInTheDocument()
    expect(screen.getByText('out of 24')).toBeInTheDocument()
  })

  it('has the ambient glow ring only for Moderate and High risk', () => {
    const glow = (c) => c.querySelectorAll('circle[style*="blur"]').length
    const { container, rerender } = render(<SOFAGauge predictData={data()} isRefreshing={false} />)
    expect(glow(container)).toBe(1)
    rerender(<SOFAGauge predictData={data({ risk_color: '#28a745' })} isRefreshing={false} />)
    expect(glow(container)).toBe(0)
  })
})

describe('score animation', () => {
  it('animates from 0 to the score on first load', () => {
    render(<SOFAGauge predictData={data({ sofa_score: 8 })} isRefreshing={false} />)
    expect(animate).toHaveBeenCalledTimes(1)
    expect(animate.mock.calls[0][1]).toMatchObject({ val: 8 })
  })

  it('does not animate without a score', () => {
    render(<SOFAGauge predictData={data({ sofa_score: undefined })} isRefreshing={false} />)
    expect(animate).not.toHaveBeenCalled()
  })

  it('animates from the previous score on the next reading', () => {
    const { rerender } = render(<SOFAGauge predictData={data({ sofa_score: 4 })} isRefreshing={false} />)
    rerender(<SOFAGauge predictData={data({ sofa_score: 9 })} isRefreshing={false} />)
    expect(animate.mock.calls.at(-1)[0].val).toBe(4)
    expect(animate.mock.calls.at(-1)[1].val).toBe(9)
  })

  it('restarts the count from 0 when the patient changes', () => {
    const { container, rerender } = render(<SOFAGauge predictData={data({ patient_id: 1, sofa_score: 12 })} isRefreshing={false} />)
    rerender(<SOFAGauge predictData={data({ patient_id: 2, sofa_score: 3 })} isRefreshing={false} />)
    // the second animation starts from 0, not from patient 1's 12
    expect(animate.mock.calls.at(-1)[0].val).toBe(0)
    expect(container.textContent).toContain('out of 24')
  })

  it('pauses the animation when unmounted', () => {
    const pause = vi.fn()
    animate.mockReturnValueOnce({ pause })
    const { unmount } = render(<SOFAGauge predictData={data()} isRefreshing={false} />)
    unmount()
    expect(pause).toHaveBeenCalled()
  })
})

describe('reduced motion', () => {
  it('fills the arc straight away when the user prefers reduced motion', () => {
    window.matchMedia.mockImplementation(q => ({
      matches: q.includes('prefers-reduced-motion'), media: q, addListener() {}, removeListener() {},
      addEventListener() {}, removeEventListener() {}, dispatchEvent() {},
    }))
    const { container } = render(<SOFAGauge predictData={data({ sofa_score: 12 })} isRefreshing={false} />)
    const arcs = [...container.querySelectorAll('circle[stroke-dasharray]')]
    expect(arcs.length).toBeGreaterThan(0)
  })
})
