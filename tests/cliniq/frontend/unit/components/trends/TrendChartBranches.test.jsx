/*
 * TrendChart: the real props are dataKey/label/unit/color/normalLo/normalHi.
 * (The older TrendChart.test.jsx passes `vital`, which the component ignores.)
 */
import { render, screen } from '@testing-library/react'
import TrendChart from '@/components/trends/TrendChart.jsx'

const props = { dataKey: 'HR', label: 'Heart Rate', unit: 'bpm', color: '#e03a3a', normalLo: 60, normalHi: 100 }
const series = (...vals) => vals.map((v, i) => ({ time: `2026-01-01 12:00:${String(i).padStart(2, '0')}`, HR: v }))

describe('TrendChart header', () => {
  it('shows the label and its normal range', () => {
    render(<TrendChart {...props} data={series(80)} />)
    expect(screen.getByText('Heart Rate')).toBeInTheDocument()
    expect(screen.getByText('60–100 bpm')).toBeInTheDocument()
  })

  it('shows the latest value as an integer', () => {
    render(<TrendChart {...props} data={series(70, 75, 82)} />)
    expect(screen.getByText('82')).toBeInTheDocument()
  })

  it('shows a decimal latest value with one place', () => {
    render(<TrendChart {...props} data={series(70, 82.46)} />)
    expect(screen.getByText('82.5')).toBeInTheDocument()
  })

  it('shows no latest value for empty data', () => {
    render(<TrendChart {...props} data={[]} />)
    expect(screen.queryByText('bpm', { selector: 'span' })).not.toBeInTheDocument()
  })

  it('shows no latest value when data is null', () => {
    render(<TrendChart {...props} data={null} />)
    expect(screen.getByText('Heart Rate')).toBeInTheDocument()
  })

  it('ignores null readings when picking the latest value', () => {
    render(<TrendChart {...props} data={[{ time: 't', HR: 90 }, { time: 't2', HR: null }]} />)
    expect(screen.getByText('90')).toBeInTheDocument()
  })
})

describe('TrendChart abnormal flag', () => {
  it('flags a latest value above the normal range', () => {
    render(<TrendChart {...props} data={series(80, 120)} />)
    expect(screen.getByText(/↑ abnormal/)).toBeInTheDocument()
  })

  it('flags a latest value below the normal range', () => {
    render(<TrendChart {...props} data={series(80, 40)} />)
    expect(screen.getByText(/↓ abnormal/)).toBeInTheDocument()
  })

  it('does not flag a normal latest value', () => {
    render(<TrendChart {...props} data={series(80, 85)} />)
    expect(screen.queryByText(/abnormal/)).not.toBeInTheDocument()
  })

  it('does not flag the latest value when only an earlier reading was abnormal', () => {
    render(<TrendChart {...props} data={series(130, 85)} />)
    expect(screen.queryByText(/abnormal/)).not.toBeInTheDocument()
  })

  it('tints the border when any reading in the window is abnormal', () => {
    // The component appends a hex alpha ('#e03a3a30'); the browser reports it as rgba(...).
    const { container, rerender } = render(<TrendChart {...props} data={series(130, 85)} />)
    const border = () => container.firstChild.style.border
    expect(border()).toMatch(/rgba\(224, 58, 58/)          // tinted with the chart colour
    rerender(<TrendChart {...props} data={series(80, 85)} />)
    expect(border()).not.toMatch(/rgba\(224, 58, 58/)      // plain neutral border
  })

  it('treats the range edges as normal', () => {
    render(<TrendChart {...props} data={series(60, 100)} />)
    expect(screen.queryByText(/abnormal/)).not.toBeInTheDocument()
  })
})

describe('TrendChart robustness', () => {
  it('renders the chart container for valid data', () => {
    render(<TrendChart {...props} data={series(70, 80)} />)
    expect(screen.getAllByTestId('chart').length).toBeGreaterThan(0)
  })

  it('renders for a single reading', () => {
    expect(() => render(<TrendChart {...props} data={series(75)} />)).not.toThrow()
  })

  it('renders when a reading has no timestamp', () => {
    expect(() => render(<TrendChart {...props} data={[{ HR: 70 }, { HR: 72, time: null }]} />)).not.toThrow()
  })

  it('renders for a time string with no date part', () => {
    expect(() => render(<TrendChart {...props} data={[{ time: '12:00:05', HR: 70 }]} />)).not.toThrow()
  })
})
