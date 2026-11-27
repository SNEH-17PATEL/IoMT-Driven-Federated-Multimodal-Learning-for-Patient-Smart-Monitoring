/*
 * TrendChart's hover tooltip. recharts is replaced by stubs in the global test setup, so the
 * tooltip never renders by itself. Instead the test captures the element TrendChart hands to
 * <Tooltip content={...}> and renders it directly, the way recharts would on hover.
 */
import { render, screen } from '@testing-library/react'
import { cloneElement } from 'react'

let captured
vi.mock('recharts', () => {
  const stub = ({ children }) => <div data-testid="chart">{children}</div>
  return {
    ResponsiveContainer: stub, LineChart: stub, AreaChart: stub, BarChart: stub,
    Line: () => null, Area: () => null, Bar: () => null, XAxis: () => null, YAxis: () => null,
    CartesianGrid: () => null, Legend: () => null, ReferenceLine: () => null, ReferenceArea: () => null,
    Tooltip: (props) => { captured = props; return null },
  }
})
import TrendChart from '@/components/trends/TrendChart.jsx'

const mountChart = () => {
  render(<TrendChart dataKey="HR" label="Heart Rate" unit="bpm" color="#e03a3a" normalLo={60} normalHi={100}
                     data={[{ time: '2026-01-01 12:00:00', HR: 80 }]} />)
  return captured.content                         // <CustomTooltip unit color />
}
const show = (el, props) => render(cloneElement(el, props))

describe('TrendChart tooltip', () => {
  it('receives the unit and colour of the chart', () => {
    const el = mountChart()
    expect(el.props).toMatchObject({ unit: 'bpm', color: '#e03a3a' })
  })

  it('shows the time label and the value to one decimal with its unit', () => {
    const el = mountChart()
    // render the tooltip alone (not next to the chart header, which also says "bpm")
    const { container } = show(el, { active: true, label: '12:00:05', payload: [{ value: 82.46 }] })
    expect(container).toHaveTextContent('12:00:05')
    expect(container).toHaveTextContent('82.5bpm')
  })

  it('shows a dash when the point has no value', () => {
    const el = mountChart()
    show(el, { active: true, label: 't', payload: [{ value: null }] })
    expect(screen.getByText('—')).toBeInTheDocument()
  })

  it.each([
    ['inactive', { active: false, label: 't', payload: [{ value: 1 }] }],
    ['no payload', { active: true, label: 't', payload: undefined }],
    ['empty payload', { active: true, label: 't', payload: [] }],
  ])('renders nothing when %s', (_n, props) => {
    const el = mountChart()
    const { container } = show(el, props)
    expect(container).toBeEmptyDOMElement()
  })

  it('uses the chart colour for the value', () => {
    const el = mountChart()
    show(el, { active: true, label: 't', payload: [{ value: 90 }] })
    expect(screen.getByText('90.0').style.color).toBe('rgb(224, 58, 58)')
  })
})
