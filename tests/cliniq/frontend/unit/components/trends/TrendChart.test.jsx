import { render, screen } from '@testing-library/react'
import TrendChart from '@/components/trends/TrendChart.jsx'

const vitalsData = [
  { time: '12:00', HR: 72, RR: 16, SpO2: 98, Temp: 37, SBP: 115, DBP: 72, MAP: 86 },
  { time: '12:30', HR: 78, RR: 18, SpO2: 97, Temp: 37.1, SBP: 118, DBP: 75, MAP: 89 },
]

describe('TrendChart', () => {
  it('renders without crash with valid data', () => {
    render(<TrendChart data={vitalsData} vital="HR" />)
    expect(document.body).toBeTruthy()
  })

  it('renders chart container', () => {
    const { container } = render(<TrendChart data={vitalsData} vital="HR" />)
    expect(container.firstChild).not.toBeNull()
  })

  it('handles empty data array gracefully', () => {
    render(<TrendChart data={[]} vital="HR" />)
    expect(document.body).toBeTruthy()
  })

  it('handles null data gracefully', () => {
    render(<TrendChart data={null} vital="HR" />)
    expect(document.body).toBeTruthy()
  })

  it('renders with SpO2 vital', () => {
    render(<TrendChart data={vitalsData} vital="SpO2" />)
    expect(document.body).toBeTruthy()
  })
})
