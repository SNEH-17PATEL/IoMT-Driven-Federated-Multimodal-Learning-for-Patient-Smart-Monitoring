import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import App from '@/App.jsx'

// The two pages are heavy (queries, polling, charts). The router wiring is what App owns,
// so replace each page with a marker that also shows the URL parameter it received.
vi.mock('@/pages/SelectPatient.jsx', () => ({
  default: () => <div data-testid="page-select">select patient page</div>,
}))
vi.mock('@/pages/MonitorDashboard.jsx', async () => {
  const { useParams } = await import('react-router-dom')
  return {
    default: () => {
      const { patientId } = useParams()
      return <div data-testid="page-monitor">monitor page for {patientId}</div>
    },
  }
})

const renderAt = (path) =>
  render(<MemoryRouter initialEntries={[path]}><App /></MemoryRouter>)

describe('App routing', () => {
  it('shows the patient selection page at "/"', () => {
    renderAt('/')
    expect(screen.getByTestId('page-select')).toBeInTheDocument()
    expect(screen.queryByTestId('page-monitor')).not.toBeInTheDocument()
  })

  it('shows the monitor dashboard at "/monitor/:patientId" and passes the id', () => {
    renderAt('/monitor/2')
    expect(screen.getByTestId('page-monitor')).toHaveTextContent('monitor page for 2')
    expect(screen.queryByTestId('page-select')).not.toBeInTheDocument()
  })

  it('redirects an unknown URL back to the patient selection page', () => {
    renderAt('/does/not/exist')
    expect(screen.getByTestId('page-select')).toBeInTheDocument()
  })

  it('redirects "/monitor" (no id) back to the selection page', () => {
    renderAt('/monitor')
    expect(screen.getByTestId('page-select')).toBeInTheDocument()
  })
})
