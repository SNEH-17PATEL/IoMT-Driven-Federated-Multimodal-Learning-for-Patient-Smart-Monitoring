import { render, screen, fireEvent } from '@testing-library/react'
import ErrorBoundary from '@/components/ErrorBoundary.jsx'

function Boom({ message = 'kaboom' }) {
  throw new Error(message)
}

describe('ErrorBoundary', () => {
  let errorSpy
  beforeEach(() => {
    // React and the boundary both log the caught error; keep test output clean
    errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
  })
  afterEach(() => errorSpy.mockRestore())

  it('renders its children when nothing goes wrong', () => {
    render(<ErrorBoundary><p>all good</p></ErrorBoundary>)
    expect(screen.getByText('all good')).toBeInTheDocument()
    expect(screen.queryByText('Something went wrong')).not.toBeInTheDocument()
  })

  it('shows the fallback screen when a child throws while rendering', () => {
    render(<ErrorBoundary><Boom /></ErrorBoundary>)
    expect(screen.getByRole('heading', { name: 'Something went wrong' })).toBeInTheDocument()
    expect(screen.getByText(/Reload the page to continue/i)).toBeInTheDocument()
  })

  it('does not render the broken child any more', () => {
    render(<ErrorBoundary><p>should vanish</p><Boom /></ErrorBoundary>)
    expect(screen.queryByText('should vanish')).not.toBeInTheDocument()
  })

  it('logs the error with the ClinIQ prefix', () => {
    render(<ErrorBoundary><Boom message="specific failure" /></ErrorBoundary>)
    const call = errorSpy.mock.calls.find(c => c[0] === '[ClinIQ] Render error:')
    expect(call).toBeTruthy()
    expect(call[1].message).toBe('specific failure')
    expect(typeof call[2]).toBe('string')            // the component stack
  })

  it('Reload button reloads the page', () => {
    const reload = vi.fn()
    const original = window.location
    Object.defineProperty(window, 'location', { configurable: true, value: { ...original, reload } })
    try {
      render(<ErrorBoundary><Boom /></ErrorBoundary>)
      fireEvent.click(screen.getByRole('button', { name: 'Reload' }))
      expect(reload).toHaveBeenCalledTimes(1)
    } finally {
      Object.defineProperty(window, 'location', { configurable: true, value: original })
    }
  })
})
