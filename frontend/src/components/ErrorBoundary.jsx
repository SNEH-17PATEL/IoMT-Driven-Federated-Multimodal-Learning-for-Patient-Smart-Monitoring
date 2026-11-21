import { Component } from 'react'

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  componentDidCatch(error, info) {
    console.error('[ClinIQ] Render error:', error, info.componentStack)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div
          style={{
            minHeight: '100vh',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 16,
            background: 'var(--base, #060d18)',
            color: 'var(--t1, #e8edf4)',
            fontFamily: 'var(--font-display, system-ui, sans-serif)',
            padding: 32,
            textAlign: 'center',
          }}
        >
          <svg width="48" height="48" viewBox="0 0 48 48" fill="none" stroke="var(--danger-l, #f85149)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M24 6L42 38H6L24 6z"/>
            <line x1="24" y1="20" x2="24" y2="29"/>
            <circle cx="24" cy="33" r="1" fill="currentColor" stroke="none"/>
          </svg>
          <h1 style={{ fontSize: 20, fontWeight: 700, margin: 0 }}>Something went wrong</h1>
          <p style={{ fontSize: 13, color: 'var(--t3, rgba(232,237,244,0.45))', maxWidth: 400, margin: 0 }}>
            An unexpected error occurred in the dashboard. Reload the page to continue.
          </p>
          <button
            onClick={() => window.location.reload()}
            style={{
              marginTop: 8,
              padding: '10px 24px',
              borderRadius: 10,
              background: 'var(--blue-l, #4fa3e8)',
              color: '#fff',
              border: 'none',
              fontWeight: 700,
              fontSize: 14,
              cursor: 'pointer',
              fontFamily: 'inherit',
            }}
          >
            Reload
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
