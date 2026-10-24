import { useTheme } from '../../contexts/ThemeContext.jsx'

function SunIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 20 20" fill="none" stroke="currentColor"
      strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
      <circle cx="10" cy="10" r="4"/>
      <line x1="10" y1="1.5" x2="10" y2="3.5"/>
      <line x1="10" y1="16.5" x2="10" y2="18.5"/>
      <line x1="1.5" y1="10" x2="3.5" y2="10"/>
      <line x1="16.5" y1="10" x2="18.5" y2="10"/>
      <line x1="4.1" y1="4.1" x2="5.5" y2="5.5"/>
      <line x1="14.5" y1="14.5" x2="15.9" y2="15.9"/>
      <line x1="4.1" y1="15.9" x2="5.5" y2="14.5"/>
      <line x1="14.5" y1="5.5" x2="15.9" y2="4.1"/>
    </svg>
  )
}

function MoonIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 20 20" fill="none" stroke="currentColor"
      strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
      <path d="M17.5 12.5A8 8 0 0 1 7.5 2.5a7 7 0 1 0 10 10z"/>
    </svg>
  )
}

export default function ThemeToggle({ compact = false }) {
  const { theme, toggleTheme } = useTheme()
  const isDark = theme === 'dark'

  return (
    <button
      onClick={toggleTheme}
      aria-label={`Switch to ${isDark ? 'light' : 'dark'} mode`}
      title={`Switch to ${isDark ? 'light' : 'dark'} mode`}
      className="flex items-center gap-1.5 rounded-lg transition-all duration-200"
      style={{
        padding: compact ? '6px 8px' : '6px 12px',
        background: 'var(--s2)',
        border: '1px solid var(--b2)',
        color: 'var(--t2)',
        cursor: 'pointer',
        fontSize: '12px',
        fontWeight: 600,
      }}
      onMouseEnter={e => {
        e.currentTarget.style.borderColor = 'var(--blue-l)'
        e.currentTarget.style.color = 'var(--blue-l)'
        e.currentTarget.style.background = 'var(--s3)'
      }}
      onMouseLeave={e => {
        e.currentTarget.style.borderColor = 'var(--b2)'
        e.currentTarget.style.color = 'var(--t2)'
        e.currentTarget.style.background = 'var(--s2)'
      }}
    >
      {isDark ? <SunIcon /> : <MoonIcon />}
      {!compact && (
        <span style={{ userSelect: 'none' }}>
          {isDark ? 'Light' : 'Dark'}
        </span>
      )}
    </button>
  )
}
