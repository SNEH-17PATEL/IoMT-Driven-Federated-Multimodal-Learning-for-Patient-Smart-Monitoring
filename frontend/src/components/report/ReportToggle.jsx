const MODES = [
  {
    key: 'hospital',
    label: 'Hospital',
    icon: () => (
      <svg width="11" height="11" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <rect x="2" y="1" width="12" height="14" rx="1.5"/>
        <line x1="8" y1="5" x2="8" y2="11"/>
        <line x1="5" y1="8" x2="11" y2="8"/>
      </svg>
    ),
  },
  {
    key: 'homecare',
    label: 'Home Care',
    icon: () => (
      <svg width="11" height="11" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M1.5 7L8 1.5 14.5 7V14H10V10H6v4H1.5V7z"/>
      </svg>
    ),
  },
]

export default function ReportToggle({ careMode, setCareMode }) {
  return (
    <div
      className="flex items-center p-1 rounded-xl flex-shrink-0"
      style={{
        background: 'var(--s2)',
        border: '1px solid var(--b1)',
        boxShadow: 'inset 0 1px 3px rgba(0,0,0,0.06)',
      }}
    >
      {MODES.map(({ key, label, icon: Icon }) => {
        const active = careMode === key
        return (
          <button
            key={key}
            onClick={() => setCareMode(key)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200"
            style={{
              background: active
                ? 'linear-gradient(135deg, var(--blue-l), var(--cyan-l))'
                : 'transparent',
              color: active ? '#ffffff' : 'var(--t3)',
              fontFamily: 'var(--font-display)',
              boxShadow: active ? '0 2px 8px rgba(32,112,204,0.28), 0 1px 2px rgba(0,0,0,0.14)' : 'none',
              letterSpacing: active ? '0.01em' : '0',
              cursor: 'pointer',
              border: 'none',
            }}
          >
            <Icon />
            {label}
          </button>
        )
      })}
    </div>
  )
}
