/**
 * Vitest global setup file — must live inside frontend/ so Vite's
 * module resolver can find node_modules (e.g. @testing-library/jest-dom).
 */
import '@testing-library/jest-dom'
import React from 'react'

// Make React globally available for any JSX that OXC/esbuild transforms
// to classic-runtime (React.createElement) for files outside the Vite root.
globalThis.React = React

// ── CSS custom properties used by components ──────────────────────────────
Object.assign(document.documentElement.style, {
  '--t1': '#fff', '--t2': '#ccc', '--t3': '#aaa', '--t4': '#888',
  '--s1': '#0d1a2a', '--s2': '#122236',
  '--b1': 'rgba(255,255,255,0.08)',
  '--r-xl': '16px', '--r-lg': '12px',
  '--ok-l': '#28a745', '--danger-l': '#dc3545', '--warn-l': '#e6a817',
  '--warn': '#e6a817', '--warn-bg': 'rgba(230,168,23,0.08)',
  '--blue-l': '#60a5fa', '--cyan-l': '#38bdf8',
  '--font-display': 'system-ui, sans-serif',
  '--sh-sm': '0 2px 8px rgba(0,0,0,0.2)',
})

// ── framer-motion ─────────────────────────────────────────────────────────
vi.mock('framer-motion', () => ({
  motion: new Proxy({}, {
    get: (_, tag) =>
      React.forwardRef(({ children, ...rest }, ref) => {
        const filtered = Object.fromEntries(
          Object.entries(rest).filter(([k]) =>
            !['initial', 'animate', 'exit', 'transition', 'variants',
              'whileHover', 'whileTap', 'whileDrag', 'whileInView',
              'onHoverStart', 'onHoverEnd', 'layoutId'].includes(k)
          )
        )
        return React.createElement(tag, { ...filtered, ref }, children)
      }),
  }),
  AnimatePresence: ({ children }) => children,
  useReducedMotion: () => false,
  useAnimation: () => ({ start: vi.fn() }),
  useInView: () => true,
}))

// ── animejs ───────────────────────────────────────────────────────────────
// createTimeline().add(...).add(...) chains — each add() must return the timeline
vi.mock('animejs', () => {
  // createTimeline().add(...).add(...).pause() — all must chain/exist
  const makeAnim = () => ({ pause: vi.fn(), play: vi.fn(), reverse: vi.fn(), seek: vi.fn(), cancel: vi.fn() })
  const makeTl = () => {
    const tl = { pause: vi.fn(), play: vi.fn() }
    tl.add = vi.fn(() => tl)
    return tl
  }
  return {
    animate: vi.fn(() => makeAnim()),
    createTimeline: vi.fn(() => makeTl()),
    default: {
      animate: vi.fn(() => makeAnim()),
      createTimeline: vi.fn(() => makeTl()),
    },
  }
})

// ── AOS ───────────────────────────────────────────────────────────────────
vi.mock('aos', () => ({
  default: { init: vi.fn(), refresh: vi.fn() },
}))

// ── recharts ──────────────────────────────────────────────────────────────
vi.mock('recharts', () => {
  const stub = ({ children }) =>
    React.createElement('div', { 'data-testid': 'chart' }, children)
  return {
    ResponsiveContainer: stub,
    LineChart: stub,
    AreaChart: stub,
    BarChart: stub,
    Line: () => null,
    Area: () => null,
    Bar: () => null,
    XAxis: () => null,
    YAxis: () => null,
    CartesianGrid: () => null,
    Tooltip: () => null,
    Legend: () => null,
    ReferenceLine: () => null,
    ReferenceArea: () => null,
  }
})

// ── @mediapipe/tasks-vision ────────────────────────────────────────────────
vi.mock('@mediapipe/tasks-vision', () => ({
  FaceLandmarker: { createFromOptions: vi.fn() },
  FilesetResolver: { forVisionTasks: vi.fn() },
  DrawingUtils: vi.fn(),
}))

// ── IntersectionObserver ──────────────────────────────────────────────────
global.IntersectionObserver = class IntersectionObserver {
  constructor(cb) { this._cb = cb }
  observe(el) { this._cb([{ isIntersecting: true, target: el }]) }
  unobserve() {}
  disconnect() {}
}

// ── ResizeObserver ────────────────────────────────────────────────────────
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

// ── window.matchMedia ─────────────────────────────────────────────────────
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
})
