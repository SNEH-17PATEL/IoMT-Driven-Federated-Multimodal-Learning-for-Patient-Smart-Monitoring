/*
 * All three tabs of the LLM (AI report) validation drawer plus its header, tab dot and
 * keyboard handling. Basic open/close and the judge_used message are covered in
 * LLMValidationDrawer.test.jsx.
 */
import { render, screen, fireEvent } from '@testing-library/react'
import LLMValidationDrawer from '@/components/validation/LLMValidationDrawer.jsx'

const report = (over = {}) => ({
  reliability_score: 0.82, reliability_label: 'High', reliability_color: '#28a745',
  judge_used: true, violations: [], ...over,
})
const open = (data, props = {}) => {
  const onClose = props.onClose ?? vi.fn()
  const utils = render(<LLMValidationDrawer open onClose={onClose} reportData={data} careMode="hospital" {...props} />)
  return { ...utils, onClose }
}
const tab = (name) => fireEvent.click(screen.getByRole('button', { name: new RegExp(name) }))

describe('header', () => {
  it('shows the reliability badge with label and 3-decimal score', () => {
    open(report())
    expect(screen.getAllByText('High Reliability').length).toBeGreaterThan(0)
    expect(screen.getAllByText('0.820').length).toBeGreaterThan(0)   // header badge + overview hero
  })

  it('has no badge before a report exists', () => {
    open(null)
    // The header badge is the only place a bare 3-decimal score appears, and with a real
    // report it does (see the first test in this block); with no report it must be absent.
    expect(screen.queryByText(/^\d\.\d{3}$/)).not.toBeInTheDocument()
    expect(screen.getByText(/Awaiting report generation/)).toBeInTheDocument()
  })

  it('shows the care mode', () => {
    open(report())
    expect(screen.getByText(/Hospital Mode/)).toBeInTheDocument()
  })

  it('shows Home Care mode', () => {
    open(report(), { careMode: 'homecare' })
    expect(screen.getByText(/Home Care Mode/)).toBeInTheDocument()
  })

  it('shows a singular violation count', () => {
    open(report({ violations: ['one'] }))
    expect(screen.getByText('1 violation')).toBeInTheDocument()
  })

  it('shows a plural violation count', () => {
    open(report({ violations: ['one', 'two'] }))
    expect(screen.getByText('2 violations')).toBeInTheDocument()
  })

  it('omits the violation count when there are none', () => {
    open(report())
    expect(screen.queryByText(/^\d+ violations?$/)).not.toBeInTheDocument()
  })

  it('puts a warning dot on the Findings tab only when there are violations', () => {
    const { unmount } = open(report())
    const dot = () => screen.getByRole('button', { name: /Findings/ }).querySelector('span')
    expect(dot()).toBeNull()
    unmount()
    open(report({ violations: ['x'] }))
    expect(dot()).not.toBeNull()
  })

  it('closes on Escape and from the ✕ button', () => {
    const { onClose } = open(report())
    fireEvent.keyDown(window, { key: 'Escape' })
    fireEvent.click(screen.getByLabelText('Close panel'))
    expect(onClose).toHaveBeenCalledTimes(2)
  })

  it('ignores other keys', () => {
    const { onClose } = open(report())
    fireEvent.keyDown(window, { key: 'a' })
    expect(onClose).not.toHaveBeenCalled()
  })

  it('goes back to Overview when reopened', () => {
    const onClose = vi.fn()
    const { rerender } = render(<LLMValidationDrawer open onClose={onClose} reportData={report()} careMode="hospital" />)
    tab('9 Methods')
    expect(screen.getByText(/7 Deterministic checks/)).toBeInTheDocument()
    rerender(<LLMValidationDrawer open={false} onClose={onClose} reportData={report()} careMode="hospital" />)
    rerender(<LLMValidationDrawer open onClose={onClose} reportData={report()} careMode="hospital" />)
    expect(screen.getByText('Score thresholds')).toBeInTheDocument()
  })
})

describe('Overview tab', () => {
  it.each([
    ['High', /passed all critical safety checks/, '✅'],
    ['Moderate', /Minor gaps detected/, '⚠️'],
    ['Low', /Significant failures detected/, '❌'],
  ])('%s label gives the matching explanation', (label, text, icon) => {
    open(report({ reliability_label: label }))
    expect(screen.getByText(text)).toBeInTheDocument()
    expect(document.body.textContent).toContain(icon)
  })

  it('shows placeholders (no icon) when the label and score are unknown', () => {
    open(report({ reliability_label: undefined, reliability_score: undefined }))
    expect(screen.getAllByText('—').length).toBeGreaterThan(0)          // score and label placeholders
    expect(screen.getByText(/Awaiting report generation/)).toBeInTheDocument()
    expect(document.body.textContent).not.toMatch(/[✅⚠️❌]\s*—\s*Reliability/)
  })

  it('explains the three score thresholds and the weight bar', () => {
    open(report())
    expect(screen.getByText('≥ 0.75')).toBeInTheDocument()
    expect(screen.getByText('0.55 – 0.75')).toBeInTheDocument()
    expect(screen.getByText('< 0.55')).toBeInTheDocument()
    expect(screen.getByText(/7 deterministic \(65%\) · 2 judge LLM \(35%\)/)).toBeInTheDocument()
  })

  it('draws nine weight segments that add up to 100', () => {
    const { container } = open(report())
    const segs = [...container.querySelectorAll('div[title$="%"]')]
    expect(segs).toHaveLength(9)
    const total = segs.reduce((s, el) => s + Number(el.title.match(/(\d+)%/)[1]), 0)
    expect(total).toBe(100)
  })
})

describe('9 Methods tab', () => {
  it('lists the 7 deterministic and 2 judge methods with weights', () => {
    open(report())
    tab('9 Methods')
    for (const n of ['Factual Grounding', 'Clinical Hard Rules', 'SHAP-LLM Coherence', 'Response Structure',
      'Severity Calibration', 'Contraindication', 'Numeric Accuracy', 'G-Eval Clinical Quality', 'RAGAS Faithfulness'])
      expect(screen.getByText(n)).toBeInTheDocument()
    expect(screen.getByText('22%')).toBeInTheDocument()
    expect(screen.getByText(/2 Judge LLM checks/)).toBeInTheDocument()
  })

  it.each([
    [0.90, /All methods performing well/],
    [0.70, /Some methods below optimal/],
    [0.40, /Multiple methods flagged/],
  ])('score %s gives the right hint', (score, hint) => {
    open(report({ reliability_score: score }))
    tab('9 Methods')
    expect(screen.getByText(hint, { exact: false })).toBeInTheDocument()
    expect(screen.getByText(new RegExp(`Combined score: ${score.toFixed(3)}`))).toBeInTheDocument()
  })

  it('shows no combined-score hint without a score', () => {
    open(report({ reliability_score: undefined }))
    tab('9 Methods')
    expect(screen.queryByText(/Combined score/)).not.toBeInTheDocument()
  })
})

describe('Findings tab', () => {
  it('shows the all-clear message when there are no violations', () => {
    open(report())
    tab('Findings')
    expect(screen.getByText('No violations detected for this reading')).toBeInTheDocument()
  })

  it('lists each violation', () => {
    open(report({ violations: ['SpO₂ < 90% → must mention oxygen therapy', 'Hallucinated number in response: 187'] }))
    tab('Findings')
    expect(screen.getByText(/must mention oxygen therapy/)).toBeInTheDocument()
    expect(screen.getByText(/Hallucinated number in response: 187/)).toBeInTheDocument()
    expect(screen.queryByText('No violations detected for this reading')).not.toBeInTheDocument()
  })

  it('explains how the score affects clinical trust and carries the disclaimer', () => {
    open(report())
    tab('Findings')
    expect(screen.getByText('Score ≥ 0.75')).toBeInTheDocument()
    expect(screen.getByText('Score 0.55–0.75')).toBeInTheDocument()
    expect(screen.getByText('Score < 0.55')).toBeInTheDocument()
    expect(screen.getByText(/requires qualified clinical judgment/)).toBeInTheDocument()
  })

  it('copes with a null report', () => {
    open(null)
    tab('Findings')
    expect(screen.getByText('No violations detected for this reading')).toBeInTheDocument()
  })
})
