/*
 * Hover behaviour of the four floating action buttons.
 *
 * Each button expands to show a text label while the pointer is over it and collapses
 * again afterwards. That is driven by framer-motion's onHoverStart / onHoverEnd, which the
 * global test setup (vitest-setup.js) stubs out — so this file puts the REAL library back
 * with importActual and dispatches real pointer events.
 */
import { render, screen, fireEvent, act } from '@testing-library/react'

vi.mock('framer-motion', async () => await vi.importActual('framer-motion'))

import HistoryFAB from '@/components/history/HistoryFAB.jsx'
import CVMonitorFAB from '@/components/cv/CVMonitorFAB.jsx'
import LLMValidationFAB from '@/components/validation/LLMValidationFAB.jsx'
import SOFAValidationFAB from '@/components/validation/SOFAValidationFAB.jsx'

// Real framer-motion listens for pointer events from a mouse pointer.
const enter = async (el) => { await act(async () => { fireEvent.pointerEnter(el, { pointerType: 'mouse' }) }) }
const leave = async (el) => { await act(async () => { fireEvent.pointerLeave(el, { pointerType: 'mouse' }) }) }

async function expectLabelOnHover(ui, labelPattern) {
  render(ui)
  const btn = screen.getByRole('button')
  expect(screen.queryByText(labelPattern)).not.toBeInTheDocument()   // collapsed to start with
  await enter(btn)
  expect(await screen.findByText(labelPattern)).toBeInTheDocument()  // expanded
  await leave(btn)
  await vi.waitFor(() => expect(screen.queryByText(labelPattern)).not.toBeInTheDocument())
}

describe('floating buttons expand on hover', () => {
  it('History shows "Patient History" and the record count', async () => {
    render(<HistoryFAB onClick={() => {}} recordCount={7} />)
    const btn = screen.getByRole('button')
    await enter(btn)
    expect(await screen.findByText(/Patient History/)).toBeInTheDocument()
    expect(screen.getByText('7')).toBeInTheDocument()
  })

  it('History hides the count badge when there are no records', async () => {
    render(<HistoryFAB onClick={() => {}} recordCount={0} />)
    await enter(screen.getByRole('button'))
    expect(await screen.findByText(/Patient History/)).toBeInTheDocument()
    expect(screen.queryByText('0')).not.toBeInTheDocument()
  })

  it('History label collapses again when the pointer leaves', () =>
    expectLabelOnHover(<HistoryFAB onClick={() => {}} recordCount={2} />, /Patient History/))

  it('Camera shows "Live CV Monitor" when closed', () =>
    expectLabelOnHover(<CVMonitorFAB onClick={() => {}} isActive={false} />, /Live CV Monitor/))

  it('Camera shows "CV Live — Click to Close" while active', () =>
    expectLabelOnHover(<CVMonitorFAB onClick={() => {}} isActive />, /CV Live — Click to Close/))

  it('LLM validation shows "AI Validation" once a report exists', () =>
    expectLabelOnHover(<LLMValidationFAB onClick={() => {}} hasData />, /^AI Validation$/))

  it('LLM validation shows "Awaiting Report…" before the first report', () =>
    // pointer-events are disabled in this state, but the label logic is still reachable
    expectLabelOnHover(<LLMValidationFAB onClick={() => {}} hasData={false} />, /^Awaiting Report…$/))

  it('SOFA validation shows "SOFA Validation"', () =>
    expectLabelOnHover(<SOFAValidationFAB onClick={() => {}} />, /SOFA Validation/))

  it('clicking still works after the button has been hovered', async () => {
    const onClick = vi.fn()
    render(<SOFAValidationFAB onClick={onClick} />)
    const btn = screen.getByRole('button')
    await enter(btn)
    await screen.findByText(/SOFA Validation/)
    fireEvent.click(btn)
    expect(onClick).toHaveBeenCalledTimes(1)
  })
})
