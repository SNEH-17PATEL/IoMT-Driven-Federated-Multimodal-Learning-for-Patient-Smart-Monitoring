/*
 * ReportSection — how the AI text is turned into UI:
 *   "Heading:"        -> a small uppercase sub-heading with a rule
 *   "• item" / "- x"  -> a bullet row (marker stripped)
 *   "**bold**"        -> <strong>
 *   a line with ⚡    -> a highlighted callout
 *   blank lines       -> skipped
 */
import { render, screen, fireEvent } from '@testing-library/react'
import ReportSection from '@/components/report/ReportSection.jsx'

const show = (content, name = 'IMMEDIATE ACTIONS') =>
  render(<ReportSection name={name} content={content} color="#dc3545" subtitle="Next steps" />)

describe('line types', () => {
  it('renders "Heading:" lines as sub-headings without the colon', () => {
    show('Respiratory:\n• Give oxygen')
    expect(screen.getByText('Respiratory')).toBeInTheDocument()
    expect(screen.queryByText('Respiratory:')).not.toBeInTheDocument()
  })

  it('turns heading text into upper-case styling (not different text)', () => {
    show('Hemodynamics:')
    expect(screen.getByText('Hemodynamics').style.textTransform).toBe('uppercase')
  })

  it('does not treat a long line ending in a colon as a heading', () => {
    const long = 'This is a very long sentence that happens to end with a colon and keeps going on:'
    show(long)
    expect(screen.getByText(long)).toBeInTheDocument()
    expect(screen.getByText(long).style.textTransform).not.toBe('uppercase')
  })

  it('does not treat a bullet ending in a colon as a heading', () => {
    show('• Note this:')
    const el = screen.getByText('Note this:')
    expect(el.style.textTransform).not.toBe('uppercase')
  })

  it.each(['• first', '- first', '* first'])('strips the "%s" bullet marker', (line) => {
    show(line)
    expect(screen.getByText('first')).toBeInTheDocument()
    expect(screen.queryByText(line)).not.toBeInTheDocument()
  })

  it('shows plain lines as they are', () => {
    show('Just a sentence.')
    expect(screen.getByText('Just a sentence.')).toBeInTheDocument()
  })

  it('renders **bold** segments as <strong>', () => {
    const { container } = show('Check **MAP > 65** now')
    const strong = container.querySelector('strong')
    expect(strong).toHaveTextContent('MAP > 65')
    expect(container.textContent).toContain('Check MAP > 65 now')
  })

  it('handles several bold segments in one line', () => {
    const { container } = show('**A** and **B**')
    expect([...container.querySelectorAll('strong')].map(s => s.textContent)).toEqual(['A', 'B'])
  })

  it('leaves an unmatched ** alone', () => {
    const { container } = show('price is **not closed')
    expect(container.querySelector('strong')).toBeNull()
    expect(container.textContent).toContain('**not closed')
  })

  it('highlights a ⚡ line as a callout with bold support', () => {
    const { container } = show('⚡ **Urgent:** start vasopressors')
    const callout = screen.getByText(/start vasopressors/).closest('div')
    expect(callout.style.borderLeft).toContain('2.5px solid')
    expect(container.querySelector('strong')).toHaveTextContent('Urgent:')
  })

  it('skips blank lines', () => {
    const { container } = show('one\n\n   \ntwo')
    expect(container.querySelectorAll('.flex-1.p-4 > div')).toHaveLength(2)
  })

  it('renders a mixed section in order', () => {
    const { container } = show('Respiratory:\n• Oxygen\n⚡ Escalate\nPlain line')
    const text = container.querySelector('.flex-1.p-4').textContent
    expect(text.indexOf('Respiratory')).toBeLessThan(text.indexOf('Oxygen'))
    expect(text.indexOf('Oxygen')).toBeLessThan(text.indexOf('Escalate'))
    expect(text.indexOf('Escalate')).toBeLessThan(text.indexOf('Plain line'))
  })
})

describe('empty and fallback content', () => {
  it.each([undefined, null, '', '   \n  '])('shows "No assessment available." for %j', (c) => {
    show(c)
    expect(screen.getByText('No assessment available.')).toBeInTheDocument()
  })
})

describe('header', () => {
  it.each([
    ['CURRENT CONDITION', 'Current Condition'],
    ['PROBABLE CAUSE', 'Probable Cause'],
    ['RISK FORECAST', 'Risk Forecast'],
    ['IMMEDIATE ACTIONS', 'Immediate Actions'],
  ])('%s gets an icon and the title "%s"', (name, title) => {
    const { container } = show('x', name)
    expect(screen.getByText(title)).toBeInTheDocument()
    expect(container.querySelector('svg')).toBeTruthy()
  })

  it('an unknown section name has no icon but still shows its name', () => {
    const { container } = show('x', 'CUSTOM')
    expect(screen.getByText('CUSTOM')).toBeInTheDocument()
    expect(container.querySelector('svg')).toBeNull()
  })

  it('omits the subtitle when not given', () => {
    render(<ReportSection name="RISK FORECAST" content="x" color="#fff" />)
    expect(screen.queryByText('Next steps')).not.toBeInTheDocument()
  })
})

describe('hover lift', () => {
  it('raises the card on hover and drops it on leave', () => {
    const { container } = show('x')
    const card = container.firstChild
    fireEvent.mouseEnter(card)
    expect(card.style.transform).toBe('translateY(-2px)')
    fireEvent.mouseLeave(card)
    expect(card.style.transform).toBe('translateY(0)')
  })
})
