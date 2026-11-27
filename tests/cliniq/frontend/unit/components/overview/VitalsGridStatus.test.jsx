/*
 * VitalsGrid — the two cards the grid derives itself (GCS Eye and Stress Score), the clinical
 * note card, and how the grid passes API status through to each vital card.
 */
import { render, screen } from '@testing-library/react'
import VitalsGrid from '@/components/overview/VitalsGrid.jsx'

const pd = (over = {}) => ({
  vitals: { HR: 80, RR: 16, SpO2: 98, Temp: 37, SBP: 118, DBP: 76, MAP: 90, GCS_eye: 4, stress: 1 },
  vitals_status: {}, clinical_note: 'stable overnight', ...over,
})
const grid = (over) => render(<VitalsGrid predictData={pd(over)} />)
const vitals = (v) => pd({ vitals: { ...pd().vitals, ...v } })

describe('GCS Eye card', () => {
  it.each([[1, 'None'], [2, 'To Pain'], [3, 'To Voice'], [4, 'Spontaneous']])(
    'GCS %i shows "%s"', (n, label) => {
      render(<VitalsGrid predictData={vitals({ GCS_eye: n })} />)
      expect(screen.getByText(label)).toBeInTheDocument()
    })

  // The big GCS number is coloured by status: low scores (E1, E2) use the warning colour.
  it.each([[1, true], [2, true], [3, false], [4, false]])('GCS %i is highlighted = %s', (n, highlighted) => {
    render(<VitalsGrid predictData={vitals({ GCS_eye: n })} />)
    const bigNumber = screen.getAllByText(String(n)).find(el => el.className.includes('num') && el.style.fontSize === '3.1rem')
    expect(bigNumber).toBeTruthy()
    expect(bigNumber.style.color).toBe(highlighted ? 'var(--warn-l)' : 'var(--t1)')
  })

  it('shows a dash when GCS is missing', () => {
    render(<VitalsGrid predictData={vitals({ GCS_eye: undefined })} />)
    expect(screen.getAllByText('—').length).toBeGreaterThan(0)
  })

  it('shows a dash label for an out-of-range GCS value', () => {
    render(<VitalsGrid predictData={vitals({ GCS_eye: 7 })} />)
    expect(screen.getAllByText('—').length).toBeGreaterThan(0)
  })
})

describe('Stress card', () => {
  it.each([
    [0.5, 'Low Stress'], [3.4, 'Low Stress'],
    [3.5, 'Elevated'], [6.4, 'Elevated'],
    [6.5, 'High Stress'], [10, 'High Stress'],
  ])('stress %s is "%s" (bands at 3.5 and 6.5)', (v, label) => {
    render(<VitalsGrid predictData={vitals({ stress: v })} />)
    expect(screen.getByText(label)).toBeInTheDocument()
  })

  it('shows a dash label and value when stress is missing', () => {
    render(<VitalsGrid predictData={vitals({ stress: undefined })} />)
    expect(screen.getByText('Stress Score')).toBeInTheDocument()
    expect(screen.queryByText(/Low Stress|Elevated|High Stress/)).not.toBeInTheDocument()
  })

  it('fills the 10-dot scale up to the rounded value', () => {
    const { container } = render(<VitalsGrid predictData={vitals({ stress: 4.4 })} />)
    const dots = [...container.querySelectorAll('div[style*="opacity"]')].filter(d => ['1', '0.25'].includes(d.style.opacity) && d.style.borderRadius)
    const lit = dots.filter(d => d.style.opacity === '1').length
    expect(lit).toBeGreaterThanOrEqual(4)
  })

  it('shows the scale end labels', () => {
    render(<VitalsGrid predictData={vitals({ stress: 2 })} />)
    expect(screen.getByText('0 calm')).toBeInTheDocument()
    expect(screen.getByText(/10 sev/)).toBeInTheDocument()
  })
})

describe('clinical note card', () => {
  it('shows the note', () => {
    grid()
    expect(screen.getByText('stable overnight')).toBeInTheDocument()
  })

  it.each([undefined, null, '', '   ', 'No clinical notes provided'])('shows the empty message for %j', (note) => {
    render(<VitalsGrid predictData={pd({ clinical_note: note })} />)
    expect(screen.getByText('No clinical notes on record for this reading.')).toBeInTheDocument()
  })
})

describe('status pass-through to the vital cards', () => {
  it('shows each API status', () => {
    render(<VitalsGrid predictData={pd({ vitals_status: { HR: 'high', SpO2: 'low', RR: 'normal' } })} />)
    expect(screen.getByText('Above Normal')).toBeInTheDocument()
    expect(screen.getByText('Below Normal')).toBeInTheDocument()
  })

  it('defaults a vital with no status to normal', () => {
    render(<VitalsGrid predictData={pd({ vitals_status: {} })} />)
    expect(screen.queryByText('Above Normal')).not.toBeInTheDocument()
    expect(screen.queryByText('Below Normal')).not.toBeInTheDocument()
  })

  it('copes with a prediction that has no vitals or statuses at all', () => {
    expect(() => render(<VitalsGrid predictData={{}} />)).not.toThrow()
  })
})
