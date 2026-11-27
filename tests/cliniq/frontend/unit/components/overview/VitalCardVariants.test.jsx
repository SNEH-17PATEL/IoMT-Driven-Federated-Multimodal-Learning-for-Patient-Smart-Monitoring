/*
 * VitalCard — standard and wide variants, every status, every icon animation family,
 * the missing-value display and the value-change pop.
 */
import { render, screen } from '@testing-library/react'
import VitalCard from '@/components/overview/VitalCard.jsx'

const Icon = () => <svg data-testid="icon" />
const card = (props = {}) =>
  render(<VitalCard Icon={Icon} label="Heart Rate" value={80} unit="bpm" status="normal"
                    normalLo={60} normalHi={100} {...props} />)

describe('standard card', () => {
  it('shows label, value, unit and status', () => {
    card()
    expect(screen.getByText('Heart Rate')).toBeInTheDocument()
    expect(screen.getByText('80')).toBeInTheDocument()
    expect(screen.getByText('bpm')).toBeInTheDocument()
    expect(screen.getByText('Normal')).toBeInTheDocument()
  })

  it.each([
    ['high', 'Above Normal', '↑'],
    ['low', 'Below Normal', '↓'],
  ])('%s status shows "%s"', (status, label) => {
    card({ status, value: 130 })
    expect(screen.getByText(label)).toBeInTheDocument()
  })

  it('treats an unknown status as normal', () => {
    card({ status: 'weird' })
    expect(screen.getByText('Normal')).toBeInTheDocument()
  })

  it('shows decimals with one place and integers as they are', () => {
    const { rerender } = card({ value: 37.26 })
    expect(screen.getByText('37.3')).toBeInTheDocument()
    rerender(<VitalCard Icon={Icon} label="Heart Rate" value={72} unit="bpm" status="normal" normalLo={60} normalHi={100} />)
    expect(screen.getByText('72')).toBeInTheDocument()
  })

  it.each([null, undefined])('shows a dash when the value is %s', (v) => {
    card({ value: v })
    expect(screen.getByText('—')).toBeInTheDocument()
  })

  it('draws the icon', () => {
    card()
    expect(screen.getByTestId('icon')).toBeInTheDocument()
  })

  it('abnormal cards get a tinted border, normal ones do not', () => {
    const { container, rerender } = card({ status: 'high', value: 130 })
    const border = () => container.firstChild.style.border
    expect(border()).toContain('color-mix')
    rerender(<VitalCard Icon={Icon} label="Heart Rate" value={80} unit="bpm" status="normal" normalLo={60} normalHi={100} />)
    expect(border()).toContain('var(--b1)')
  })
})

describe('icon animation by vital', () => {
  const anim = (label, status = 'normal') => {
    const { container } = card({ label, status })
    return container.querySelector('[style*="animation"]')?.style.animation ?? ''
  }

  it.each([
    ['Heart Rate', 'heartbeat'],
    ['SpO₂', 'dropSway'],
    ['SpO2 sat', 'dropSway'],
    ['Resp. Rate', 'breathe'],
    ['Temperature', 'floatY'],
    ['Sys. BP', 'bpNudgeUp'],
    ['SBP', 'bpNudgeUp'],
    ['MAP', 'mapSpin'],
    ['Dia. BP', 'bpNudgeDn'],
    ['DBP', 'bpNudgeDn'],
    ['Something else', 'iconPulse'],
  ])('"%s" uses the %s animation', (label, name) => {
    expect(anim(label)).toContain(name)
  })

  it('abnormal values animate faster than normal ones', () => {
    const secs = (s) => parseFloat(s.match(/([\d.]+)s/)[1])
    expect(secs(anim('Heart Rate', 'high'))).toBeLessThan(secs(anim('Heart Rate', 'normal')))
  })
})

describe('wide card', () => {
  const wide = (props = {}) =>
    render(<VitalCard wide Icon={Icon} label="Heart Rate" value={80} unit="bpm" status="normal"
                      normalLo={60} normalHi={100} {...props} />)

  it('shows label, value, unit and status', () => {
    wide()
    expect(screen.getByText('Heart Rate')).toBeInTheDocument()
    expect(screen.getByText('80')).toBeInTheDocument()
    expect(screen.getByText('Normal')).toBeInTheDocument()
  })

  it.each([['high', 'Above Normal'], ['low', 'Below Normal']])('%s -> %s', (status, label) => {
    wide({ status })
    expect(screen.getByText(label)).toBeInTheDocument()
  })

  it('shows a dash for a missing value and decimals to one place', () => {
    const { rerender } = wide({ value: null })
    expect(screen.getByText('—')).toBeInTheDocument()
    rerender(<VitalCard wide Icon={Icon} label="Heart Rate" value={98.76} unit="bpm" status="normal" normalLo={60} normalHi={100} />)
    expect(screen.getByText('98.8')).toBeInTheDocument()
  })

  it('treats an unknown status as normal', () => {
    wide({ status: 'weird' })
    expect(screen.getByText('Normal')).toBeInTheDocument()
  })
})

describe('value change', () => {
  it('re-renders with the new value when it changes (the pop animation runs on the value node)', () => {
    const { rerender } = card({ value: 70 })
    rerender(<VitalCard Icon={Icon} label="Heart Rate" value={95} unit="bpm" status="normal" normalLo={60} normalHi={100} />)
    expect(screen.getByText('95')).toBeInTheDocument()
    expect(screen.queryByText('70')).not.toBeInTheDocument()
  })
})
