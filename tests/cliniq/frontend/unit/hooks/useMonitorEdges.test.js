/*
 * useMonitor — edge cases: camera values reaching the request, error wording, resuming the
 * row index, ignoring a response for a different patient, and timers.
 */
import { renderHook, act, waitFor } from '@testing-library/react'
import { useMonitor } from '@/hooks/useMonitor.js'

vi.mock('@/api/client.js', () => ({ runPredict: vi.fn() }))
import { runPredict } from '@/api/client.js'

const ok = (patient_id = 1, extra = {}) => ({ data: { patient_id, sofa_score: 4, ...extra } })

beforeEach(() => { runPredict.mockReset(); sessionStorage.clear() })
afterEach(() => { vi.useRealTimers() })

describe('camera data', () => {
  it('sends the camera values passed at call time', async () => {
    runPredict.mockResolvedValue(ok())
    const cv = { stress: 5, gcs_eye: 3 }
    renderHook(() => useMonitor(1, cv))
    await waitFor(() => expect(runPredict).toHaveBeenCalledWith(1, 0, cv))
  })

  it('uses the NEWEST camera values for the next request (not the ones from first render)', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] })
    runPredict.mockResolvedValue(ok())
    const { rerender } = renderHook(({ cv }) => useMonitor(1, cv), { initialProps: { cv: null } })
    await act(async () => { await Promise.resolve() })
    rerender({ cv: { stress: 7, gcs_eye: 2 } })
    await act(async () => { vi.advanceTimersByTime(90_000); await Promise.resolve() })
    const last = runPredict.mock.calls.at(-1)
    expect(last[2]).toEqual({ stress: 7, gcs_eye: 2 })
  })
})

describe('errors', () => {
  it('includes the server detail in the message', async () => {
    runPredict.mockRejectedValue({ response: { data: { detail: 'No sample data' } } })
    const { result } = renderHook(() => useMonitor(1))
    await waitFor(() => expect(result.current.error).toBe('Refresh failed (No sample data). Retrying in 90s.'))
  })

  it('uses the Error message when there is no response body', async () => {
    runPredict.mockRejectedValue(new Error('Network Error'))
    const { result } = renderHook(() => useMonitor(1))
    await waitFor(() => expect(result.current.error).toBe('Refresh failed (Network Error). Retrying in 90s.'))
  })

  it('uses a generic message when there is no detail at all', async () => {
    runPredict.mockRejectedValue({})
    const { result } = renderHook(() => useMonitor(1))
    await waitFor(() => expect(result.current.error).toBe('Refresh failed — showing last reading. Retrying in 90s.'))
  })

  it('keeps the previous reading when a later refresh fails', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] })
    runPredict.mockResolvedValueOnce(ok(1, { sofa_score: 3.3 }))
    const { result } = renderHook(() => useMonitor(1))
    await act(async () => { await Promise.resolve() })
    expect(result.current.predictData.sofa_score).toBe(3.3)
    runPredict.mockRejectedValue(new Error('boom'))
    await act(async () => { vi.advanceTimersByTime(90_000); await Promise.resolve() })
    expect(result.current.predictData.sofa_score).toBe(3.3)
    expect(result.current.error).toMatch(/Refresh failed/)
  })

  it('clears the error after a good refresh', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] })
    runPredict.mockRejectedValueOnce(new Error('boom')).mockResolvedValue(ok())
    const { result } = renderHook(() => useMonitor(1))
    await act(async () => { await Promise.resolve() })
    expect(result.current.error).not.toBeNull()
    await act(async () => { vi.advanceTimersByTime(90_000); await Promise.resolve() })
    expect(result.current.error).toBeNull()
  })
})

describe('row index', () => {
  it('resumes from the row saved in sessionStorage', async () => {
    sessionStorage.setItem('rowIdx_2', '7')
    runPredict.mockResolvedValue(ok(2))
    renderHook(() => useMonitor(2))
    await waitFor(() => expect(runPredict).toHaveBeenCalledWith(2, 7, null))
  })

  it('saves the next row after each successful reading', async () => {
    runPredict.mockResolvedValue(ok(1))
    renderHook(() => useMonitor(1))
    await waitFor(() => expect(sessionStorage.getItem('rowIdx_1')).toBe('1'))
  })

  it('does not advance the row when the request fails', async () => {
    runPredict.mockRejectedValue(new Error('x'))
    const { result } = renderHook(() => useMonitor(1))
    await waitFor(() => expect(result.current.error).not.toBeNull())
    expect(sessionStorage.getItem('rowIdx_1')).toBeNull()
  })

  it('keeps a separate row for each patient', async () => {
    sessionStorage.setItem('rowIdx_1', '3')
    sessionStorage.setItem('rowIdx_3', '9')
    runPredict.mockResolvedValue(ok(3))
    renderHook(() => useMonitor(3))
    await waitFor(() => expect(runPredict).toHaveBeenCalledWith(3, 9, null))
  })
})

describe('stale responses', () => {
  it('ignores a reply that belongs to a different patient', async () => {
    runPredict.mockResolvedValue(ok(99))                 // server answers for patient 99, hook is on 1
    const { result } = renderHook(() => useMonitor(1))
    await waitFor(() => expect(runPredict).toHaveBeenCalled())
    await act(async () => { await Promise.resolve() })
    expect(result.current.predictData).toBeNull()
  })

  it('shows nothing from the old patient after switching', async () => {
    runPredict.mockImplementation(async (pid) => ok(pid, { sofa_score: pid }))
    const { result, rerender } = renderHook(({ id }) => useMonitor(id), { initialProps: { id: 1 } })
    await waitFor(() => expect(result.current.predictData?.patient_id).toBe(1))
    rerender({ id: 2 })
    await waitFor(() => expect(result.current.predictData?.patient_id).toBe(2))
  })
})

describe('timers', () => {
  it('counts down once a second and fires a new request at 1', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] })
    runPredict.mockResolvedValue(ok())
    const { result } = renderHook(() => useMonitor(1))
    await act(async () => { await Promise.resolve() })
    expect(result.current.countdown).toBe(90)
    await act(async () => { vi.advanceTimersByTime(10_000) })
    expect(result.current.countdown).toBe(80)
    const before = runPredict.mock.calls.length
    await act(async () => { vi.advanceTimersByTime(80_000); await Promise.resolve() })
    expect(runPredict.mock.calls.length).toBeGreaterThan(before)
  })

  it('stops the timer when the component unmounts', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] })
    runPredict.mockResolvedValue(ok())
    const { unmount } = renderHook(() => useMonitor(1))
    await act(async () => { await Promise.resolve() })
    unmount()
    const calls = runPredict.mock.calls.length
    await act(async () => { vi.advanceTimersByTime(200_000) })
    expect(runPredict.mock.calls.length).toBe(calls)
  })
})
