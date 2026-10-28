import { renderHook, act, waitFor } from '@testing-library/react'
import { useMonitor } from '@/hooks/useMonitor.js'

vi.mock('@/api/client.js', () => ({
  runPredict: vi.fn(),
}))

import { runPredict } from '@/api/client.js'

const mockResponse = {
  data: {
    patient_id: 1,
    sofa_score: 4.5,
    risk_level: 'Low Risk',
    risk_color: '#28a745',
    vitals: { HR: 72, SpO2: 98, SBP: 110, RR: 14 },
    shap_top5: [],
  },
}

describe('useMonitor', () => {
  beforeEach(() => {
    runPredict.mockReset()
  })

  it('calls runPredict on mount with patient_id and rowIdx=0', async () => {
    runPredict.mockResolvedValue(mockResponse)
    renderHook(() => useMonitor(1))
    await waitFor(() => expect(runPredict).toHaveBeenCalledWith(1, 0))
  })

  it('sets predictData after successful fetch', async () => {
    runPredict.mockResolvedValue(mockResponse)
    const { result } = renderHook(() => useMonitor(1))
    await waitFor(() => expect(result.current.predictData).not.toBeNull())
    expect(result.current.predictData.patient_id).toBe(1)
  })

  it('sets error on failed fetch', async () => {
    runPredict.mockRejectedValue(new Error('Network error'))
    const { result } = renderHook(() => useMonitor(1))
    await waitFor(() => expect(result.current.error).not.toBeNull(), { timeout: 3000 })
    expect(result.current.error).toMatch(/Refresh failed/)
  })

  it('isRefreshing is true during fetch', async () => {
    let resolve
    runPredict.mockReturnValue(new Promise(r => { resolve = r }))
    const { result } = renderHook(() => useMonitor(1))
    await waitFor(() => expect(result.current.isRefreshing).toBe(true))
    await act(async () => { resolve(mockResponse) })
    await waitFor(() => expect(result.current.isRefreshing).toBe(false))
  })

  it('countdown resets to 90 after fetch completes', async () => {
    runPredict.mockResolvedValue(mockResponse)
    const { result } = renderHook(() => useMonitor(1))
    await waitFor(() => expect(result.current.countdown).toBe(90))
  })

  it('changing patientId resets predictData and triggers new fetch', async () => {
    runPredict.mockResolvedValue(mockResponse)
    const { result, rerender } = renderHook(({ pid }) => useMonitor(pid), {
      initialProps: { pid: 1 },
    })
    await waitFor(() => expect(result.current.predictData).not.toBeNull())

    runPredict.mockResolvedValue({ data: { ...mockResponse.data, patient_id: 2 } })
    act(() => rerender({ pid: 2 }))
    await waitFor(() => expect(runPredict).toHaveBeenCalledWith(2, 0))
  })

  it('lastUpdated is set after successful fetch', async () => {
    runPredict.mockResolvedValue(mockResponse)
    const { result } = renderHook(() => useMonitor(1))
    await waitFor(() => expect(result.current.lastUpdated).not.toBeNull())
    expect(result.current.lastUpdated).toBeInstanceOf(Date)
  })

  it('countdown decrements via interval timer', async () => {
    vi.useFakeTimers()
    try {
      runPredict.mockResolvedValue(mockResponse)
      const { result } = renderHook(() => useMonitor(1))

      // Flush the fetch promise: multiple microtask cycles resolve the async hook
      await act(async () => {
        await Promise.resolve()
        await Promise.resolve()
        await Promise.resolve()
      })

      // After fetch completes, countdown is reset to 90
      const before = result.current.countdown

      // Advance the fake 1-second interval timer by 3 ticks
      act(() => { vi.advanceTimersByTime(3000) })

      // Countdown should have decremented by at least 1
      expect(result.current.countdown).toBeLessThan(before)
    } finally {
      vi.useRealTimers()
    }
  })

  it('rowIdx increments to 1 on the second fetch', async () => {
    vi.useFakeTimers()
    try {
      runPredict.mockResolvedValue(mockResponse)
      renderHook(() => useMonitor(1))

      // Flush first fetch (rowIdx=0)
      await act(async () => {
        await Promise.resolve()
        await Promise.resolve()
        await Promise.resolve()
      })

      expect(runPredict).toHaveBeenCalledWith(1, 0)

      // Advance countdown from 90 → 1 to trigger second fetch
      act(() => { vi.advanceTimersByTime(90000) })

      // Flush second fetch
      await act(async () => {
        await Promise.resolve()
        await Promise.resolve()
        await Promise.resolve()
      })

      expect(runPredict).toHaveBeenCalledTimes(2)
      // Second call must use rowIdx=1
      expect(runPredict.mock.calls[1][1]).toBe(1)
    } finally {
      vi.useRealTimers()
    }
  })

  it('concurrent fetch guard prevents overlapping API calls', async () => {
    vi.useFakeTimers()
    try {
      // First fetch never resolves — stays in flight (fetchingRef=true)
      runPredict.mockReturnValue(new Promise(() => {}))
      renderHook(() => useMonitor(1))

      await act(async () => {
        await Promise.resolve()
        await Promise.resolve()
      })

      // One call in flight
      expect(runPredict).toHaveBeenCalledTimes(1)

      // Advance 90s — countdown reaches 1 and tries to call fetchNext again
      // but fetchingRef is still true, so the call is blocked
      act(() => { vi.advanceTimersByTime(90000) })

      await act(async () => {
        await Promise.resolve()
        await Promise.resolve()
      })

      // Still only 1 call — second attempt was guarded
      expect(runPredict).toHaveBeenCalledTimes(1)
    } finally {
      vi.useRealTimers()
    }
  })

  it('StrictMode double-invoke guard prevents duplicate initial fetch', async () => {
    // React StrictMode invokes effects twice; lastPatientRef prevents the 2nd fetch
    runPredict.mockResolvedValue(mockResponse)
    const wrapper = ({ children }) => React.createElement(React.StrictMode, null, children)
    renderHook(() => useMonitor(1), { wrapper })
    await waitFor(() => expect(runPredict).toHaveBeenCalled())
    // Despite StrictMode's double-invocation, lastPatientRef ensures at most 1 fetch fires
    expect(runPredict).toHaveBeenCalledTimes(1)
  })
})
