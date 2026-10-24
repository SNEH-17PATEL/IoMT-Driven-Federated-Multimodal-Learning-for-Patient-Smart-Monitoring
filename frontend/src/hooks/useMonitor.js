import { useState, useEffect, useRef, useCallback } from 'react'
import { runPredict } from '../api/client.js'

export function useMonitor(patientId) {
  const [predictData,  setPredictData]  = useState(null)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [lastUpdated,  setLastUpdated]  = useState(null)
  const [countdown,    setCountdown]    = useState(90)
  const [error,        setError]        = useState(null)

  const rowIdxRef      = useRef(0)
  const patientIdRef   = useRef(patientId)
  const fetchingRef    = useRef(false)   // guard: prevent overlapping fetches
  patientIdRef.current = patientId

  // Stable fetch function — reads latest values from refs
  const fetchNext = useCallback(async () => {
    // Don't start a new fetch if one is already in flight
    if (fetchingRef.current) return
    fetchingRef.current = true
    setIsRefreshing(true)
    try {
      const result = await runPredict(patientIdRef.current, rowIdxRef.current)
      setPredictData(result.data)
      setLastUpdated(new Date())
      setError(null)
      rowIdxRef.current += 1
    } catch (err) {
      console.error('Predict failed:', err)
      setError('Refresh failed — showing last reading. Retrying in 90s.')
    } finally {
      setIsRefreshing(false)
      fetchingRef.current = false
      // Reset countdown only after the fetch completes so the timer reflects
      // how long until the NEXT reading, not time elapsed during this one.
      setCountdown(90)
    }
  }, [])

  // Initial fetch + state reset when patient changes
  useEffect(() => {
    rowIdxRef.current   = 0
    fetchingRef.current = false
    setCountdown(90)
    setError(null)
    setPredictData(null)
    fetchNext()
  }, [patientId, fetchNext])

  // 1-second countdown — triggers fetchNext when it reaches 0
  useEffect(() => {
    const tick = setInterval(() => {
      setCountdown(c => {
        if (c <= 1) {
          fetchNext()
          // Don't reset here — fetchNext resets it after the call completes
          // so the counter freezes at ~0-1 during the in-flight predict call
          return 1
        }
        return c - 1
      })
    }, 1000)
    return () => clearInterval(tick)
  }, [patientId, fetchNext])

  return { predictData, isRefreshing, lastUpdated, countdown, error }
}
