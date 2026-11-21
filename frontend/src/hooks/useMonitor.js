import { useState, useEffect, useRef, useCallback } from 'react'
import { runPredict } from '../api/client.js'

export function useMonitor(patientId, cvData = null) {
  const [predictData,  setPredictData]  = useState(null)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [lastUpdated,  setLastUpdated]  = useState(null)
  const [countdown,    setCountdown]    = useState(90)
  const [error,        setError]        = useState(null)

  const rowIdxRef      = useRef(
    // H7: resume from where we left off across page visits
    parseInt(sessionStorage.getItem(`rowIdx_${patientId}`) ?? '0', 10)
  )
  const patientIdRef   = useRef(patientId)
  const cvDataRef      = useRef(cvData)
  const fetchingRef    = useRef(false)  // guard: prevent overlapping fetches
  const lastPatientRef = useRef(null)   // StrictMode guard: skip duplicate initial fetch
  patientIdRef.current = patientId
  cvDataRef.current    = cvData         // always latest CV reading without recreating fetchNext

  // Stable fetch function — reads latest values from refs
  const fetchNext = useCallback(async () => {
    if (fetchingRef.current) return
    fetchingRef.current = true
    setIsRefreshing(true)
    const currentPatient = patientIdRef.current
    try {
      const result = await runPredict(currentPatient, rowIdxRef.current, cvDataRef.current)
      // H6: discard response if patient changed while in-flight
      if (result.data?.patient_id !== patientIdRef.current) return
      setPredictData(result.data)
      setLastUpdated(new Date())
      setError(null)
      rowIdxRef.current += 1
      // H7: persist so page refresh resumes from the same row
      sessionStorage.setItem(`rowIdx_${currentPatient}`, rowIdxRef.current)
    } catch (err) {
      console.error('Predict failed:', err)
      const detail = err?.response?.data?.detail ?? err?.message ?? null
      setError(detail
        ? `Refresh failed (${detail}). Retrying in 90s.`
        : 'Refresh failed — showing last reading. Retrying in 90s.'
      )
    } finally {
      setIsRefreshing(false)
      fetchingRef.current = false
      setCountdown(90)
    }
  }, [])

  // Initial fetch + state reset when patient changes
  // lastPatientRef prevents React StrictMode's double-invocation from firing two API calls
  useEffect(() => {
    if (lastPatientRef.current === patientId) return
    lastPatientRef.current = patientId
    // H7: load the persisted row index for this patient (or start at 0)
    rowIdxRef.current = parseInt(sessionStorage.getItem(`rowIdx_${patientId}`) ?? '0', 10)
    fetchingRef.current = false
    setCountdown(90)
    setError(null)
    setPredictData(null)
    fetchNext()
  }, [patientId, fetchNext])

  // H5: keep the countdown tick pure — no side effects inside the updater.
  // A separate effect watches countdown and fires fetchNext when it bottoms out.
  useEffect(() => {
    const tick = setInterval(() => {
      setCountdown(c => (c <= 1 ? 1 : c - 1))
    }, 1000)
    return () => clearInterval(tick)
  }, [patientId])

  useEffect(() => {
    if (countdown === 1) fetchNext()
  }, [countdown, fetchNext])

  return { predictData, isRefreshing, lastUpdated, countdown, error }
}
