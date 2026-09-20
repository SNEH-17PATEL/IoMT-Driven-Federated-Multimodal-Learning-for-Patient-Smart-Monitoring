import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";

const CYCLE_SECONDS = 30;

// Mirrors the original Streamlit app's loop: read one row, run the full
// prediction/SHAP/LLM pipeline, display it, wait 30s, repeat.
export function useMonitorLoop(patientId) {
  const [reading, setReading] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [secondsLeft, setSecondsLeft] = useState(CYCLE_SECONDS);

  const timerRef = useRef(null);
  const cancelledRef = useRef(false);

  const runCycle = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.getReading(patientId);
      if (cancelledRef.current) return;
      if (data.error) {
        setError(data.message || data.error);
      } else {
        setError(null);
        setReading(data);
      }
    } catch (e) {
      if (!cancelledRef.current) setError(e.message);
    }
    if (cancelledRef.current) return;
    setLoading(false);
    setSecondsLeft(CYCLE_SECONDS);
  }, [patientId]);

  useEffect(() => {
    cancelledRef.current = false;
    setReading(null);
    setError(null);
    api.startMonitoring(patientId).finally(() => {
      if (!cancelledRef.current) runCycle();
    });
    return () => {
      cancelledRef.current = true;
      clearInterval(timerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [patientId]);

  useEffect(() => {
    if (loading || error) {
      clearInterval(timerRef.current);
      return;
    }
    clearInterval(timerRef.current);
    timerRef.current = setInterval(() => {
      setSecondsLeft((s) => {
        if (s <= 1) {
          runCycle();
          return 0;
        }
        return s - 1;
      });
    }, 1000);
    return () => clearInterval(timerRef.current);
  }, [loading, error, runCycle]);

  return { reading, loading, error, secondsLeft, retry: runCycle };
}
