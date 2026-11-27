/*
 * runPredict — the camera fields. They must be sent ONLY when the camera has a value,
 * because the backend treats a missing field as "use the stored database value".
 */
import { vi, describe, it, expect, beforeEach } from 'vitest'

vi.mock('axios', () => {
  const instance = { get: vi.fn(), post: vi.fn() }
  return { default: { create: vi.fn(() => instance) } }
})
import axios from 'axios'
import { runPredict } from '@/api/client.js'

const api = axios.create.mock.results[0]?.value
const body = () => api.post.mock.calls.at(-1)[1]

describe('runPredict camera fields', () => {
  beforeEach(() => { api.post.mockReset().mockResolvedValue({ data: {} }) })

  it('sends only patient and row when there is no camera data', () => {
    runPredict(1, 4)
    expect(api.post).toHaveBeenCalledWith('/predict', { patient_id: 1, row_idx: 4 })
  })

  it('treats null camera data like none', () => {
    runPredict(2, 0, null)
    expect(body()).toEqual({ patient_id: 2, row_idx: 0 })
  })

  it('sends both camera values when both exist', () => {
    runPredict(1, 0, { gcs_eye: 3, stress: 6.5 })
    expect(body()).toEqual({ patient_id: 1, row_idx: 0, cv_gcs_eye: 3, cv_stress: 6.5 })
  })

  it('sends only GCS when stress is missing', () => {
    runPredict(1, 0, { gcs_eye: 2 })
    expect(body()).toEqual({ patient_id: 1, row_idx: 0, cv_gcs_eye: 2 })
  })

  it('sends only stress when GCS is missing', () => {
    runPredict(1, 0, { stress: 4.2 })
    expect(body()).toEqual({ patient_id: 1, row_idx: 0, cv_stress: 4.2 })
  })

  it('a stress of exactly 0 IS sent (zero is a real reading, not "missing")', () => {
    runPredict(1, 0, { gcs_eye: 4, stress: 0 })
    expect(body().cv_stress).toBe(0)
  })

  it.each([undefined, null])('a %s value is not sent', (v) => {
    runPredict(1, 0, { gcs_eye: v, stress: v })
    expect(body()).toEqual({ patient_id: 1, row_idx: 0 })
  })
})
