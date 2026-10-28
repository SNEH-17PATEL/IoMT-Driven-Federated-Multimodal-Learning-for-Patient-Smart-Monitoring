import { vi, describe, it, expect, beforeEach } from 'vitest'

// Mock axios — return a shared instance from create() to intercept all API calls
vi.mock('axios', () => {
  const instance = { get: vi.fn(), post: vi.fn() }
  return {
    default: {
      create: vi.fn(() => instance),
    },
  }
})

// Both imports must come AFTER vi.mock (hoisted, but both resolve before tests run)
import axios from 'axios'
import { getPatients, getPatient, runPredict, generateReport, getHistory, getVitals } from '@/api/client.js'

// client.js calls axios.create() on import — retrieve the resulting instance
const api = axios.create.mock.results[0]?.value

describe('API client', () => {
  beforeEach(() => {
    api.get.mockReset()
    api.post.mockReset()
  })

  it('getPatients calls GET /patients', () => {
    api.get.mockResolvedValue({ data: [] })
    getPatients()
    expect(api.get).toHaveBeenCalledWith('/patients')
  })

  it('getPatient(2) calls GET /patients/2', () => {
    api.get.mockResolvedValue({ data: {} })
    getPatient(2)
    expect(api.get).toHaveBeenCalledWith('/patients/2')
  })

  it('runPredict(1, 5) calls POST /predict with correct body', () => {
    api.post.mockResolvedValue({ data: {} })
    runPredict(1, 5)
    expect(api.post).toHaveBeenCalledWith('/predict', { patient_id: 1, row_idx: 5 })
  })

  it('generateReport calls POST /report with correct fields', () => {
    api.post.mockResolvedValue({ data: {} })
    const pred = { patient_id: 1, sofa_score: 4.0 }
    generateReport(1, 'hospital', pred)
    expect(api.post).toHaveBeenCalledWith('/report', {
      patient_id: 1,
      care_mode: 'hospital',
      predict_result: pred,
    })
  })

  it('getHistory(1, 5) calls GET /history/1?limit=5', () => {
    api.get.mockResolvedValue({ data: [] })
    getHistory(1, 5)
    expect(api.get).toHaveBeenCalledWith('/history/1?limit=5')
  })

  it('getVitals(1, 20) calls GET /vitals/1?limit=20', () => {
    api.get.mockResolvedValue({ data: [] })
    getVitals(1, 20)
    expect(api.get).toHaveBeenCalledWith('/vitals/1?limit=20')
  })

  it('getHistory uses default limit of 5', () => {
    api.get.mockResolvedValue({ data: [] })
    getHistory(2)
    expect(api.get).toHaveBeenCalledWith('/history/2?limit=5')
  })

  it('getVitals uses default limit of 20', () => {
    api.get.mockResolvedValue({ data: [] })
    getVitals(3)
    expect(api.get).toHaveBeenCalledWith('/vitals/3?limit=20')
  })
})
