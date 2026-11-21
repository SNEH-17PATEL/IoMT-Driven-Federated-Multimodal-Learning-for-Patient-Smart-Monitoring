import axios from 'axios'

// H17: 90 s covers the LLM report call (main call can take 10–30 s)
const api = axios.create({ baseURL: '/api', timeout: 90_000 })

export const getPatients = () =>
  api.get('/patients')

export const getPatient = (id) =>
  api.get(`/patients/${id}`)

export const runPredict = (patientId, rowIdx, cvData = null) =>
  api.post('/predict', {
    patient_id:  patientId,
    row_idx:     rowIdx,
    ...(cvData?.gcs_eye != null && { cv_gcs_eye: cvData.gcs_eye }),
    ...(cvData?.stress  != null && { cv_stress:  cvData.stress  }),
  })

export const generateReport = (patientId, careMode) =>
  api.post('/report', { patient_id: patientId, care_mode: careMode })

export const getHistory = (patientId, limit = 50) =>
  api.get(`/history/${patientId}?limit=${limit}`)

export const getVitals = (patientId, limit = 20) =>
  api.get(`/vitals/${patientId}?limit=${limit}`)
