import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export const getPatients = () =>
  api.get('/patients')

export const getPatient = (id) =>
  api.get(`/patients/${id}`)

export const runPredict = (patientId, rowIdx) =>
  api.post('/predict', { patient_id: patientId, row_idx: rowIdx })

export const generateReport = (patientId, careMode, predictResult) =>
  api.post('/report', { patient_id: patientId, care_mode: careMode, predict_result: predictResult })

export const getHistory = (patientId, limit = 5) =>
  api.get(`/history/${patientId}?limit=${limit}`)

export const getVitals = (patientId, limit = 20) =>
  api.get(`/vitals/${patientId}?limit=${limit}`)
