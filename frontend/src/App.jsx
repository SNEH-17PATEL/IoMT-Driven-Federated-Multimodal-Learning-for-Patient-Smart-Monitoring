import { Routes, Route, Navigate } from 'react-router-dom'
import SelectPatient from './pages/SelectPatient.jsx'
import MonitorDashboard from './pages/MonitorDashboard.jsx'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<SelectPatient />} />
      <Route path="/monitor/:patientId" element={<MonitorDashboard />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
