import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { getVitals } from '../../api/client.js'
import TrendChart from './TrendChart.jsx'

const gridVariants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.09, delayChildren: 0.05 } },
}
const chartVariants = {
  hidden: { opacity: 0, y: 22, scale: 0.97 },
  visible: { opacity: 1, y: 0, scale: 1, transition: { type: 'spring', stiffness: 280, damping: 26 } },
}

const CHARTS = [
  { dataKey: 'HR',   label: 'Heart Rate',   unit: 'bpm',  color: '#e03a3a', normalLo: 60,   normalHi: 100  },
  { dataKey: 'RR',   label: 'Resp. Rate',   unit: '/min', color: '#2878e8', normalLo: 12,   normalHi: 20   },
  { dataKey: 'SpO2', label: 'SpO₂',         unit: '%',    color: '#0f9259', normalLo: 95,   normalHi: 100  },
  { dataKey: 'Temp', label: 'Temperature',  unit: '°C',   color: '#d97a10', normalLo: 36.0, normalHi: 37.5 },
  { dataKey: 'SBP',  label: 'Sys. BP',      unit: 'mmHg', color: '#8840c8', normalLo: 90,   normalHi: 120  },
  { dataKey: 'MAP',  label: 'MAP',          unit: 'mmHg', color: '#009688', normalLo: 70,   normalHi: 100  },
]

function SkeletonChart() {
  return (
    <div className="rounded-xl p-4 skeleton" style={{ border: '1px solid var(--b1)', height: 203 }} />
  )
}

export default function TrendsSection({ patientId }) {
  const { data: vitalsData, isLoading } = useQuery({
    queryKey: ['vitals', patientId],
    queryFn: () => getVitals(patientId, 20).then(r => r.data),
    refetchInterval: 30_000,
  })

  return (
    <motion.div
      className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4"
      variants={gridVariants}
      initial="hidden"
      animate="visible"
    >
      {isLoading
        ? CHARTS.map(c => <SkeletonChart key={c.dataKey} />)
        : CHARTS.map((c, i) => (
            <motion.div
              key={c.dataKey}
              variants={chartVariants}
              data-aos="fade-up"
              data-aos-delay={i * 70}
              data-aos-duration="550"
            >
              <TrendChart
                data={vitalsData ?? []}
                dataKey={c.dataKey}
                label={c.label}
                unit={c.unit}
                color={c.color}
                normalLo={c.normalLo}
                normalHi={c.normalHi}
              />
            </motion.div>
          ))
      }
    </motion.div>
  )
}
