import { motion } from "framer-motion";
import { Activity, Droplets, Eye, Gauge, Heart, Thermometer, Wind } from "lucide-react";
import AnimatedNumber from "./ui/AnimatedNumber";

const CARDS = [
  { key: "HR", label: "Heart Rate", Icon: Heart, unit: "bpm", decimals: 0, rangeKey: "HR" },
  { key: "RR", label: "Breathing Rate", Icon: Wind, unit: "br/min", decimals: 0, rangeKey: "RR" },
  { key: "SpO2", label: "Blood Oxygen", Icon: Droplets, unit: "%", decimals: 1, rangeKey: "SpO2" },
  { key: "Temp", label: "Temperature", Icon: Thermometer, unit: "°C", decimals: 1, rangeKey: "Temp" },
  { key: "SBP", label: "Blood Pressure (Top)", Icon: Gauge, unit: "mmHg", decimals: 0, rangeKey: "SBP" },
  { key: "DBP", label: "Blood Pressure (Bottom)", Icon: Gauge, unit: "mmHg", decimals: 0, rangeKey: "DBP" },
  { key: "MAP", label: "Avg. Blood Pressure", Icon: Activity, unit: "mmHg", decimals: 0, rangeKey: "MAP" },
];

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.05 } },
};
const item = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0, transition: { duration: 0.35, ease: [0.16, 1, 0.3, 1] } },
};

function VitalCard({ label, Icon, value, decimals, unit, lo, hi }) {
  const inRange = value >= lo && value <= hi;
  const status = inRange ? "Normal" : value < lo ? "Below Normal" : "Above Normal";
  const color = inRange ? "var(--low)" : "var(--high)";
  const glow = inRange ? "glow-low" : "glow-high";

  return (
    <motion.div
      variants={item}
      whileHover={{ y: -3 }}
      className={`rounded-lg panel-inset p-3.5 text-center flex flex-col justify-between min-h-[110px] transition-shadow ${glow}`}
    >
      <div className="flex items-center justify-center gap-1.5 text-[10px] font-mono font-semibold uppercase tracking-wide text-[color:var(--text-faint)]">
        <Icon size={12} style={{ color }} /> {label}
      </div>
      <div className="text-3xl font-bold font-mono text-[color:var(--text)] leading-none my-1.5">
        <AnimatedNumber value={value} decimals={decimals} />
        <span className="text-xs font-medium text-[color:var(--text-faint)] ml-1">{unit}</span>
      </div>
      <div className="text-[11px] font-bold" style={{ color }}>
        {inRange ? "✓" : "⚠"} {status}
      </div>
    </motion.div>
  );
}

export default function VitalsGrid({ vitals, normalRanges }) {
  return (
    <div>
      <div className="text-sm font-bold text-[color:var(--text)] mb-2 flex items-center gap-1.5">
        <Activity size={14} className="text-[color:var(--accent)]" /> Latest Vitals
      </div>
      <motion.div variants={container} initial="hidden" animate="show" className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
        {CARDS.map((c) => {
          const [lo, hi] = normalRanges[c.rangeKey];
          return (
            <VitalCard
              key={c.key}
              label={c.label}
              Icon={c.Icon}
              value={vitals[c.key]}
              decimals={c.decimals}
              unit={c.unit}
              lo={lo}
              hi={hi}
            />
          );
        })}
        <motion.div
          variants={item}
          whileHover={{ y: -3 }}
          className="rounded-lg panel-inset glow-accent p-3.5 text-center flex flex-col justify-between min-h-[110px]"
        >
          <div className="flex items-center justify-center gap-1.5 text-[10px] font-mono font-semibold uppercase tracking-wide text-[color:var(--text-faint)]">
            <Eye size={12} className="text-[color:var(--accent)]" /> Alertness
          </div>
          <div className="text-lg font-bold text-[color:var(--text)] leading-tight my-1.5">
            {vitals.GCS_label}
          </div>
          <div className="text-[11px] font-bold text-[color:var(--accent)]">
            {vitals.GCS_eye}/4
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
}
