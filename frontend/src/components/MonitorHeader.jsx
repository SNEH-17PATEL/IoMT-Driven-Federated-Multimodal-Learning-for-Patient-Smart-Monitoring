import { motion } from "framer-motion";

export default function MonitorHeader({ patient, clock }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      className="relative overflow-hidden rounded-xl text-white px-6 py-4 panel-raised"
    >
      <div className="pointer-events-none absolute -top-10 right-10 w-40 h-40 rounded-full bg-[color:var(--violet)]/15 blur-3xl" />
      <div className="pointer-events-none absolute -bottom-10 left-1/3 w-48 h-24 rounded-full bg-[color:var(--accent)]/12 blur-3xl" />

      <div className="relative flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-3">
          <span className="live-dot" />
          <span className="live-badge bg-[color:var(--high)]/12 border border-[color:var(--high)]/40 text-[color:var(--high)] text-[11px] font-mono font-extrabold tracking-widest px-3 py-1 rounded">
            LIVE
          </span>
          <span className="text-lg font-bold font-display">
            {patient.icon} {patient.name}
          </span>
          <span className="text-xs text-[color:var(--text-dim)] bg-white/5 border border-white/8 px-3 py-1 rounded-full">
            {patient.description}
          </span>
        </div>
        <div className="flex gap-6 text-xs text-[color:var(--text-dim)]">
          <Stat label="Time" value={clock} mono accent />
        </div>
      </div>
    </motion.div>
  );
}

function Stat({ label, value, mono, accent }) {
  return (
    <div className="text-center">
      <div className="text-[10px] font-mono uppercase tracking-wide text-[color:var(--text-faint)]">
        {label}
      </div>
      <div
        className={`font-bold text-lg leading-tight font-display ${mono ? "font-mono" : ""} ${
          accent ? "text-[color:var(--accent)]" : "text-[color:var(--text)]"
        }`}
      >
        {value}
      </div>
    </div>
  );
}
