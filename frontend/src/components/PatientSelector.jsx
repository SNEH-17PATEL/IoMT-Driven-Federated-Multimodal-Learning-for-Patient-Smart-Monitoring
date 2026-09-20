import { motion } from "framer-motion";
import { ArrowRight, Radio } from "lucide-react";

const RISK_STYLES = {
  1: { color: "#2de6a3", label: "LOW RISK", glow: "glow-low" },
  2: { color: "#ffb020", label: "MODERATE RISK", glow: "glow-mod" },
  3: { color: "#ff3b5c", label: "HIGH RISK", glow: "glow-high" },
};

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.1, delayChildren: 0.15 } },
};
const item = {
  hidden: { opacity: 0, y: 24 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.16, 1, 0.3, 1] } },
};

export default function PatientSelector({ patients, onSelect }) {
  return (
    <div className="relative mx-auto max-w-6xl px-6 py-16 w-full">
      <motion.div
        initial={{ opacity: 0, y: -14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        className="text-center mb-14"
      >
        <div className="inline-flex items-center gap-2 rounded-full border border-[color:var(--accent)]/25 bg-[color:var(--accent)]/5 text-[color:var(--accent)] px-4 py-1.5 text-[11px] font-mono font-semibold tracking-[0.15em] uppercase mb-6">
          <Radio size={12} className="animate-pulse" /> ICU Clinical Decision Support · Console Online
        </div>
        <h1 className="text-4xl sm:text-5xl font-bold font-display tracking-tight text-[color:var(--text)]">
          Select a patient to <span className="gradient-text">monitor</span>
        </h1>
        <p className="mt-5 text-[color:var(--text-dim)] max-w-xl mx-auto text-[15px] leading-relaxed">
          Vitals update automatically every{" "}
          <strong className="text-[color:var(--text)] font-mono font-semibold">30 seconds</strong>,
          with a simple health score and a plain-English explanation of what's going on.
        </p>
      </motion.div>

      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="grid grid-cols-1 md:grid-cols-3 gap-6"
      >
        {patients.map((p) => {
          const s = RISK_STYLES[p.id] ?? RISK_STYLES[1];
          return (
            <motion.div
              key={p.id}
              variants={item}
              whileHover={{ y: -6 }}
              className={`group relative panel rounded-xl p-6 transition-shadow duration-300 hover:${s.glow}`}
            >
              <span
                className="inline-flex items-center gap-1.5 text-[10px] font-mono font-bold tracking-[0.12em] px-2.5 py-1 rounded border"
                style={{ color: s.color, borderColor: `${s.color}4d`, background: `${s.color}14` }}
              >
                <span className="w-1.5 h-1.5 rounded-full" style={{ background: s.color }} />
                {s.label}
              </span>

              <h3 className="text-xl font-bold font-display text-[color:var(--text)] mt-4">
                Patient {p.id}
              </h3>
              <p className="text-sm font-semibold mt-1" style={{ color: s.color }}>
                {p.description.split(" — ")[0]}
              </p>
              <p className="text-xs text-[color:var(--text-dim)] mt-2.5 leading-relaxed min-h-[48px]">
                {p.long_desc}
              </p>

              <motion.button
                whileTap={{ scale: 0.97 }}
                onClick={() => onSelect(p.id)}
                className="mt-5 w-full group/btn rounded-lg font-semibold text-sm py-3 flex items-center justify-center gap-2 border transition-colors"
                style={{
                  color: s.color,
                  borderColor: `${s.color}4d`,
                  background: `${s.color}0f`,
                }}
              >
                Start Monitoring
                <ArrowRight size={15} className="transition-transform group-hover/btn:translate-x-1" />
              </motion.button>
            </motion.div>
          );
        })}
      </motion.div>
    </div>
  );
}
