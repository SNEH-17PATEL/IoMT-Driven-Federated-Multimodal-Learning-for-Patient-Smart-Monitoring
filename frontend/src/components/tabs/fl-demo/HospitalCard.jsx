import { motion } from "framer-motion";
import { Hospital, Loader2 } from "lucide-react";

export default function HospitalCard({ h, gradient, training }) {
  return (
    <motion.div
      layout
      whileHover={{ y: -3 }}
      className="relative rounded-lg p-4 panel-inset overflow-hidden"
    >
      <div className={`absolute inset-x-0 top-0 h-0.5 bg-gradient-to-r ${gradient}`} />
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <div className={`w-8 h-8 rounded-md bg-gradient-to-br ${gradient} text-white flex items-center justify-center`}>
            <Hospital size={15} />
          </div>
          <span className="text-sm font-bold text-[color:var(--text)]">{h.name}</span>
        </div>
        {training ? (
          <span className="text-[10px] font-mono font-bold text-[color:var(--violet)] flex items-center gap-1">
            <Loader2 size={11} className="animate-spin" /> training…
          </span>
        ) : (
          <span className="text-[10px] font-mono font-bold text-[color:var(--text-faint)]">idle</span>
        )}
      </div>
      <div className="text-[10px] font-mono text-[color:var(--text-faint)] mb-3">
        {h.n_train ?? "—"} train · {h.n_val ?? "—"} val samples (synthetic, local-only)
      </div>
      <div className="grid grid-cols-3 gap-2 text-center">
        <Metric label="Loss" value={h.loss} />
        <Metric label="MAE" value={h.mae} />
        <Metric label="R²" value={h.r2} />
      </div>
    </motion.div>
  );
}

function Metric({ label, value }) {
  return (
    <div>
      <div className="text-[9px] font-mono uppercase text-[color:var(--text-faint)]">{label}</div>
      <motion.div
        key={value}
        initial={{ opacity: 0, y: -4 }}
        animate={{ opacity: 1, y: 0 }}
        className="font-bold font-mono text-[color:var(--text)]"
      >
        {value ?? "—"}
      </motion.div>
    </div>
  );
}
