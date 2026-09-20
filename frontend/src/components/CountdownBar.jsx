import { motion } from "framer-motion";
import { Loader2 } from "lucide-react";
import CircularProgress from "./ui/CircularProgress";

export default function CountdownBar({ loading, secondsLeft, total = 30 }) {
  const pct = loading ? 1 : (total - secondsLeft) / total;

  return (
    <motion.div
      layout
      className="mt-3 rounded-lg text-[color:var(--text)] px-5 py-3 flex items-center gap-3 text-xs panel-raised"
    >
      <CircularProgress pct={loading ? 1 : pct} size={30} strokeWidth={3} color={loading ? "#a78bfa" : "#22d3ee"} trackColor="rgba(255,255,255,0.1)">
        {loading ? (
          <Loader2 size={13} className="animate-spin text-[color:var(--violet)]" />
        ) : (
          <span className="text-[9px] font-mono font-bold text-[color:var(--accent)]">{secondsLeft}</span>
        )}
      </CircularProgress>

      <span className="font-mono font-bold tracking-wide uppercase">Continuous Monitoring</span>
      <span className="text-[color:var(--text-faint)]">|</span>
      {loading ? (
        <span className="italic text-[color:var(--text-dim)]">Processing new reading…</span>
      ) : (
        <span className="text-[color:var(--text-dim)]">
          Next reading in{" "}
          <b className="text-[color:var(--accent)] font-mono">{String(secondsLeft).padStart(2, "0")}s</b>
        </span>
      )}
      <div className="ml-auto w-40 h-1.5 rounded-full bg-white/8 overflow-hidden">
        <motion.div
          className="h-full bg-gradient-to-r from-[color:var(--accent)] to-[color:var(--violet)]"
          animate={{ width: `${pct * 100}%` }}
          transition={{ duration: 0.5 }}
        />
      </div>
    </motion.div>
  );
}
