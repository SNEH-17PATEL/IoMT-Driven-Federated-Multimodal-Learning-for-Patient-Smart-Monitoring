import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle, ChevronDown, ClipboardList } from "lucide-react";
import SofaGauge from "../charts/SofaGauge";
import SeverityScale from "../charts/SeverityScale";
import TrendChart from "../charts/TrendChart";

export default function RiskAssessmentTab({ reading }) {
  const [historyOpen, setHistoryOpen] = useState(false);
  const [advisoryOpen, setAdvisoryOpen] = useState(true);
  const { sofa_score, risk_text, risk_icon, vitals_window, history, alert_threshold } = reading;

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
        <div className="lg:col-span-2">
          <SofaGauge score={sofa_score} riskText={risk_text} riskIcon={risk_icon} />
        </div>
        <div className="lg:col-span-3">
          <SeverityScale score={sofa_score} />
        </div>
      </div>

      <AnimatePresence>
        {sofa_score >= alert_threshold && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="rounded-lg border border-[color:var(--high)]/30 bg-[color:var(--high)]/6 overflow-hidden"
          >
            <button
              onClick={() => setAdvisoryOpen((v) => !v)}
              className="w-full text-left px-4 py-3 font-bold text-[color:var(--high)] text-sm flex justify-between items-center"
            >
              <span className="flex items-center gap-1.5">
                <AlertTriangle size={15} /> This score needs attention
              </span>
              <motion.span animate={{ rotate: advisoryOpen ? 180 : 0 }}>
                <ChevronDown size={16} />
              </motion.span>
            </button>
            <AnimatePresence>
              {advisoryOpen && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="px-4 pb-4 text-xs text-[color:var(--text-dim)] leading-relaxed space-y-2"
                >
                  <p>
                    This score can be off by a few points in either direction — it's a helpful
                    signal, not an exact reading.
                  </p>
                  <p>
                    <b className="text-[color:var(--text)]">What to do:</b> check on the patient
                    now, and contact their doctor or care team if anything feels wrong. Use the{" "}
                    <b className="text-[color:var(--text)]">AI Report</b> tab for more detail. A
                    fresh reading comes in every 30 seconds — keep an eye on whether the score is
                    going up or down.
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="rounded-lg panel p-4">
        <TrendChart vitalsWindow={vitals_window} />
      </div>

      {history.length > 1 && (
        <div className="rounded-lg panel overflow-hidden">
          <button
            onClick={() => setHistoryOpen((v) => !v)}
            className="w-full text-left px-4 py-3 font-bold text-[color:var(--text)] text-sm flex justify-between items-center"
          >
            <span className="flex items-center gap-1.5">
              <ClipboardList size={15} className="text-[color:var(--accent)]" /> Reading History (
              {history.length}, newest first)
            </span>
            <motion.span animate={{ rotate: historyOpen ? 180 : 0 }}>
              <ChevronDown size={16} />
            </motion.span>
          </button>
          <AnimatePresence>
            {historyOpen && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="px-2 pb-3 overflow-x-auto"
              >
                <table className="w-full text-xs border-collapse font-mono">
                  <thead>
                    <tr className="text-[color:var(--text-faint)] text-left">
                      {Object.keys(history[0]).map((k) => (
                        <th key={k} className="px-2 py-1 font-semibold">
                          {k}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {history.map((row, i) => {
                      const risk = String(row.Risk || "");
                      const color = risk.includes("High")
                        ? "var(--high)"
                        : risk.includes("Moderate")
                        ? "var(--mod)"
                        : "var(--low)";
                      return (
                        <tr key={i} style={{ background: `color-mix(in srgb, ${color} 8%, transparent)` }}>
                          {Object.values(row).map((v, j) => (
                            <td key={j} className="px-2 py-1 whitespace-nowrap text-[color:var(--text-dim)]">
                              {String(v)}
                            </td>
                          ))}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
}
