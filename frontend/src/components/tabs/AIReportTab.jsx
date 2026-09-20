import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  AlertCircle, Brain, CheckCircle2, ChevronDown, Eye, Search,
  ShieldAlert, Siren, Stethoscope, XCircle,
} from "lucide-react";
import CircularProgress from "../ui/CircularProgress";
import AnimatedNumber from "../ui/AnimatedNumber";
import InfoTip from "../ui/InfoTip";
import { EXPLAIN } from "../../copy";

const SEC_CFG = {
  "CURRENT CONDITION": { Icon: Stethoscope, color: "#22d3ee", sub: "Patient status right now" },
  "PROBABLE CAUSE": { Icon: Search, color: "#ffb020", sub: "Why this is happening" },
  "RISK FORECAST": { Icon: Eye, color: "#a78bfa", sub: "Predicted trajectory if untreated" },
  "IMMEDIATE ACTIONS": { Icon: Siren, color: "#ff3b5c", sub: "Critical interventions — next 30 minutes" },
};

function reliabilityMeta(consistency) {
  if (consistency >= 0.8)
    return {
      color: "#2de6a3", Icon: CheckCircle2,
      label: "High Reliability",
      msg: "All 3 AI responses agree on clinical findings, diagnoses, and recommended interventions.",
    };
  if (consistency >= 0.6)
    return {
      color: "#ffb020", Icon: AlertCircle,
      label: "Moderate Reliability",
      msg: "Responses agree on core findings with some variation in secondary recommendations. Review with care.",
    };
  return {
    color: "#ff3b5c", Icon: XCircle,
    label: "Low Reliability",
    msg: "Significant disagreement across responses on clinical findings or interventions. Use clinical judgment.",
  };
}

export default function AIReportTab({ reading }) {
  const [showAll, setShowAll] = useState(false);
  const { llm } = reading;
  const rel = reliabilityMeta(llm.consistency);
  const sections = llm.sections || {};
  const hasSections = Object.keys(sections).length > 0;
  const validCount = llm.response_valid.filter(Boolean).length;

  return (
    <div className="space-y-5">
      <div className="rounded-xl border px-6 py-5" style={{ borderColor: `${rel.color}4d`, background: `${rel.color}0d` }}>
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className="flex flex-col items-center gap-1">
              <CircularProgress pct={llm.consistency} size={76} strokeWidth={7} color={rel.color} trackColor="rgba(255,255,255,0.08)">
                <span className="text-lg font-bold font-mono" style={{ color: rel.color }}>
                  <AnimatedNumber value={llm.consistency} decimals={2} />
                </span>
              </CircularProgress>
              <div className="text-[9px] font-mono font-bold uppercase tracking-wide text-[color:var(--text-faint)] flex items-center gap-1">
                Agreement <InfoTip text={EXPLAIN.consistency} size={11} />
              </div>
            </div>
            <div>
              <div className="text-[11px] font-mono font-bold uppercase tracking-widest flex items-center gap-1.5" style={{ color: rel.color }}>
                <Brain size={13} /> AI Clinical Assessment
              </div>
              <div className="text-xl font-bold font-display mt-1 flex items-center gap-1.5" style={{ color: rel.color }}>
                <rel.Icon size={20} /> {rel.label}
              </div>
              <p className="text-xs text-[color:var(--text-dim)] max-w-md mt-1">{rel.msg}</p>
            </div>
          </div>
          <div className="flex flex-col items-end gap-1.5">
            <div className="text-[9px] font-mono font-bold uppercase tracking-wide text-[color:var(--text-faint)]">
              3 AI Opinions
            </div>
            <div className="flex gap-1.5">
            {llm.response_valid.map((v, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.08 }}
                className="text-center rounded-lg border px-3 py-2 panel-inset"
                style={{ borderColor: v ? "rgba(45,230,163,0.4)" : "rgba(255,59,92,0.4)" }}
              >
                <div className="font-black" style={{ color: v ? "#2de6a3" : "#ff3b5c" }}>
                  {v ? "✓" : "✗"}
                </div>
                <div className="text-[9px] font-mono text-[color:var(--text-faint)]">R{i + 1}</div>
              </motion.div>
            ))}
            </div>
          </div>
        </div>
      </div>

      {hasSections ? (
        Object.entries(sections).map(([name, content], i) => {
          const cfg = SEC_CFG[name] ?? { Icon: ShieldAlert, color: "#8b94a3", sub: "" };
          return (
            <motion.div
              key={name}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.06 }}
              className="rounded-lg border-l-2 panel-inset overflow-hidden"
              style={{ borderColor: cfg.color }}
            >
              <div className="flex items-center gap-3 px-4 pt-3">
                <div
                  className="w-9 h-9 rounded-lg flex items-center justify-center shrink-0 border"
                  style={{ color: cfg.color, borderColor: `${cfg.color}4d`, background: `${cfg.color}14` }}
                >
                  <cfg.Icon size={16} />
                </div>
                <div>
                  <div className="text-xs font-extrabold font-mono uppercase tracking-wide" style={{ color: cfg.color }}>
                    {name}
                  </div>
                  <div className="text-[10px] opacity-70" style={{ color: cfg.color }}>
                    {cfg.sub}
                  </div>
                </div>
              </div>
              <div className="px-4 pb-4 pt-2 text-sm text-[color:var(--text-dim)] whitespace-pre-line leading-relaxed">
                {content}
              </div>
            </motion.div>
          );
        })
      ) : (
        <div className="text-sm text-[color:var(--text-dim)] whitespace-pre-line rounded-lg panel p-4">
          {llm.main_response}
        </div>
      )}

      {llm.responses?.length > 0 && (
        <div className="rounded-lg panel overflow-hidden">
          <button
            onClick={() => setShowAll((v) => !v)}
            className="w-full text-left px-4 py-3 font-bold text-[color:var(--text)] text-sm flex justify-between items-center"
          >
            <span>Read all 3 AI opinions in full — {validCount}/3 gave a complete answer</span>
            <motion.span animate={{ rotate: showAll ? 180 : 0 }}>
              <ChevronDown size={16} />
            </motion.span>
          </button>
          <AnimatePresence>
            {showAll && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="px-4 pb-4 space-y-3"
              >
                {llm.responses.map((resp, i) => {
                  const valid = llm.response_valid[i];
                  return (
                    <div
                      key={i}
                      className="rounded-lg border px-4 py-3 text-xs panel-inset"
                      style={{ borderColor: valid ? "rgba(255,255,255,0.1)" : "rgba(255,176,32,0.4)" }}
                    >
                      <div
                        className="font-bold mb-2 pb-2 border-b"
                        style={{ color: valid ? "var(--text)" : "var(--mod)", borderColor: "rgba(255,255,255,0.08)" }}
                      >
                        {valid ? `✓ Response ${i + 1}` : `⚠ Response ${i + 1} — incomplete`}
                      </div>
                      <div className="whitespace-pre-line text-[color:var(--text-dim)]">{resp}</div>
                    </div>
                  );
                })}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}

      <div className="rounded-r-lg border-l-2 border-[color:var(--accent)] bg-[color:var(--accent)]/6 px-4 py-3">
        <div className="text-xs font-bold text-[color:var(--accent)] mb-1">Clinical Disclaimer</div>
        <p className="text-xs text-[color:var(--text-dim)] leading-relaxed">
          This system is a <b className="text-[color:var(--text)]">decision support tool only</b>. It does not diagnose disease or
          replace the clinical judgment of qualified healthcare professionals. All AI-generated
          outputs must be reviewed by a licensed clinician before any clinical action is taken.
        </p>
      </div>
    </div>
  );
}
