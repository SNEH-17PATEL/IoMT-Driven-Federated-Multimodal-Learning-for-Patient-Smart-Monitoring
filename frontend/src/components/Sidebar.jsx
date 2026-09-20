import { motion } from "framer-motion";
import { Activity, HelpCircle, Radio, ShieldCheck, Square } from "lucide-react";
import InfoTip from "./ui/InfoTip";
import { EXPLAIN } from "../copy";

function InfoRow({ label, value, tip }) {
  return (
    <div className="flex items-start justify-between gap-3 text-[11px]">
      <span className="text-[color:var(--text-faint)] flex items-center gap-1">
        {label}
        {tip && <InfoTip text={tip} side="top" size={12} />}
      </span>
      <span className="text-[color:var(--text)] font-mono font-semibold text-right">{value}</span>
    </div>
  );
}

export default function Sidebar({ patient, modelInfo, onStop, onHelp }) {
  return (
    <aside className="w-72 shrink-0 relative overflow-hidden text-[color:var(--text-dim)] min-h-screen flex flex-col bg-[#08090e] border-r border-white/8">
      <div className="pointer-events-none absolute -top-24 -left-16 w-64 h-64 rounded-full bg-[color:var(--accent)]/10 blur-3xl" />
      <div className="pointer-events-none absolute bottom-0 -right-10 w-56 h-56 rounded-full bg-[color:var(--violet)]/10 blur-3xl" />

      <div className="relative px-6 py-7 border-b border-white/8">
        <motion.div
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-3"
        >
          <div className="w-10 h-10 rounded-lg bg-[#0e141d] border border-[color:var(--accent)]/40 flex items-center justify-center">
            <Activity size={18} className="text-[color:var(--accent)]" strokeWidth={2.5} />
          </div>
          <div>
            <div className="font-bold font-display text-[color:var(--text)] leading-tight tracking-tight">
              ICU Console
            </div>
            <div className="text-[10.5px] font-mono text-[color:var(--text-faint)]">
              CONTINUOUS PATIENT MONITORING
            </div>
          </div>
        </motion.div>
        {onHelp && (
          <button
            onClick={onHelp}
            className="mt-4 w-full flex items-center justify-center gap-1.5 text-xs font-semibold rounded-md border border-white/10 hover:border-[color:var(--accent)]/50 hover:bg-[color:var(--accent)]/8 text-[color:var(--text-dim)] hover:text-[color:var(--accent)] py-2 transition-colors"
          >
            <HelpCircle size={13} /> How this works
          </button>
        )}
      </div>

      {patient && (
        <motion.div
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative mx-4 mt-5 rounded-lg p-4 panel"
        >
          <div className="flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-widest text-[color:var(--accent)] font-bold mb-2">
            <Radio size={12} className="animate-pulse" /> Now Watching
          </div>
          <div className="text-sm font-bold text-[color:var(--text)]">
            {patient.icon} {patient.name}
          </div>
          <button
            onClick={onStop}
            className="mt-3 w-full flex items-center justify-center gap-1.5 text-xs font-semibold rounded-md border border-white/10 hover:border-[color:var(--high)]/50 hover:bg-[color:var(--high)]/8 text-[color:var(--text-dim)] hover:text-[color:var(--high)] py-2 transition-colors"
          >
            <Square size={12} /> Stop &amp; Pick Another Patient
          </button>
        </motion.div>
      )}

      {modelInfo && (
        <div className="relative mx-4 mt-4 rounded-lg p-4 panel space-y-2.5">
          <div className="flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-widest font-bold text-[color:var(--violet)] mb-1">
            <ShieldCheck size={12} /> About This AI
          </div>
          <InfoRow
            label="Accuracy"
            tip={EXPLAIN.mae}
            value={`±${Math.round(modelInfo.final_mae)} pts`}
          />
          <InfoRow
            label="Trained on"
            value={`${modelInfo.train_samples?.toLocaleString()} real patients`}
          />
          <InfoRow
            label="Built by"
            tip={EXPLAIN.federatedLearning}
            value={`${modelInfo.hospitals} hospitals`}
          />
          <InfoRow
            label="Data shared?"
            value={<span className="text-[color:var(--low)]">Never</span>}
          />
        </div>
      )}

      <div className="mt-auto relative px-6 py-5 text-[10.5px] font-mono text-[color:var(--text-faint)] flex items-center gap-1.5 border-t border-white/8">
        <Activity size={12} /> A support tool to help you keep watch — not a diagnosis.
      </div>
    </aside>
  );
}
