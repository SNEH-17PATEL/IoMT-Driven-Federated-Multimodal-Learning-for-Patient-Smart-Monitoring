import { motion } from "framer-motion";
import { CheckCircle2, Settings } from "lucide-react";
import InfoTip from "../../ui/InfoTip";
import { EXPLAIN } from "../../../copy";

const STEPS = [
  ["①", "Compute update", "local_weights − global_weights", "#22d3ee"],
  ["②", "Clip L2 norm", "Bounds any patient's max influence (sensitivity S)", "#ffb020"],
  ["③", "Add noise", "Gaussian N(0, (σ·S)²) to every weight parameter", "#a78bfa"],
  ["④", "Transmit", "Server receives noisy update — cannot trace individuals", "#2de6a3"],
];

export default function PrivacyPanel({ modelInfo: m }) {
  if (m.differential_privacy) {
    return (
      <div className="rounded-lg border border-[color:var(--low)]/35 bg-[color:var(--low)]/8 px-5 py-4">
        <div className="text-sm font-extrabold text-[color:var(--low)] mb-1 flex items-center gap-1.5">
          <CheckCircle2 size={16} /> Extra Privacy Protection: ON
          <InfoTip text={EXPLAIN.differentialPrivacy} />
        </div>
        <div className="flex flex-wrap gap-4 text-xs text-[color:var(--text-dim)] font-mono">
          <span><b className="text-[color:var(--text)]">σ:</b> {m.dp_sigma}</span>
          <span><b className="text-[color:var(--text)]">S:</b> {m.dp_sensitivity}</span>
          <span><b className="text-[color:var(--text)]">ε:</b> ≈{m.dp_epsilon}</span>
          <span><b className="text-[color:var(--text)]">δ:</b> {m.dp_delta}</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="rounded-lg border border-[color:var(--mod)]/35 bg-[color:var(--mod)]/8 px-5 py-3.5">
        <div className="text-sm font-bold text-[color:var(--mod)] flex items-center gap-1.5">
          <Settings size={15} /> Extra Privacy Protection: Off for now
          <InfoTip text={EXPLAIN.differentialPrivacy} />
        </div>
        <p className="text-xs text-[color:var(--text-dim)] mt-1">
          Model trained with plain FedAvg (no noise). Enable by setting{" "}
          <code className="font-mono">USE_DP = True</code> in train_federated.py and retraining.
        </p>
      </div>
      <div className="text-xs font-bold text-[color:var(--text)]">
        How it would work (technical, for the curious):
      </div>
      {STEPS.map(([step, title, desc, color], i) => (
        <motion.div
          key={step}
          initial={{ opacity: 0, x: -10 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ delay: i * 0.08 }}
          className="flex items-start gap-3 rounded-lg panel-inset px-3.5 py-2.5"
        >
          <div
            className="rounded-full w-6 h-6 flex items-center justify-center text-[11px] font-extrabold shrink-0 border"
            style={{ color, borderColor: `${color}4d`, background: `${color}14` }}
          >
            {step}
          </div>
          <div>
            <div className="text-xs font-bold" style={{ color }}>{title}</div>
            <div className="text-[11px] text-[color:var(--text-faint)]">{desc}</div>
          </div>
        </motion.div>
      ))}
    </div>
  );
}
