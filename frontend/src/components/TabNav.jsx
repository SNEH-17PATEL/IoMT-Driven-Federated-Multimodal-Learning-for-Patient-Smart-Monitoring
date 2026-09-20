import { motion } from "framer-motion";
import { Activity, Brain, Microscope, ShieldCheck, Zap } from "lucide-react";

const TABS = [
  { key: "risk", label: "Patient Status", sub: "Risk Assessment", Icon: Activity },
  { key: "explain", label: "Why This Score?", sub: "Explainability (SHAP)", Icon: Microscope },
  { key: "report", label: "AI Report", sub: "AI Clinical Report", Icon: Brain },
  { key: "flinfo", label: "How It Was Trained", sub: "Federated Learning Info", Icon: ShieldCheck },
  { key: "fldemo", label: "Watch AI Learn", sub: "Live FL Demo", Icon: Zap },
];

export default function TabNav({ active, onChange }) {
  return (
    <div className="flex flex-wrap gap-1 relative">
      {TABS.map((t) => {
        const isActive = active === t.key;
        return (
          <button
            key={t.key}
            onClick={() => onChange(t.key)}
            title={t.sub}
            className={`relative flex items-center gap-1.5 text-sm font-semibold px-4 py-2.5 rounded-t-lg transition-colors ${
              isActive ? "text-[color:var(--text)]" : "text-[color:var(--text-faint)] hover:text-[color:var(--text-dim)]"
            }`}
          >
            {isActive && (
              <motion.div
                layoutId="tab-pill"
                className="absolute inset-0 bg-[color:var(--bg-2)] rounded-t-lg border border-b-0 border-white/10"
                transition={{ type: "spring", stiffness: 400, damping: 32 }}
              />
            )}
            {isActive && (
              <motion.div
                layoutId="tab-underline"
                className="absolute left-3 right-3 -bottom-px h-[2px] bg-[color:var(--accent)]"
              />
            )}
            <span className="relative flex items-center gap-1.5">
              <t.Icon size={15} className={isActive ? "text-[color:var(--accent)]" : ""} />
              {t.label}
            </span>
          </button>
        );
      })}
    </div>
  );
}
