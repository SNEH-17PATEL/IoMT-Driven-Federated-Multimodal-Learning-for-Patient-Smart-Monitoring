import { motion, AnimatePresence } from "framer-motion";
import { X, MousePointerClick, HeartPulse, Brain, FileText, ShieldCheck } from "lucide-react";

const STEPS = [
  {
    Icon: MousePointerClick,
    title: "1. Pick a patient",
    body: "Choose one of the three example patients below. Each one tells a different story — from a smooth recovery to a serious emergency.",
    color: "#22d3ee",
  },
  {
    Icon: HeartPulse,
    title: "2. Watch their vitals",
    body: "Every 30 seconds, a fresh reading comes in — heart rate, oxygen, blood pressure and more — just like a real bedside monitor.",
    color: "#ff3b5c",
  },
  {
    Icon: Brain,
    title: "3. The AI gives a health score",
    body: "An AI trained across multiple hospitals turns those vitals into a single 0–24 score. Lower is healthier, higher means more urgent care is needed.",
    color: "#a78bfa",
  },
  {
    Icon: FileText,
    title: "4. Get a plain-English report",
    body: "Three independent AI opinions are compared and combined into one clear write-up: what's happening, why, and what to do next.",
    color: "#ffb020",
  },
  {
    Icon: ShieldCheck,
    title: "5. See the technology (optional)",
    body: "Curious how it was built? The last two tabs explain — and even let you watch, live — how hospitals train this AI together without ever sharing patient data.",
    color: "#2de6a3",
  },
];

export default function HelpModal({ open, onClose }) {
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm"
          onClick={onClose}
        >
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.97 }}
            transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
            onClick={(e) => e.stopPropagation()}
            className="relative w-full max-w-lg rounded-xl panel-raised overflow-hidden max-h-[85vh] flex flex-col"
          >
            <div className="px-6 py-5 text-[color:var(--text)] relative shrink-0 border-b border-white/8">
              <div className="text-lg font-bold font-display">How this works</div>
              <p className="text-xs text-[color:var(--text-dim)] mt-1">
                No medical or tech background needed — here's the whole idea in five steps.
              </p>
              <button
                onClick={onClose}
                className="absolute top-4 right-4 text-[color:var(--text-dim)] hover:text-[color:var(--text)] rounded-full p-1 hover:bg-white/8"
                aria-label="Close"
              >
                <X size={18} />
              </button>
            </div>

            <div className="p-5 space-y-3 overflow-y-auto">
              {STEPS.map((s, i) => (
                <motion.div
                  key={s.title}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.06 }}
                  className="flex items-start gap-3 rounded-lg panel-inset p-3.5"
                >
                  <div
                    className="w-9 h-9 rounded-lg shrink-0 flex items-center justify-center border"
                    style={{ color: s.color, borderColor: `${s.color}4d`, background: `${s.color}14` }}
                  >
                    <s.Icon size={17} />
                  </div>
                  <div>
                    <div className="text-sm font-bold text-[color:var(--text)]">{s.title}</div>
                    <div className="text-xs text-[color:var(--text-dim)] mt-0.5 leading-relaxed">{s.body}</div>
                  </div>
                </motion.div>
              ))}
            </div>

            <div className="px-5 pb-5 shrink-0">
              <button
                onClick={onClose}
                className="w-full rounded-lg border border-[color:var(--accent)]/40 bg-[color:var(--accent)]/10 hover:bg-[color:var(--accent)]/18 text-[color:var(--accent)] font-bold text-sm py-3 transition-colors"
              >
                Got it — let's go
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
