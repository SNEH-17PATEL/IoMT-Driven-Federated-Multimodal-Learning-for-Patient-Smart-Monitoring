import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { HelpCircle } from "lucide-react";

// A small "?" affordance that reveals a plain-language explanation on click —
// used everywhere a technical term (SOFA, SHAP, FedAvg, ...) would otherwise
// need to be looked up elsewhere.
export default function InfoTip({ text, side = "top", size = 14, className = "", light = false }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    if (!open) return;
    function onDocClick(e) {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    }
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [open]);

  return (
    <span className={`relative inline-flex align-middle ${className}`} ref={ref}>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          setOpen((v) => !v);
        }}
        className={`transition-colors rounded-full ${
          light ? "text-white/70 hover:text-white" : "text-[color:var(--text-faint)] hover:text-[color:var(--accent)]"
        }`}
        aria-label="What does this mean?"
      >
        <HelpCircle size={size} />
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: side === "top" ? 4 : -4, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: side === "top" ? 4 : -4, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className={`absolute z-50 ${
              side === "top" ? "bottom-full mb-2.5" : "top-full mt-2.5"
            } left-1/2 -translate-x-1/2 w-60 rounded-lg bg-[color:var(--bg-3)] text-[color:var(--text)] text-[11.5px] leading-relaxed px-3.5 py-3 shadow-2xl border border-white/12`}
          >
            {text}
            <div
              className={`absolute left-1/2 -translate-x-1/2 w-2.5 h-2.5 bg-[color:var(--bg-3)] rotate-45 ${
                side === "top" ? "top-full -mt-1.5" : "bottom-full -mb-1.5"
              }`}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </span>
  );
}
