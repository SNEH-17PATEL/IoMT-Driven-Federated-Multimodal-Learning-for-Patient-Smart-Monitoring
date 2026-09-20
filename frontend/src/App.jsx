import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Activity, HelpCircle } from "lucide-react";
import { api } from "./api";

import ConsoleBackdrop from "./components/ui/ConsoleBackdrop";
import HelpModal from "./components/ui/HelpModal";
import PatientSelector from "./components/PatientSelector";
import MonitorView from "./components/MonitorView";

const HELP_SEEN_KEY = "icu_monitor_help_seen";

function TopBar({ onHelp }) {
  return (
    <div className="border-b border-white/8 bg-[#0a0d13]/80 backdrop-blur px-6 py-3.5 flex items-center gap-3 sticky top-0 z-10">
      <div className="w-8 h-8 rounded-md bg-[#0e141d] border border-[color:var(--accent)]/40 flex items-center justify-center">
        <Activity size={15} className="text-[color:var(--accent)]" strokeWidth={2.5} />
      </div>
      <span className="font-bold font-display text-[15px] text-[color:var(--text)] tracking-tight">
        ICU CLINICAL CONSOLE
      </span>
      <span className="text-[11px] font-mono text-[color:var(--text-faint)] ml-1 hidden sm:inline">
        v2.0 · AI-ASSISTED MONITORING
      </span>
      <button
        onClick={onHelp}
        className="ml-auto flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-[color:var(--accent)] hover:text-[color:var(--accent-2)] border border-[color:var(--accent)]/25 hover:border-[color:var(--accent)]/50 bg-[color:var(--accent)]/5 hover:bg-[color:var(--accent)]/10 rounded-md px-3 py-1.5 transition-colors"
      >
        <HelpCircle size={13} /> How this works
      </button>
    </div>
  );
}

export default function App() {
  const [patients, setPatients] = useState([]);
  const [modelInfo, setModelInfo] = useState(null);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [ready, setReady] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);

  useEffect(() => {
    Promise.all([api.getPatients(), api.getModelInfo()])
      .then(([p, m]) => {
        setPatients(p);
        setModelInfo(m);
      })
      .finally(() => setReady(true));

    try {
      if (!localStorage.getItem(HELP_SEEN_KEY)) {
        setHelpOpen(true);
      }
    } catch {
      // localStorage unavailable — just skip the auto-open, no big deal
    }
  }, []);

  function closeHelp() {
    setHelpOpen(false);
    try {
      localStorage.setItem(HELP_SEEN_KEY, "1");
    } catch {
      // ignore
    }
  }

  if (!ready) {
    return (
      <div className="min-h-screen flex items-center justify-center relative">
        <ConsoleBackdrop />
        <div className="relative z-[1] flex items-center gap-2.5 text-[color:var(--text-dim)] text-sm font-mono">
          <motion.span
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            className="inline-block w-3.5 h-3.5 border-2 border-[color:var(--accent)]/30 border-t-[color:var(--accent)] rounded-full"
          />
          BOOTING ICU CONSOLE…
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex relative">
      <ConsoleBackdrop />
      <HelpModal open={helpOpen} onClose={closeHelp} />
      <div className="relative z-[1] flex-1 flex">
        <AnimatePresence mode="wait">
          {selectedPatient === null ? (
            <motion.div
              key="selector"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex-1 flex flex-col"
            >
              <TopBar onHelp={() => setHelpOpen(true)} />
              <PatientSelector patients={patients} onSelect={setSelectedPatient} />
            </motion.div>
          ) : (
            <motion.div
              key="monitor"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex-1 flex"
            >
              <MonitorView
                patientId={selectedPatient}
                patients={patients}
                modelInfo={modelInfo}
                onStop={() => setSelectedPatient(null)}
                onHelp={() => setHelpOpen(true)}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
