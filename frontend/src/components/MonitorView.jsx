import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle, ClipboardList, RotateCcw } from "lucide-react";
import { useMonitorLoop } from "../hooks/useMonitorLoop";

import Sidebar from "./Sidebar";
import MonitorHeader from "./MonitorHeader";
import CountdownBar from "./CountdownBar";
import VitalsGrid from "./VitalsGrid";
import TabNav from "./TabNav";
import RiskAssessmentTab from "./tabs/RiskAssessmentTab";
import ExplainabilityTab from "./tabs/ExplainabilityTab";
import AIReportTab from "./tabs/AIReportTab";
import FLInfoTab from "./tabs/FLInfoTab";
import FLLiveDemoTab from "./tabs/FLLiveDemoTab";

function useClock() {
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return now.toLocaleTimeString("en-US", { hour12: false });
}

function AlertBanner({ alert }) {
  if (!alert) return null;
  const isError = alert.level === "error";
  const color = isError ? "var(--high)" : "var(--mod)";
  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-lg px-4 py-3 text-sm font-semibold border flex items-center gap-2"
      style={{ color, borderColor: `${color}4d`, background: `${color}14` }}
    >
      <AlertTriangle size={16} className="shrink-0" />
      {alert.message}
    </motion.div>
  );
}

function LoadingCard() {
  return (
    <div className="rounded-lg panel px-6 py-16 text-center relative overflow-hidden">
      <div className="shimmer absolute inset-0" />
      <div className="relative text-[color:var(--accent)] font-mono font-semibold text-sm animate-pulse">
        ⚙ Loading first reading — running SHAP + AI clinical analysis…
      </div>
    </div>
  );
}

const TAB_CONTENT = {
  risk: RiskAssessmentTab,
  explain: ExplainabilityTab,
  report: AIReportTab,
};

export default function MonitorView({ patientId, patients, modelInfo, onStop, onHelp }) {
  const { reading, loading, error, secondsLeft, retry } = useMonitorLoop(patientId);
  const clock = useClock();
  const [tab, setTab] = useState("risk");
  const patientCfg = patients.find((p) => p.id === patientId);

  const ReadingTab = TAB_CONTENT[tab];

  return (
    <div className="flex-1 flex">
      <Sidebar patient={patientCfg} modelInfo={modelInfo} onStop={onStop} onHelp={onHelp} />
      <main className="flex-1 px-6 py-6 max-w-6xl mx-auto w-full space-y-4">
        <MonitorHeader patient={patientCfg} clock={clock} />
        <CountdownBar loading={loading} secondsLeft={secondsLeft} />

        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="rounded-lg border border-[color:var(--high)]/30 bg-[color:var(--high)]/8 text-[color:var(--high)] px-4 py-3 text-sm flex items-center justify-between"
            >
              <span className="flex items-center gap-2">
                <AlertTriangle size={16} /> {error}
              </span>
              <button
                onClick={retry}
                className="flex items-center gap-1.5 rounded-md bg-[color:var(--high)]/15 hover:bg-[color:var(--high)]/25 border border-[color:var(--high)]/40 text-[color:var(--high)] text-xs font-semibold px-3 py-1.5"
              >
                <RotateCcw size={12} /> Retry
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {!reading && !error && <LoadingCard />}

        {reading && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
            <AlertBanner alert={reading.alert} />
            <VitalsGrid vitals={reading.vitals} normalRanges={reading.normal_ranges} />

            <details className="rounded-lg panel px-4 py-2 text-xs text-[color:var(--text-dim)]">
              <summary className="cursor-pointer font-semibold text-[color:var(--text)] flex items-center gap-1.5 select-none">
                <ClipboardList size={13} className="text-[color:var(--accent)]" /> Clinical Notes
              </summary>
              <p className="mt-2 leading-relaxed">{reading.clinical_note}</p>
              <p className="mt-1 text-[color:var(--text-faint)]">Stress Score: {reading.vitals.stress}/10</p>
            </details>

            <TabNav active={tab} onChange={setTab} />
            <div className="relative panel-raised rounded-b-xl rounded-tr-xl p-5 min-h-[200px] overflow-hidden">
              <motion.div
                key={tab}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
              >
                {tab === "flinfo" ? (
                  <FLInfoTab modelInfo={modelInfo} />
                ) : tab === "fldemo" ? (
                  <FLLiveDemoTab />
                ) : (
                  ReadingTab && <ReadingTab reading={reading} />
                )}
              </motion.div>
            </div>
          </motion.div>
        )}
      </main>
    </div>
  );
}
