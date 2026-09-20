import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { AlertTriangle, CheckCircle2, Radio, Zap } from "lucide-react";
import FLProgressChart from "../charts/FLProgressChart";
import FlowDiagram from "./fl-demo/FlowDiagram";
import HospitalCard from "./fl-demo/HospitalCard";
import DemoControls from "./fl-demo/DemoControls";
import { flDemoSocketUrl } from "../../api";
import { HOSPITAL_GRADIENTS } from "../../theme";

const initialForm = { rounds: 15, epochs: 1, hospital_size: 500 };

export default function FLLiveDemoTab() {
  const [status, setStatus] = useState("idle"); // idle | connecting | running | done
  const [initInfo, setInitInfo] = useState(null);
  const [currentRound, setCurrentRound] = useState(0);
  const [hospitals, setHospitals] = useState({});
  const [trainingIds, setTrainingIds] = useState(new Set());
  const [aggregating, setAggregating] = useState(false);
  const [history, setHistory] = useState([]);
  const [summary, setSummary] = useState(null);
  const [form, setForm] = useState(initialForm);
  const wsRef = useRef(null);
  const hospitalIdsRef = useRef([]);
  const aggTimerRef = useRef(null);

  useEffect(() => () => {
    wsRef.current?.close();
    clearTimeout(aggTimerRef.current);
  }, []);

  function start() {
    setStatus("connecting");
    setHistory([]);
    setSummary(null);
    setCurrentRound(0);

    const ws = new WebSocket(flDemoSocketUrl());
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(JSON.stringify({ action: "start", ...form }));
    };

    ws.onmessage = (evt) => {
      const msg = JSON.parse(evt.data);
      switch (msg.type) {
        case "init": {
          setInitInfo(msg);
          const h = {};
          msg.hospitals.forEach((hh) => (h[hh.id] = { ...hh, loss: null, mae: null, r2: null }));
          hospitalIdsRef.current = msg.hospitals.map((hh) => hh.id);
          setHospitals(h);
          setStatus("running");
          break;
        }
        case "round_start":
          setCurrentRound(msg.round);
          setTrainingIds(new Set(hospitalIdsRef.current));
          break;
        case "hospital_update":
          setHospitals((prev) => ({
            ...prev,
            [msg.hospital.id]: { ...prev[msg.hospital.id], ...msg.hospital },
          }));
          setTrainingIds((prev) => {
            const next = new Set(prev);
            next.delete(msg.hospital.id);
            return next;
          });
          break;
        case "aggregated":
          setHistory((prev) => [...prev, { round: msg.round, ...msg.global, is_best: msg.is_best }]);
          setAggregating(true);
          clearTimeout(aggTimerRef.current);
          aggTimerRef.current = setTimeout(() => setAggregating(false), 650);
          break;
        case "done":
          setSummary(msg);
          setStatus("done");
          break;
        case "cancelled":
          setStatus("idle");
          break;
        default:
          break;
      }
    };

    ws.onerror = () => setStatus("idle");
  }

  function stop() {
    wsRef.current?.send(JSON.stringify({ action: "cancel" }));
    wsRef.current?.close();
    setStatus("idle");
  }

  const isRunning = status === "running" || status === "connecting";
  const hospitalList = Object.values(hospitals);

  return (
    <div className="space-y-6">
      <motion.div
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative overflow-hidden rounded-xl panel-raised p-6"
      >
        <div className="pointer-events-none absolute -top-14 -left-10 w-56 h-56 rounded-full bg-[color:var(--violet)]/15 blur-3xl" />
        <div className="pointer-events-none absolute bottom-0 right-0 w-56 h-40 rounded-full bg-[color:var(--accent)]/12 blur-3xl" />
        <div className="relative flex items-center gap-2 text-xl font-bold font-display mb-1 text-[color:var(--text)]">
          <Zap size={20} className="text-[color:var(--mod)]" /> Watch the AI Learn — Live
        </div>
        <p className="relative text-sm text-[color:var(--text-dim)] max-w-2xl">
          Press play below and watch, step by step, as three hospitals each teach the AI using
          their own patients, then combine what they learned into one smarter, shared AI —
          without any hospital ever sending patient data anywhere.
        </p>
        <p className="relative flex items-start gap-1.5 text-xs text-[color:var(--mod)] bg-[color:var(--mod)]/8 border border-[color:var(--mod)]/30 rounded-lg px-3 py-2 mt-3 max-w-2xl">
          <AlertTriangle size={14} className="mt-0.5 shrink-0" />
          This demo uses made-up practice data so it runs instantly in your browser — the real
          hospital data is protected and too large to ship here. For the real model's actual
          results, see the <b className="whitespace-nowrap">How It Was Trained</b> tab.
        </p>
      </motion.div>

      <DemoControls form={form} setForm={setForm} isRunning={isRunning} onStart={start} onStop={stop} />

      {status !== "idle" && (
        <>
          <div className="flex items-center justify-between rounded-lg panel-raised px-5 py-3">
            <div className="flex items-center gap-2 text-sm font-bold font-mono text-[color:var(--text)]">
              {isRunning && <Radio size={15} className="text-[color:var(--high)] animate-pulse" />}
              Round {currentRound || 0} / {initInfo?.rounds ?? form.rounds}
            </div>
            {summary && (
              <div className="text-xs text-[color:var(--low)] font-semibold flex items-center gap-1.5">
                <CheckCircle2 size={14} /> Done — best round {summary.best_round}, MAE{" "}
                {summary.best_mae}
              </div>
            )}
          </div>

          <FlowDiagram hospitals={hospitalList} trainingIds={trainingIds} aggregating={aggregating} />

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {hospitalList.map((h, i) => (
              <HospitalCard key={h.id} h={h} gradient={HOSPITAL_GRADIENTS[i % HOSPITAL_GRADIENTS.length]} training={trainingIds.has(h.id)} />
            ))}
          </div>

          {history.length > 0 && (
            <div className="rounded-lg panel p-4">
              <div className="text-sm font-bold text-[color:var(--text)] mb-2">
                Global Model — FedAvg Aggregation Across Rounds
              </div>
              <FLProgressChart history={history} />
            </div>
          )}
        </>
      )}
    </div>
  );
}
