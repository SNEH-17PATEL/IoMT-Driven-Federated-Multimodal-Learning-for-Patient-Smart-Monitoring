import { motion } from "framer-motion";
import { Play, Square } from "lucide-react";

function Field({ label, value, onChange, options, disabled }) {
  return (
    <label className="text-xs">
      <div className="text-[color:var(--text-faint)] font-mono font-semibold mb-1 uppercase tracking-wide">{label}</div>
      <select
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="rounded-md border border-white/12 px-2.5 py-1.5 text-sm disabled:opacity-50 bg-[color:var(--bg-2)] text-[color:var(--text)] font-mono"
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </label>
  );
}

export default function DemoControls({ form, setForm, isRunning, onStart, onStop }) {
  return (
    <div className="flex flex-wrap items-end gap-4 rounded-lg panel p-4">
      <Field label="How many rounds" value={form.rounds} onChange={(v) => setForm((f) => ({ ...f, rounds: v }))} options={[8, 15, 25, 40]} disabled={isRunning} />
      <Field label="Practice per round" value={form.epochs} onChange={(v) => setForm((f) => ({ ...f, epochs: v }))} options={[1, 2, 3]} disabled={isRunning} />
      <Field label="Patients per hospital" value={form.hospital_size} onChange={(v) => setForm((f) => ({ ...f, hospital_size: v }))} options={[300, 500, 1000]} disabled={isRunning} />

      {!isRunning ? (
        <motion.button
          whileTap={{ scale: 0.97 }}
          onClick={onStart}
          className="ml-auto flex items-center gap-1.5 rounded-lg border border-[color:var(--accent)]/40 bg-[color:var(--accent)]/12 hover:bg-[color:var(--accent)]/20 text-[color:var(--accent)] font-bold text-sm px-5 py-2.5 transition-colors"
        >
          <Play size={15} /> Start Watching
        </motion.button>
      ) : (
        <motion.button
          whileTap={{ scale: 0.97 }}
          onClick={onStop}
          className="ml-auto flex items-center gap-1.5 rounded-lg border border-[color:var(--high)]/40 bg-[color:var(--high)]/12 hover:bg-[color:var(--high)]/20 text-[color:var(--high)] font-bold text-sm px-5 py-2.5 transition-colors"
        >
          <Square size={13} /> Stop
        </motion.button>
      )}
    </div>
  );
}
