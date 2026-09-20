import { motion } from "framer-motion";
import { Lock, Server } from "lucide-react";
import { HOSPITAL_GRADIENTS } from "../../../theme";
import InfoTip from "../../ui/InfoTip";
import { EXPLAIN } from "../../../copy";

function HospitalCard({ name, sub, gradient, delay }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      whileHover={{ y: -4 }}
      className="rounded-lg p-4 text-center flex-1 relative overflow-hidden panel-inset"
    >
      <div className={`absolute inset-x-0 top-0 h-0.5 bg-gradient-to-r ${gradient}`} />
      <div className={`w-10 h-10 mx-auto rounded-lg bg-gradient-to-br ${gradient} flex items-center justify-center text-lg mb-2`}>
        🏥
      </div>
      <div className="text-xs font-bold text-[color:var(--text)]">{name}</div>
      <div className="text-[10px] text-[color:var(--text-faint)]">{sub}</div>
      <div className="text-[11px] font-mono font-bold mt-1.5 text-[color:var(--text-dim)]">~16K patients</div>
      <div className="text-[9px] font-mono text-[color:var(--text-faint)] mt-2 bg-black/30 rounded-full px-2 py-1 flex items-center justify-center gap-1">
        <Lock size={9} /> PRIVATE data
      </div>
    </motion.div>
  );
}

export default function ProtocolDiagram({ modelInfo: m }) {
  const names = m.hospital_names ?? ["General ICU", "Mixed ICU", "Cardiac/Trauma ICU"];

  return (
    <div className="relative rounded-xl overflow-hidden panel-raised p-6">
      <div className="pointer-events-none absolute -top-16 -left-10 w-48 h-48 rounded-full bg-[color:var(--accent)]/12 blur-3xl" />
      <div className="pointer-events-none absolute bottom-0 right-0 w-56 h-40 rounded-full bg-[color:var(--violet)]/12 blur-3xl" />

      <div className="relative flex justify-center mb-4">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="rounded-xl border-2 border-[color:var(--accent)]/50 bg-[color:var(--accent)]/8 px-9 py-4 text-center glow-accent"
        >
          <Server className="mx-auto mb-1 text-[color:var(--accent)] animate-float" size={22} />
          <div className="text-sm font-bold text-[color:var(--accent)] flex items-center justify-center gap-1.5">
            Shared AI Brain <InfoTip text={EXPLAIN.federatedLearning} />
          </div>
          <div className="text-[11px] text-[color:var(--text-faint)]">Starts the same for every hospital</div>
        </motion.div>
      </div>

      <div className="relative text-center text-[color:var(--accent)] text-xs font-mono font-semibold mb-3">
        ① Send a copy of the AI to each hospital ↓↓↓
      </div>

      <div className="relative flex flex-col sm:flex-row gap-3">
        {names.map((n, i) => (
          <HospitalCard
            key={n}
            name={`Hospital ${i}`}
            sub={n}
            gradient={HOSPITAL_GRADIENTS[i % HOSPITAL_GRADIENTS.length]}
            delay={i * 0.1}
          />
        ))}
      </div>

      <div className="relative text-center text-[11px] text-[color:var(--text-dim)] my-4 leading-relaxed">
        ② Each hospital teaches it using only their own patients (nothing leaves the building)
        <br />
        <span className="text-[color:var(--accent)] font-semibold">
          ③ Only what was learned comes back — zero patient records, ever ↑↑↑
        </span>
      </div>

      <div className="relative flex justify-center">
        <motion.div
          whileHover={{ scale: 1.02 }}
          className="rounded-xl border-2 border-[color:var(--violet)]/50 bg-[color:var(--violet)]/8 px-8 py-4 text-center"
        >
          <div className="text-sm font-bold text-[color:var(--violet)] flex items-center justify-center gap-1.5">
            ④ Combine everyone's learning into one smarter AI
            <InfoTip text={`Technical name: ${m.aggregation}`} />
          </div>
          <div className="text-[11px] text-[color:var(--text-faint)] mt-1">
            → Repeat {m.num_rounds} times until the AI gets really good
          </div>
        </motion.div>
      </div>
    </div>
  );
}
