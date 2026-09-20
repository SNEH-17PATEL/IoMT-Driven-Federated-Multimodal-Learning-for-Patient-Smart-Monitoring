import { motion } from "framer-motion";
import { BrainCircuit, Lock, ShieldCheck, Sliders } from "lucide-react";
import ProtocolDiagram from "./fl-info/ProtocolDiagram";
import ConfigTiles from "./fl-info/ConfigTiles";
import PerformanceTiles from "./fl-info/PerformanceTiles";
import ArchitectureDiagram from "./fl-info/ArchitectureDiagram";
import PrivacyPanel from "./fl-info/PrivacyPanel";

function SectionHeading({ Icon, children }) {
  return (
    <div className="text-base font-bold font-display text-[color:var(--text)] mb-3 flex items-center gap-1.5">
      <Icon size={17} className="text-[color:var(--violet)]" /> {children}
    </div>
  );
}

export default function FLInfoTab({ modelInfo }) {
  const m = modelInfo;

  return (
    <div className="space-y-8">
      <motion.div
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative overflow-hidden rounded-xl panel-raised p-6"
      >
        <div className="pointer-events-none absolute -top-10 right-0 w-56 h-56 rounded-full bg-[color:var(--violet)]/15 blur-3xl" />
        <div className="relative flex items-center gap-2 text-xl font-bold font-display mb-1 text-[color:var(--text)]">
          <Lock size={20} className="text-[color:var(--accent)]" /> Federated Learning — How This Model Was
          Trained
        </div>
        <p className="relative text-sm text-[color:var(--text-dim)] max-w-2xl">
          Privacy-preserving collaborative AI across {m.hospitals} hospital ICUs. Patient data{" "}
          <b className="text-[color:var(--accent)]">never leaves</b> each hospital — only model weights are
          shared.
        </p>
      </motion.div>

      <ProtocolDiagram modelInfo={m} />

      <div>
        <SectionHeading Icon={Sliders}>Training Configuration</SectionHeading>
        <ConfigTiles modelInfo={m} />
      </div>

      <div>
        <SectionHeading Icon={BrainCircuit}>Global Model Performance</SectionHeading>
        <PerformanceTiles modelInfo={m} />
      </div>

      <div>
        <SectionHeading Icon={BrainCircuit}>Model Architecture (PyTorch DNN)</SectionHeading>
        <ArchitectureDiagram modelInfo={m} />
      </div>

      <div>
        <SectionHeading Icon={ShieldCheck}>Differential Privacy</SectionHeading>
        <PrivacyPanel modelInfo={m} />
      </div>
    </div>
  );
}
