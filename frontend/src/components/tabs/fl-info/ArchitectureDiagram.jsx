import { motion } from "framer-motion";
import { Wrench } from "lucide-react";

const LAYER_COLOR = {
  INPUT: "#22d3ee",
  Linear: "#2de6a3",
  OUTPUT: "#ff6b85",
};

export default function ArchitectureDiagram({ modelInfo: m }) {
  const layers = [
    ["INPUT", `${m.input_features} features`, "Vitals (trends + latest + GCS) + SOFA-vocab TF-IDF"],
    ["Linear", `${m.input_features} → 128`, "Fully connected · ReLU activation"],
    ["Linear", "128 → 64", "Fully connected · ReLU activation"],
    ["Linear", "64 → 32", "Fully connected · ReLU activation"],
    ["Linear", "32 → 1", "Output layer · No activation (regression)"],
    ["OUTPUT", "SOFA (0–24)", "Predicted SOFA score · clip(0, 24)"],
  ];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
      <div className="lg:col-span-3 space-y-1">
        {layers.map(([type, dim, desc], i) => (
          <div key={i}>
            <motion.div
              initial={{ opacity: 0, x: -12 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.06 }}
              className="rounded-lg panel-inset flex items-center gap-3 px-4 py-2.5"
            >
              <span
                className="rounded font-mono text-[10px] font-extrabold px-2.5 py-1 whitespace-nowrap border"
                style={{ color: LAYER_COLOR[type], borderColor: `${LAYER_COLOR[type]}4d`, background: `${LAYER_COLOR[type]}14` }}
              >
                {type}
              </span>
              <span className="text-sm font-bold font-mono min-w-[90px] text-[color:var(--text)]">{dim}</span>
              <span className="text-xs text-[color:var(--text-faint)]">{desc}</span>
            </motion.div>
            {i < layers.length - 1 && (
              <div className="text-center text-[color:var(--text-faint)] text-lg leading-none py-0.5">↓</div>
            )}
          </div>
        ))}
      </div>
      <div className="lg:col-span-2 rounded-lg panel p-4 text-xs text-[color:var(--text-dim)] leading-relaxed">
        <div className="text-sm font-bold text-[color:var(--text)] mb-2 flex items-center gap-1.5">
          <Wrench size={14} className="text-[color:var(--accent)]" /> Training Details
        </div>
        <p><b className="text-[color:var(--text)]">Optimizer:</b> {m.optimizer ?? "AdamW (weight_decay=1e-4)"}</p>
        <p><b className="text-[color:var(--text)]">Loss:</b> {m.loss_weights ?? "Weighted MSE — high-SOFA patients get more gradient weight"}</p>
        <p><b className="text-[color:var(--text)]">No Dropout</b> — causes FL divergence; regularised by AdamW instead</p>
        <p><b className="text-[color:var(--text)]">FL Framework:</b> Flower (flwr)</p>
        <p><b className="text-[color:var(--text)]">Best round:</b> {m.best_round} of {m.num_rounds}</p>
      </div>
    </div>
  );
}
