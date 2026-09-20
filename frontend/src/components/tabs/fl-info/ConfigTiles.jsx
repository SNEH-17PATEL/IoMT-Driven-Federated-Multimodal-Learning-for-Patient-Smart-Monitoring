import { motion } from "framer-motion";

const COLORS = ["#22d3ee", "#a78bfa", "#2de6a3", "#ffb020", "#67e8f9", "#f472b6", "#ffcc4d", "#ff6b85"];

export default function ConfigTiles({ modelInfo: m }) {
  const tiles = [
    ["FL Rounds", m.num_rounds, "rounds of federation"],
    ["Hospitals", m.hospitals, "ICU sites"],
    ["Epochs / Round", m.epochs_per_round, "local training epochs"],
    ["Aggregation", m.aggregation, "weight averaging method"],
    ["Training Samples", m.train_samples?.toLocaleString(), "real ICU patients"],
    ["Test Samples", m.test_samples?.toLocaleString(), "held-out patients"],
    ["Best Round", m.best_round, "lowest eval loss"],
    ["Split Type", m.split_type, "data distribution"],
  ];

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
      {tiles.map(([label, value, sub], i) => {
        const color = COLORS[i % COLORS.length];
        return (
          <motion.div
            key={label}
            initial={{ opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: i * 0.04 }}
            whileHover={{ y: -3 }}
            className="rounded-lg p-3.5 text-center panel-inset"
            style={{ boxShadow: `inset 0 2px 0 0 ${color}` }}
          >
            <div className="text-[9px] font-mono font-bold uppercase tracking-wide text-[color:var(--text-faint)]">{label}</div>
            <div className="text-lg font-bold font-display leading-tight mt-1 break-words" style={{ color }}>{value}</div>
            <div className="text-[9px] text-[color:var(--text-faint)] mt-0.5">{sub}</div>
          </motion.div>
        );
      })}
    </div>
  );
}
