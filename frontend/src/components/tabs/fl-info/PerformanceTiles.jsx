import { motion } from "framer-motion";
import AnimatedNumber from "../../ui/AnimatedNumber";

const COLORS = ["#22d3ee", "#2de6a3", "#67e8f9", "#ff6b85"];

export default function PerformanceTiles({ modelInfo: m }) {
  const tiles = [
    ["Mean Abs. Error", m.final_mae, 3, "SOFA points", "Average prediction error on held-out ICU patients"],
    ["R² Score", m.final_r2, 3, "variance", "Proportion of variance explained by the model"],
    ["Pred Range Min", m.pred_range_min, 1, "SOFA", "Lowest predicted SOFA across test set"],
    ["Pred Range Max", m.pred_range_max, 1, "SOFA", "Highest predicted SOFA across test set"],
  ];

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3.5">
      {tiles.map(([label, value, decimals, unit, desc], i) => {
        const color = COLORS[i];
        return (
          <motion.div
            key={label}
            initial={{ opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: i * 0.05 }}
            whileHover={{ y: -3 }}
            className="rounded-lg p-4 text-center panel-inset"
            style={{ boxShadow: `inset 0 2px 0 0 ${color}` }}
          >
            <div className="text-[9px] font-mono font-bold uppercase tracking-wide text-[color:var(--text-faint)] mb-1">{label}</div>
            <div className="text-3xl font-bold font-mono" style={{ color }}>
              <AnimatedNumber value={value} decimals={decimals} />
            </div>
            <div className="text-[10px] font-mono text-[color:var(--text-faint)]">{unit}</div>
            <div className="text-[9px] text-[color:var(--text-dim)] mt-1.5 leading-relaxed">{desc}</div>
          </motion.div>
        );
      })}
    </div>
  );
}
