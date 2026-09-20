import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  AlertOctagon, Brain, ChevronDown, Droplets, FlaskConical, Frown,
  Gauge, Heart, Info, Microscope, Thermometer, Wind,
} from "lucide-react";
import InfoTip from "../ui/InfoTip";
import { EXPLAIN } from "../../copy";

const RISK_ICONS = {
  "Low oxygen levels": Droplets,
  "Respiratory distress": Wind,
  "Organ failure": AlertOctagon,
  "Sepsis / Infection": FlaskConical,
  "Respiratory failure (intubated)": Wind,
  "Haemodynamic instability": Heart,
  Hypotension: Gauge,
  "Abnormal heart rate": Heart,
  "Altered mental status": Brain,
  "Neurological deterioration": Brain,
  "High physiological stress": Frown,
};

function featureIcon(feature) {
  const f = feature.toLowerCase();
  if (f.includes("spo2")) return Droplets;
  if (f.includes("hr")) return Heart;
  if (f.includes("rr")) return Wind;
  if (f.includes("temp")) return Thermometer;
  if (f.includes("sbp") || f.includes("dbp") || f.includes("map")) return Gauge;
  if (f.includes("gcs")) return Brain;
  if (f.includes("stress")) return Frown;
  return FlaskConical;
}

function ShapBar({ item, index }) {
  const inc = item.increases_risk;
  const color = inc ? "#ff3b5c" : "#2de6a3";
  const Icon = featureIcon(item.feature);

  return (
    <motion.div
      initial={{ opacity: 0, x: -16 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.04 }}
      className="flex items-center gap-4 rounded-r-lg border-l-2 px-4 py-3 panel-inset"
      style={{ borderColor: color }}
    >
      <div
        className="w-9 h-9 rounded-lg flex items-center justify-center shrink-0 border"
        style={{ color, borderColor: `${color}4d`, background: `${color}14` }}
      >
        <Icon size={16} />
      </div>
      <div className="min-w-[150px] shrink-0">
        <div className="text-sm font-bold text-[color:var(--text)]">{item.label}</div>
        <div className="text-[11px] font-mono text-[color:var(--text-faint)]">{item.value}</div>
      </div>
      <div className="flex-1 min-w-[80px] h-2.5 rounded-full bg-white/6 overflow-hidden">
        <motion.div
          className="h-full rounded-full"
          style={{ background: color }}
          initial={{ width: 0 }}
          animate={{ width: `${item.bar_pct}%` }}
          transition={{ duration: 0.7, delay: index * 0.04 + 0.1, ease: [0.16, 1, 0.3, 1] }}
        />
      </div>
      <div className="min-w-[110px] text-right shrink-0">
        <div className="text-sm font-bold font-mono" style={{ color }}>{item.abs_impact.toFixed(4)}</div>
        <div className="text-[11px] font-bold" style={{ color }}>
          {inc ? "↑ Increases Risk" : "↓ Reduces Risk"}
        </div>
      </div>
    </motion.div>
  );
}

export default function ExplainabilityTab({ reading }) {
  const [aboutOpen, setAboutOpen] = useState(false);
  const { shap, clinical_explanations, key_risks } = reading;

  if (!shap || shap.length === 0) {
    return (
      <div className="rounded-lg border border-[color:var(--accent)]/25 bg-[color:var(--accent)]/6 text-[color:var(--accent)] text-sm px-4 py-3 flex items-center gap-2">
        <Info size={16} /> SHAP background data not found. Run{" "}
        <code className="font-mono">python train_federated.py</code> once to generate SHAP background samples, then
        restart the backend.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <div className="text-base font-bold font-display text-[color:var(--text)] flex items-center gap-1.5">
          <Microscope size={17} className="text-[color:var(--violet)]" /> What drove this score?
          <InfoTip text={EXPLAIN.shap} />
        </div>
        <p className="text-xs text-[color:var(--text-dim)] mb-3">
          These are the vital signs that mattered most for this reading. A longer bar means it
          had a bigger effect — red pushed the score up (worse), green pulled it down (better).
        </p>
        <div className="space-y-2">
          {shap.map((item, i) => (
            <ShapBar key={i} item={item} index={i} />
          ))}
        </div>
      </div>

      <div>
        <div className="text-base font-bold font-display text-[color:var(--text)] mb-2">
          Clinical Interpretations
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {clinical_explanations.map((exp, i) => {
            const inc = exp.includes("increasing risk");
            const color = inc ? "#ff3b5c" : "#2de6a3";
            return (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
                className="flex items-start gap-2 rounded-md border px-3 py-2 text-xs text-[color:var(--text-dim)]"
                style={{ borderColor: `${color}33`, background: `${color}0d` }}
              >
                <span className="font-black" style={{ color }}>
                  {inc ? "↑" : "↓"}
                </span>
                <span>{exp}</span>
              </motion.div>
            );
          })}
        </div>
      </div>

      <div>
        <div className="text-base font-bold font-display text-[color:var(--text)] mb-2">
          Key Risk Factors
        </div>
        {key_risks.length > 0 ? (
          <div className="flex flex-wrap gap-2.5">
            {key_risks.map((risk, i) => {
              const Icon = RISK_ICONS[risk] ?? AlertOctagon;
              return (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, scale: 0.85 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: i * 0.06, type: "spring", stiffness: 300, damping: 20 }}
                  whileHover={{ y: -3 }}
                  className="flex items-center gap-2.5 rounded-lg border border-[color:var(--high)]/30 bg-[color:var(--high)]/6 px-4 py-2.5"
                >
                  <div className="w-8 h-8 rounded-md border border-[color:var(--high)]/40 bg-[color:var(--high)]/12 text-[color:var(--high)] flex items-center justify-center shrink-0">
                    <Icon size={15} />
                  </div>
                  <div>
                    <div className="text-xs font-extrabold text-[color:var(--high)]">{risk}</div>
                    <div className="text-[10px] text-[color:var(--text-faint)]">SHAP-identified risk factor</div>
                  </div>
                </motion.div>
              );
            })}
          </div>
        ) : (
          <div className="rounded-lg bg-[color:var(--low)]/8 border border-[color:var(--low)]/30 text-[color:var(--low)] text-xs px-3 py-2">
            No specific clinical risk factors flagged by the model for this reading.
          </div>
        )}
      </div>

      <div className="rounded-lg panel overflow-hidden">
        <button
          onClick={() => setAboutOpen((v) => !v)}
          className="w-full text-left px-4 py-3 font-bold text-[color:var(--text)] text-sm flex justify-between items-center"
        >
          <span className="flex items-center gap-1.5">
            <Info size={15} className="text-[color:var(--accent)]" /> About SHAP — understanding
            counterintuitive results
          </span>
          <motion.span animate={{ rotate: aboutOpen ? 180 : 0 }}>
            <ChevronDown size={16} />
          </motion.span>
        </button>
        <AnimatePresence>
          {aboutOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="px-4 pb-4 text-xs text-[color:var(--text-dim)] leading-relaxed space-y-2"
            >
              <p>
                <b className="text-[color:var(--text)]">What SHAP values represent:</b> SHAP (SHapley Additive exPlanations)
                quantifies how much each feature <i>shifted</i> this patient's predicted SOFA
                away from the model's baseline. ↑ means the feature pushed the prediction toward
                a higher (worse) SOFA score; ↓ means it pushed it lower (toward better).
              </p>
              <p>
                <b className="text-[color:var(--text)]">Why some results may seem counterintuitive:</b> SHAP explains what the{" "}
                <i>model</i> learned — not established clinical logic. MIMIC-III training data
                contains correlations that can differ from clinical intuition — e.g. a high
                Stress Score might appear as "reducing risk" because responsive/agitated
                patients often had lower SOFA than unresponsive ones in the training cohort.
              </p>
              <p>
                <b className="text-[color:var(--text)]">How to use SHAP correctly:</b> use the ranking to understand which signals
                drove this prediction; don't interpret individual directions as clinical ground
                truth. The AI Report tab integrates all information for holistic reasoning.
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
