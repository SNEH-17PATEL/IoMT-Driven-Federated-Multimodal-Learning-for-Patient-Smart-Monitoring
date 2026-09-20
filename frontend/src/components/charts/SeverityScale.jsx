import { motion } from "framer-motion";
import { Gauge } from "lucide-react";

export default function SeverityScale({ score }) {
  const pct = Math.min(Math.max(score / 24, 0), 1);
  const lowW = (4 / 24) * 300;
  const modW = (5 / 24) * 300;
  const hiW = 300 - lowW - modW;
  const marker = pct * 300;
  const markerColor = score < 5 ? "#2de6a3" : score < 10 ? "#ffb020" : "#ff3b5c";

  return (
    <div className="rounded-lg panel px-5 py-4">
      <div className="text-[10.5px] font-mono font-bold uppercase tracking-wide text-[color:var(--text-faint)] mb-2.5 flex items-center gap-1.5">
        <Gauge size={13} className="text-[color:var(--accent)]" /> SOFA Severity Scale — Current Reading
      </div>
      <svg width="100%" viewBox="0 0 300 54" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="sev-low" x1="0" x2="1">
            <stop offset="0%" stopColor="#2de6a3" />
            <stop offset="100%" stopColor="#0fae76" />
          </linearGradient>
          <linearGradient id="sev-mod" x1="0" x2="1">
            <stop offset="0%" stopColor="#ffcc4d" />
            <stop offset="100%" stopColor="#e08900" />
          </linearGradient>
          <linearGradient id="sev-hi" x1="0" x2="1">
            <stop offset="0%" stopColor="#ff6b85" />
            <stop offset="100%" stopColor="#e0173d" />
          </linearGradient>
        </defs>
        <rect x="0" y="14" width={lowW} height="20" fill="url(#sev-low)" rx="4" opacity="0.9" />
        <rect x={lowW} y="14" width={modW} height="20" fill="url(#sev-mod)" opacity="0.9" />
        <rect x={lowW + modW} y="14" width={hiW} height="20" fill="url(#sev-hi)" rx="4" opacity="0.9" />

        <motion.g initial={false} animate={{ x: marker }} transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}>
          <circle cx={0} cy={24} r={9} fill={markerColor} opacity={0.25}>
            <animate attributeName="r" values="9;13;9" dur="1.6s" repeatCount="indefinite" />
          </circle>
          <polygon points="0,7 -5,14 5,14" fill="#06080c" stroke={markerColor} strokeWidth="1.4" />
          <rect x={-2.5} y="12" width="5" height="24" fill="#06080c" rx="2.5" stroke={markerColor} strokeWidth="1.4" />
        </motion.g>

        <text x="4" y="50" fontSize="9" fill="#2de6a3" fontWeight="700" fontFamily="monospace">
          LOW (0-4)
        </text>
        <text x={lowW + 4} y="50" fontSize="9" fill="#ffb020" fontWeight="700" fontFamily="monospace">
          MOD (5-9)
        </text>
        <text x={lowW + modW + 4} y="50" fontSize="9" fill="#ff3b5c" fontWeight="700" fontFamily="monospace">
          HIGH (10+)
        </text>
        <text x="295" y="50" fontSize="9" fill="#545d6c" textAnchor="end" fontFamily="monospace">
          24
        </text>
      </svg>
    </div>
  );
}
