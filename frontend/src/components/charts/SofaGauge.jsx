import { motion } from "framer-motion";
import AnimatedNumber from "../ui/AnimatedNumber";
import InfoTip from "../ui/InfoTip";
import { RISK, riskTier } from "../../theme";
import { EXPLAIN } from "../../copy";

export default function SofaGauge({ score, riskText, riskIcon }) {
  const tier = riskTier(score);
  const palette = RISK[tier];
  const pct = Math.min(score / 24, 1);

  const size = 220;
  const stroke = 14;
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  // 270-degree arc (like a speedometer), starting bottom-left
  const arcFraction = 0.75;

  return (
    <div className="relative rounded-xl panel-raised h-full flex flex-col items-center justify-center overflow-hidden p-6" style={{ color: palette.solid }}>
      <div
        className="pointer-events-none absolute -top-10 -right-10 w-32 h-32 rounded-full blur-3xl opacity-25"
        style={{ background: palette.solid }}
      />
      <div className="text-[10.5px] font-mono font-bold tracking-[2px] uppercase relative flex items-center gap-1.5 text-[color:var(--text-faint)]">
        Overall Health Score
        <InfoTip text={EXPLAIN.sofa} />
      </div>

      <div className="relative mt-2" style={{ width: size, height: size * 0.82 }}>
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="absolute -top-4 left-0">
          <defs>
            <linearGradient id={`sofa-grad-${tier}`} x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor={palette.from} />
              <stop offset="100%" stopColor={palette.to} />
            </linearGradient>
          </defs>
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="rgba(255,255,255,0.08)"
            strokeWidth={stroke}
            strokeLinecap="round"
            strokeDasharray={`${circumference * arcFraction} ${circumference}`}
            transform={`rotate(135 ${size / 2} ${size / 2})`}
          />
          <motion.circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={`url(#sofa-grad-${tier})`}
            strokeWidth={stroke}
            strokeLinecap="round"
            strokeDasharray={circumference * arcFraction}
            initial={false}
            animate={{ strokeDashoffset: circumference * arcFraction * (1 - pct) }}
            transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
            transform={`rotate(135 ${size / 2} ${size / 2})`}
          />
        </svg>

        <div className="absolute inset-0 flex flex-col items-center justify-center pt-2">
          <div className="leading-none font-display">
            <span className="text-6xl font-bold font-mono" style={{ color: palette.solid }}>
              <AnimatedNumber value={score} decimals={1} />
            </span>
            <span className="text-xl font-semibold opacity-50" style={{ color: palette.solid }}>
              /24
            </span>
          </div>
          <div className="text-sm font-bold font-mono mt-1.5 uppercase tracking-wide" style={{ color: palette.solid }}>
            {riskIcon} {riskText}
          </div>
        </div>
      </div>
    </div>
  );
}
