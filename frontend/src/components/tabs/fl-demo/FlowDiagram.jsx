import { motion } from "framer-motion";
import { Server } from "lucide-react";
import { HOSPITAL_GRADIENTS } from "../../../theme";

const NODE_COLORS = ["#2de6a3", "#ffb020", "#ff6b85", "#67e8f9", "#a78bfa"];

// A small, live "wiring diagram": a central FL server with animated
// connectors to each hospital node. Dashes flow outward while the server
// is aggregating, and each hospital node glows while it trains locally.
export default function FlowDiagram({ hospitals, trainingIds, aggregating }) {
  const n = hospitals.length;
  const width = 640;
  const height = 260;
  const serverX = width / 2;
  const serverY = 40;
  const nodeY = height - 60;
  const spacing = width / (n + 1);

  return (
    <div className="rounded-lg panel-raised p-4 overflow-x-auto">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ minWidth: 480 }}>
        {hospitals.map((h, i) => {
          const x = spacing * (i + 1);
          const isTraining = trainingIds.has(h.id);
          const color = NODE_COLORS[i % NODE_COLORS.length];
          return (
            <g key={h.id}>
              <line
                x1={serverX}
                y1={serverY + 20}
                x2={x}
                y2={nodeY - 22}
                stroke={isTraining || aggregating ? color : "rgba(255,255,255,0.12)"}
                strokeWidth={2}
                className={isTraining || aggregating ? "flow-dash" : ""}
                strokeDasharray={isTraining || aggregating ? "6 6" : undefined}
              />
            </g>
          );
        })}

        {/* Server node */}
        <g>
          <motion.circle
            cx={serverX}
            cy={serverY}
            r={22}
            fill="rgba(34,211,238,0.12)"
            stroke="#22d3ee"
            strokeWidth={2}
            animate={aggregating ? { r: [22, 28, 22] } : { r: 22 }}
            transition={{ duration: 0.6, repeat: aggregating ? Infinity : 0 }}
          />
          <text x={serverX} y={serverY + 5} textAnchor="middle" fontSize="16">
            🖥️
          </text>
        </g>

        {hospitals.map((h, i) => {
          const x = spacing * (i + 1);
          const isTraining = trainingIds.has(h.id);
          const color = NODE_COLORS[i % NODE_COLORS.length];
          return (
            <g key={`node-${h.id}`}>
              <motion.circle
                cx={x}
                cy={nodeY}
                r={16}
                fill={`${color}22`}
                stroke={color}
                strokeWidth={2}
                animate={isTraining ? { r: [16, 19, 16] } : { r: 16 }}
                transition={{ duration: 0.7, repeat: isTraining ? Infinity : 0 }}
              />
              <text x={x} y={nodeY + 32} textAnchor="middle" fontSize="10" fill="#c5cbd4" fontWeight="700" fontFamily="monospace">
                {h.name}
              </text>
            </g>
          );
        })}
      </svg>

      <div className="flex items-center justify-center gap-2 mt-1 text-[11px] font-mono text-[color:var(--accent)] font-semibold">
        <Server size={13} /> Global FL Server {aggregating && "· aggregating…"}
      </div>
    </div>
  );
}

export { HOSPITAL_GRADIENTS };
