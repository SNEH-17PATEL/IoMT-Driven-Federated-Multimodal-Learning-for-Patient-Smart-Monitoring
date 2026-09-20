import "../../chartSetup";
import { Line } from "react-chartjs-2";

const PANELS = [
  { key: "HR", label: "Heart Rate", unit: "bpm", color: "#ff6b85", lo: 60, hi: 100 },
  { key: "RR", label: "Breathing Rate", unit: "br/min", color: "#67e8f9", lo: 12, hi: 20 },
  { key: "SpO2", label: "Blood Oxygen", unit: "%", color: "#2de6a3", lo: 95, hi: 100 },
  { key: "Temp", label: "Temperature", unit: "°C", color: "#ffb020", lo: 36.5, hi: 37.5 },
  { key: "SBP", label: "BP (Top)", unit: "mmHg", color: "#a78bfa", lo: 100, hi: 120 },
  { key: "DBP", label: "BP (Bottom)", unit: "mmHg", color: "#22d3ee", lo: 60, hi: 80 },
  { key: "MAP", label: "Avg. BP", unit: "mmHg", color: "#f472b6", lo: 70, hi: 100 },
];

function normalBandPlugin(lo, hi) {
  return {
    id: `band-${lo}-${hi}`,
    beforeDatasetsDraw(chart) {
      const { ctx, chartArea, scales } = chart;
      if (!chartArea) return;
      const yLo = scales.y.getPixelForValue(lo);
      const yHi = scales.y.getPixelForValue(hi);
      ctx.save();
      ctx.fillStyle = "rgba(45,230,163,0.08)";
      ctx.fillRect(chartArea.left, yHi, chartArea.right - chartArea.left, yLo - yHi);
      ctx.restore();
    },
  };
}

function MiniTrendChart({ panel, values }) {
  const labels = values.map((_, i) => i + 1);
  const data = {
    labels,
    datasets: [
      {
        label: panel.label,
        data: values,
        borderColor: panel.color,
        backgroundColor: panel.color,
        pointRadius: 2,
        borderWidth: 2,
        tension: 0.25,
      },
    ],
  };
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (ctx) => `${panel.label}: ${ctx.parsed.y} ${panel.unit}`,
        },
      },
    },
    scales: {
      x: { display: false },
      y: {
        ticks: { font: { size: 9, family: "'JetBrains Mono', monospace" } },
        grid: { color: "rgba(255,255,255,0.06)" },
      },
    },
  };

  return (
    <div className="rounded-lg panel-inset p-2" style={{ borderTop: `2px solid ${panel.color}` }}>
      <div className="text-[10px] font-mono font-semibold text-[color:var(--text-dim)] mb-1 text-center">
        {panel.label} <span className="text-[color:var(--text-faint)] font-normal">({panel.unit})</span>
      </div>
      <div style={{ height: 110 }}>
        <Line data={data} options={options} plugins={[normalBandPlugin(panel.lo, panel.hi)]} />
      </div>
    </div>
  );
}

export default function TrendChart({ vitalsWindow }) {
  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <div className="text-sm font-bold text-[color:var(--text)]">
          Vital Signs — Last {vitalsWindow.length} Readings
        </div>
        <div className="text-[11px] font-mono text-[color:var(--low)] flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded-sm bg-[color:var(--low)]/25 border border-[color:var(--low)]/40" /> normal range
        </div>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
        {PANELS.map((panel) => (
          <MiniTrendChart
            key={panel.key}
            panel={panel}
            values={vitalsWindow.map((r) => r[panel.key])}
          />
        ))}
      </div>
    </div>
  );
}
