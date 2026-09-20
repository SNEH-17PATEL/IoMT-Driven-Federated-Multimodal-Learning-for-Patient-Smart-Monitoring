import "../../chartSetup";
import { Line } from "react-chartjs-2";

export default function FLProgressChart({ history }) {
  const labels = history.map((h) => `R${h.round}`);
  const data = {
    labels,
    datasets: [
      {
        label: "Global MAE (SOFA pts)",
        data: history.map((h) => h.mae),
        borderColor: "#ff6b85",
        backgroundColor: "#ff6b85",
        yAxisID: "y",
        tension: 0.35,
        pointRadius: 3,
        borderWidth: 2.5,
      },
      {
        label: "Global R²",
        data: history.map((h) => h.r2),
        borderColor: "#22d3ee",
        backgroundColor: "#22d3ee",
        yAxisID: "y1",
        tension: 0.35,
        pointRadius: 3,
        borderWidth: 2.5,
      },
    ],
  };

  const monoFont = { size: 10, family: "'JetBrains Mono', monospace" };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 250 },
    plugins: { legend: { position: "top", labels: { boxWidth: 10, font: { size: 11 } } } },
    scales: {
      x: { grid: { display: false }, ticks: { font: monoFont } },
      y: {
        type: "linear",
        position: "left",
        title: { display: true, text: "MAE (lower is better)", font: { size: 10 } },
        grid: { color: "rgba(255,255,255,0.06)" },
        ticks: { font: monoFont },
      },
      y1: {
        type: "linear",
        position: "right",
        grid: { drawOnChartArea: false },
        title: { display: true, text: "R² (higher is better)", font: { size: 10 } },
        ticks: { font: monoFont },
      },
    },
  };

  return (
    <div style={{ height: 260 }}>
      <Line data={data} options={options} />
    </div>
  );
}
