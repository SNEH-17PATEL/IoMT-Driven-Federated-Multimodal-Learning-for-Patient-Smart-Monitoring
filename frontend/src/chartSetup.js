import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Filler
);

// Dark console defaults so every chart matches the surrounding UI without
// repeating color options in each component.
ChartJS.defaults.color = "#8b94a3";
ChartJS.defaults.borderColor = "rgba(255,255,255,0.08)";
ChartJS.defaults.font.family = "'Inter', ui-sans-serif, sans-serif";
ChartJS.defaults.plugins.tooltip.backgroundColor = "#0a0d13";
ChartJS.defaults.plugins.tooltip.titleColor = "#e8edf4";
ChartJS.defaults.plugins.tooltip.bodyColor = "#c5cbd4";
ChartJS.defaults.plugins.tooltip.borderColor = "rgba(255,255,255,0.12)";
ChartJS.defaults.plugins.tooltip.borderWidth = 1;
ChartJS.defaults.plugins.tooltip.padding = 8;
ChartJS.defaults.plugins.tooltip.displayColors = false;

export default ChartJS;
