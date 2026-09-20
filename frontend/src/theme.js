// Central design tokens for the clinical console: a near-black telemetry
// surface with phosphor-coded risk accents, so every component agrees on
// the same palette instead of re-inventing colors inline.

export const RISK = {
  low: {
    text: "text-[#2de6a3]",
    solid: "#2de6a3",
    from: "#2de6a3",
    to: "#0fae76",
    bg: "bg-[#0a1712]",
    border: "border-[#2de6a3]/30",
    ring: "ring-[#2de6a3]/30",
    chip: "bg-[#2de6a3]/10 text-[#2de6a3]",
    glow: "glow-low",
    gradient: "from-[#2de6a3] to-[#0fae76]",
  },
  moderate: {
    text: "text-[#ffb020]",
    solid: "#ffb020",
    from: "#ffcc4d",
    to: "#e08900",
    bg: "bg-[#1a1408]",
    border: "border-[#ffb020]/30",
    ring: "ring-[#ffb020]/30",
    chip: "bg-[#ffb020]/10 text-[#ffb020]",
    glow: "glow-mod",
    gradient: "from-[#ffcc4d] to-[#e08900]",
  },
  high: {
    text: "text-[#ff3b5c]",
    solid: "#ff3b5c",
    from: "#ff6b85",
    to: "#e0173d",
    bg: "bg-[#1a0a0e]",
    border: "border-[#ff3b5c]/30",
    ring: "ring-[#ff3b5c]/30",
    chip: "bg-[#ff3b5c]/10 text-[#ff3b5c]",
    glow: "glow-high",
    gradient: "from-[#ff6b85] to-[#e0173d]",
  },
};

export function riskTier(score) {
  if (score < 5) return "low";
  if (score < 10) return "moderate";
  return "high";
}

export const BRAND_GRADIENT = "from-cyan-400 via-cyan-300 to-violet-400";

export const HOSPITAL_GRADIENTS = [
  "from-[#2de6a3] to-[#0fae76]",
  "from-[#ffb020] to-[#e0173d]",
  "from-[#22d3ee] to-[#a78bfa]",
  "from-[#67e8f9] to-[#22d3ee]",
  "from-[#a78bfa] to-[#ff3b5c]",
];
