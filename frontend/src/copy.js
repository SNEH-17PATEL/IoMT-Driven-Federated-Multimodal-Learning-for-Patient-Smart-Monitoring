// Plain-language explanations for technical terms, kept in one place so
// wording stays consistent everywhere an InfoTip is used.
export const EXPLAIN = {
  sofa:
    "SOFA stands for \"organ failure score.\" Doctors use it to measure how well a patient's organs (lungs, heart, kidneys, etc.) are working. 0 = healthy, 24 = very severe. Higher is worse.",
  shap:
    "SHAP shows which vital signs pushed the AI's score up or down the most for this specific patient — like a receipt explaining the AI's math.",
  consistency:
    "We ask the AI the same question 3 separate times. This score shows how much those 3 answers agree with each other. Higher = more trustworthy; lower = the AI is unsure, so use extra caution.",
  fedavg:
    "Federated Averaging (FedAvg): each hospital trains the AI on its own patients, then only the learned patterns (not the patient data) are combined into one shared, smarter AI.",
  mae:
    "Mean Absolute Error — on average, how far off the AI's score is from the real answer, measured in SOFA points. Lower is better.",
  r2:
    "R² (R-squared) — how much of the up-and-down pattern in patient outcomes the AI can explain. 1.0 would be a perfect prediction; 0 means it's guessing.",
  differentialPrivacy:
    "Differential Privacy adds a small amount of random \"noise\" to the AI's learning so no single patient's data can ever be reverse-engineered from it.",
  federatedLearning:
    "Federated Learning lets multiple hospitals teach one shared AI together — without any hospital ever sending its patients' actual data anywhere.",
  vitalsWindow:
    "The AI looks at the last 20 readings (about the last 3+ hours) to spot trends, not just a single snapshot.",
};
