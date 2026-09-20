const BASE = "/api";

async function getJSON(path) {
  const res = await fetch(BASE + path);
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
  return res.json();
}

async function postJSON(path, body) {
  const res = await fetch(BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`POST ${path} failed: ${res.status}`);
  return res.json();
}

export const api = {
  getPatients: () => getJSON("/patients"),
  getModelInfo: () => getJSON("/model-info"),
  startMonitoring: (patientId) => postJSON(`/monitor/${patientId}/start`),
  getReading: (patientId) => getJSON(`/monitor/${patientId}/reading`),
};

export function flDemoSocketUrl() {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/fl-demo`;
}
