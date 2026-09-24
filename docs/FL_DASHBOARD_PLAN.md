# Federated Learning Simulation Dashboard — Design Plan

**Project:** ICU CDSS — Multimodal Intelligence System with Federated Learning  
**Interface:** `fl_dashboard.py` — standalone Streamlit app (separate from `app.py`)  
**Status:** Planning phase — not yet implemented  
**Last updated:** 2026-09-20

---

## 1. Purpose & Goal

Build a second Streamlit interface that **visually simulates the federated learning
process** — exactly what happens when you run `server.py` + 3 × `client.py` in
separate terminals, but shown live in a browser instead of terminal logs.

**Primary audience:** Both technical reviewers (capstone evaluators) and non-technical
observers who want to understand how federated learning protects patient privacy while
training a shared model.

**Core message the interface must communicate:**
> "Three hospitals each train the model on their own private data. Only the model
> weights (not any patient records) travel to the server. The server combines all
> three sets of weights into one improved global model. This repeats for N rounds."

---

## 2. Relationship to Existing App

| Interface | File | Purpose | Run command |
|---|---|---|---|
| Patient Monitor | `app.py` | Live SOFA prediction for 3 sample patients | `streamlit run app.py` |
| FL Dashboard | `fl_dashboard.py` | Simulate & visualize the FL training process | `streamlit run fl_dashboard.py` |

These are **two completely separate Streamlit apps** — two different browser tabs,
two different terminal processes. They share the same Python utilities (`model_utils.py`,
`data/fl_training/` CSVs, `models/` artifacts) but run independently.

---

## 3. Decisions Confirmed

| # | Question | Answer |
|---|---|---|
| Q1 | Real FL or simulation? | **Hybrid** — Real PyTorch FL, 5–10 rounds, no BigQuery needed |
| Q2 | Data source? | Existing `data/fl_training/client_0/1/2.csv` on disk |
| Q3 | Control style? | **Speed slider** — auto-play, user controls pace (0.5s–5s per phase) |
| Q4 | Weight visualization? | **All three** — numbers + histogram overlay + layer heatmap |
| Q5 | After simulation? | Results shown only — **no files saved**, no model overwrite |
| Q6 | App structure? | **Two separate apps** — `app.py` and `fl_dashboard.py` |
| Q7 | Initial weights? | **Random PyTorch init** — `ICUModel(input_dim)` with no pretrained weights loaded (matches actual `server.py` behaviour) |
| Q8 | Aggregation algorithms? | **All** — FedAvg, FedYogi, FedProx (see open question Q12 below) |
| Q9 | Client training order? | **Sequential** — Hospital 0 → Hospital 1 → Hospital 2 → aggregate |
| Q10 | Weight histogram? | **Before aggregation vs After aggregation overlay** — one per round |
| Q11 | App coexistence? | **Two separate Streamlit apps** |

---

## 4. What One FL Round Looks Like (the 6 phases)

This is what the dashboard must visualize, one phase at a time:

```
Round N begins
│
├── Phase 1: DISTRIBUTE
│     Server broadcasts current global weights → all 3 hospitals
│     Visual: arrows from Server node → Hospital 0, 1, 2 (animated)
│     Data shown: weight mean, std of global model
│
├── Phase 2: TRAIN — Hospital 0 (General ICU)
│     Hospital 0 sets global weights, trains locally (3 epochs, FedProx)
│     Visual: Hospital 0 node glows/pulses, loss ticks down
│     Data shown: epoch-by-epoch loss for Hospital 0
│
├── Phase 3: TRAIN — Hospital 1 (Mixed ICU)
│     Hospital 1 trains on its local data
│     Visual: Hospital 1 node pulses
│     Data shown: epoch-by-epoch loss for Hospital 1
│
├── Phase 4: TRAIN — Hospital 2 (Cardiac/Trauma ICU)
│     Hospital 2 trains on its local data
│     Visual: Hospital 2 node pulses
│     Data shown: epoch-by-epoch loss for Hospital 2
│
├── Phase 5: UPLOAD + AGGREGATE
│     All 3 hospitals send updated weights → Server
│     Server aggregates (FedAvg / FedYogi / FedProx)
│     Visual: arrows from Hospital 0, 1, 2 → Server node (animated)
│     Data shown: weight histogram before vs after aggregation
│
└── Phase 6: ROUND COMPLETE
      Global model updated, metrics logged
      Visual: Server node brightens, global loss updates
      Data shown: round summary, best round tracker
```

---

## 5. UI Layout — Screen Wireframe

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  🔬 Federated Learning Simulation Dashboard                                  │
│  ICU CDSS — Real FL Training Visualization · MIMIC-III Client Data           │
├──────────────────────────────────────────────────────────────────────────────┤
│  CONTROL PANEL                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  Rounds: [===●===] 10    Speed: [●=======] 0.5s/phase                  │ │
│  │  Algorithm: [FedAvg ▼]                                                  │ │
│  │  [▶ Start Simulation]  [⏹ Stop]  [↺ Reset]                              │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
├──────────────────────────────────────┬───────────────────────────────────────┤
│  NETWORK VISUALIZATION               │  LIVE METRICS                         │
│                                      │                                        │
│         ┌──────────────┐             │  Round: 3 / 10                         │
│         │  🖥 SERVER   │             │  Phase: Training Hospital 1             │
│         │  Global Model│             │  Global Loss: 7.234 → 6.891 (↓ 4.7%)  │
│         └──────┬───────┘             │  Best Round: 2 (loss: 6.834)           │
│          ↓  ↓  ↓   ↑  ↑  ↑          │                                        │
│    ┌────┐ ┌────┐ ┌────┐              │  [Loss Curves — Plotly line chart]     │
│    │ H0 │ │ H1 │ │ H2 │             │   Hospital 0 ──── (blue)               │
│    │ 🏥 │ │ 🏥 │ │ 🏥 │             │   Hospital 1 ──── (orange)             │
│    └────┘ └────┘ └────┘              │   Hospital 2 ──── (green)              │
│   General Mixed  Cardiac             │   Global    ──── (white/bold)          │
│                                      │                                        │
│  [Phase badge: 🟡 Training H1...]    │   Round: 1  2  3  4 ...               │
├──────────────────────────────────────┴───────────────────────────────────────┤
│  WEIGHT STATISTICS — Round 3                                                 │
│  ┌────────────────────┬────────────────────┬────────────────────┐            │
│  │  Hospital 0        │  Hospital 1        │  Hospital 2        │            │
│  │  23,041 params     │  23,041 params     │  23,041 params     │            │
│  │  mean:  0.0023     │  mean:  0.0019     │  mean:  0.0027     │            │
│  │  std:   0.041      │  std:   0.038      │  std:   0.043      │            │
│  │  Δ mean: +0.0012   │  Δ mean: +0.0008   │  Δ mean: +0.0015   │            │
│  └────────────────────┴────────────────────┴────────────────────┘            │
├──────────────────────────────────────┬───────────────────────────────────────┤
│  WEIGHT HISTOGRAM — Round 3          │  LAYER 1 WEIGHTS (108 → 128)          │
│                                      │                                        │
│  [Plotly histogram]                  │  [Plotly heatmap: 108 × 128]           │
│   Before agg: blue bars              │  Each cell = one weight value          │
│   After agg:  orange overlay         │  Rows = input features                 │
│                                      │  Cols = hidden neurons                 │
│  Aggregation narrows distribution    │  Color intensity = weight magnitude    │
│  → weights converge toward consensus │  Updates each round                    │
├──────────────────────────────────────┴───────────────────────────────────────┤
│  ROUND HISTORY LOG                                                           │
│  ┌──────┬───────────┬───────────┬───────────┬───────────────────────────┐   │
│  │Round │ H0 Loss   │ H1 Loss   │ H2 Loss   │ Global Loss   │ Best?     │   │
│  │  1   │  7.843    │  7.612    │  7.901    │  7.785        │           │   │
│  │  2   │  7.102    │  6.934    │  7.234    │  6.834        │ ✓ Best    │   │
│  │  3   │  6.891    │  6.712    │  7.012    │  6.872        │           │   │
│  └──────┴───────────┴───────────┴───────────┴───────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Technical Architecture

### 6.1 Backend — Custom FL Loop (no Flower framework)

The dashboard does **not** use `flwr`, `server.py`, or `client.py` directly.
Instead, it runs a self-contained FL simulation using the same core components:
- `model_utils.ICUModel` — exact same DNN architecture
- `model_utils.train_model` — exact same training function (with FedProx)
- `model_utils.get_weights` / `set_weights` — weight serialization
- Manual aggregation functions (FedAvg, FedYogi) implemented in `fl_dashboard.py`

This gives complete control over:
- Pausing between phases to update the UI
- Extracting weight histograms at any point
- Running at any speed without Flower's communication overhead

### 6.2 Threading Model

```
Main Thread (Streamlit)              Background Thread (FL Engine)
─────────────────────────            ────────────────────────────────
Reads _FL_STATE dict                 Writes _FL_STATE dict
Renders UI based on state            Runs FL phases one by one
Auto-refreshes every 1s              Sleeps between phases (speed slider)
Shows current phase animation        Updates metrics after each phase
```

`_FL_STATE` is a **module-level Python dict** (not `st.session_state`) —
shared across Streamlit reruns, thread-safe via a `threading.Lock`.

```python
_FL_STATE = {
    "running":        False,
    "current_round":  0,
    "current_phase":  "idle",   # idle | distributing | training_0/1/2 | aggregating | done
    "client_losses":  [[], [], []],
    "global_losses":  [],
    "client_weights": [None, None, None],
    "global_weights": None,
    "pre_agg_weights": None,    # snapshot before aggregation (for histogram)
    "post_agg_weights": None,   # snapshot after aggregation (for histogram)
    "weight_stats":   [],       # per-round: [{"mean": ..., "std": ..., "delta": ...}, ...]
    "best_round":     0,
    "best_loss":      float("inf"),
    "round_log":      [],       # list of dicts for the history table
    "error":          None,
}
```

### 6.3 Auto-Refresh Mechanism

Uses `streamlit-autorefresh` library:
```python
from streamlit_autorefresh import st_autorefresh
# Only auto-refresh while simulation is running
if _FL_STATE["running"]:
    st_autorefresh(interval=1000, key="fl_refresh")  # refresh every 1 second
```

### 6.4 Aggregation Implementations

Three algorithms will be implemented natively (no Flower dependency):

**FedAvg** — weighted average by sample count:
```
w_global = Σ (n_k / N) × w_k    for each client k
```

**FedYogi** — server-side adaptive optimizer:
```
Δ = w_aggregated - w_global_prev
m_t = β1·m_{t-1} + (1-β1)·Δ
v_t = v_{t-1} + (1-β2)·(Δ²-v_{t-1})·sign(Δ²-v_{t-1})
w_global = w_global_prev + η · m_t / (√v_t + τ)
```

**FedProx** — proximal regularization (client-side, shown in training phase):
```
Loss_local = MSE_loss + (μ/2) · ||w_local - w_global||²
```
This is already built into `model_utils.train_model` — no extra code needed.

### 6.5 Data Loading

```python
# Load once at startup (cached)
@st.cache_resource
def load_client_data():
    scaler = joblib.load("models/scaler.pkl")
    feature_cols = list(scaler.feature_names_in_)
    datasets = []
    for i in range(3):
        df = pd.read_csv(f"data/fl_training/client_{i}.csv")
        X = df[feature_cols].values.astype(np.float32)
        y = df["sofa_score"].values.astype(np.float32)
        datasets.append((X, y))
    return datasets, feature_cols
```

---

## 7. Component Breakdown

| Component | What it shows | Library |
|---|---|---|
| Network diagram | Server + 3 hospital nodes, animated arrows per phase | Custom HTML/CSS + CSS animations |
| Phase badge | Current phase label + colour | Custom HTML |
| Loss curves | 4 lines (3 clients + global) updating each round | Plotly `go.Scatter` |
| Weight stats cards | mean, std, Δ per client | Custom HTML tiles |
| Weight histogram | Before/after aggregation overlay | Plotly `go.Histogram` |
| Layer 1 heatmap | 108×128 weight matrix | Plotly `go.Heatmap` |
| Round history table | All rounds with losses + best marker | `st.dataframe` with styling |
| Control panel | Sliders, buttons, algorithm selector | Native Streamlit widgets |

---

## 8. Visual Style

Matches the dark ICU aesthetic of `app.py`:
- Background: dark gradient (`#0f2027` → `#2c5364`)
- Accent: `#00d2ff` (cyan) for server, green/amber/red for hospitals
- Animated arrows: CSS `@keyframes` for flowing dots along paths
- Phase badge: colour changes per phase (blue = distribute, amber = training, purple = aggregate, green = done)
- Font: same monospace/bold style as existing app

---

## 9. File Structure

```
icu_monitor/
├── app.py                          # Patient monitoring (existing, unchanged)
├── fl_dashboard.py                 # FL simulation (NEW)
├── model_utils.py                  # Shared — ICUModel, train_model, etc.
├── data/
│   └── fl_training/
│       ├── client_0.csv            # Hospital 0 data (already exists)
│       ├── client_1.csv            # Hospital 1 data (already exists)
│       └── client_2.csv            # Hospital 2 data (already exists)
├── models/
│   ├── scaler.pkl                  # Used to get feature_cols
│   └── federated_model.pth         # NOT overwritten by simulation
└── docs/
    └── FL_DASHBOARD_PLAN.md        # This file
```

---

## 10. Open Questions / Decisions Pending

### Q12 — Aggregation algorithm selection ✅ DECIDED

**Dropdown selector** — user picks ONE before starting. 4 choices:

### Q16 — FedProx dropdown treatment ✅ DECIDED

FedProx is a **client-side regularization term**, not a server aggregation algorithm.
The 4 dropdown choices correctly pair server aggregation + client regularization:

| Dropdown choice | Server aggregation | Client training |
|---|---|---|
| FedAvg | Weighted average | Standard (no proximal term) |
| FedYogi | Yogi adaptive optimizer | Standard (no proximal term) |
| FedAvg + FedProx | Weighted average | + proximal term μ=0.5 |
| **FedYogi + FedProx** | Yogi adaptive optimizer | + proximal term μ=0.5 — **actual setup** |

The UI will label the 4th option as "← your actual training setup" so the user
understands which combination was used to produce the final model.

**FedYogi hyperparameters** (from `train_federated.py`): η=0.01, β1=0.9, β2=0.99, τ=0.001  
**FedProx μ** (from `train_federated.py` and `client.py`): μ=0.5

---

### Q13 — Show client data statistics at start? ✅ YES (defaulted)

Before the simulation begins, show a summary panel:
- Number of training samples per hospital
- SOFA score range (min / mean / max) per hospital
- Helps non-tech viewers understand WHY each hospital's weights differ

### Q14 — Round history table? ✅ YES (defaulted)

Scrollable table at the bottom: `Round | H0 Loss | H1 Loss | H2 Loss | Global Loss | Best?`
Colour-coded rows: green for best round, normal for others.

### Q15 — Explanatory tooltips for non-tech people? ✅ YES (defaulted)

Collapsible `st.expander()` panels below each major section:
- "What are model weights?" — below the weight stats panel
- "What does aggregation mean?" — below the histogram panel
- "Why not share patient data directly?" — below the network diagram

---

## 11. Implementation Order (when ready to build)

1. `fl_dashboard.py` skeleton — page config, CSS, layout columns
2. Control panel — sliders, buttons, algorithm dropdown
3. FL backend — `_FL_STATE` dict, background thread, FL loop with all phases
4. Network visualization — static first (server + 3 hospital nodes), then add CSS animations
5. Loss curves — Plotly line chart, updates each round
6. Weight stats cards — per-client numbers after each training phase
7. Weight histogram — before/after overlay, updates after aggregation phase
8. Layer 1 heatmap — 108×128 weight matrix, updates each round
9. Round history table — `st.dataframe` with colour coding
10. Auto-refresh mechanism — `streamlit-autorefresh` conditional on `_FL_STATE["running"]`
11. Polish — animations, phase transitions, dark theme, ICU aesthetic
12. Final testing — full 10-round run, verify all panels update correctly

---

## 12. Dependencies to Add

```
streamlit-autorefresh    # auto-refresh while FL is running
```

All other dependencies already present: `streamlit`, `torch`, `numpy`, `pandas`,
`plotly`, `joblib`, `scikit-learn`.

---

**All questions resolved. Ready for implementation.**
