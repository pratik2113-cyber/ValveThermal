# ValveThermal AI
## Composite Valve CFD Temperature Distribution Simulator

**Hackathon project** — Temperature distribution in composite valves using CFD + AI

---

## What This Does

| Feature | Description |
|---|---|
| **2D CFD Solver** | Finite Difference Method solves the heat equation across all composite layers |
| **AI Surrogate** | Gradient Boosting surrogate replaces slow CFD — 100× faster, instant predictions |
| **Safety Classifier** | ML model classifies each layer: SAFE / WARNING / CRITICAL |
| **Claude AI Copilot** | Ask Claude for design recommendations, failure analysis, material suggestions |
| **Interactive Dashboard** | Real-time Streamlit UI with 2D heatmap, heat flux vectors, safety cards |

---

## Project Structure

```
cfd_valve_project/
├── app.py               ← Main Streamlit dashboard (run this)
├── cfd_engine.py        ← 2D FDM CFD physics solver
├── surrogate_model.py   ← AI surrogate model (GBR)
├── claude_advisor.py    ← Claude AI engineering copilot
├── requirements.txt     ← Python dependencies
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Anthropic API key
```bash
# Linux / Mac
export ANTHROPIC_API_KEY="your-api-key-here"

# Windows
set ANTHROPIC_API_KEY=your-api-key-here
```
Get your key at: https://console.anthropic.com

### 3. Run the app
```bash
streamlit run app.py
```

---

## How to Use

### Step 1: Configure your valve
- Set **T_inlet** (hot fluid temperature, K)
- Set **T_outlet** (ambient/coolant temperature, K)
- Set convection coefficients **h_inner** and **h_outer**
- Choose **2, 3, or 4 composite layers**
- For each layer: select material, thickness (mm), fiber angle (°)

### Step 2: Choose solver mode
- **Full CFD**: Runs 2D FDM simulation (~3 seconds), gives complete temperature field
- **AI Surrogate**: Instant prediction (<0.1s), requires one-time training

### Step 3: Analyse results
- View **2D temperature heatmap** with hotspot marker
- Check **layer safety cards** (SAFE / WARNING / CRITICAL)
- Explore **heat flux vector field**

### Step 4: Ask Claude
- Type any engineering question in the Claude copilot box
- Claude explains results, suggests improvements, recommends materials
- Full multi-turn conversation history maintained

---

## Physics Model

The CFD engine solves the **2D steady-state heat equation**:

```
∂/∂x(k ∂T/∂x) + ∂/∂y(k ∂T/∂y) + Q = 0
```

**Boundary conditions:**
- Inner wall (left): Convective — `q = h_inner × (T_fluid - T_wall)`
- Outer wall (right): Convective — `q = h_outer × (T_wall - T_ambient)`
- Top/Bottom: Adiabatic (insulated) — `dT/dn = 0`

**Fiber angle effect on conductivity:**
```
k_eff = k_base × (cos²θ + 0.1 × sin²θ)
```

**Solver:** Gauss-Seidel iterative method with convergence tolerance 1e-4

---

## Materials Library

| Material | k (W/mK) | T_max (K) |
|---|---|---|
| Carbon Fiber Composite | 5.0 | 450 |
| Glass Fiber Composite | 0.35 | 350 |
| Stainless Steel | 16.0 | 1200 |
| Titanium Alloy | 7.0 | 900 |
| Ceramic Matrix | 3.5 | 1500 |
| Polymer Matrix | 0.25 | 250 |
| Aluminum Alloy | 150.0 | 600 |
| Inconel 718 | 11.4 | 1000 |

---

## What Makes This Stand Out

1. **Real 2D CFD physics** — not a simplified 1D model
2. **AI surrogate** — trained on 300 CFD simulations, 100× faster inference
3. **Multi-layer composites** — fiber angle effects on thermal conductivity
4. **Per-layer safety margins** — with temperature limits per material
5. **Claude AI copilot** — natural language engineering recommendations
6. **Heat flux vectors** — full thermal field visualization
7. **Multi-turn conversation** — ask follow-up questions to Claude

---

## Tech Stack

- **Physics:** NumPy (FDM solver)
- **ML:** scikit-learn (Gradient Boosting surrogate + safety classifier)
- **AI:** Anthropic Claude (engineering copilot)
- **UI:** Streamlit + Plotly
- **Persistence:** joblib (model serialization)
