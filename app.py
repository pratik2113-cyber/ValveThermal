"""
app.py
------
ValveThermal AI — Composite Valve CFD Temperature Distribution Simulator
with Real-Time AI Surrogate + Gemini Engineering Copilot

Run: streamlit run app.py

FIXES APPLIED:
  #1  Auto-fire AI bug fixed — Gemini only called on explicit button click
  #2  API key handled via gemini_advisor.py (no crash on missing key)
  #4  CFD vectorized — see cfd_engine.py
  #5  Retrain Surrogate button always available
  #6  Conversation history capped at 20 messages (in gemini_advisor.py)
  #7  get_quick_tip() wired up for WARNING/CRITICAL layer cards
  #8  Replaced claude_advisor import with gemini_advisor
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import os
import time

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ValveThermal AI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg:       #0a0c10;
    --panel:    #111318;
    --card:     #181c24;
    --border:   #222836;
    --bright:   #2d3448;
    --red:      #ef4444;
    --amber:    #f59e0b;
    --green:    #22c55e;
    --blue:     #3b82f6;
    --cyan:     #06b6d4;
    --orange:   #f97316;
    --text:     #f1f5f9;
    --muted:    #64748b;
    --dim:      #374151;
    --mono:     'JetBrains Mono', monospace;
    --sans:     'Inter', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: var(--sans);
    color: var(--text);
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] {
    background: var(--panel) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: var(--mono);
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--cyan);
    border-bottom: 1px solid var(--border);
    padding-bottom: 4px;
    margin: 1.2rem 0 0.4rem;
}
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div {
    background: var(--card) !important;
    border: 1px solid var(--bright) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
    border-radius: 4px !important;
}
[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--bright);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    position: relative;
    overflow: hidden;
}
[data-testid="stMetric"]::before {
    content:''; position:absolute; top:0; left:0;
    width:3px; height:100%;
    background: var(--orange); border-radius:8px 0 0 8px;
}
[data-testid="stMetricLabel"] p {
    font-family: var(--mono) !important;
    font-size: 0.58rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}
[data-testid="stAlert"] {
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    border-radius: 6px !important;
}
[data-testid="stTextArea"] textarea {
    background: var(--card) !important;
    border: 1px solid var(--bright) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
}
[data-testid="stButton"] button {
    background: linear-gradient(135deg, #f97316, #ef4444) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #3b82f6, #06b6d4) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
}
label[data-testid="stWidgetLabel"] p {
    font-family: var(--sans) !important;
    font-size: 0.76rem !important;
    color: var(--muted) !important;
}
hr { border-top: 1px solid var(--border) !important; }
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--bright); border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    from cfd_engine import (
        simulate_valve_temperature, ValveLayer, generate_training_data, MATERIALS
    )
except ImportError:
    st.error("cfd_engine.py not found. Place all files in the same directory.")
    st.stop()

try:
    from surrogate_model import (
        train_surrogate, train_safety_classifier, predict_fast,
        SURROGATE_PATH, CLASSIFIER_PATH
    )
except ImportError:
    st.error("surrogate_model.py not found.")
    st.stop()

# FIX #8: Import from gemini_advisor instead of claude_advisor
try:
    from gemini_advisor import (
        get_gemini_analysis,
        get_material_recommendation,
        get_quick_tip,
    )
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from report_generator import generate_pdf_report
    REPORT_AVAILABLE = True
except ImportError:
    REPORT_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────
LAYER_COLORS  = ["#f97316", "#06b6d4", "#a855f7", "#22c55e"]
SAFETY_COLORS = {"SAFE": "#22c55e", "WARNING": "#f59e0b", "CRITICAL": "#ef4444"}
SAFETY_ICONS  = {"SAFE": "✦", "WARNING": "⚠", "CRITICAL": "✕"}
MATERIAL_LIST = list(MATERIALS.keys())

# ── Session state init ────────────────────────────────────────────────────────
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "last_cfd_result" not in st.session_state:
    st.session_state.last_cfd_result = None
if "last_layers_info" not in st.session_state:
    st.session_state.last_layers_info = []
if "surrogate_ready" not in st.session_state:
    st.session_state.surrogate_ready = os.path.exists(SURROGATE_PATH)
if "quick_tips_cache" not in st.session_state:
    st.session_state.quick_tips_cache = {}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:0.5rem 0 1rem">
        <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;
                    font-weight:700;color:#f1f5f9;letter-spacing:0.05em;">
            VALVE<span style="color:#f97316;">THERMAL</span>
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.55rem;
                    color:#374151;letter-spacing:0.3em;margin-top:3px;">
            CFD · AI SURROGATE · GEMINI COPILOT
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── Operating conditions ──────────────────────────────────────────────
    st.markdown("### 🌡 Operating Conditions")
    c1, c2 = st.columns(2)
    with c1:
        T_inlet  = st.number_input("T_inlet (K)", value=850.0, min_value=300.0,
                                   max_value=2000.0, step=10.0, format="%.0f")
    with c2:
        T_outlet = st.number_input("T_outlet (K)", value=350.0, min_value=200.0,
                                   max_value=800.0, step=10.0, format="%.0f")

    c3, c4 = st.columns(2)
    with c3:
        h_inner = st.number_input("h_inner (W/m²K)", value=500.0,
                                  min_value=10.0, max_value=5000.0,
                                  step=50.0, format="%.0f")
    with c4:
        h_outer = st.number_input("h_outer (W/m²K)", value=25.0,
                                  min_value=5.0, max_value=500.0,
                                  step=5.0, format="%.0f")

    valve_height = st.number_input("Valve height (m)", value=0.05,
                                   min_value=0.01, max_value=0.5,
                                   step=0.005, format="%.3f")

    # ── Layer configuration ───────────────────────────────────────────────
    st.markdown("### 🧱 Composite Layers")
    n_layers = st.selectbox("Number of layers", [2, 3, 4], index=1)

    DEFAULTS = [
        {"mat": "Carbon Fiber Composite", "thick": 0.005, "angle": 0},
        {"mat": "Ceramic Matrix",         "thick": 0.008, "angle": 45},
        {"mat": "Stainless Steel",        "thick": 0.003, "angle": 0},
        {"mat": "Polymer Matrix",         "thick": 0.004, "angle": 90},
    ]

    layers_config = []
    for i in range(n_layers):
        d = DEFAULTS[i]
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:6px;
                    margin:0.8rem 0 0.3rem">
            <div style="width:9px;height:9px;border-radius:2px;
                        background:{LAYER_COLORS[i]};flex-shrink:0"></div>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;
                         letter-spacing:0.15em;text-transform:uppercase;
                         color:#64748b">Layer {i+1}</span>
        </div>
        """, unsafe_allow_html=True)
        mat   = st.selectbox(f"Material L{i+1}", MATERIAL_LIST,
                             index=MATERIAL_LIST.index(d["mat"]),
                             key=f"mat_{i}")
        lc1, lc2 = st.columns(2)
        with lc1:
            thick = st.number_input(f"L{i+1} (mm)", value=d["thick"]*1000,
                                    min_value=0.1, max_value=50.0,
                                    step=0.1, format="%.1f", key=f"thick_{i}")
        with lc2:
            angle = st.number_input(f"θ{i+1} (°)", value=float(d["angle"]),
                                    min_value=0.0, max_value=90.0,
                                    step=5.0, format="%.0f", key=f"ang_{i}")
        layers_config.append({"material": mat, "thickness": thick/1000, "angle": angle})

    # ── Solver mode ───────────────────────────────────────────────────────
    st.markdown("### ⚡ Solver Mode")
    solver_mode = st.radio(
        "Choose solver",
        ["Full CFD (accurate, ~1s)", "AI Surrogate (instant, <0.1s)"],
        index=0,
    )
    use_surrogate = "Surrogate" in solver_mode

    if use_surrogate and not st.session_state.surrogate_ready:
        if st.button("Train AI Surrogate (one-time ~30s)", key="train_btn"):
            with st.spinner("Generating 300 CFD training simulations..."):
                X_train, y_train = generate_training_data(n_samples=300)
            with st.spinner("Training surrogate neural network..."):
                train_surrogate(X_train, y_train, verbose=False)
                train_safety_classifier(X_train, y_train[:, 0], verbose=False)
            st.session_state.surrogate_ready = True
            st.session_state.quick_tips_cache = {}
            st.success("Surrogate model trained!")
            st.rerun()

    # FIX #5: Always-available retrain button
    if st.session_state.surrogate_ready:
        if st.button("🔄 Retrain Surrogate", key="retrain_btn",
                     help="Re-train with new data (clears existing model)"):
            with st.spinner("Generating training data..."):
                X_train, y_train = generate_training_data(n_samples=300)
            with st.spinner("Retraining surrogate..."):
                train_surrogate(X_train, y_train, verbose=False)
                train_safety_classifier(X_train, y_train[:, 0], verbose=False)
            st.session_state.quick_tips_cache = {}
            st.success("Surrogate retrained!")
            st.rerun()

    st.divider()
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.55rem;
                color:#374151;text-align:center;line-height:1.8">
        FDM solver · GBR surrogate · Gemini AI<br>
        cfd_engine · surrogate_model · gemini_advisor
    </div>
    """, unsafe_allow_html=True)

# ── Build layer objects ───────────────────────────────────────────────────────
valve_layers = [
    ValveLayer(
        name=cfg["material"],
        material=cfg["material"],
        thickness=cfg["thickness"],
        angle=cfg["angle"],
    )
    for cfg in layers_config
]

# ── Validation ────────────────────────────────────────────────────────────────
errors = []
if T_inlet <= T_outlet:
    errors.append("T_inlet must be greater than T_outlet.")
for i, cfg in enumerate(layers_config):
    if cfg["thickness"] <= 0:
        errors.append(f"Layer {i+1}: thickness must be > 0.")

# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:0.4rem 0 1.2rem">
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                letter-spacing:0.25em;text-transform:uppercase;
                color:#f97316;margin-bottom:0.3rem">
        Composite Valve CFD Analysis
    </div>
    <h1 style="font-family:'Inter',sans-serif;font-size:1.9rem;
               font-weight:600;color:#f1f5f9;margin:0;line-height:1.2">
        Temperature Distribution<br>
        <span style="color:#374151;font-weight:300">Simulator &amp; Gemini AI Copilot</span>
    </h1>
</div>
""", unsafe_allow_html=True)

# ── Errors ────────────────────────────────────────────────────────────────────
if errors:
    for e in errors:
        st.error(e)
    st.stop()

# ── Run simulation ────────────────────────────────────────────────────────────
cfd_result       = None
surrogate_result = None
sim_time_ms      = 0

if use_surrogate:
    if not st.session_state.surrogate_ready:
        st.warning("Train the AI surrogate first (button in sidebar).")
        st.stop()

    t0 = time.time()
    avg_k     = np.mean([lay.conductivity for lay in valve_layers])
    total_L   = sum(lay.thickness for lay in valve_layers)
    avg_angle = np.mean([lay.angle for lay in valve_layers])

    surrogate_result = predict_fast(
        total_thickness=total_L,
        avg_k=avg_k,
        n_layers=len(valve_layers),
        T_inlet=T_inlet,
        T_outlet=T_outlet,
        h_inner=h_inner,
        h_outer=h_outer,
        fiber_angle=avg_angle,
    )
    sim_time_ms = (time.time() - t0) * 1000

else:
    t0 = time.time()
    with st.spinner("Running CFD simulation..."):
        cfd_result = simulate_valve_temperature(
            layers=valve_layers,
            T_inlet=T_inlet, T_outlet=T_outlet,
            h_inner=h_inner, h_outer=h_outer,
            valve_height=valve_height,
            nx_per_layer=25, ny=35,
            max_iter=3000, tol=1e-4,
        )
    sim_time_ms = (time.time() - t0) * 1000
    st.session_state.last_cfd_result = cfd_result

# Pull result values
if cfd_result:
    T_max     = cfd_result["max_temperature"]
    T_avg     = cfd_result["avg_temperature"]
    hotspot_x = cfd_result["hotspot_x"]
    safety    = cfd_result["safety_status"]
    overall   = "CRITICAL" if "CRITICAL" in safety else ("WARNING" if "WARNING" in safety else "SAFE")
    result_src = f"Full CFD ({cfd_result['iterations']} iters, residual {cfd_result['residual']:.2e})"
else:
    T_max     = surrogate_result["T_max"]
    T_avg     = surrogate_result["T_avg"]
    hotspot_x = surrogate_result["hotspot_x"]
    overall   = surrogate_result["safety_label"]
    result_src = surrogate_result["method"]
    safety    = []

# Store for Gemini
st.session_state.last_layers_info = [
    {**cfg, "conductivity": valve_layers[i].conductivity}
    for i, cfg in enumerate(layers_config)
]

# ── Metric row ────────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("MAX TEMPERATURE", f"{T_max:.0f} K",
              f"{T_max-273:.0f} °C", delta_color="off")
with m2:
    st.metric("AVG TEMPERATURE", f"{T_avg:.0f} K",
              f"{T_avg-273:.0f} °C", delta_color="off")
with m3:
    st.metric("HOTSPOT POSITION", f"{hotspot_x*1000:.1f} mm",
              "from inner wall", delta_color="off")
with m4:
    st.metric("WALL THICKNESS",
              f"{sum(c['thickness'] for c in layers_config)*1000:.1f} mm",
              f"{n_layers} layers", delta_color="off")
with m5:
    ov_c = SAFETY_COLORS[overall]
    ov_i = SAFETY_ICONS[overall]
    st.markdown(f"""
    <div style="background:#181c24;border:1px solid #2d3448;border-radius:8px;
                padding:1rem 1.2rem;position:relative;overflow:hidden">
        <div style="position:absolute;top:0;left:0;width:3px;height:100%;
                    background:{ov_c};border-radius:8px 0 0 8px"></div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;
                    letter-spacing:0.15em;text-transform:uppercase;
                    color:#64748b;margin-bottom:4px">SAFETY VERDICT</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;
                    font-weight:700;color:{ov_c}">{ov_i} {overall}</div>
        <div style="font-family:'Inter',sans-serif;font-size:0.7rem;
                    color:#374151;margin-top:3px">{sim_time_ms:.1f} ms · {result_src[:28]}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ── Temperature field visualization ──────────────────────────────────────────
if cfd_result:
    T_field = cfd_result["temperature_field"]
    x_g     = cfd_result["x_grid"]
    y_g     = cfd_result["y_grid"]
    bounds  = cfd_result["layer_boundaries"]

    fig = go.Figure()

    # Heatmap
    fig.add_trace(go.Heatmap(
        z=T_field,
        x=x_g * 1000,   # mm
        y=y_g * 1000,
        colorscale=[
            [0.00, "#0a0c10"],
            [0.20, "#1e3a5f"],
            [0.40, "#1d4ed8"],
            [0.60, "#f59e0b"],
            [0.80, "#ef4444"],
            [1.00, "#ffffff"],
        ],
        colorbar=dict(
            title=dict(text="Temperature (K)",
                       font=dict(family="JetBrains Mono", size=11, color="#64748b")),
            tickfont=dict(family="JetBrains Mono", size=10, color="#64748b"),
            bgcolor="#111318", bordercolor="#222836", borderwidth=1,
        ),
        hovertemplate="x: %{x:.2f}mm<br>y: %{y:.2f}mm<br>T: %{z:.1f}K<extra></extra>",
    ))

    # Layer boundary lines
    for i, xb in enumerate(bounds[1:-1]):
        fig.add_vline(
            x=xb * 1000, line_dash="dash",
            line_color="rgba(255,255,255,0.3)", line_width=1.5,
        )

    # Layer labels
    total_L_mm = sum(c["thickness"] for c in layers_config) * 1000
    valve_h_mm = valve_height * 1000
    for i, (x0, x1) in enumerate(zip(
        [b * 1000 for b in bounds[:-1]],
        [b * 1000 for b in bounds[1:]],
    )):
        mid = (x0 + x1) / 2
        fig.add_annotation(
            x=mid, y=valve_h_mm * 1.05,
            text=f"<b>L{i+1}</b><br>{layers_config[i]['material'][:12]}<br>{layers_config[i]['thickness']*1000:.1f}mm",
            showarrow=False,
            font=dict(color=LAYER_COLORS[i], family="JetBrains Mono", size=9),
            align="center", xanchor="center", yanchor="bottom",
        )

    # Hotspot marker
    fig.add_trace(go.Scatter(
        x=[cfd_result["hotspot_x"] * 1000],
        y=[cfd_result["hotspot_y"] * 1000],
        mode="markers",
        marker=dict(symbol="x", size=14, color="#ef4444",
                    line=dict(width=2, color="white")),
        name="Hotspot",
        hovertemplate=f"Hotspot: {T_max:.0f}K<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="2D Temperature Distribution — Composite Valve Cross-Section",
                   font=dict(family="Inter", size=15, color="#f1f5f9"),
                   x=0.0, xanchor="left"),
        xaxis=dict(title=dict(text="Wall Thickness (mm)",
                   font=dict(family="JetBrains Mono", size=11, color="#64748b")),
                   tickfont=dict(family="JetBrains Mono", size=10, color="#374151"),
                   gridcolor="rgba(255,255,255,0.04)",
                   range=[-total_L_mm * 0.02, total_L_mm * 1.02]),
        yaxis=dict(title=dict(text="Valve Height (mm)",
                   font=dict(family="JetBrains Mono", size=11, color="#64748b")),
                   tickfont=dict(family="JetBrains Mono", size=10, color="#374151"),
                   gridcolor="rgba(255,255,255,0.04)"),
        plot_bgcolor="#0a0c10", paper_bgcolor="#111318",
        margin=dict(l=60, r=20, t=80, b=60),
        font=dict(family="Inter", color="#f1f5f9"),
        showlegend=True,
        legend=dict(bgcolor="#181c24", bordercolor="#222836", borderwidth=1,
                    font=dict(family="JetBrains Mono", size=10, color="#64748b")),
        hovermode="closest",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Heat flux vectors ─────────────────────────────────────────────────
    with st.expander("Heat Flux Vector Field"):
        skip = 3
        qx   = cfd_result["heat_flux_x"][::skip, ::skip]
        qy   = cfd_result["heat_flux_y"][::skip, ::skip]
        xv   = x_g[::skip] * 1000
        yv   = y_g[::skip] * 1000

        fig2 = go.Figure(go.Cone(
            x=np.tile(xv, len(yv)),
            y=np.repeat(yv, len(xv)),
            z=np.zeros(len(xv) * len(yv)),
            u=qx.flatten(), v=qy.flatten(),
            w=np.zeros(qx.size),
            sizemode="scaled", sizeref=0.5,
            colorscale="Oranges", showscale=False,
        ))
        fig2.update_layout(
            scene=dict(
                xaxis_title="x (mm)", yaxis_title="y (mm)",
                zaxis=dict(showticklabels=False, showgrid=False),
                bgcolor="#0a0c10",
            ),
            plot_bgcolor="#0a0c10", paper_bgcolor="#111318",
            height=350,
            title=dict(text="Heat Flux Vectors",
                       font=dict(family="Inter", size=13, color="#f1f5f9")),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True,
                        config={"displayModeBar": False})

else:
    # Surrogate mode — show 1D temperature profile estimate
    total_L  = sum(c["thickness"] for c in layers_config)
    x_est    = np.linspace(0, total_L, 100) * 1000
    T_est    = T_inlet + (T_outlet - T_inlet) * (x_est / (total_L * 1000))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_est, y=T_est,
        mode="lines", line=dict(color="#f97316", width=3),
        fill="tozeroy", fillcolor="rgba(249,115,22,0.07)",
        hovertemplate="x: %{x:.2f}mm<br>T: %{y:.1f}K<extra></extra>",
        name="Estimated profile",
    ))
    x_pos = 0
    for i, cfg in enumerate(layers_config):
        x1 = x_pos + cfg["thickness"] * 1000
        r, g, b = (int(LAYER_COLORS[i][j:j+2], 16) for j in (1, 3, 5))
        fig.add_vrect(x0=x_pos, x1=x1,
                      fillcolor=f"rgba({r},{g},{b},0.06)", line_width=0)
        fig.add_annotation(x=(x_pos+x1)/2, y=T_inlet,
                           text=f"L{i+1}", yshift=14, showarrow=False,
                           font=dict(color=LAYER_COLORS[i], size=10,
                                     family="JetBrains Mono"))
        x_pos = x1

    fig.update_layout(
        title=dict(text="Estimated Temperature Profile (AI Surrogate Mode)",
                   font=dict(family="Inter", size=15, color="#f1f5f9"),
                   x=0.0, xanchor="left"),
        xaxis=dict(title=dict(text="Wall Thickness (mm)",
                   font=dict(family="JetBrains Mono", size=11, color="#64748b")),
                   tickfont=dict(family="JetBrains Mono", size=10, color="#374151"),
                   gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(title=dict(text="Temperature (K)",
                   font=dict(family="JetBrains Mono", size=11, color="#64748b")),
                   tickfont=dict(family="JetBrains Mono", size=10, color="#374151"),
                   gridcolor="rgba(255,255,255,0.04)"),
        plot_bgcolor="#0a0c10", paper_bgcolor="#111318",
        margin=dict(l=60, r=20, t=70, b=60),
        font=dict(family="Inter", color="#f1f5f9"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── Layer safety cards ────────────────────────────────────────────────────────
if safety:
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;
                letter-spacing:0.2em;text-transform:uppercase;color:#64748b;
                margin:0.5rem 0 0.7rem">Layer Safety Status</div>
    """, unsafe_allow_html=True)
    cols = st.columns(len(safety))
    for i, (col, (status, avg_T)) in enumerate(
        zip(cols, zip(safety, cfd_result["layer_avg_temps"]))
    ):
        sc  = SAFETY_COLORS[status]
        si  = SAFETY_ICONS[status]
        mat = layers_config[i]["material"]
        T_limit = MATERIALS[mat]["T_max"]
        margin  = ((T_limit - cfd_result["layer_avg_temps"][i]) / T_limit) * 100

        # FIX #7: Get quick tip from Gemini for WARNING/CRITICAL layers
        quick_tip = ""
        if GEMINI_AVAILABLE and status in ("WARNING", "CRITICAL"):
            tip_key = f"{mat}_{status}_{T_max:.0f}"
            if tip_key not in st.session_state.quick_tips_cache:
                try:
                    tip = get_quick_tip(
                        hotspot_temp=cfd_result["layer_avg_temps"][i],
                        T_limit=T_limit,
                        layer_name=mat,
                    )
                    st.session_state.quick_tips_cache[tip_key] = tip
                except Exception:
                    st.session_state.quick_tips_cache[tip_key] = ""
            quick_tip = st.session_state.quick_tips_cache.get(tip_key, "")

        tip_html = f"""
            <div style="font-family:'Inter',sans-serif;font-size:0.68rem;
                        color:#94a3b8;margin-top:6px;font-style:italic;
                        line-height:1.4">💡 {quick_tip}</div>
        """ if quick_tip else ""

        with col:
            st.markdown(f"""
            <div style="background:#181c24;border:1px solid #2d3448;
                        border-left:3px solid {sc};border-radius:8px;
                        padding:0.85rem 1rem">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;
                            letter-spacing:0.12em;text-transform:uppercase;
                            color:#64748b;margin-bottom:4px">Layer {i+1}</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.9rem;
                            font-weight:700;color:{sc}">{si} {status}</div>
                <div style="font-family:'Inter',sans-serif;font-size:0.72rem;
                            color:#94a3b8;margin-top:3px">{avg_T:.0f}K avg</div>
                <div style="height:3px;background:#222836;border-radius:2px;
                            margin-top:6px;overflow:hidden">
                    <div style="width:{min(100, max(0, 100-margin)):.0f}%;
                                height:100%;background:{sc};border-radius:2px"></div>
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;
                            color:#374151;margin-top:3px">{margin:.1f}% margin to T_max</div>
                {tip_html}
            </div>
            """, unsafe_allow_html=True)

# ── Surrogate mode safety probabilities ──────────────────────────────────────
elif surrogate_result:
    proba = surrogate_result.get("safety_proba", {})
    if proba:
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;
                    letter-spacing:0.2em;text-transform:uppercase;color:#64748b;
                    margin:0.5rem 0 0.7rem">Predicted Safety Probabilities</div>
        """, unsafe_allow_html=True)
        pc = st.columns(3)
        for col, (cls, p) in zip(pc, proba.items()):
            sc = SAFETY_COLORS.get(cls, "#64748b")
            si = SAFETY_ICONS.get(cls, "?")
            active = cls == surrogate_result["safety_label"]
            with col:
                st.markdown(f"""
                <div style="background:#181c24;border:1px solid {'#2d3448' if not active else sc};
                            border-radius:8px;padding:0.85rem 1rem;
                            opacity:{'1.0' if active else '0.45'}">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:1.2rem;
                                color:{sc}">{si}</div>
                    <div style="font-family:'Inter',sans-serif;font-size:0.78rem;
                                color:#f1f5f9;margin-top:2px">{cls}</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;
                                font-weight:700;color:{sc};margin-top:4px">{p*100:.1f}%</div>
                    <div style="height:3px;background:#222836;border-radius:2px;
                                margin-top:6px;overflow:hidden">
                        <div style="width:{p*100:.1f}%;height:100%;
                                    background:{sc};border-radius:2px"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ── Gemini AI Copilot section ─────────────────────────────────────────────────
st.markdown("""
<div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;
            letter-spacing:0.2em;text-transform:uppercase;color:#06b6d4;
            border-top:1px solid #222836;padding-top:1rem;margin-bottom:0.8rem">
    ✦ Gemini AI Engineering Copilot
</div>
""", unsafe_allow_html=True)

if not GEMINI_AVAILABLE:
    st.warning(
        "gemini_advisor.py not found or google-generativeai not installed.\n\n"
        "Run: `pip install google-generativeai`"
    )
else:
    result_for_gemini = {
        "T_max":        T_max,
        "T_avg":        T_avg,
        "hotspot_x":    hotspot_x,
        "safety_label": overall,
    }
    if cfd_result:
        result_for_gemini.update({
            "max_temperature": cfd_result["max_temperature"],
            "avg_temperature": cfd_result["avg_temperature"],
            "safety_status":   cfd_result["safety_status"],
            "layer_avg_temps": cfd_result["layer_avg_temps"],
            "residual":        cfd_result["residual"],
        })

    ca1, ca2 = st.columns([3, 1])
    with ca1:
        user_q = st.text_area(
            "Ask Gemini about your valve design",
            placeholder="e.g. Why is there a hotspot in layer 2? What material should I use instead? How do I reduce thermal stress?",
            height=80,
            key="gemini_question",
        )
    with ca2:
        st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
        run_gemini = st.button("Ask Gemini ▶", use_container_width=True, key="ask_gemini_btn")
        clear_hist = st.button("Clear history", use_container_width=True, key="clear_hist_btn")

    if clear_hist:
        st.session_state.conversation_history = []
        st.success("Conversation cleared.")

    # FIX #1: Only call Gemini when button is explicitly clicked AND question is not empty
    if run_gemini:
        if not user_q.strip():
            st.warning("Please enter a question before clicking Ask Gemini.")
        else:
            with st.spinner("Gemini is analysing your valve design..."):
                try:
                    reply, new_history = get_gemini_analysis(
                        layers_info=st.session_state.last_layers_info or layers_config,
                        cfd_result=result_for_gemini,
                        T_inlet=T_inlet, T_outlet=T_outlet,
                        h_inner=h_inner, h_outer=h_outer,
                        user_question=user_q,
                        conversation_history=st.session_state.conversation_history,
                    )
                    st.session_state.conversation_history = new_history
                except Exception as e:
                    st.error(f"Gemini API error: {e}")
                    reply = None

            if reply:
                st.markdown(f"""
                <div style="background:#181c24;border:1px solid #2d3448;
                            border-left:3px solid #06b6d4;border-radius:8px;
                            padding:1.1rem 1.3rem;margin-top:0.5rem">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;
                                letter-spacing:0.15em;text-transform:uppercase;
                                color:#06b6d4;margin-bottom:0.6rem">Gemini's Analysis</div>
                    <div style="font-family:'Inter',sans-serif;font-size:0.85rem;
                                color:#94a3b8;line-height:1.75;white-space:pre-wrap">{reply}</div>
                </div>
                """, unsafe_allow_html=True)

    # Show conversation history
    if len(st.session_state.conversation_history) > 2:
        with st.expander(f"Conversation history ({len(st.session_state.conversation_history)//2} exchanges)"):
            for msg in st.session_state.conversation_history:
                role    = msg["role"].upper()
                color   = "#06b6d4" if role == "ASSISTANT" else "#f59e0b"
                prefix  = "GEMINI" if role == "ASSISTANT" else "YOU"
                content = msg["content"][:500] + ("..." if len(msg["content"]) > 500 else "")
                st.markdown(f"""
                <div style="margin-bottom:0.6rem;padding:0.6rem 0.8rem;
                            background:#111318;border-radius:6px;
                            border-left:2px solid {color}">
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;
                                 color:{color};letter-spacing:0.1em">{prefix}</span>
                    <div style="font-family:'Inter',sans-serif;font-size:0.78rem;
                                color:#64748b;margin-top:4px;white-space:pre-wrap">{content}</div>
                </div>
                """, unsafe_allow_html=True)

# ── PDF Report Download ──────────────────────────────────────────────────────
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;
            letter-spacing:0.2em;text-transform:uppercase;color:#a855f7;
            border-top:1px solid #222836;padding-top:1rem;margin-bottom:0.8rem">
    ⬇ Export Analysis Report
</div>
""", unsafe_allow_html=True)

if not REPORT_AVAILABLE:
    st.warning("report_generator.py not found. Cannot generate PDF.")
else:
    rp1, rp2 = st.columns([3, 1])
    with rp1:
        st.markdown("""
        <div style="background:#181c24;border:1px solid #2d3448;
                    border-left:3px solid #a855f7;border-radius:8px;
                    padding:0.9rem 1.1rem">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;
                        letter-spacing:0.15em;text-transform:uppercase;
                        color:#a855f7;margin-bottom:4px">📄 PDF Report Contents</div>
            <div style="font-family:'Inter',sans-serif;font-size:0.8rem;
                        color:#94a3b8;line-height:1.7">
                Cover page · Executive summary · Operating conditions table ·
                Layer configuration · 2D temperature heatmap · Layer safety analysis ·
                Temperature vs T_max bar chart · Gemini AI conversation · Materials appendix
            </div>
        </div>
        """, unsafe_allow_html=True)
    with rp2:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        gen_report = st.button("Generate PDF ▶",
                               use_container_width=True,
                               key="gen_report_btn")

    if gen_report:
        with st.spinner("Building your PDF report... this takes a few seconds"):
            try:
                pdf_bytes = generate_pdf_report(
                    layers_config        = layers_config,
                    layers_info          = st.session_state.last_layers_info or layers_config,
                    cfd_result           = cfd_result,
                    surrogate_result     = surrogate_result,
                    T_inlet              = T_inlet,
                    T_outlet             = T_outlet,
                    h_inner              = h_inner,
                    h_outer              = h_outer,
                    valve_height         = valve_height,
                    sim_time_ms          = sim_time_ms,
                    result_src           = result_src,
                    overall_safety       = overall,
                    gemini_conversation  = st.session_state.conversation_history,
                )
                from datetime import datetime
                fname = f"ValveThermal_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                st.success("Report ready! Click the button below to download.")
                st.download_button(
                    label        = "⬇ Download PDF Report",
                    data         = pdf_bytes,
                    file_name    = fname,
                    mime         = "application/pdf",
                    use_container_width = True,
                    key          = "download_pdf_btn",
                )
            except Exception as e:
                st.error(f"Report generation failed: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="font-family:'JetBrains Mono',monospace;font-size:0.55rem;
            color:#374151;text-align:center;letter-spacing:0.12em;
            border-top:1px solid #222836;padding-top:1rem">
    VALVETHERMAL AI  ·  2D FDM CFD Solver  ·  GBR Surrogate  ·  Gemini AI Copilot  ·  PDF Reports<br>
    cfd_engine.py · surrogate_model.py · gemini_advisor.py · report_generator.py · app.py
</div>
""", unsafe_allow_html=True)
