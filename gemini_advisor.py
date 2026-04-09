"""
gemini_advisor.py
-----------------
Google Gemini AI Engineering Copilot for composite valve thermal analysis.

Uses the Google Generative AI API to:
  1. Interpret CFD simulation results in plain English
  2. Generate design improvement recommendations
  3. Explain failure risks and root causes
  4. Suggest material substitutions
  5. Answer follow-up engineering questions
"""

import os
import json

# ── API Key setup ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyA5mdBfcLiMafhqcYhqeODgUgttB6ZNaDA"

try:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_MODEL_NAME = "gemini-2.0-flash"
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

from cfd_engine import MATERIALS

# Max conversation turns to keep (2 messages per turn)
MAX_HISTORY_MESSAGES = 20


def build_system_prompt() -> str:
    """Return the system prompt for the Gemini engineering advisor."""
    material_list = "\n".join(
        f"  - {name}: k={props['k']} W/mK, T_max={props['T_max']}K"
        for name, props in MATERIALS.items()
    )
    return f"""You are an expert thermal engineering consultant specializing in \
composite valve design and Computational Fluid Dynamics (CFD).

You have deep knowledge of:
- Heat transfer: conduction, convection, radiation
- Composite materials: fiber-reinforced polymers, ceramic matrix, metal matrix
- Valve design for aerospace, automotive, and industrial applications
- CFD simulation interpretation and validation
- Thermal failure modes: delamination, oxidation, creep, fatigue

Available materials in this system:
{material_list}

When analyzing simulation results:
1. Identify critical hotspots and explain WHY they form
2. Assess safety margins relative to material temperature limits
3. Suggest SPECIFIC design changes with quantified expected improvements
4. Use engineering terminology but explain concepts clearly
5. Be direct and actionable — this is a design decision tool

Keep responses concise (under 300 words) but technically precise.
Always mention temperature values in Kelvin and Celsius.
Format recommendations as numbered action items when possible."""


def _trim_history(history: list[dict], max_messages: int = MAX_HISTORY_MESSAGES) -> list[dict]:
    """Keep only the most recent max_messages entries (fix #6: unbounded history)."""
    if len(history) > max_messages:
        return history[-max_messages:]
    return history


def get_gemini_analysis(
    layers_info:          list[dict],
    cfd_result:           dict,
    T_inlet:              float,
    T_outlet:             float,
    h_inner:              float,
    h_outer:              float,
    user_question:        str = "",
    conversation_history: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """
    Get Gemini's engineering analysis and recommendations.

    Parameters
    ----------
    layers_info          : list of layer dicts with material, thickness, angle
    cfd_result           : result dict from CFD simulation or surrogate
    T_inlet              : hot fluid temperature [K]
    T_outlet             : ambient temperature [K]
    h_inner              : convection coefficient inner [W/m²K]
    h_outer              : convection coefficient outer [W/m²K]
    user_question        : optional specific question from user
    conversation_history : previous messages for multi-turn conversation

    Returns
    -------
    (response_text, updated_history)
    """
    if not _GEMINI_AVAILABLE:
        raise ImportError(
            "google-generativeai is not installed. "
            "Run: pip install google-generativeai"
        )

    # Build structured context
    layers_text = "\n".join(
        f"  Layer {i+1}: {lay['material']}, "
        f"thickness={lay['thickness']*1000:.1f}mm, "
        f"fiber_angle={lay.get('angle', 0):.0f}°, "
        f"k={lay.get('conductivity', MATERIALS.get(lay['material'], {}).get('k', 0)):.2f} W/mK"
        for i, lay in enumerate(layers_info)
    )

    safety_text = ""
    if "safety_status" in cfd_result and "layer_avg_temps" in cfd_result:
        for i, (status, avg_t) in enumerate(
            zip(cfd_result["safety_status"], cfd_result["layer_avg_temps"])
        ):
            emoji = "🔴" if status == "CRITICAL" else ("🟡" if status == "WARNING" else "🟢")
            safety_text += f"  {emoji} Layer {i+1}: {status} (avg {avg_t:.0f}K)\n"

    t_max_val = cfd_result.get('T_max', cfd_result.get('max_temperature', 0))
    t_avg_val = cfd_result.get('T_avg', cfd_result.get('avg_temperature', 0))

    context = f"""
=== COMPOSITE VALVE THERMAL ANALYSIS RESULTS ===

OPERATING CONDITIONS:
  Hot fluid inlet: {T_inlet:.0f}K ({T_inlet-273:.0f}°C)
  Ambient/coolant: {T_outlet:.0f}K ({T_outlet-273:.0f}°C)
  Inner convection (h): {h_inner:.0f} W/m²K
  Outer convection (h): {h_outer:.0f} W/m²K

VALVE LAYUP:
{layers_text}

SIMULATION RESULTS:
  Max temperature : {t_max_val:.1f}K ({t_max_val-273:.1f}°C)
  Avg temperature : {t_avg_val:.1f}K
  Hotspot location: x = {cfd_result.get('hotspot_x', 0):.4f}m from inner wall
  Solver          : {cfd_result.get('residual', 'surrogate prediction')}

LAYER SAFETY STATUS:
{safety_text if safety_text else "  (Run full CFD for per-layer safety analysis)"}

OVERALL SAFETY: {cfd_result.get('safety_label', 'See layer status above')}
"""

    # Determine the user message
    if user_question.strip():
        user_msg = f"{context}\n\nMY QUESTION: {user_question}"
    else:
        user_msg = f"""{context}

Please provide:
1. A brief assessment of this valve design's thermal performance
2. The most critical concern (if any) and its root cause
3. Top 3 specific design recommendations to improve thermal management
4. Material substitution suggestions if needed"""

    # Handle conversation history — trim to cap (fix #6)
    if conversation_history is None:
        conversation_history = []
    history = _trim_history(conversation_history, MAX_HISTORY_MESSAGES)

    # Build Gemini chat history format
    # Gemini uses {"role": "user"/"model", "parts": ["text"]}
    gemini_history = []
    for msg in history:
        role = "model" if msg["role"] == "assistant" else "user"
        gemini_history.append({
            "role": role,
            "parts": [msg["content"]],
        })

    system_prompt = build_system_prompt()
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL_NAME,
        system_instruction=system_prompt,
    )
    chat = model.start_chat(history=gemini_history)
    response = chat.send_message(user_msg)
    reply = response.text

    # Update conversation history (store in unified role format)
    updated_history = history + [
        {"role": "user",      "content": user_msg},
        {"role": "assistant", "content": reply},
    ]
    updated_history = _trim_history(updated_history, MAX_HISTORY_MESSAGES)

    return reply, updated_history


def get_material_recommendation(
    current_material: str,
    max_temperature:  float,
    constraint:       str = "cost",   # "cost" | "weight" | "performance"
) -> str:
    """Ask Gemini for a material substitution recommendation."""
    if not _GEMINI_AVAILABLE:
        raise ImportError("google-generativeai not installed.")

    mat_info = MATERIALS.get(current_material, {})
    prompt = f"""A composite valve layer made of {current_material} \
(k={mat_info.get('k','?')} W/mK, T_max={mat_info.get('T_max','?')}K) \
is reaching {max_temperature:.0f}K in service.

The design constraint priority is: {constraint}.

From these available materials:
{json.dumps({k: {kk: vv for kk, vv in v.items()} for k, v in MATERIALS.items()}, indent=2)}

Recommend the best replacement material and explain why in 3 sentences."""

    model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text


def get_quick_tip(hotspot_temp: float, T_limit: float, layer_name: str) -> str:
    """Get a single quick engineering tip from Gemini (fix #7: was dead code)."""
    if not _GEMINI_AVAILABLE:
        return "Install google-generativeai for AI tips."

    margin = ((T_limit - hotspot_temp) / T_limit) * 100
    prompt = f"""In one sentence, give the most important design tip for a composite valve \
where the {layer_name} layer is at {hotspot_temp:.0f}K with only a {margin:.1f}% safety margin \
(T_limit = {T_limit:.0f}K)."""

    model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text.strip()


# Backwards-compat aliases so existing code referencing "claude" works
get_claude_analysis = get_gemini_analysis
