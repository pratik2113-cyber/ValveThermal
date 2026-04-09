"""
report_generator.py
--------------------
Professional PDF Report Generator for ValveThermal AI.

Generates a complete multi-page engineering report including:
  - Cover page with project branding
  - Executive summary with safety verdict
  - Operating conditions & layer configuration tables
  - 2D temperature heatmap (matplotlib)
  - Per-layer safety analysis table
  - Gemini AI recommendations
  - Materials library appendix

Usage:
    pdf_bytes = generate_pdf_report(
        layers_config, layers_info, cfd_result,
        T_inlet, T_outlet, h_inner, h_outer,
        valve_height, sim_time_ms, result_src,
        overall_safety, gemini_conversation
    )
"""

import io
import numpy as np
from datetime import datetime

# ── Matplotlib (non-interactive backend) ──────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

# ── ReportLab Platypus ────────────────────────────────────────────────────────
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors as rl_colors
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF

# ── Material data (duplicated to avoid circular import) ───────────────────────
MATERIALS = {
    "Carbon Fiber Composite": {"k": 5.0,   "rho": 1600, "cp": 900,  "T_max": 450},
    "Glass Fiber Composite":  {"k": 0.35,  "rho": 1800, "cp": 1200, "T_max": 350},
    "Stainless Steel":        {"k": 16.0,  "rho": 7900, "cp": 500,  "T_max": 1200},
    "Titanium Alloy":         {"k": 7.0,   "rho": 4500, "cp": 520,  "T_max": 900},
    "Ceramic Matrix":         {"k": 3.5,   "rho": 2700, "cp": 800,  "T_max": 1500},
    "Polymer Matrix":         {"k": 0.25,  "rho": 1400, "cp": 1300, "T_max": 250},
    "Aluminum Alloy":         {"k": 150.0, "rho": 2700, "cp": 900,  "T_max": 600},
    "Inconel 718":            {"k": 11.4,  "rho": 8190, "cp": 435,  "T_max": 1000},
}

# ── Brand colours ─────────────────────────────────────────────────────────────
C_ORANGE  = rl_colors.HexColor("#f97316")
C_CYAN    = rl_colors.HexColor("#06b6d4")
C_BG      = rl_colors.HexColor("#0a0c10")
C_PANEL   = rl_colors.HexColor("#111318")
C_CARD    = rl_colors.HexColor("#181c24")
C_BORDER  = rl_colors.HexColor("#2d3448")
C_TEXT    = rl_colors.HexColor("#f1f5f9")
C_MUTED   = rl_colors.HexColor("#64748b")
C_DIM     = rl_colors.HexColor("#374151")
C_GREEN   = rl_colors.HexColor("#22c55e")
C_AMBER   = rl_colors.HexColor("#f59e0b")
C_RED     = rl_colors.HexColor("#ef4444")
C_WHITE   = rl_colors.white
C_BLACK   = rl_colors.black

SAFETY_COLOR = {"SAFE": C_GREEN, "WARNING": C_AMBER, "CRITICAL": C_RED}
SAFETY_ICON  = {"SAFE": "✦ SAFE", "WARNING": "⚠ WARNING", "CRITICAL": "✕ CRITICAL"}
LAYER_HEX    = ["#f97316", "#06b6d4", "#a855f7", "#22c55e"]


# ─────────────────────────────────────────────────────────────────────────────
# Style sheet
# ─────────────────────────────────────────────────────────────────────────────
def _make_styles():
    base = getSampleStyleSheet()

    def S(name, **kw):
        return ParagraphStyle(name, **kw)

    return {
        "cover_title": S("cover_title",
            fontSize=36, textColor=C_WHITE, fontName="Helvetica-Bold",
            alignment=TA_CENTER, leading=44, spaceAfter=6),
        "cover_subtitle": S("cover_subtitle",
            fontSize=14, textColor=C_ORANGE, fontName="Helvetica",
            alignment=TA_CENTER, leading=20, spaceAfter=4),
        "cover_meta": S("cover_meta",
            fontSize=9, textColor=C_MUTED, fontName="Helvetica",
            alignment=TA_CENTER, leading=14),
        "section_label": S("section_label",
            fontSize=7, textColor=C_CYAN, fontName="Helvetica-Bold",
            alignment=TA_LEFT, spaceAfter=2, spaceBefore=14,
            leading=10),
        "section_title": S("section_title",
            fontSize=16, textColor=C_TEXT, fontName="Helvetica-Bold",
            alignment=TA_LEFT, spaceAfter=6, leading=20),
        "body": S("body",
            fontSize=9, textColor=C_MUTED, fontName="Helvetica",
            leading=14, spaceAfter=4),
        "body_dark": S("body_dark",
            fontSize=9, textColor=C_TEXT, fontName="Helvetica",
            leading=14, spaceAfter=4),
        "mono": S("mono",
            fontSize=8, textColor=C_MUTED, fontName="Courier",
            leading=12, spaceAfter=2),
        "table_header": S("table_header",
            fontSize=7, textColor=C_MUTED, fontName="Helvetica-Bold",
            alignment=TA_LEFT, leading=10),
        "table_cell": S("table_cell",
            fontSize=8.5, textColor=C_TEXT, fontName="Helvetica",
            alignment=TA_LEFT, leading=12),
        "table_cell_mono": S("table_cell_mono",
            fontSize=8, textColor=C_CYAN, fontName="Courier",
            alignment=TA_RIGHT, leading=12),
        "verdict_safe":     S("verdict_safe",     fontSize=18,
            textColor=C_GREEN,  fontName="Helvetica-Bold", alignment=TA_CENTER),
        "verdict_warning":  S("verdict_warning",  fontSize=18,
            textColor=C_AMBER,  fontName="Helvetica-Bold", alignment=TA_CENTER),
        "verdict_critical": S("verdict_critical", fontSize=18,
            textColor=C_RED,    fontName="Helvetica-Bold", alignment=TA_CENTER),
        "ai_heading": S("ai_heading",
            fontSize=9, textColor=C_CYAN, fontName="Helvetica-Bold",
            spaceAfter=4, leading=12),
        "ai_body": S("ai_body",
            fontSize=8.5, textColor=C_MUTED, fontName="Helvetica",
            leading=13, spaceAfter=3, alignment=TA_JUSTIFY),
        "footer": S("footer",
            fontSize=7, textColor=C_DIM, fontName="Helvetica",
            alignment=TA_CENTER, leading=10),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Page background callback
# ─────────────────────────────────────────────────────────────────────────────
def _draw_page_bg(canvas, doc):
    """Dark background + header bar + footer on every page."""
    W, H = A4
    canvas.saveState()

    # Background
    canvas.setFillColor(C_BG)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)

    # Top orange accent bar
    canvas.setFillColor(C_ORANGE)
    canvas.rect(0, H - 3*mm, W, 3*mm, fill=1, stroke=0)

    # Header text (skip cover page)
    if doc.page > 1:
        canvas.setFillColor(C_DIM)
        canvas.setFont("Helvetica", 7)
        canvas.drawString(1.5*cm, H - 10*mm, "VALVETHERMAL AI  ·  Thermal Analysis Report")
        canvas.drawRightString(W - 1.5*cm, H - 10*mm,
                               f"Page {doc.page}")
        # Header separator
        canvas.setStrokeColor(C_BORDER)
        canvas.setLineWidth(0.3)
        canvas.line(1.5*cm, H - 12*mm, W - 1.5*cm, H - 12*mm)

    # Footer separator + text
    canvas.setStrokeColor(C_BORDER)
    canvas.setLineWidth(0.3)
    canvas.line(1.5*cm, 14*mm, W - 1.5*cm, 14*mm)
    canvas.setFillColor(C_DIM)
    canvas.setFont("Helvetica", 6.5)
    canvas.drawCentredString(W/2, 9*mm,
        "Generated by ValveThermal AI  ·  cfd_engine.py · surrogate_model.py · gemini_advisor.py")

    canvas.restoreState()


# ─────────────────────────────────────────────────────────────────────────────
# Cover page
# ─────────────────────────────────────────────────────────────────────────────
def _cover_page(styles, layers_config, T_inlet, T_outlet,
                overall_safety, sim_time_ms, timestamp):
    story = []
    W, H = A4

    # Large top spacer
    story.append(Spacer(1, 3.5*cm))

    # Brand label
    story.append(Paragraph("COMPOSITE VALVE CFD ANALYSIS", styles["cover_subtitle"]))
    story.append(Spacer(1, 0.4*cm))

    # Main title
    story.append(Paragraph(
        '<font color="#f1f5f9">Valve</font><font color="#f97316">Thermal</font> AI',
        styles["cover_title"]))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph("Thermal Engineering Report", ParagraphStyle(
        "ct2", fontSize=13, textColor=C_MUTED, fontName="Helvetica",
        alignment=TA_CENTER, leading=18)))

    story.append(Spacer(1, 0.6*cm))

    # Decorative rule
    story.append(HRFlowable(width="60%", thickness=1.5,
                            color=C_ORANGE, hAlign="CENTER"))
    story.append(Spacer(1, 0.6*cm))

    # Summary block (table)
    sc = SAFETY_COLOR[overall_safety]
    si = SAFETY_ICON[overall_safety]

    summary_data = [
        ["Report Generated", timestamp],
        ["T_inlet / T_outlet", f"{T_inlet:.0f} K  /  {T_outlet:.0f} K"],
        ["Layers Configured", str(len(layers_config))],
        ["Materials", ", ".join(c["material"].split()[0] for c in layers_config)],
        ["Simulation Time", f"{sim_time_ms:.1f} ms"],
        ["Overall Safety", si],
    ]
    col_w = [5.5*cm, 9.5*cm]
    tbl = Table(summary_data, colWidths=col_w)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), C_PANEL),
        ("BACKGROUND",   (0, 0), (0, -1), C_CARD),
        ("TEXTCOLOR",    (0, 0), (0, -1), C_MUTED),
        ("TEXTCOLOR",    (1, 0), (1, -1), C_TEXT),
        ("TEXTCOLOR",    (1, 5), (1, 5),  sc),
        ("FONTNAME",     (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",     (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUND",(0, 0), (-1, -1), [C_PANEL, C_CARD]),
        ("GRID",         (0, 0), (-1, -1), 0.3, C_BORDER),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING",   (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 7),
        ("FONTNAME",     (1, 5), (1, 5),   "Helvetica-Bold"),
        ("FONTSIZE",     (1, 5), (1, 5),   10),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 1.2*cm))

    # Disclaimer footer
    story.append(Paragraph(
        "This report was automatically generated by ValveThermal AI. "
        "Results are based on 2D FDM simulation and/or AI surrogate predictions. "
        "Always validate critical designs with certified CFD software.",
        ParagraphStyle("disc", fontSize=7.5, textColor=C_DIM,
                       fontName="Helvetica", alignment=TA_CENTER, leading=12)))

    story.append(PageBreak())
    return story


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib: Temperature heatmap → PNG bytes
# ─────────────────────────────────────────────────────────────────────────────
def _render_heatmap(cfd_result, layers_config, valve_height):
    """Render the 2D temperature heatmap as PNG bytes."""
    T_field = cfd_result["temperature_field"]
    x_g     = cfd_result["x_grid"] * 1000        # → mm
    y_g     = cfd_result["y_grid"] * 1000
    bounds  = [b * 1000 for b in cfd_result["layer_boundaries"]]
    T_max   = cfd_result["max_temperature"]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "vt", ["#0a0c10", "#1e3a5f", "#1d4ed8", "#f59e0b", "#ef4444", "#ffffff"])

    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.patch.set_facecolor("#111318")
    ax.set_facecolor("#0a0c10")

    im = ax.pcolormesh(x_g, y_g, T_field, cmap=cmap, shading="auto")
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.ax.yaxis.set_tick_params(color="#64748b", labelcolor="#64748b", labelsize=8)
    cbar.set_label("Temperature (K)", color="#64748b", fontsize=9)
    cbar.ax.set_facecolor("#111318")

    LAYER_COLORS_MPL = ["#f97316", "#06b6d4", "#a855f7", "#22c55e"]
    # Layer boundary lines
    for xb in bounds[1:-1]:
        ax.axvline(xb, color="white", alpha=0.35, linewidth=1.2, linestyle="--")

    # Layer labels
    for i, (x0, x1) in enumerate(zip(bounds[:-1], bounds[1:])):
        mid = (x0 + x1) / 2
        mat = layers_config[i]["material"].split()[0]
        ax.text(mid, y_g[-1] * 1.04, f"L{i+1}\n{mat}",
                ha="center", va="bottom", fontsize=7,
                color=LAYER_COLORS_MPL[i % 4], fontweight="bold")

    # Hotspot
    hx = cfd_result["hotspot_x"] * 1000
    hy = cfd_result["hotspot_y"] * 1000
    ax.plot(hx, hy, "x", color="#ef4444", markersize=12,
            markeredgewidth=2.5, label=f"Hotspot {T_max:.0f}K")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.4,
              labelcolor="white", facecolor="#181c24", edgecolor="#2d3448")

    ax.set_xlabel("Wall Thickness (mm)", color="#64748b", fontsize=9)
    ax.set_ylabel("Valve Height (mm)",   color="#64748b", fontsize=9)
    ax.tick_params(colors="#374151", labelsize=8)
    ax.spines[:].set_color("#222836")
    ax.set_title("2D Temperature Distribution — Composite Valve Cross-Section",
                 color="#f1f5f9", fontsize=10, pad=10)

    plt.tight_layout(pad=0.8)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=160, facecolor="#111318")
    plt.close(fig)
    buf.seek(0)
    return buf


def _render_layer_bar_chart(layers_config, layer_avg_temps, safety_status):
    """Bar chart: per-layer average temperatures vs T_max limits."""
    fig, ax = plt.subplots(figsize=(9, 3.2))
    fig.patch.set_facecolor("#111318")
    ax.set_facecolor("#0a0c10")

    n = len(layers_config)
    x = np.arange(n)
    bar_w = 0.35

    avgs   = layer_avg_temps
    limits = [MATERIALS[c["material"]]["T_max"] for c in layers_config]
    status_colors = [{"SAFE": "#22c55e", "WARNING": "#f59e0b",
                      "CRITICAL": "#ef4444"}[s] for s in safety_status]

    bars1 = ax.bar(x - bar_w/2, avgs, bar_w,
                   color=status_colors, alpha=0.85, label="Avg Temp (K)")
    bars2 = ax.bar(x + bar_w/2, limits, bar_w,
                   color="#374151", alpha=0.7, label="T_max limit (K)")

    for bar, val in zip(bars1, avgs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f"{val:.0f}K", ha="center", va="bottom",
                fontsize=8, color="#f1f5f9", fontweight="bold")
    for bar, val in zip(bars2, limits):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f"{val:.0f}K", ha="center", va="bottom",
                fontsize=8, color="#64748b")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {i+1}\n{c['material'].split()[0]}"
                        for i, c in enumerate(layers_config)],
                       fontsize=8, color="#94a3b8")
    ax.set_ylabel("Temperature (K)", color="#64748b", fontsize=9)
    ax.tick_params(colors="#374151", labelsize=8)
    ax.spines[:].set_color("#222836")
    ax.set_title("Layer Average Temperature vs. Material T_max Limit",
                 color="#f1f5f9", fontsize=10, pad=8)
    ax.legend(fontsize=8, framealpha=0.4, labelcolor="white",
              facecolor="#181c24", edgecolor="#2d3448")

    plt.tight_layout(pad=0.8)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, facecolor="#111318")
    plt.close(fig)
    buf.seek(0)
    return buf


def _render_surrogate_profile(layers_config, T_inlet, T_outlet):
    """1D estimated temperature profile for surrogate mode."""
    total_L = sum(c["thickness"] for c in layers_config) * 1000  # mm
    x_est = np.linspace(0, total_L, 200)
    T_est = T_inlet + (T_outlet - T_inlet) * (x_est / total_L)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    fig.patch.set_facecolor("#111318")
    ax.set_facecolor("#0a0c10")

    ax.plot(x_est, T_est, color="#f97316", linewidth=2.5, label="Est. Temperature Profile")
    ax.fill_between(x_est, T_est, alpha=0.08, color="#f97316")

    LAYER_COLORS_MPL = ["#f97316", "#06b6d4", "#a855f7", "#22c55e"]
    x_pos = 0
    for i, cfg in enumerate(layers_config):
        x1 = x_pos + cfg["thickness"] * 1000
        hex_c = LAYER_COLORS_MPL[i % 4]
        r, g, b = int(hex_c[1:3], 16)/255, int(hex_c[3:5], 16)/255, int(hex_c[5:7], 16)/255
        ax.axvspan(x_pos, x1, alpha=0.07, color=(r, g, b))
        ax.text((x_pos+x1)/2, T_inlet + 15, f"L{i+1}", ha="center",
                fontsize=9, color=hex_c, fontweight="bold")
        x_pos = x1

    ax.set_xlabel("Wall Thickness (mm)", color="#64748b", fontsize=9)
    ax.set_ylabel("Temperature (K)",     color="#64748b", fontsize=9)
    ax.tick_params(colors="#374151", labelsize=8)
    ax.spines[:].set_color("#222836")
    ax.set_title("Estimated Temperature Profile (AI Surrogate Mode)",
                 color="#f1f5f9", fontsize=10, pad=8)
    ax.legend(fontsize=8, framealpha=0.4, labelcolor="white",
              facecolor="#181c24", edgecolor="#2d3448")

    plt.tight_layout(pad=0.8)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, facecolor="#111318")
    plt.close(fig)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# Section helpers
# ─────────────────────────────────────────────────────────────────────────────
def _section(styles, label, title):
    return [
        Paragraph(label.upper(), styles["section_label"]),
        Paragraph(title, styles["section_title"]),
        HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceAfter=6),
    ]


def _dark_table(data, col_widths, row_colors=None):
    """Styled dark table with alternating rows."""
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0),  C_CARD),
        ("TEXTCOLOR",  (0, 0), (-1, 0),  C_MUTED),
        ("FONTNAME",   (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0),  7.5),
        ("GRID",       (0, 0), (-1, -1), 0.3, C_BORDER),
        ("LEFTPADDING", (0,0), (-1,-1),  8),
        ("RIGHTPADDING",(0,0), (-1,-1),  8),
        ("TOPPADDING",  (0,0), (-1,-1),  6),
        ("BOTTOMPADDING",(0,0),(-1,-1),  6),
        ("ROWBACKGROUND",(0,1),(-1,-1),  [C_PANEL, C_CARD]),
        ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 1), (-1, -1), 8.5),
        ("TEXTCOLOR",  (0, 1), (-1, -1), C_TEXT),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
    ]
    tbl.setStyle(TableStyle(style))
    return tbl


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf_report(
    layers_config:        list[dict],
    layers_info:          list[dict],
    cfd_result:           dict | None,
    surrogate_result:     dict | None,
    T_inlet:              float,
    T_outlet:             float,
    h_inner:              float,
    h_outer:              float,
    valve_height:         float,
    sim_time_ms:          float,
    result_src:           str,
    overall_safety:       str,
    gemini_conversation:  list[dict] | None = None,
) -> bytes:
    """
    Generate a complete PDF thermal analysis report.

    Returns
    -------
    bytes — the raw PDF file content for st.download_button
    """
    buf       = io.BytesIO()
    timestamp = datetime.now().strftime("%d %B %Y  |  %H:%M:%S")
    styles    = _make_styles()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=1.5*cm, rightMargin=1.5*cm,
        topMargin=1.8*cm,  bottomMargin=2.0*cm,
        title="ValveThermal AI — Thermal Analysis Report",
        author="ValveThermal AI",
        subject="Composite Valve CFD Thermal Report",
    )

    story = []

    # ── 1. Cover ──────────────────────────────────────────────────────────
    story += _cover_page(styles, layers_config, T_inlet, T_outlet,
                         overall_safety, sim_time_ms, timestamp)

    # ── 2. Executive Summary ──────────────────────────────────────────────
    story += _section(styles, "01 · Executive Summary", "Simulation Overview")

    T_max = (cfd_result["max_temperature"] if cfd_result
             else surrogate_result["T_max"])
    T_avg = (cfd_result["avg_temperature"] if cfd_result
             else surrogate_result["T_avg"])
    hotspot_x = (cfd_result["hotspot_x"] if cfd_result
                 else surrogate_result.get("hotspot_x", 0))

    sc     = SAFETY_COLOR[overall_safety]
    si     = SAFETY_ICON[overall_safety]
    solver = "Full 2D CFD (FDM)" if cfd_result else "AI Surrogate (GBR)"

    exec_data = [
        ["METRIC", "VALUE", "UNIT"],
        ["Maximum Temperature",  f"{T_max:.1f}",        "K"],
        ["Maximum Temperature",  f"{T_max-273:.1f}",    "°C"],
        ["Average Temperature",  f"{T_avg:.1f}",        "K"],
        ["Hotspot Position",     f"{hotspot_x*1000:.2f}", "mm from inner wall"],
        ["Total Wall Thickness", f"{sum(c['thickness'] for c in layers_config)*1000:.1f}", "mm"],
        ["Number of Layers",     str(len(layers_config)), "—"],
        ["Solver Method",        solver,                 "—"],
        ["Simulation Time",      f"{sim_time_ms:.1f}",  "ms"],
        ["Overall Safety",       si,                    "—"],
    ]
    col_w = [7*cm, 5*cm, 5.5*cm]
    tbl = _dark_table(exec_data, col_w)
    # Color the safety row
    safety_row = len(exec_data) - 1
    tbl.setStyle(TableStyle([
        ("TEXTCOLOR", (1, safety_row), (1, safety_row), sc),
        ("FONTNAME",  (1, safety_row), (1, safety_row), "Helvetica-Bold"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.5*cm))

    # Verdict badge (drawn inline)
    verdict_style = styles[f"verdict_{overall_safety.lower()}"]
    story.append(KeepTogether([
        Paragraph("OVERALL SAFETY VERDICT", styles["section_label"]),
        Table([[Paragraph(si, verdict_style)]],
              colWidths=["100%"],
              style=TableStyle([
                  ("BACKGROUND",    (0,0), (-1,-1), C_CARD),
                  ("GRID",          (0,0), (-1,-1), 1.2, sc),
                  ("TOPPADDING",    (0,0), (-1,-1), 14),
                  ("BOTTOMPADDING", (0,0), (-1,-1), 14),
                  ("ALIGN",         (0,0), (-1,-1), "CENTER"),
              ])),
    ]))
    story.append(PageBreak())

    # ── 3. Operating Conditions ───────────────────────────────────────────
    story += _section(styles, "02 · Configuration",
                      "Operating Conditions & Layer Setup")

    cond_data = [
        ["PARAMETER", "VALUE"],
        ["Hot Fluid Inlet (T_inlet)",  f"{T_inlet:.0f} K  ({T_inlet-273:.0f} °C)"],
        ["Ambient / Coolant (T_outlet)", f"{T_outlet:.0f} K  ({T_outlet-273:.0f} °C)"],
        ["ΔT across wall",             f"{T_inlet-T_outlet:.0f} K"],
        ["Inner Convection h",         f"{h_inner:.0f} W/m²K"],
        ["Outer Convection h",         f"{h_outer:.0f} W/m²K"],
        ["Valve Height",               f"{valve_height*1000:.1f} mm"],
        ["Solver",                     solver],
    ]
    story.append(_dark_table(cond_data, [8*cm, 9.5*cm]))
    story.append(Spacer(1, 0.6*cm))

    # Layer configuration table
    story.append(Paragraph("LAYER CONFIGURATION", styles["section_label"]))
    layer_hdr = ["#", "Material", "Thickness", "Fiber Angle", "k (W/mK)", "T_max (K)"]
    layer_rows = [layer_hdr] + [
        [
            str(i+1),
            c["material"],
            f"{c['thickness']*1000:.1f} mm",
            f"{c['angle']:.0f}°",
            f"{MATERIALS[c['material']]['k']}",
            f"{MATERIALS[c['material']]['T_max']}",
        ]
        for i, c in enumerate(layers_config)
    ]
    col_w = [0.8*cm, 5.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm]
    lay_tbl = _dark_table(layer_rows, col_w)
    # Color-code each layer number cell
    for i in range(len(layers_config)):
        hex_c = LAYER_HEX[i % 4]
        r, g, b = (int(hex_c[j:j+2], 16)/255 for j in (1, 3, 5))
        lay_tbl.setStyle(TableStyle([
            ("TEXTCOLOR", (0, i+1), (0, i+1),
             rl_colors.Color(r, g, b)),
            ("FONTNAME",  (0, i+1), (0, i+1), "Helvetica-Bold"),
        ]))
    story.append(lay_tbl)
    story.append(PageBreak())

    # ── 4. Temperature Analysis ───────────────────────────────────────────
    story += _section(styles, "03 · Thermal Analysis",
                      "Temperature Distribution Results")

    if cfd_result:
        story.append(Paragraph(
            "The 2D temperature field was computed using the Finite Difference Method (FDM) "
            "with Jacobi iteration. The colormap below shows temperature in Kelvin — "
            "from deep blue (cold) through amber to white (hottest).",
            styles["body"]))
        story.append(Spacer(1, 0.3*cm))

        # Heatmap image
        hm_buf = _render_heatmap(cfd_result, layers_config, valve_height)
        img = Image(hm_buf, width=17*cm, height=7.2*cm)
        story.append(img)
        story.append(Spacer(1, 0.4*cm))

        # CFD convergence info
        conv_data = [
            ["SOLVER PARAMETER", "VALUE"],
            ["Algorithm",         "2D Gauss-Seidel / Jacobi FDM"],
            ["Grid Points (x)",   str(len(layers_config) * 25)],
            ["Grid Points (y)",   "35"],
            ["Iterations Used",   str(cfd_result.get("iterations", "—"))],
            ["Final Residual",    f"{cfd_result.get('residual', 0):.2e}"],
            ["Convergence",       "Yes" if cfd_result.get("residual", 1) < 1e-4
                                  else "Max iterations reached"],
        ]
        story.append(_dark_table(conv_data, [8*cm, 9.5*cm]))

    else:
        story.append(Paragraph(
            "The AI Surrogate model (Gradient Boosting Regressor) was used for instant prediction. "
            "The profile below shows the estimated 1D temperature distribution across the wall.",
            styles["body"]))
        story.append(Spacer(1, 0.3*cm))
        sp_buf = _render_surrogate_profile(layers_config, T_inlet, T_outlet)
        img = Image(sp_buf, width=17*cm, height=6*cm)
        story.append(img)
        story.append(Spacer(1, 0.4*cm))

        if surrogate_result:
            proba = surrogate_result.get("safety_proba", {})
            if proba:
                story.append(Paragraph("PREDICTED SAFETY PROBABILITIES", styles["section_label"]))
                prob_data = [["CLASS", "PROBABILITY"]] + [
                    [cls, f"{p*100:.1f}%"] for cls, p in proba.items()
                ]
                story.append(_dark_table(prob_data, [8*cm, 9.5*cm]))

    story.append(PageBreak())

    # ── 5. Layer Safety Analysis ──────────────────────────────────────────
    story += _section(styles, "04 · Safety Analysis",
                      "Per-Layer Thermal Safety Assessment")

    if cfd_result and cfd_result.get("safety_status"):
        safety_statuses = cfd_result["safety_status"]
        avg_temps       = cfd_result["layer_avg_temps"]

        story.append(Paragraph(
            "Each layer is classified based on how its peak temperature compares to "
            "the material's rated maximum temperature (T_max). "
            "WARNING is triggered above 85% of T_max; CRITICAL above 100%.",
            styles["body"]))
        story.append(Spacer(1, 0.3*cm))

        # Safety table
        safety_hdr = ["Layer", "Material", "Avg Temp (K)", "T_max (K)",
                      "Margin %", "Status"]
        safety_rows = [safety_hdr]
        for i, (cfg, avg_T, status) in enumerate(
            zip(layers_config, avg_temps, safety_statuses)
        ):
            T_lim  = MATERIALS[cfg["material"]]["T_max"]
            margin = ((T_lim - avg_T) / T_lim) * 100
            safety_rows.append([
                f"L{i+1}",
                cfg["material"],
                f"{avg_T:.1f}",
                f"{T_lim}",
                f"{margin:.1f}%",
                SAFETY_ICON[status],
            ])

        s_tbl = _dark_table(safety_rows, [1*cm, 5.5*cm, 2.5*cm, 2.2*cm, 2*cm, 2.8*cm])
        # Color status cells
        for i, status in enumerate(safety_statuses):
            sc = SAFETY_COLOR[status]
            s_tbl.setStyle(TableStyle([
                ("TEXTCOLOR", (5, i+1), (5, i+1), sc),
                ("FONTNAME",  (5, i+1), (5, i+1), "Helvetica-Bold"),
                ("FONTSIZE",  (5, i+1), (5, i+1), 9),
            ]))
        story.append(s_tbl)
        story.append(Spacer(1, 0.5*cm))

        # Bar chart
        story.append(Paragraph("TEMPERATURE vs. MATERIAL LIMIT", styles["section_label"]))
        bar_buf = _render_layer_bar_chart(layers_config, avg_temps, safety_statuses)
        story.append(Image(bar_buf, width=17*cm, height=6*cm))

    else:
        story.append(Paragraph(
            "Full per-layer safety analysis requires the Full CFD solver mode. "
            "Run with 'Full CFD' selected to get detailed layer-by-layer breakdown.",
            styles["body"]))
        if surrogate_result:
            story.append(Spacer(1, 0.4*cm))
            story.append(Paragraph(
                f"AI Surrogate Safety Prediction: "
                f"<font color='#{_hex(SAFETY_COLOR[overall_safety])}'>"
                f"{SAFETY_ICON[overall_safety]}</font>",
                styles["body_dark"]))

    story.append(PageBreak())

    # ── 6. Gemini AI Recommendations ─────────────────────────────────────
    story += _section(styles, "05 · AI Engineering Advisor",
                      "Gemini AI Recommendations")

    if gemini_conversation and len(gemini_conversation) >= 2:
        story.append(Paragraph(
            "The following is the Gemini AI engineering consultation recorded during "
            "this analysis session. Each exchange is shown in full.",
            styles["body"]))
        story.append(Spacer(1, 0.3*cm))

        exchange_n = 0
        for i in range(0, len(gemini_conversation) - 1, 2):
            user_msg = gemini_conversation[i]
            ai_msg   = gemini_conversation[i+1] if i+1 < len(gemini_conversation) else None
            exchange_n += 1

            # User question — extract just the question part after "MY QUESTION:"
            content = user_msg.get("content", "")
            if "MY QUESTION:" in content:
                q_text = content.split("MY QUESTION:")[-1].strip()
            else:
                q_text = "(Automatic analysis request)"

            story.append(KeepTogether([
                Paragraph(f"Exchange {exchange_n}", styles["ai_heading"]),
                Table(
                    [[Paragraph(f"YOU: {q_text[:400]}", ParagraphStyle(
                        "q", fontSize=8.5, textColor=C_AMBER,
                        fontName="Helvetica-Bold", leading=13))]],
                    colWidths=["100%"],
                    style=TableStyle([
                        ("BACKGROUND",    (0,0), (-1,-1), C_PANEL),
                        ("LEFTPADDING",   (0,0), (-1,-1), 10),
                        ("RIGHTPADDING",  (0,0), (-1,-1), 10),
                        ("TOPPADDING",    (0,0), (-1,-1), 7),
                        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
                        ("LINEAFTER",     (0,0), (0,-1),  2, C_AMBER),
                    ])
                ),
                Spacer(1, 3),
            ]))

            if ai_msg:
                ai_content = ai_msg.get("content", "")[:1200]
                story.append(KeepTogether([
                    Table(
                        [[Paragraph(ai_content, styles["ai_body"])]],
                        colWidths=["100%"],
                        style=TableStyle([
                            ("BACKGROUND",    (0,0), (-1,-1), C_CARD),
                            ("LEFTPADDING",   (0,0), (-1,-1), 10),
                            ("RIGHTPADDING",  (0,0), (-1,-1), 10),
                            ("TOPPADDING",    (0,0), (-1,-1), 8),
                            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
                            ("LINEAFTER",     (0,0), (0,-1),  2, C_CYAN),
                        ])
                    ),
                    Spacer(1, 8),
                ]))
    else:
        story.append(Paragraph(
            "No Gemini AI conversation was recorded in this session. "
            "Use the 'Ask Gemini ▶' feature in the app to get AI engineering recommendations, "
            "which will be included in the next report you generate.",
            styles["body"]))

    story.append(PageBreak())

    # ── 7. Materials Library ──────────────────────────────────────────────
    story += _section(styles, "06 · Appendix",
                      "Materials Library Reference")

    story.append(Paragraph(
        "Thermal and physical properties of all materials available in ValveThermal AI.",
        styles["body"]))
    story.append(Spacer(1, 0.3*cm))

    mat_hdr  = ["Material", "k (W/mK)", "ρ (kg/m³)", "cp (J/kgK)", "T_max (K)"]
    mat_rows = [mat_hdr] + [
        [name,
         str(props["k"]),
         str(props["rho"]),
         str(props["cp"]),
         str(props["T_max"])]
        for name, props in MATERIALS.items()
    ]
    # Highlight materials in use
    used = {c["material"] for c in layers_config}
    m_tbl = _dark_table(mat_rows, [6.5*cm, 2.2*cm, 2.2*cm, 2.4*cm, 2.2*cm])
    for i, (name, _) in enumerate(MATERIALS.items()):
        if name in used:
            m_tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, i+1), (-1, i+1), rl_colors.HexColor("#1e293b")),
                ("TEXTCOLOR",  (0, i+1), (0,  i+1), C_ORANGE),
                ("FONTNAME",   (0, i+1), (0,  i+1), "Helvetica-Bold"),
            ]))
    story.append(m_tbl)
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(
        "★ Materials highlighted in orange are currently used in this simulation.",
        ParagraphStyle("note", fontSize=7.5, textColor=C_ORANGE,
                       fontName="Helvetica", leading=10)))

    story.append(Spacer(1, 0.8*cm))

    # Physics reference
    story += _section(styles, "06b · Appendix",
                      "Physics & Solver Reference")
    story.append(Paragraph(
        "Governing Equation (2D Steady-State Heat Conduction):",
        styles["body_dark"]))
    story.append(Paragraph(
        "∂²T/∂x² + ∂²T/∂y² + Q/k = 0",
        styles["mono"]))
    story.append(Spacer(1, 0.2*cm))

    phys_rows = [
        ["BOUNDARY", "CONDITION"],
        ["Inner wall (left)",  "Convective: q = h_inner × (T_fluid − T_wall)"],
        ["Outer wall (right)", "Convective: q = h_outer × (T_wall − T_ambient)"],
        ["Top / Bottom",       "Adiabatic (Neumann): dT/dn = 0"],
        ["Fiber angle effect", "k_eff = k_base × (cos²θ + 0.1 sin²θ)"],
        ["Solver algorithm",   "Jacobi iterative (vectorized NumPy)"],
        ["Surrogate model",    "Gradient Boosting Regressor (scikit-learn)"],
        ["AI Copilot",         "Google Gemini 2.0 Flash"],
    ]
    story.append(_dark_table(phys_rows, [5.5*cm, 12*cm]))

    # ── Build PDF ─────────────────────────────────────────────────────────
    doc.build(story, onFirstPage=_draw_page_bg, onLaterPages=_draw_page_bg)
    return buf.getvalue()


def _hex(rl_color):
    """Convert ReportLab color to hex string (for inline HTML)."""
    try:
        r = int(rl_color.red   * 255)
        g = int(rl_color.green * 255)
        b = int(rl_color.blue  * 255)
        return f"{r:02x}{g:02x}{b:02x}"
    except Exception:
        return "64748b"
