"""
Interactive Dash dashboard — faithful port from Android DashboardFragment.

Features:
  - 3D scatter plot with clusters + Catmull-Rom spline path
  - Flow field particle animation (Turbo speed coloring, 20 FPS)
  - Ball probe following MDN flow field with trail
  - Bidirectional sliders: manual probe OR auto-track ball position
  - Mark control points + explain via LLM
  - Toggle: flow animation, ball flow, show points, show path
  - Dark theme (#1E1E1E matching Android glClearColor)
"""

from __future__ import annotations

import json
import logging
from typing import Optional, List

import numpy as np

try:
    import plotly.graph_objects as go
    from dash import Dash, html, dcc, Input, Output, State, callback_context, no_update
except ImportError as _e:
    raise ImportError(
        "Visualization dependencies not installed. "
        "Install them with: pip install plotly dash\n"
        "Or install the full package: pip install tracescope"
    ) from _e

from tracescope.models.analysis import AnalysisResult
from tracescope.visualization.scatter3d import (
    plot_clusters_3d, catmull_rom_spline, CLUSTER_COLORS, _apply_dark_theme,
)
from tracescope.visualization.flow_field import FlowFieldSystem, turbo_colormap
from tracescope.visualization.probe import probe_point

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════
# Module-level state (not serialized across callbacks)
# ═══════════════════════════════════════════════════

_flow_system: Optional[FlowFieldSystem] = None
_result: Optional[AnalysisResult] = None
_explainer = None


# ═══════════════════════════════════════════════════
# DARK THEME STYLES
# ═══════════════════════════════════════════════════

_BG = "#1E1E1E"
_CARD_BG = "#2A2A2A"
_BORDER = "#3A3A3A"
_TEXT = "#E0E0E0"
_ACCENT = "#FF6B35"

_CARD_STYLE = {
    "backgroundColor": _CARD_BG,
    "border": f"1px solid {_BORDER}",
    "borderRadius": "8px",
    "padding": "12px",
    "marginBottom": "10px",
}


def _slider_to_raw(slider_val: float, axis_idx: int) -> float:
    """Convert slider [0..1000] to raw 3D coordinate."""
    pts = _result.projected_3d
    axis_min = pts[:, axis_idx].min()
    axis_max = pts[:, axis_idx].max()
    return axis_min + (slider_val / 1000.0) * (axis_max - axis_min)


def _raw_to_slider(raw_val: float, axis_idx: int) -> int:
    """Convert raw 3D coordinate to slider [0..1000]."""
    pts = _result.projected_3d
    axis_min = pts[:, axis_idx].min()
    axis_max = pts[:, axis_idx].max()
    span = axis_max - axis_min
    if span == 0:
        return 500
    pct = (raw_val - axis_min) / span
    return int(np.clip(pct * 1000, 0, 1000))


# ═══════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════

def _build_layout(result: AnalysisResult) -> html.Div:
    """Build the full Dash layout with all controls."""
    axis_labels = result.axis_info.labels
    has_flow = result.velocity_grid is not None

    # Cluster legend items
    cluster_items = []
    for c in range(result.clusters.n_clusters):
        color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        label = (
            result.cluster_labels[c]
            if c < len(result.cluster_labels)
            else f"Cluster {c}"
        )
        count = sum(1 for l in result.clusters.labels if l == c)
        cluster_items.append(
            html.Div([
                html.Span("● ", style={"color": color, "fontSize": "16px"}),
                html.Strong(label, style={"color": _TEXT}),
                html.Span(f" ({count})", style={"color": "#888"}),
            ], style={"marginBottom": "3px"})
        )

    # Flow toggle options
    flow_options = []
    if has_flow:
        flow_options = [
            {"label": " Enable Flow Animation", "value": "flow_animation"},
            {"label": " Ball Flow (follow MDN)", "value": "ball_flow"},
        ]

    display_options = [
        {"label": " Show Data Points", "value": "show_points"},
        {"label": " Show Path", "value": "show_path"},
    ]

    return html.Div([
        # Hidden state stores
        dcc.Store(id="flow-state", data={
            "ball_position": [0, 0, 0],
            "ball_trail": [],
            "control_points": [],
            "frame_index": 0,
        }),
        dcc.Store(id="control-points-store", data=[]),

        # Animation timer (50ms = 20 FPS)
        dcc.Interval(
            id="flow-interval",
            interval=50,
            n_intervals=0,
            disabled=True,
        ),

        # Header
        html.H2("TraceScope Dashboard", style={
            "textAlign": "center", "color": _TEXT, "margin": "0 0 15px 0",
            "fontWeight": "300", "letterSpacing": "2px",
        }),

        # Main flex container
        html.Div([
            # LEFT: 3D scatter plot
            html.Div([
                dcc.Graph(
                    id="scatter3d",
                    figure=_build_base_figure(result),
                    style={"height": "650px"},
                    config={"scrollZoom": True},
                ),
            ], style={"flex": "3", "minWidth": "500px"}),

            # RIGHT: Controls panel
            html.Div([
                # Flow controls
                html.Div([
                    html.H4("⚡ Flow Controls" if has_flow else "⚡ Flow (not available)",
                             style={"color": _ACCENT, "margin": "0 0 8px 0", "fontSize": "14px"}),
                    dcc.Checklist(
                        id="flow-toggles",
                        options=flow_options,
                        value=[],
                        style={"color": _TEXT, "fontSize": "13px"},
                        inputStyle={"marginRight": "6px"},
                        labelStyle={"display": "block", "marginBottom": "4px"},
                    ) if has_flow else html.Div(
                        "Train with train_flow=True to enable",
                        style={"color": "#666", "fontSize": "12px", "fontStyle": "italic"},
                    ),
                ], style=_CARD_STYLE),

                # Display toggles
                html.Div([
                    html.H4("👁 Display",
                             style={"color": _ACCENT, "margin": "0 0 8px 0", "fontSize": "14px"}),
                    dcc.Checklist(
                        id="display-toggles",
                        options=display_options,
                        value=["show_points", "show_path"],
                        style={"color": _TEXT, "fontSize": "13px"},
                        inputStyle={"marginRight": "6px"},
                        labelStyle={"display": "block", "marginBottom": "4px"},
                    ),
                ], style=_CARD_STYLE),

                # Probe sliders
                html.Div([
                    html.H4("🎯 Probe",
                             style={"color": _ACCENT, "margin": "0 0 8px 0", "fontSize": "14px"}),
                    # X slider
                    html.Div(id="slider-x-label",
                             children=f"{axis_labels[0]} 50%",
                             style={"color": _TEXT, "fontSize": "12px", "marginBottom": "2px"}),
                    dcc.Slider(
                        id="slider-x", min=0, max=1000, value=500, step=1,
                        marks={}, updatemode="drag",
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    # Y slider
                    html.Div(id="slider-y-label",
                             children=f"{axis_labels[1]} 50%",
                             style={"color": _TEXT, "fontSize": "12px", "marginBottom": "2px"}),
                    dcc.Slider(
                        id="slider-y", min=0, max=1000, value=500, step=1,
                        marks={}, updatemode="drag",
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    # Z slider
                    html.Div(id="slider-z-label",
                             children=f"{axis_labels[2]} 50%",
                             style={"color": _TEXT, "fontSize": "12px", "marginBottom": "2px"}),
                    dcc.Slider(
                        id="slider-z", min=0, max=1000, value=500, step=1,
                        marks={}, updatemode="drag",
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    # Buttons
                    html.Div([
                        html.Button("Mark Point", id="btn-mark", n_clicks=0,
                                    style={"marginRight": "6px", "padding": "4px 12px",
                                           "backgroundColor": _ACCENT, "color": "white",
                                           "border": "none", "borderRadius": "4px", "cursor": "pointer"}),
                        html.Button("Clear", id="btn-clear", n_clicks=0,
                                    style={"marginRight": "6px", "padding": "4px 12px",
                                           "backgroundColor": "#555", "color": "white",
                                           "border": "none", "borderRadius": "4px", "cursor": "pointer"}),
                        html.Button("Explain", id="btn-explain", n_clicks=0,
                                    style={"padding": "4px 12px",
                                           "backgroundColor": "#4A90D9", "color": "white",
                                           "border": "none", "borderRadius": "4px", "cursor": "pointer"}),
                    ], style={"marginTop": "8px"}),
                ], style=_CARD_STYLE),

                # Clusters
                html.Div([
                    html.H4("🔵 Clusters",
                             style={"color": _ACCENT, "margin": "0 0 8px 0", "fontSize": "14px"}),
                    html.Div(cluster_items),
                ], style=_CARD_STYLE),

                # Info panel
                html.Div([
                    html.H4("📊 Info",
                             style={"color": _ACCENT, "margin": "0 0 8px 0", "fontSize": "14px"}),
                    html.Div(id="info-panel", style={
                        "whiteSpace": "pre-wrap", "color": _TEXT,
                        "fontSize": "12px", "maxHeight": "200px", "overflowY": "auto",
                    }),
                ], style=_CARD_STYLE),

            ], style={
                "flex": "1", "minWidth": "280px", "maxWidth": "350px",
                "padding": "0 0 0 10px", "overflowY": "auto",
                "maxHeight": "680px",
            }),
        ], style={"display": "flex", "gap": "10px"}),

    ], style={
        "padding": "15px", "fontFamily": "'Segoe UI', sans-serif",
        "backgroundColor": _BG, "minHeight": "100vh",
    })


def _build_base_figure(result: AnalysisResult) -> go.Figure:
    """Build the initial figure with clusters + path."""
    return plot_clusters_3d(result, show_path=True, use_spline=True)


# ═══════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════

def _register_callbacks(app: Dash, result: AnalysisResult):
    """Register all Dash callbacks."""

    # ── Callback 1: Flow interval enable/disable ─────────────────
    @app.callback(
        Output("flow-interval", "disabled"),
        Input("flow-toggles", "value"),
    )
    def toggle_flow_interval(flow_toggles):
        flow_toggles = flow_toggles or []
        # Enable interval when flow animation is checked
        return "flow_animation" not in flow_toggles

    # ── Callback 2: Flow state update (animation tick) ───────────
    @app.callback(
        Output("flow-state", "data"),
        Input("flow-interval", "n_intervals"),
        State("flow-state", "data"),
        State("flow-toggles", "value"),
        prevent_initial_call=True,
    )
    def update_flow_state(n_intervals, state, flow_toggles):
        if _flow_system is None:
            return no_update

        flow_toggles = flow_toggles or []

        # Step flow particles
        pos, colors, alphas, speeds = _flow_system.step()

        # Convert colors to Plotly rgba strings (subsample for performance)
        max_render = 4000  # Render at most 4000 particles
        step_size = max(1, len(pos) // max_render)
        render_pos = pos[::step_size]
        render_colors = colors[::step_size]
        render_alphas = alphas[::step_size]

        particle_colors = [
            f"rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},{a:.2f})"
            for c, a in zip(render_colors, render_alphas)
        ]

        state["particle_x"] = render_pos[:, 0].tolist()
        state["particle_y"] = render_pos[:, 1].tolist()
        state["particle_z"] = render_pos[:, 2].tolist()
        state["particle_colors"] = particle_colors
        state["frame_index"] = n_intervals

        # Ball flow
        if "ball_flow" in flow_toggles:
            _flow_system.advance_ball()
            bp = _flow_system.ball_pos.tolist()
            state["ball_position"] = bp
            # Trail: last 100 positions
            trail = state.get("ball_trail", [])
            trail.append(bp)
            if len(trail) > 100:
                trail = trail[-100:]
            state["ball_trail"] = trail
        else:
            # Ball stays where sliders put it
            state["ball_trail"] = []

        return state

    # ── Callback 3: Main figure render ───────────────────────────
    @app.callback(
        Output("scatter3d", "figure"),
        Input("flow-state", "data"),
        Input("slider-x", "value"),
        Input("slider-y", "value"),
        Input("slider-z", "value"),
        Input("scatter3d", "clickData"),
        Input("display-toggles", "value"),
        Input("control-points-store", "data"),
        State("flow-toggles", "value"),
    )
    def render_figure(flow_data, sx, sy, sz, click_data,
                      display_toggles, control_points, flow_toggles):
        display_toggles = display_toggles or []
        flow_toggles = flow_toggles or []

        fig = go.Figure()
        pts = result.projected_3d
        labels = result.clusters.labels
        axis_labels = result.axis_info.labels

        # ── Cluster scatter points ──────────────────────────
        if "show_points" in display_toggles:
            for c in range(result.clusters.n_clusters):
                mask = [i for i, l in enumerate(labels) if l == c]
                if not mask:
                    continue
                cluster_pts = pts[mask]
                cluster_texts = [result.session.entries[i].text[:80] for i in mask]
                cluster_roles = [result.session.entries[i].role for i in mask]
                hover = [f"[{r}] {t}..." for r, t in zip(cluster_roles, cluster_texts)]
                color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
                label = (
                    result.cluster_labels[c]
                    if c < len(result.cluster_labels) else f"Cluster {c}"
                )
                fig.add_trace(go.Scatter3d(
                    x=cluster_pts[:, 0], y=cluster_pts[:, 1], z=cluster_pts[:, 2],
                    mode="markers",
                    marker=dict(size=6, color=color, opacity=0.8),
                    name=label, text=hover, hoverinfo="text",
                ))

        # ── Catmull-Rom spline path ─────────────────────────
        if "show_path" in display_toggles and len(pts) >= 2:
            spline_pts = catmull_rom_spline(pts, samples_per_segment=20)
            fig.add_trace(go.Scatter3d(
                x=spline_pts[:, 0], y=spline_pts[:, 1], z=spline_pts[:, 2],
                mode="lines",
                line=dict(color="rgba(255,255,255,0.5)", width=4),
                name="Path", showlegend=True, hoverinfo="skip",
            ))

        # ── Flow particles ──────────────────────────────────
        if ("flow_animation" in flow_toggles and flow_data
                and flow_data.get("particle_x")):
            fig.add_trace(go.Scatter3d(
                x=flow_data["particle_x"],
                y=flow_data["particle_y"],
                z=flow_data["particle_z"],
                mode="markers",
                marker=dict(size=2, color=flow_data["particle_colors"], opacity=1.0),
                name="Flow particles", showlegend=False, hoverinfo="skip",
            ))

        # ── Ball marker ─────────────────────────────────────
        if flow_data and flow_data.get("ball_position"):
            bp = flow_data["ball_position"]
            fig.add_trace(go.Scatter3d(
                x=[bp[0]], y=[bp[1]], z=[bp[2]],
                mode="markers",
                marker=dict(size=12, color="rgb(255,100,50)", opacity=0.9,
                            line=dict(width=1, color="white")),
                name="Ball", showlegend=False,
            ))

        # ── Ball trail ──────────────────────────────────────
        if flow_data and flow_data.get("ball_trail") and len(flow_data["ball_trail"]) >= 2:
            trail = np.array(flow_data["ball_trail"])
            fig.add_trace(go.Scatter3d(
                x=trail[:, 0], y=trail[:, 1], z=trail[:, 2],
                mode="lines",
                line=dict(color="yellow", width=3),
                name="Trail", showlegend=False, hoverinfo="skip",
            ))

        # ── Probe marker (from sliders, when not in ball flow) ──
        if "ball_flow" not in flow_toggles:
            px = _slider_to_raw(sx, 0)
            py = _slider_to_raw(sy, 1)
            pz = _slider_to_raw(sz, 2)
            fig.add_trace(go.Scatter3d(
                x=[px], y=[py], z=[pz],
                mode="markers",
                marker=dict(size=10, color="yellow", symbol="diamond",
                            line=dict(width=1, color="white")),
                name="Probe", showlegend=False,
            ))

        # ── Control points ──────────────────────────────────
        if control_points and len(control_points) > 0:
            cp = np.array(control_points)
            fig.add_trace(go.Scatter3d(
                x=cp[:, 0], y=cp[:, 1], z=cp[:, 2],
                mode="markers",
                marker=dict(size=8, color="white", symbol="cross",
                            line=dict(width=1, color=_ACCENT)),
                name="Marked", showlegend=False,
            ))
            # Connecting line between control points
            if len(cp) >= 2:
                spline = catmull_rom_spline(cp, samples_per_segment=10)
                fig.add_trace(go.Scatter3d(
                    x=spline[:, 0], y=spline[:, 1], z=spline[:, 2],
                    mode="lines",
                    line=dict(color=_ACCENT, width=3, dash="dash"),
                    name="Marked path", showlegend=False, hoverinfo="skip",
                ))

        # ── Apply dark theme ────────────────────────────────
        _apply_dark_theme(fig, axis_labels)
        fig.update_layout(
            showlegend=True,
            margin=dict(l=0, r=0, t=10, b=0),
            uirevision="constant",  # Preserve camera angle across updates
        )

        return fig

    # ── Callback 4: Slider sync (bidirectional) + info panel ─────
    @app.callback(
        Output("info-panel", "children"),
        Output("slider-x", "value"),
        Output("slider-y", "value"),
        Output("slider-z", "value"),
        Output("slider-x-label", "children"),
        Output("slider-y-label", "children"),
        Output("slider-z-label", "children"),
        Output("control-points-store", "data"),
        Input("slider-x", "value"),
        Input("slider-y", "value"),
        Input("slider-z", "value"),
        Input("scatter3d", "clickData"),
        Input("btn-explain", "n_clicks"),
        Input("btn-mark", "n_clicks"),
        Input("btn-clear", "n_clicks"),
        Input("flow-state", "data"),
        State("flow-toggles", "value"),
        State("control-points-store", "data"),
        prevent_initial_call=True,
    )
    def update_info_and_sliders(sx, sy, sz, click_data,
                                explain_clicks, mark_clicks, clear_clicks,
                                flow_data, flow_toggles, control_points):
        ctx = callback_context
        if not ctx.triggered:
            return (no_update,) * 8

        triggered = ctx.triggered[0]["prop_id"]
        flow_toggles = flow_toggles or []
        control_points = control_points or []
        axis_labels = result.axis_info.labels

        # Default outputs (no update)
        info_text = no_update
        out_sx, out_sy, out_sz = no_update, no_update, no_update
        lbl_x = no_update
        lbl_y = no_update
        lbl_z = no_update
        out_cp = no_update

        # ── Ball flow state → update sliders to track ball ────
        if "flow-state" in triggered and "ball_flow" in flow_toggles:
            if flow_data and flow_data.get("ball_position"):
                bp = flow_data["ball_position"]
                out_sx = _raw_to_slider(bp[0], 0)
                out_sy = _raw_to_slider(bp[1], 1)
                out_sz = _raw_to_slider(bp[2], 2)
                lbl_x = f"{axis_labels[0]} {round(out_sx / 10)}%"
                lbl_y = f"{axis_labels[1]} {round(out_sy / 10)}%"
                lbl_z = f"{axis_labels[2]} {round(out_sz / 10)}%"

                # Compute info for ball position
                info_text = _build_probe_info(bp[0], bp[1], bp[2])

            return info_text, out_sx, out_sy, out_sz, lbl_x, lbl_y, lbl_z, out_cp

        # ── Click on a scatter point ──────────────────────────
        if "scatter3d.clickData" in triggered and click_data:
            point = click_data["points"][0]
            px_c, py_c, pz_c = point["x"], point["y"], point["z"]

            # Find nearest entry
            dists = np.linalg.norm(
                result.projected_3d - np.array([px_c, py_c, pz_c]), axis=1
            )
            idx = int(np.argmin(dists))
            entry = result.session.entries[idx]
            cluster_id = result.clusters.labels[idx]
            cluster_label = (
                result.cluster_labels[cluster_id]
                if cluster_id < len(result.cluster_labels)
                else f"Cluster {cluster_id}"
            )

            info_text = (
                f"Message #{entry.step_index} [{entry.role}]\n"
                f"Cluster: {cluster_label}\n\n"
                f"{entry.text}"
            )

            # Jump sliders to clicked point
            out_sx = _raw_to_slider(px_c, 0)
            out_sy = _raw_to_slider(py_c, 1)
            out_sz = _raw_to_slider(pz_c, 2)
            lbl_x = f"{axis_labels[0]} {round(out_sx / 10)}%"
            lbl_y = f"{axis_labels[1]} {round(out_sy / 10)}%"
            lbl_z = f"{axis_labels[2]} {round(out_sz / 10)}%"

            # Set ball position if flow system exists
            if _flow_system is not None:
                _flow_system.set_ball_position(px_c, py_c, pz_c)

            return info_text, out_sx, out_sy, out_sz, lbl_x, lbl_y, lbl_z, out_cp

        # ── Slider dragged (manual probe) ─────────────────────
        if any(s in triggered for s in ["slider-x", "slider-y", "slider-z"]):
            px = _slider_to_raw(sx, 0)
            py = _slider_to_raw(sy, 1)
            pz = _slider_to_raw(sz, 2)

            # Update ball position if not flowing
            if "ball_flow" not in flow_toggles and _flow_system is not None:
                _flow_system.set_ball_position(px, py, pz)

            info_text = _build_probe_info(px, py, pz)
            lbl_x = f"{axis_labels[0]} {round(sx / 10)}%"
            lbl_y = f"{axis_labels[1]} {round(sy / 10)}%"
            lbl_z = f"{axis_labels[2]} {round(sz / 10)}%"

            return info_text, no_update, no_update, no_update, lbl_x, lbl_y, lbl_z, out_cp

        # ── Mark Point button ─────────────────────────────────
        if "btn-mark" in triggered:
            px = _slider_to_raw(sx, 0)
            py = _slider_to_raw(sy, 1)
            pz = _slider_to_raw(sz, 2)
            control_points.append([float(px), float(py), float(pz)])
            out_cp = control_points
            info_text = f"Marked point #{len(control_points)} at ({px:.2f}, {py:.2f}, {pz:.2f})"
            return info_text, no_update, no_update, no_update, no_update, no_update, no_update, out_cp

        # ── Clear Points button ───────────────────────────────
        if "btn-clear" in triggered:
            out_cp = []
            info_text = "Control points cleared."
            return info_text, no_update, no_update, no_update, no_update, no_update, no_update, out_cp

        # ── Explain button ────────────────────────────────────
        if "btn-explain" in triggered:
            px = _slider_to_raw(sx, 0)
            py = _slider_to_raw(sy, 1)
            pz = _slider_to_raw(sz, 2)

            if _explainer and len(control_points) > 0:
                # Multi-point explanation
                try:
                    info_text = _build_multi_explain(control_points)
                except Exception as e:
                    info_text = f"Explain error: {e}"
            elif _explainer:
                # Single-point explanation
                try:
                    from tracescope.visualization.probe import probe_with_explanation
                    probe_info = probe_with_explanation(result, _explainer, px, py, pz)
                    info_text = (
                        f"LLM Explanation:\n{probe_info['explanation']}\n\n"
                        f"Nearest messages:\n"
                    )
                    for item in probe_info["nearest_texts"][:3]:
                        info_text += f"  [{item['role']}] {item['text'][:100]}...\n"
                except Exception as e:
                    info_text = f"Explain error: {e}"
            else:
                info_text = "No explainer configured. Pass explainer= to launch_dashboard()."

            return info_text, no_update, no_update, no_update, no_update, no_update, no_update, out_cp

        return (no_update,) * 8


def _build_probe_info(x: float, y: float, z: float) -> str:
    """Build probe info string with axis %, cluster distances, nearest texts."""
    probe_info = probe_point(_result, x, y, z)

    lines = ["Probe Position:"]
    for axis_name, pct in probe_info["axis_percentages"].items():
        lines.append(f"  {axis_name}: {pct:.1f}%")
    lines.append("\nCluster Distances:")
    for name, pct in probe_info["cluster_distances"].items():
        lines.append(f"  {name}: {pct:.1f}% closeness")
    lines.append("\nNearest Messages:")
    for item in probe_info["nearest_texts"][:3]:
        lines.append(f"  [{item['role']}] {item['text'][:80]}...")
    return "\n".join(lines)


def _build_multi_explain(control_points: list) -> str:
    """Build multi-point path explanation via LLM."""
    lines = ["Control Points Path:\n"]
    for i, cp in enumerate(control_points):
        probe_info = probe_point(_result, cp[0], cp[1], cp[2])
        pcts = probe_info["axis_percentages"]
        dists = probe_info["cluster_distances"]
        pct_str = ", ".join(f"{k}: {v:.0f}%" for k, v in pcts.items())
        dist_str = ", ".join(f"{k}: {v:.0f}%" for k, v in dists.items())
        lines.append(f"  Point {i}: {pct_str}")
        lines.append(f"    Clusters: {dist_str}")

    if _explainer:
        try:
            # Use multi-point explanation
            axis_labels = list(probe_point(_result, *control_points[0])["axis_percentages"].keys())
            all_pcts = []
            all_dists = []
            for cp in control_points:
                pi = probe_point(_result, cp[0], cp[1], cp[2])
                all_pcts.append(list(pi["axis_percentages"].values()))
                all_dists.append(pi["cluster_distances"])

            control_points_for_llm = []
            for pcts_vals, dists_dict in zip(all_pcts, all_dists):
                control_points_for_llm.append({
                    "axis_pcts": [int(v) for v in pcts_vals],
                    "cluster_distances": [(k, int(v)) for k, v in dists_dict.items()],
                })
            explanation = _explainer.explain_probe_multi(
                axis_labels=axis_labels,
                control_points=control_points_for_llm,
            )
            lines.append(f"\nLLM Analysis:\n{explanation}")
        except Exception as e:
            lines.append(f"\nExplain error: {e}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════

def launch_dashboard(
    result: AnalysisResult,
    port: int = 8050,
    debug: bool = False,
    explainer=None,
) -> None:
    """Launch the interactive Dash dashboard.

    Args:
        result: AnalysisResult from the pipeline.
        port: Port to serve on.
        debug: Enable Dash debug mode.
        explainer: Optional SemanticExplainer for probe explanations.
    """
    global _flow_system, _result, _explainer
    _result = result
    _explainer = explainer

    # Initialize FlowFieldSystem if velocity grid is available
    if result.velocity_grid is not None:
        logger.info("Initializing FlowFieldSystem with %d³ velocity grid",
                     result.velocity_grid.shape[0])
        _flow_system = FlowFieldSystem(
            result.velocity_grid,
            result.axis_min,
            result.axis_max,
            particle_grid=20,  # 8000 particles for web performance
        )
        # Set ball to center
        center = (result.axis_min + result.axis_max) / 2
        _flow_system.set_ball_position(*center)
        logger.info("FlowFieldSystem ready: %d particles", _flow_system.particle_count)
    else:
        _flow_system = None
        logger.info("No velocity grid available — flow animation disabled")

    app = Dash(__name__)
    app.layout = _build_layout(result)
    _register_callbacks(app, result)

    print(f"\n{'='*60}")
    print(f"  TraceScope Dashboard")
    print(f"  http://localhost:{port}")
    print(f"  Entries: {len(result.session)}")
    print(f"  Clusters: {result.clusters.n_clusters}")
    print(f"  Flow: {'enabled' if _flow_system else 'disabled'}")
    print(f"{'='*60}\n")

    app.run(port=port, debug=debug)
