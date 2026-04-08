"""
3D scatter plot with Catmull-Rom spline paths — faithful port from Android.

Features ported from My3DScatterRenderer.java:
  - Catmull-Rom spline interpolation (20 samples/segment, exact b0-b3 coefficients)
  - Android cluster color palette (10 colors)
  - Dark background (#1E1E1E matching Android's glClearColor 0.12, 0.12, 0.12)
  - Proper point sizing
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

# Plotly is only needed for the plotting functions, not for catmull_rom_spline.
# Defer import so gl_renderer can import catmull_rom_spline without plotly.
go = None  # lazy-loaded

def _require_plotly():
    global go
    if go is not None:
        return
    try:
        import plotly.graph_objects as _go
        go = _go
    except ImportError as _e:
        raise ImportError(
            "Visualization dependencies not installed. "
            "Install them with: pip install plotly\n"
            "Or install the full package: pip install tracescope"
        ) from _e

from tracescope.models.analysis import AnalysisResult


# ═══════════════════════════════════════════════════
# Android cluster palette (exact RGB from DashboardFragment.java line 2254)
# ═══════════════════════════════════════════════════
CLUSTER_COLORS = [
    "rgb(255, 0, 0)",       # red
    "rgb(0, 255, 0)",       # green
    "rgb(0, 0, 255)",       # blue
    "rgb(255, 255, 0)",     # yellow
    "rgb(255, 0, 255)",     # magenta
    "rgb(0, 255, 255)",     # cyan
    "rgb(255, 128, 0)",     # orange
    "rgb(128, 0, 255)",     # purple
    "rgb(0, 128, 255)",     # sky blue
    "rgb(128, 255, 128)",   # pastel green
]

# Hex versions for use where needed
CLUSTER_COLORS_HEX = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF",
    "#00FFFF", "#FF8000", "#8000FF", "#0080FF", "#80FF80",
]


def catmull_rom_spline(
    points: np.ndarray,
    samples_per_segment: int = 20,
) -> np.ndarray:
    """Catmull-Rom spline interpolation.

    Exact coefficients from Android My3DScatterRenderer.java lines 414-456:
        b0 = -0.5*t³ +     t² - 0.5*t
        b1 =  1.5*t³ - 2.5*t² + 1
        b2 = -1.5*t³ + 2.0*t² + 0.5*t
        b3 =  0.5*t³ - 0.5*t²

    Mirrors first/last control points for boundary handling.

    Args:
        points: (N, 3) array of control points.
        samples_per_segment: Number of interpolated samples per segment (default 20).

    Returns:
        (M, 3) array of interpolated points.
    """
    N = len(points)
    if N < 2:
        return points.copy()
    if N == 2:
        # Simple linear interpolation
        result = []
        for j in range(samples_per_segment + 1):
            t = j / samples_per_segment
            result.append(points[0] * (1 - t) + points[1] * t)
        return np.array(result)

    # Mirror endpoints for boundary handling
    # p_extended = [mirror_start, p0, p1, ..., pN-1, mirror_end]
    mirror_start = 2 * points[0] - points[1]
    mirror_end = 2 * points[-1] - points[-2]
    extended = np.vstack([mirror_start, points, mirror_end])

    result = []
    # Iterate over segments: for each group of 4 consecutive points (p0, p1, p2, p3)
    # the spline passes through p1 → p2
    for seg in range(len(extended) - 3):
        p0 = extended[seg]
        p1 = extended[seg + 1]
        p2 = extended[seg + 2]
        p3 = extended[seg + 3]

        for j in range(samples_per_segment):
            t = j / samples_per_segment
            t2 = t * t
            t3 = t2 * t

            # Catmull-Rom basis functions (exact Android coefficients)
            b0 = -0.5 * t3 + t2 - 0.5 * t
            b1 = 1.5 * t3 - 2.5 * t2 + 1.0
            b2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
            b3 = 0.5 * t3 - 0.5 * t2

            point = b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3
            result.append(point)

    # Add the last point
    result.append(extended[-2])  # Last original point

    return np.array(result)


def _apply_dark_theme(fig, axis_labels: List[str]) -> None:
    """Apply dark theme matching Android's glClearColor(0.12, 0.12, 0.12)."""
    bg_color = "rgb(30, 30, 30)"
    grid_color = "rgb(60, 60, 60)"

    axis_common = dict(
        backgroundcolor=bg_color,
        gridcolor=grid_color,
        color="white",
        showbackground=True,
        zerolinecolor=grid_color,
    )

    fig.update_layout(
        scene=dict(
            bgcolor=bg_color,
            xaxis=dict(title=axis_labels[0] if len(axis_labels) > 0 else "X", **axis_common),
            yaxis=dict(title=axis_labels[1] if len(axis_labels) > 1 else "Y", **axis_common),
            zaxis=dict(title=axis_labels[2] if len(axis_labels) > 2 else "Z", **axis_common),
        ),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color="white"),
        legend=dict(font=dict(color="white")),
    )


def _score_to_plotly_colors(values: List[Optional[float]], fallback_color: str = "rgb(128,128,128)") -> List[str]:
    """Convert score values to plotly color strings using red-yellow-green gradient."""
    from tracescope.visualization.flow_field import score_colormap

    colors = []
    for v in values:
        if v is None:
            colors.append(fallback_color)
        else:
            rgb = score_colormap(np.array([v]))[0]
            colors.append(f"rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})")
    return colors


def plot_clusters_3d(
    result: AnalysisResult,
    show_path: bool = True,
    marker_size: int = 6,
    path_width: int = 5,
    use_spline: bool = True,
    color_by_score: Optional[str] = None,
):
    """Create a 3D scatter plot with clusters colored and Catmull-Rom spline path.

    Args:
        result: AnalysisResult from the pipeline.
        show_path: Whether to draw the conversation path.
        marker_size: Size of scatter points.
        path_width: Width of the path line.
        use_spline: Use Catmull-Rom spline (True) or straight lines (False).
        color_by_score: Optional score channel name to color points by score
                        instead of cluster. Uses red-yellow-green gradient.

    Returns:
        Plotly Figure object.
    """
    _require_plotly()
    fig = go.Figure()
    pts = result.projected_3d
    labels = result.clusters.labels
    axis_labels = result.axis_info.labels
    n_clusters = result.clusters.n_clusters

    if color_by_score and color_by_score in result.score_channels:
        # ── Score-based coloring ──────────────────────────
        entry_scores = result.get_entry_scores(color_by_score)
        path_score_map = result.get_path_scores(color_by_score)

        # For each entry: use entry score if present, else path score, else None
        final_scores = []
        for i, e in enumerate(result.session.entries):
            s = entry_scores[i]
            if s is None and e.path_id is not None:
                s = path_score_map.get(e.path_id)
            final_scores.append(s)

        point_colors = _score_to_plotly_colors(final_scores)
        hover = [
            f"[{result.session.entries[i].role}] {result.session.entries[i].text[:100]}... "
            f"({color_by_score}: {final_scores[i]:.2f})" if final_scores[i] is not None
            else f"[{result.session.entries[i].role}] {result.session.entries[i].text[:100]}..."
            for i in range(len(pts))
        ]

        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                marker=dict(size=marker_size, color=point_colors, opacity=0.8),
                name=f"Score: {color_by_score}",
                text=hover,
                hoverinfo="text",
            )
        )
    else:
        # ── Cluster-based coloring (default) ──────────────
        for c in range(n_clusters):
            mask = [i for i, l in enumerate(labels) if l == c]
            if not mask:
                continue
            cluster_pts = pts[mask]
            cluster_texts = [result.session.entries[i].text[:100] for i in mask]
            cluster_roles = [result.session.entries[i].role for i in mask]

            hover = [
                f"[{role}] {text}..."
                for role, text in zip(cluster_roles, cluster_texts)
            ]

            color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
            label = (
                result.cluster_labels[c]
                if c < len(result.cluster_labels)
                else f"Cluster {c}"
            )

            fig.add_trace(
                go.Scatter3d(
                    x=cluster_pts[:, 0],
                    y=cluster_pts[:, 1],
                    z=cluster_pts[:, 2],
                    mode="markers",
                    marker=dict(size=marker_size, color=color, opacity=0.8),
                    name=label,
                    text=hover,
                    hoverinfo="text",
                )
            )

    # Path line (Catmull-Rom spline or straight)
    if show_path and len(pts) >= 2:
        if use_spline:
            spline_pts = catmull_rom_spline(pts, samples_per_segment=20)
        else:
            spline_pts = pts

        # Color path by score if available
        if color_by_score and color_by_score in result.score_channels:
            path_score_map = result.get_path_scores(color_by_score)
            entry_scores = result.get_entry_scores(color_by_score)
            # Build per-entry score, falling back to path score
            per_entry = []
            for i, e in enumerate(result.session.entries):
                s = entry_scores[i]
                if s is None and e.path_id is not None:
                    s = path_score_map.get(e.path_id)
                per_entry.append(s if s is not None else 0.5)
            path_colors = _score_to_plotly_colors(per_entry)
            # For spline: repeat colors for interpolated points
            if use_spline and len(per_entry) >= 2:
                n_seg = len(per_entry) - 1
                expanded = []
                for seg_i in range(n_seg):
                    for j in range(20):
                        t = j / 20.0
                        v = per_entry[seg_i] * (1 - t) + per_entry[seg_i + 1] * t
                        expanded.append(v)
                expanded.append(per_entry[-1])
                # Trim to match spline length
                expanded = expanded[:len(spline_pts)]
                while len(expanded) < len(spline_pts):
                    expanded.append(per_entry[-1])
                path_colors = _score_to_plotly_colors(expanded)

            fig.add_trace(
                go.Scatter3d(
                    x=spline_pts[:, 0], y=spline_pts[:, 1], z=spline_pts[:, 2],
                    mode="lines",
                    line=dict(color=path_colors, width=path_width),
                    name="Path",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=spline_pts[:, 0],
                    y=spline_pts[:, 1],
                    z=spline_pts[:, 2],
                    mode="lines",
                    line=dict(color="rgba(255,255,255,0.7)", width=path_width),
                    name="Path",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

    # Apply dark theme
    _apply_dark_theme(fig, axis_labels)

    fig.update_layout(
        title="TraceScope: Conversation in Semantic Space",
        showlegend=True,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


def plot_multi_paths(
    results: List[AnalysisResult],
    labels: Optional[List[str]] = None,
):
    """Overlay multiple conversation paths in the same 3D space."""
    _require_plotly()
    fig = go.Figure()

    for i, result in enumerate(results):
        pts = result.projected_3d
        label = labels[i] if labels and i < len(labels) else result.session.label
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]

        # Spline path
        if len(pts) >= 2:
            spline_pts = catmull_rom_spline(pts, samples_per_segment=20)
        else:
            spline_pts = pts

        fig.add_trace(
            go.Scatter3d(
                x=spline_pts[:, 0],
                y=spline_pts[:, 1],
                z=spline_pts[:, 2],
                mode="lines",
                line=dict(color=color, width=4),
                name=label,
            )
        )

        # Add scatter points
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(size=4, color=color, opacity=0.7),
                name=f"{label} (points)",
                showlegend=False,
            )
        )

    axis_labels = results[0].axis_info.labels if results else ["X", "Y", "Z"]
    _apply_dark_theme(fig, axis_labels)

    fig.update_layout(
        title="TraceScope: Multiple Conversation Paths",
        showlegend=True,
    )

    return fig
