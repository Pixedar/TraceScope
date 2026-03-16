"""
GPU-accelerated 3D renderer with interactive controls — faithful port of
Android's My3DScatterRenderer.java + FlowFieldSystem.java + DashboardFragment.

GPU rendering (vispy + OpenGL):
  - Flow halo particles: circular discard, pow(1.6) alpha falloff, heat bloom
  - Cluster points: vibrant boost (*1.3 + 0.1), centre-lit spherical gradient
  - Turbo speed colormap: exact degree-5 polynomial coefficients
  - 64,000 particles (40^3), auto-rotation 7.5 deg/s, dark background

Interactive controls (Qt panel):
  - Flow / Ball / Auto-Rotate toggle checkboxes
  - Show Points / Show Path toggles
  - Bidirectional X/Y/Z probe sliders (track ball OR set position manually)
  - Mark Point / Clear buttons with Catmull-Rom connecting spline
  - Cluster legend with counts
  - Info panel: axis %, cluster distances, nearest message texts

Keyboard shortcuts (when 3D canvas has focus):
  Space  flow   B  ball   P  points   L  path   A  auto-rotate
  +/-    particle size   R  reset camera   Esc  quit

Requires:  pip install vispy PyOpenGL PyQt5   (or PySide6)
"""

from __future__ import annotations

import threading
import numpy as np

try:
    from vispy import app, scene, gloo
    from vispy.visuals import Visual
    from vispy.scene.visuals import create_visual_node
except ImportError:
    raise ImportError(
        "vispy is required for the GPU renderer.\n"
        "Install with:  pip install vispy PyOpenGL PyQt5\n"
        "  (or replace PyQt5 with PySide6 / pyglet / glfw)"
    )


def _ensure_viable_backend():
    """Try to use the best available vispy backend, falling back to software
    rendering if GPU/OpenGL is not available.  Called once before building the
    canvas so that vispy is configured for the current environment.
    """
    # If a backend is already set and working, leave it alone.
    try:
        current = app.use_app()
        if current is not None:
            return
    except Exception:
        pass

    # Preferred order: PyQt5 (GPU), PySide6 (GPU), pyglet, osmesa (CPU-only)
    for backend in ("pyqt5", "pyside6", "pyglet", "osmesa"):
        try:
            app.use_app(backend)
            return
        except Exception:
            continue

    # Last resort — let vispy pick whatever it can find
    try:
        app.use_app()
    except Exception:
        pass

# Qt imports — try PyQt5 first (most common), then PySide6
try:
    from PyQt5 import QtWidgets, QtCore
except ImportError:
    try:
        from PySide6 import QtWidgets, QtCore
    except ImportError:
        QtWidgets = None  # type: ignore[assignment]

from .flow_field import FlowFieldSystem
from .scatter3d import catmull_rom_spline
from .probe import probe_point, probe_with_explanation


# ═══════════════════════════════════════════════════════════════════
#  GLSL shaders — exact port of Android My3DScatterRenderer.java
# ═══════════════════════════════════════════════════════════════════

FLOW_HALO_VERT = """
attribute vec3 a_position;
attribute vec4 a_color;
varying   vec4 v_color;
uniform   float u_base_size;

void main() {
    vec4 clip   = $transform(vec4(a_position, 1.0));
    gl_Position = clip;
    float size  = u_base_size / max(0.001, clip.w);
    gl_PointSize = clamp(size, 2.0, 64.0);
    v_color = a_color;
}
"""

FLOW_HALO_FRAG = """
varying vec4 v_color;

void main() {
    vec2  d  = gl_PointCoord - vec2(0.5);
    float r2 = dot(d, d);
    if (r2 > 0.25) discard;
    float alpha = pow(clamp((0.5 - sqrt(r2)) * 2.0, 0.0, 1.0), 1.6);
    vec3 rgb = mix(v_color.rgb * 0.65, v_color.rgb, alpha);
    gl_FragColor = vec4(rgb, alpha * v_color.a);
}
"""

CLUSTER_POINT_VERT = """
attribute vec3 a_position;
attribute vec4 a_color;
varying   vec4 v_color;
uniform   float u_point_size;

void main() {
    vec4 clip   = $transform(vec4(a_position, 1.0));
    gl_Position = clip;
    float dist  = length(clip.xyz);
    gl_PointSize = u_point_size + (1.0 / dist);
    v_color = a_color;
}
"""

CLUSTER_POINT_FRAG = """
varying vec4 v_color;

void main() {
    vec2  coord = gl_PointCoord - vec2(0.5);
    float dist  = length(coord);
    if (dist > 0.5) discard;
    float lightFactor = 0.8 + 0.2 * (1.0 - smoothstep(0.0, 0.5, dist));
    vec3  vibrant = clamp(v_color.rgb * 1.3 + vec3(0.1), vec3(0.0), vec3(1.0));
    gl_FragColor = vec4(vibrant * lightFactor, v_color.a);
}
"""


# ═══════════════════════════════════════════════════════════════════
#  Color palettes (matching Android exactly)
# ═══════════════════════════════════════════════════════════════════

CLUSTER_COLORS_GL = np.array([
    [1.0, 0.0, 0.0, 1.0],       # red
    [0.0, 1.0, 0.0, 1.0],       # green
    [0.0, 0.0, 1.0, 1.0],       # blue
    [1.0, 1.0, 0.0, 1.0],       # yellow
    [1.0, 0.0, 1.0, 1.0],       # magenta
    [0.0, 1.0, 1.0, 1.0],       # cyan
    [1.0, 0.5, 0.0, 1.0],       # orange
    [0.5, 0.0, 1.0, 1.0],       # purple
    [0.0, 0.5, 1.0, 1.0],       # sky blue
    [0.5, 1.0, 0.5, 1.0],       # pastel green
], dtype=np.float32)

AXIS_COLORS = [
    np.array([112, 73, 78, 255], dtype=np.float32) / 255,
    np.array([73, 112, 78, 255], dtype=np.float32) / 255,
    np.array([73, 78, 112, 255], dtype=np.float32) / 255,
]


# ═══════════════════════════════════════════════════════════════════
#  Custom vispy Visuals (unchanged — exact Android shaders)
# ═══════════════════════════════════════════════════════════════════

class _FlowHaloVisual(Visual):
    def __init__(self, base_size=40.0):
        Visual.__init__(self, vcode=FLOW_HALO_VERT, fcode=FLOW_HALO_FRAG)
        self._draw_mode = 'points'
        self._n = 0
        self.set_gl_state(depth_test=True, cull_face=False, blend=True,
                          blend_func=('src_alpha', 'one_minus_src_alpha'))
        self.shared_program['u_base_size'] = float(base_size)
        self.shared_program['a_position'] = gloo.VertexBuffer(np.zeros((1, 3), dtype=np.float32))
        self.shared_program['a_color'] = gloo.VertexBuffer(np.zeros((1, 4), dtype=np.float32))

    def set_data(self, positions, colors):
        self._n = len(positions)
        self.shared_program['a_position'].set_data(np.ascontiguousarray(positions, dtype=np.float32))
        self.shared_program['a_color'].set_data(np.ascontiguousarray(colors, dtype=np.float32))
        self.update()

    def set_base_size(self, size):
        self.shared_program['u_base_size'] = float(size)
        self.update()

    def _prepare_transforms(self, view):
        view.view_program.vert['transform'] = view.transforms.get_transform()

    def _prepare_draw(self, view):
        if self._n < 1:
            return False
        gloo.set_state(depth_mask=False)
        return True

    def draw(self, *args, **kwargs):
        super().draw(*args, **kwargs)
        gloo.set_state(depth_mask=True)


class _ClusterPointVisual(Visual):
    def __init__(self, point_size=7.0):
        Visual.__init__(self, vcode=CLUSTER_POINT_VERT, fcode=CLUSTER_POINT_FRAG)
        self._draw_mode = 'points'
        self._n = 0
        self.set_gl_state(depth_test=True, cull_face=False, blend=True,
                          blend_func=('src_alpha', 'one_minus_src_alpha'))
        self.shared_program['u_point_size'] = float(point_size)
        self.shared_program['a_position'] = gloo.VertexBuffer(np.zeros((1, 3), dtype=np.float32))
        self.shared_program['a_color'] = gloo.VertexBuffer(np.zeros((1, 4), dtype=np.float32))

    def set_data(self, positions, colors):
        self._n = len(positions)
        self.shared_program['a_position'].set_data(np.ascontiguousarray(positions, dtype=np.float32))
        self.shared_program['a_color'].set_data(np.ascontiguousarray(colors, dtype=np.float32))
        self.update()

    def set_point_size(self, size):
        self.shared_program['u_point_size'] = float(size)
        self.update()

    def _prepare_transforms(self, view):
        view.view_program.vert['transform'] = view.transforms.get_transform()

    def _prepare_draw(self, view):
        return self._n > 0


FlowHalo = create_visual_node(_FlowHaloVisual)
ClusterPoint = create_visual_node(_ClusterPointVisual)


# ═══════════════════════════════════════════════════════════════════
#  Qt dark theme
# ═══════════════════════════════════════════════════════════════════

DARK_QSS = """
QMainWindow, QWidget#central { background: #1E1E1E; }
QGroupBox {
    background: #2A2A2A; border: 1px solid #3A3A3A; border-radius: 6px;
    margin-top: 4px; padding: 24px 8px 8px 8px; font-size: 12px;
}
QGroupBox::title {
    subcontrol-origin: padding;
    subcontrol-position: top center;
    padding: 4px 8px;
    color: white; font-weight: bold;
}
QCheckBox { color: #E0E0E0; spacing: 6px; font-size: 12px; }
QCheckBox::indicator { width: 14px; height: 14px; }
QLabel { color: #E0E0E0; font-size: 12px; }
QSlider::groove:horizontal {
    height: 4px; background: #3A3A3A; border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 14px; height: 14px; margin: -5px 0;
    background: #FF6B35; border-radius: 7px;
}
QPushButton {
    background: #FF6B35; color: white; border: none;
    border-radius: 4px; padding: 5px 14px; font-size: 12px;
}
QPushButton:hover { background: #FF8555; }
QPushButton#btn-secondary { background: #555; }
QPushButton#btn-secondary:hover { background: #777; }
QTextEdit {
    background: #1E1E1E; color: #E0E0E0; border: 1px solid #3A3A3A;
    font-size: 11px; font-family: Consolas, monospace;
}
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical {
    width: 4px; background: transparent; margin: 0; padding: 0;
}
QScrollBar::handle:vertical {
    background: rgba(255, 255, 255, 30); border-radius: 2px; min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: transparent; height: 0px;
}
"""


# ═══════════════════════════════════════════════════════════════════
#  FlowRenderer — GPU rendering + interactive Qt controls
# ═══════════════════════════════════════════════════════════════════

class _SliderEventFilter(QtCore.QObject if QtWidgets is not None else object):
    """Detects mouse press/release on sliders to temporarily pause ball sync."""

    def __init__(self, renderer):
        if QtWidgets is not None:
            super().__init__()
        self._renderer = renderer

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.MouseButtonPress:
            self._renderer._manual_slider_override = True
        elif event.type() == QtCore.QEvent.MouseButtonRelease:
            self._renderer._manual_slider_override = False
        return False


class FlowRenderer:
    """GPU-accelerated 3D flow renderer with interactive controls panel."""

    AUTO_ROTATE_SPEED = 7.5
    BALL_RADIUS = 0.07
    BALL_COLOR = (1.0, 0.2, 0.2, 1.0)
    TRAIL_COLOR = (1.0, 1.0, 0.0, 1.0)
    PATH_COLOR = (1.0, 0.0, 0.0, 1.0)
    PROBE_COLOR = np.array([[1.0, 1.0, 0.0, 1.0]], dtype=np.float32)

    def __init__(self, result, particle_grid=40, base_size=40.0,
                 window_size=(1400, 800), title='TraceScope Flow Renderer',
                 explainer=None):
        self.result = result
        self.base_size = base_size
        self._title = title
        self._explainer = explainer

        # ── State ─────────────────────────────────────
        self._has_flow = result.velocity_grid is not None
        self.flow_active = self._has_flow
        self.show_points = True
        self.show_path = True
        self.ball_flowing = self._has_flow
        self.auto_rotate = False
        self._last_time = None
        self._frame_count = 0
        self._updating_sliders = False
        self._manual_slider_override = False
        self._control_points: list = []
        self._particle_grid = particle_grid
        self._spline_path = False  # default: straight lines; True = Catmull-Rom splines
        self._explain_pending = None  # async explain result (set by bg thread, consumed by timer)

        # ── Cluster centroids + colors for particle coloring ──
        n_cls = result.clusters.n_clusters
        self._cluster_centroids = (
            result.cluster_centroids_3d.astype(np.float32)
            if result.cluster_centroids_3d is not None
            else np.zeros((n_cls, 3), dtype=np.float32)
        )
        self._cluster_colors_arr = np.array([
            CLUSTER_COLORS_GL[c % len(CLUSTER_COLORS_GL)][:3]
            for c in range(n_cls)
        ], dtype=np.float32)  # (K, 3)

        # ── Bounds ────────────────────────────────────
        pts = result.projected_3d
        self._data_center = pts.mean(axis=0).astype(np.float32)
        self._axis_min = (np.asarray(result.axis_min, dtype=np.float32)
                          if result.axis_min is not None
                          else pts.min(axis=0).astype(np.float32) - 0.1)
        self._axis_max = (np.asarray(result.axis_max, dtype=np.float32)
                          if result.axis_max is not None
                          else pts.max(axis=0).astype(np.float32) + 0.1)

        # ── Path sample points for flow blob (computed before flow init) ──
        self._path_sample_pts = None
        if len(result.projected_3d) >= 2:
            self._path_sample_pts = catmull_rom_spline(
                result.projected_3d, 20
            ).astype(np.float32)

        # ── Flow system ──────────────────────────────
        self.flow: FlowFieldSystem | None = None
        if self._has_flow:
            self.flow = FlowFieldSystem(
                result.velocity_grid, self._axis_min, self._axis_max,
                particle_grid=particle_grid,
                path_points=self._path_sample_pts,
            )

        # Pre-allocate flow RGBA buffer for performance
        if self.flow is not None:
            self._flow_rgba = np.empty((self.flow.particle_count, 4), dtype=np.float32)
        else:
            self._flow_rgba = None

        # ── Canvas + Scene ────────────────────────────
        self.canvas = scene.SceneCanvas(
            keys='interactive', bgcolor=(0.12, 0.12, 0.12, 1.0),
            size=window_size, show=False, title=title,
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(
            fov=60, distance=5.0, center=tuple(self._data_center),
            up='+y', azimuth=0, elevation=20,
        )

        # ── Build 3D visuals ──────────────────────────
        self._build_visuals()

        # ── Build Qt window with controls ─────────────
        self.window = None
        if QtWidgets is not None:
            self._build_qt_window(window_size)

        # ── Keyboard (fallback when canvas has focus) ─
        self.canvas.events.key_press.connect(self._on_key)

        # ── Mouse double-click to select nearest data point ─
        self.canvas.events.mouse_double_click.connect(self._on_double_click)

        # ── Timer ─────────────────────────────────────
        self._timer = app.Timer(interval=1.0 / 60, connect=self._on_timer)

    # ═══════════════════════════════════════════════════
    #  3D Visual construction
    # ═══════════════════════════════════════════════════

    def _build_visuals(self):
        # Cluster points
        self.vis_clusters = ClusterPoint(point_size=7.0, parent=self.view.scene)
        self.vis_clusters.order = 0
        self._setup_cluster_data()
        self.vis_clusters.visible = self.show_points

        # Flow halo particles
        self.vis_flow = FlowHalo(base_size=self.base_size, parent=self.view.scene)
        self.vis_flow.order = 1
        self.vis_flow.visible = self.flow_active

        # Spline path — both Tube and Line visuals (toggle with Simple Lines)
        spline_pts, spline_colors = self._compute_path()
        self._spline_pts = spline_pts  # store for blob computation
        # Base alpha for path
        self._path_base_alpha = 0.88
        spline_colors[:, 3] = self._path_base_alpha

        # Tube path (no directional lighting — flat vertex colors)
        tube_points = 10
        vertex_colors = np.repeat(spline_colors, tube_points, axis=0)
        try:
            self.vis_path_tube = scene.visuals.Tube(
                points=spline_pts, radius=0.015,
                tube_points=tube_points, shading=None,
                vertex_colors=vertex_colors,
                parent=self.view.scene,
            )
        except Exception:
            self.vis_path_tube = scene.visuals.Line(
                pos=spline_pts, color=spline_colors, width=4,
                parent=self.view.scene, antialias=True,
            )
        self.vis_path_tube.order = 2
        self.vis_path_tube.visible = self.show_path and self._spline_path

        # Simple line path — straight lines between actual data points
        straight_pts, straight_colors = self._compute_straight_path()
        straight_colors[:, 3] = self._path_base_alpha
        self.vis_path_line = scene.visuals.Line(
            pos=straight_pts, color=straight_colors, width=3,
            parent=self.view.scene, antialias=True,
        )
        self.vis_path_line.order = 2
        self.vis_path_line.visible = self.show_path and not self._spline_path

        # Probe marker (yellow)
        self.vis_probe = ClusterPoint(point_size=15.0, parent=self.view.scene)
        self.vis_probe.order = 3
        self.vis_probe.set_data(
            self._data_center.reshape(1, 3),
            self.PROBE_COLOR,
        )

        # Control-point markers (white) + connecting spline
        self.vis_markers = ClusterPoint(point_size=10.0, parent=self.view.scene)
        self.vis_markers.order = 3
        self.vis_markers.visible = False
        self.vis_marked_path = scene.visuals.Line(
            pos=np.zeros((2, 3), dtype=np.float32),
            color=(1.0, 0.5, 0.2, 0.8), width=3,
            parent=self.view.scene, antialias=True,
        )
        self.vis_marked_path.order = 3
        self.vis_marked_path.visible = False

        # Ball sphere
        self.vis_ball = scene.visuals.Sphere(
            radius=self.BALL_RADIUS, cols=24, rows=24,
            color=self.BALL_COLOR, parent=self.view.scene, method='latitude',
        )
        self.vis_ball.order = 4
        self.vis_ball.visible = False

        # Ball trail
        self.vis_trail = scene.visuals.Line(
            pos=np.zeros((2, 3), dtype=np.float32),
            color=self.TRAIL_COLOR, width=4,
            parent=self.view.scene, antialias=True,
        )
        self.vis_trail.order = 4
        self.vis_trail.visible = False

        # Debug: blob boundary visualization (toggle with D key)
        # Shows a point cloud of the blob-occupied cells as semi-transparent markers
        self._debug_blob = False
        self.vis_blob_debug = ClusterPoint(point_size=4.0, parent=self.view.scene)
        self.vis_blob_debug.order = 6
        self.vis_blob_debug.visible = False
        if self.flow is not None:
            blob_pts = self.flow.get_blob_surface_points()
            if blob_pts is not None:
                blob_colors = np.ones((len(blob_pts), 4), dtype=np.float32)
                blob_colors[:, :3] = [0.2, 0.8, 0.4]  # green
                blob_colors[:, 3] = 0.15  # very transparent
                self.vis_blob_debug.set_data(blob_pts, blob_colors)

        # Axes + labels
        self._setup_axes()
        self._setup_axis_labels()

    def _setup_cluster_data(self):
        pts = np.ascontiguousarray(self.result.projected_3d, dtype=np.float32)
        colors = np.zeros((len(pts), 4), dtype=np.float32)
        for i, label in enumerate(self.result.clusters.labels):
            colors[i] = CLUSTER_COLORS_GL[label % len(CLUSTER_COLORS_GL)]
        self.vis_clusters.set_data(pts, colors)

    def _compute_path(self):
        """Compute Catmull-Rom spline with per-vertex cluster-interpolated colors."""
        pts = self.result.projected_3d
        if len(pts) < 2:
            return np.zeros((2, 3), dtype=np.float32), np.array([[1, 0, 0, 1]] * 2, dtype=np.float32)

        spline = catmull_rom_spline(pts, 20).astype(np.float32)

        # Assign cluster colors to each spline point by nearest data point
        # For each data point, get its cluster color
        data_colors = np.zeros((len(pts), 4), dtype=np.float32)
        for i, label in enumerate(self.result.clusters.labels):
            data_colors[i] = CLUSTER_COLORS_GL[label % len(CLUSTER_COLORS_GL)]

        # For each spline point, interpolate color from nearest data points
        dists = np.linalg.norm(spline[:, None, :] - pts[None, :, :], axis=2)  # (S, N)
        nearest_idx = np.argmin(dists, axis=1)
        spline_colors = data_colors[nearest_idx]

        # Smooth colors with a simple moving average (window=5) for nice transitions
        kernel = 5
        if len(spline_colors) > kernel:
            pad = kernel // 2
            padded = np.pad(spline_colors, ((pad, pad), (0, 0)), mode='edge')
            smoothed = np.zeros_like(spline_colors)
            for j in range(len(spline_colors)):
                smoothed[j] = padded[j:j + kernel].mean(axis=0)
            smoothed[:, 3] = 1.0  # keep alpha = 1
            spline_colors = smoothed

        return spline, spline_colors

    def _compute_straight_path(self):
        """Straight lines between actual data points with cluster colors."""
        pts = self.result.projected_3d
        if len(pts) < 2:
            return np.zeros((2, 3), dtype=np.float32), np.array([[1, 0, 0, 1]] * 2, dtype=np.float32)

        pts = pts.astype(np.float32)
        colors = np.zeros((len(pts), 4), dtype=np.float32)
        for i, label in enumerate(self.result.clusters.labels):
            colors[i] = CLUSTER_COLORS_GL[label % len(CLUSTER_COLORS_GL)]
        return pts, colors

    def _setup_axes(self):
        mn, mx = self._axis_min, self._axis_max
        self.vis_axes = []
        for i in range(3):
            start, end = mn.copy(), mn.copy()
            end[i] = mx[i]
            line = scene.visuals.Line(
                pos=np.array([start, end], dtype=np.float32),
                color=AXIS_COLORS[i], width=3,
                parent=self.view.scene, antialias=True,
            )
            line.order = 5
            self.vis_axes.append(line)

    # ── Axis label config (adjust here) ────────────────────
    # AXIS_LABEL_SIZE: font size for 3D axis labels (default 36)
    # Located at gl_renderer.py ~line 590 for easy manual tuning.
    AXIS_LABEL_SIZE = 39

    def _setup_axis_labels(self):
        mn, mx = self._axis_min, self._axis_max
        labels = getattr(self.result.axis_info, 'labels', ['Axis 1', 'Axis 2', 'Axis 3'])
        axis_names = ['X', 'Y', 'Z']
        self.vis_labels = []
        offset = (mx - mn) * 0.03
        for i in range(3):
            pos = mn.copy()
            pos[i] = mx[i] + offset[i]
            label_text = labels[i] if i < len(labels) else f'Axis {i+1}'
            t = scene.visuals.Text(
                text=f'{axis_names[i]}: {label_text}',
                pos=pos.reshape(1, 3).astype(np.float32),
                color=AXIS_COLORS[i][:3], font_size=self.AXIS_LABEL_SIZE,
                parent=self.view.scene, anchor_x='left',
            )
            t.order = 6
            self.vis_labels.append(t)

    # ═══════════════════════════════════════════════════
    #  Qt window + controls panel
    # ═══════════════════════════════════════════════════

    def _build_qt_window(self, window_size):
        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle(self._title)
        self.window.setStyleSheet(DARK_QSS)

        central = QtWidgets.QWidget()
        central.setObjectName('central')
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Left: vispy 3D canvas
        layout.addWidget(self.canvas.native, stretch=3)

        # Right: controls panel
        layout.addWidget(self._build_controls(), stretch=0)

        self.window.setCentralWidget(central)
        self.window.resize(window_size[0], window_size[1])

        # ── Canvas overlays ───────────────────────────
        self._build_canvas_overlays()

    def _build_controls(self):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(290)

        widget = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(widget)
        vbox.setContentsMargins(10, 8, 14, 8)
        vbox.setSpacing(6)

        axis_labels = getattr(self.result.axis_info, 'labels', ['X', 'Y', 'Z'])

        # ── 1. Flow Controls ─────────────────────────────
        grp = QtWidgets.QGroupBox("Flow Controls")
        gl = QtWidgets.QVBoxLayout()

        self.chk_flow = QtWidgets.QCheckBox("Flow Animation")
        self.chk_flow.setChecked(self.flow_active)
        self.chk_flow.setEnabled(self._has_flow)
        self.chk_flow.toggled.connect(self._qt_flow_toggle)
        gl.addWidget(self.chk_flow)

        self.chk_ball = QtWidgets.QCheckBox("Drag probe by the flow")
        self.chk_ball.setChecked(self._has_flow)
        self.chk_ball.setEnabled(self._has_flow)
        self.chk_ball.toggled.connect(self._qt_ball_toggle)
        gl.addWidget(self.chk_ball)

        grp.setLayout(gl)
        vbox.addWidget(grp)

        # ── 2. Display ───────────────────────────────────
        grp = QtWidgets.QGroupBox("Display")
        gl = QtWidgets.QVBoxLayout()

        self.chk_points = QtWidgets.QCheckBox("Show Data Points")
        self.chk_points.setChecked(True)
        self.chk_points.toggled.connect(self._qt_points_toggle)
        gl.addWidget(self.chk_points)

        self.chk_path = QtWidgets.QCheckBox("Show Path")
        self.chk_path.setChecked(True)
        self.chk_path.toggled.connect(self._qt_path_toggle)
        gl.addWidget(self.chk_path)

        self.chk_spline_path = QtWidgets.QCheckBox("Spline Path")
        self.chk_spline_path.setChecked(False)
        self.chk_spline_path.toggled.connect(self._qt_spline_path)
        gl.addWidget(self.chk_spline_path)

        self.chk_info_overlay = QtWidgets.QCheckBox("Show Info Overlay")
        self.chk_info_overlay.setChecked(True)
        self.chk_info_overlay.toggled.connect(self._qt_info_overlay_toggle)
        gl.addWidget(self.chk_info_overlay)

        grp.setLayout(gl)
        vbox.addWidget(grp)

        # ── 3. Probe Sliders ─────────────────────────────
        grp = QtWidgets.QGroupBox("Probe")
        gl = QtWidgets.QVBoxLayout()

        self._slider_labels = []
        self._sliders = []
        for i in range(3):
            lbl = QtWidgets.QLabel(f"{axis_labels[i]}: 50%")
            gl.addWidget(lbl)
            self._slider_labels.append(lbl)

            sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            sl.setRange(0, 1000)
            sl.setValue(500)
            sl.valueChanged.connect(lambda v, idx=i: self._qt_slider_changed(idx, v))
            gl.addWidget(sl)
            self._sliders.append(sl)

        self._slider_filter = _SliderEventFilter(self)
        for sl in self._sliders:
            sl.installEventFilter(self._slider_filter)

        btn_row = QtWidgets.QHBoxLayout()
        btn_mark = QtWidgets.QPushButton("Mark Point")
        btn_mark.clicked.connect(self._qt_mark)
        btn_row.addWidget(btn_mark)

        btn_clear = QtWidgets.QPushButton("Clear")
        btn_clear.setObjectName('btn-secondary')
        btn_clear.clicked.connect(self._qt_clear)
        btn_row.addWidget(btn_clear)
        gl.addLayout(btn_row)

        btn_explain = QtWidgets.QPushButton("Explain")
        btn_explain.clicked.connect(self._qt_explain)
        gl.addWidget(btn_explain)

        grp.setLayout(gl)
        vbox.addWidget(grp)

        # ── 4. Cluster Legend ─────────────────────────────
        grp = QtWidgets.QGroupBox("Clusters")
        gl = QtWidgets.QVBoxLayout()
        for c in range(self.result.clusters.n_clusters):
            col = CLUSTER_COLORS_GL[c % len(CLUSTER_COLORS_GL)]
            hexc = '#%02x%02x%02x' % (int(col[0]*255), int(col[1]*255), int(col[2]*255))
            name = (self.result.cluster_labels[c]
                    if c < len(self.result.cluster_labels) else f"Cluster {c}")
            cnt = sum(1 for l in self.result.clusters.labels if l == c)
            lbl = QtWidgets.QLabel(
                f'<span style="color:{hexc};font-size:14px;">&#9679;</span> '
                f'<b>{name}</b> ({cnt})')
            lbl.setTextFormat(QtCore.Qt.RichText)
            lbl.setWordWrap(True)
            gl.addWidget(lbl)
        grp.setLayout(gl)
        vbox.addWidget(grp)

        # ── 5. Flow Settings ────────────────────────────
        grp = QtWidgets.QGroupBox("Flow Settings")
        gl = QtWidgets.QVBoxLayout()

        self._flow_opacity = 1.0
        lbl_op = QtWidgets.QLabel("Opacity: 100%")
        gl.addWidget(lbl_op)
        sl_opacity = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sl_opacity.setRange(0, 100)
        sl_opacity.setValue(100)
        sl_opacity.valueChanged.connect(lambda v, l=lbl_op: self._qt_flow_opacity(v, l))
        gl.addWidget(sl_opacity)

        lbl_sp = QtWidgets.QLabel("Speed: 1.0x")
        gl.addWidget(lbl_sp)
        sl_speed = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sl_speed.setRange(10, 300)
        sl_speed.setValue(100)
        sl_speed.valueChanged.connect(lambda v, l=lbl_sp: self._qt_flow_speed(v, l))
        gl.addWidget(sl_speed)

        init_count = self._particle_grid ** 3
        lbl_pc = QtWidgets.QLabel(f"Particles: {init_count:,} ({self._particle_grid}\u00b3)")
        gl.addWidget(lbl_pc)
        sl_particles = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sl_particles.setRange(8, 50)
        sl_particles.setValue(self._particle_grid)
        sl_particles.valueChanged.connect(lambda v, l=lbl_pc: self._qt_particle_count(v, l))
        gl.addWidget(sl_particles)

        self._flow_color_mode = "speed"
        self._chk_entropy = QtWidgets.QCheckBox("Entropy Colors")
        self._chk_entropy.setChecked(False)
        self._chk_entropy.toggled.connect(self._qt_entropy_colors)
        gl.addWidget(self._chk_entropy)

        self._chk_cluster_colors = QtWidgets.QCheckBox("Cluster Colors")
        self._chk_cluster_colors.setChecked(False)
        self._chk_cluster_colors.toggled.connect(self._qt_cluster_colors)
        gl.addWidget(self._chk_cluster_colors)

        grp.setLayout(gl)
        grp.setEnabled(self._has_flow)
        vbox.addWidget(grp)

        # ── Explanation Panel (hidden until Explain is pressed) ──
        self._explain_group = QtWidgets.QGroupBox("Explanation")
        gl = QtWidgets.QVBoxLayout()
        self._explain_text = QtWidgets.QTextEdit()
        self._explain_text.setReadOnly(True)
        self._explain_text.setMaximumHeight(300)
        gl.addWidget(self._explain_text)
        btn_close_explain = QtWidgets.QPushButton("Close")
        btn_close_explain.setObjectName('btn-secondary')
        btn_close_explain.clicked.connect(lambda: self._explain_group.setVisible(False))
        gl.addWidget(btn_close_explain)
        self._explain_group.setLayout(gl)
        self._explain_group.setVisible(False)
        vbox.addWidget(self._explain_group)

        self.info_text = None
        vbox.addStretch()
        scroll.setWidget(widget)
        return scroll

    def _build_canvas_overlays(self):
        """Create transparent overlays on the 3D canvas for info + point details."""
        canvas_widget = self.canvas.native

        overlay_style = (
            "background: rgba(20, 20, 20, 180); color: white; "
            "font-family: Consolas, monospace; font-size: 11px; "
            "padding: 8px; border-radius: 4px;"
        )

        # ── Info overlay (bottom-left) ────────────────
        self._info_overlay = QtWidgets.QLabel(canvas_widget)
        self._info_overlay.setStyleSheet(overlay_style)
        self._info_overlay.setWordWrap(True)
        self._info_overlay.setTextFormat(QtCore.Qt.PlainText)
        self._info_overlay.move(10, 10)
        self._info_overlay.setMaximumWidth(360)
        self._info_overlay.setText("")
        self._info_overlay.adjustSize()
        self._info_overlay.show()

        # ── Point info overlay (bottom-center) ────────
        self._point_overlay = QtWidgets.QFrame(canvas_widget)
        self._point_overlay.setStyleSheet(
            "background: rgba(30, 30, 30, 220); border-radius: 8px; padding: 0px;"
        )
        point_layout = QtWidgets.QVBoxLayout(self._point_overlay)
        point_layout.setContentsMargins(10, 8, 10, 8)
        point_layout.setSpacing(4)

        # Header row with close button
        header = QtWidgets.QHBoxLayout()
        self._point_header_label = QtWidgets.QLabel("")
        self._point_header_label.setStyleSheet("color: #FF6B35; font-weight: bold; font-size: 12px;")
        header.addWidget(self._point_header_label)
        header.addStretch()
        btn_x = QtWidgets.QPushButton("X")
        btn_x.setFixedSize(20, 20)
        btn_x.setStyleSheet(
            "background: #555; color: white; border-radius: 10px; "
            "font-size: 10px; padding: 0px;"
        )
        btn_x.clicked.connect(lambda: self._point_overlay.setVisible(False))
        header.addWidget(btn_x)
        point_layout.addLayout(header)

        self._point_text_label = QtWidgets.QLabel("")
        self._point_text_label.setStyleSheet("color: #E0E0E0; font-size: 11px;")
        self._point_text_label.setWordWrap(True)
        self._point_text_label.setMaximumWidth(400)
        point_layout.addWidget(self._point_text_label)

        self._point_overlay.adjustSize()
        self._point_overlay.setVisible(False)

        # ── Explanation overlay (bottom-center, bold, larger) ──
        self._explain_overlay = QtWidgets.QLabel(canvas_widget)
        self._explain_overlay.setStyleSheet(
            "background: rgba(15, 15, 15, 200); color: white; "
            "font-family: Consolas, monospace; font-size: 13px; font-weight: bold; "
            "padding: 12px 16px; border-radius: 6px;"
        )
        self._explain_overlay.setWordWrap(True)
        self._explain_overlay.setTextFormat(QtCore.Qt.PlainText)
        self._explain_overlay.setMaximumWidth(500)
        self._explain_overlay.setText("")
        self._explain_overlay.adjustSize()
        self._explain_overlay.setVisible(False)
        # Click to dismiss
        self._explain_overlay.mousePressEvent = lambda e: self._explain_overlay.setVisible(False)

        # Connect canvas resize to reposition overlays
        self.canvas.events.resize.connect(self._reposition_overlays)

    def _reposition_overlays(self, event=None):
        """Reposition overlays on canvas resize."""
        if not hasattr(self, '_info_overlay'):
            return
        w = self.canvas.native.width()
        h = self.canvas.native.height()

        # Info overlay: bottom-left
        self._info_overlay.move(10, h - self._info_overlay.height() - 10)

        # Point overlay: bottom-center
        pw = self._point_overlay.width()
        self._point_overlay.move((w - pw) // 2, h - self._point_overlay.height() - 10)

        # Explain overlay: bottom-center, above point overlay
        if hasattr(self, '_explain_overlay') and self._explain_overlay.isVisible():
            ew = self._explain_overlay.width()
            self._explain_overlay.move(
                (w - ew) // 2,
                h - self._explain_overlay.height() - 20
            )

    def _sync_path_visibility(self):
        """Update tube/line path visibility and alpha based on current flags."""
        self.vis_path_tube.visible = self.show_path and self._spline_path
        self.vis_path_line.visible = self.show_path and not self._spline_path
        # When flow is active alongside paths, make paths more transparent
        if self.show_path and self.flow_active:
            alpha = self._path_base_alpha - 0.20  # -20% when flow is visible
        else:
            alpha = self._path_base_alpha
        self._update_path_alpha(alpha)

    def _update_path_alpha(self, alpha):
        """Update alpha on whichever path visual is active."""
        # For the simple line visual — update color array alpha
        if hasattr(self, 'vis_path_line') and self.vis_path_line.visible:
            straight_pts, straight_colors = self._compute_straight_path()
            straight_colors[:, 3] = alpha
            self.vis_path_line.set_data(pos=straight_pts, color=straight_colors)
        # Tube doesn't support easy alpha updates (mesh), so we rebuild only
        # if needed — for now, tube alpha is set at build time

    def _qt_info_overlay_toggle(self, checked):
        if hasattr(self, '_info_overlay'):
            self._info_overlay.setVisible(checked)

    def _qt_flow_opacity(self, value, label):
        self._flow_opacity = value / 100.0
        label.setText(f"Opacity: {value}%")

    def _qt_flow_speed(self, value, label):
        speed = value / 100.0
        label.setText(f"Speed: {speed:.1f}x")
        if self.flow is not None:
            self.flow.speed_multiplier = speed

    def _qt_particle_count(self, value, label):
        count = value ** 3
        label.setText(f"Particles: {count:,} ({value}\u00b3)")
        self._particle_grid = value
        if self.flow is not None:
            self.flow.set_particle_grid(value)
            self._flow_rgba = np.empty((self.flow.particle_count, 4), dtype=np.float32)

    def _qt_spline_path(self, checked):
        self._spline_path = checked
        self._sync_path_visibility()

    def _qt_entropy_colors(self, checked):
        if checked:
            self._flow_color_mode = 'entropy'
            self._chk_cluster_colors.setChecked(False)
        elif self._flow_color_mode == 'entropy':
            self._flow_color_mode = 'speed'

    def _qt_cluster_colors(self, checked):
        if checked:
            self._flow_color_mode = 'cluster'
            self._chk_entropy.setChecked(False)
        elif self._flow_color_mode == 'cluster':
            self._flow_color_mode = 'speed'

    # ═══════════════════════════════════════════════════
    #  Qt signal handlers
    # ═══════════════════════════════════════════════════

    def _qt_flow_toggle(self, checked):
        self.flow_active = checked
        self.vis_flow.visible = checked
        self._sync_path_visibility()  # update path transparency

    def _qt_ball_toggle(self, checked):
        self.ball_flowing = checked
        if self.flow is not None:
            if checked:
                self.flow.start_ball_flow()
            else:
                self.flow.stop_ball_flow()
                self.vis_ball.visible = False
                self.vis_trail.visible = False

    def _qt_path_toggle(self, checked):
        self.show_path = checked
        self._sync_path_visibility()

    def _qt_points_toggle(self, checked):
        self.show_points = checked
        self.vis_clusters.visible = checked

    def _qt_slider_changed(self, axis, value):
        if self._updating_sliders:
            return
        labels = getattr(self.result.axis_info, 'labels', ['X', 'Y', 'Z'])
        self._slider_labels[axis].setText(f"{labels[axis]}: {value / 10:.0f}%")

        pos = self._probe_pos_from_sliders()
        self.vis_probe.set_data(pos.reshape(1, 3), self.PROBE_COLOR)

        if self.flow is not None and (not self.ball_flowing or self._manual_slider_override):
            self.flow.set_ball_position(*pos)

        self._update_info(pos)

    def _qt_mark(self):
        pos = self._probe_pos_from_sliders()
        self._control_points.append(pos.copy())
        self._refresh_control_points()

    def _qt_clear(self):
        self._control_points.clear()
        self._refresh_control_points()

    def _qt_explain(self):
        if self._explainer is None:
            self._show_explain("No explainer configured.\nPass explainer= to launch_renderer().")
            return

        pos = self._probe_pos_from_sliders()
        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
        control_points = list(self._control_points)  # snapshot

        self._show_explain("Generating explanation...")
        self._explain_pending = None  # will be set by background thread

        def _run():
            import logging
            logger = logging.getLogger(__name__)
            try:
                if control_points:
                    text = self._build_multi_explain()
                else:
                    info = probe_with_explanation(
                        self.result, self._explainer, px, py, pz)
                    lines = [f"LLM Explanation:\n{info['explanation']}\n"]
                    lines.append("Nearest messages:")
                    for item in info["nearest_texts"][:3]:
                        lines.append(f"  [{item['role']}] {item['text'][:100]}...")
                    text = "\n".join(lines)
                logger.info("Explain completed successfully")
                self._explain_pending = text
            except Exception as e:
                logger.error(f"Explain failed: {e}", exc_info=True)
                self._explain_pending = f"Explain error: {e}"

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def _show_explain(self, text: str):
        """Show text in both the sidebar panel and the canvas overlay."""
        if hasattr(self, '_explain_text'):
            self._explain_text.setPlainText(text)
            self._explain_group.setVisible(True)
        # Also show on canvas overlay (bold, bottom-center)
        if hasattr(self, '_explain_overlay'):
            # Truncate for overlay display (max ~200 chars)
            display = text
            if len(display) > 300:
                display = display[:300] + "..."
            self._explain_overlay.setText(display)
            self._explain_overlay.adjustSize()
            self._explain_overlay.setVisible(True)
            self._reposition_overlays()

    def _build_multi_explain(self) -> str:
        """Build multi-point path explanation via LLM."""
        all_pcts = []
        all_dists = []
        axis_labels = None

        for cp in self._control_points:
            info = probe_point(self.result, cp[0], cp[1], cp[2])
            pcts = info["axis_percentages"]
            dists = info["cluster_distances"]
            if axis_labels is None:
                axis_labels = list(pcts.keys())
            all_pcts.append(list(pcts.values()))
            all_dists.append(dists)

        control_points_for_llm = []
        for pcts_vals, dists_dict in zip(all_pcts, all_dists):
            control_points_for_llm.append({
                "axis_pcts": [int(v) for v in pcts_vals],
                "cluster_distances": [(k, int(v)) for k, v in dists_dict.items()],
            })
        explanation = self._explainer.explain_probe_multi(
            axis_labels=axis_labels,
            control_points=control_points_for_llm,
        )
        return explanation

    # ═══════════════════════════════════════════════════
    #  Helpers
    # ═══════════════════════════════════════════════════

    def _slider_to_raw(self, val, axis):
        pts = self.result.projected_3d
        mn, mx = pts[:, axis].min(), pts[:, axis].max()
        return mn + (val / 1000.0) * (mx - mn)

    def _raw_to_slider(self, raw, axis):
        pts = self.result.projected_3d
        mn, mx = pts[:, axis].min(), pts[:, axis].max()
        span = mx - mn
        if span == 0:
            return 500
        return int(np.clip((raw - mn) / span * 1000, 0, 1000))

    def _probe_pos_from_sliders(self):
        return np.array([
            self._slider_to_raw(self._sliders[i].value(), i) for i in range(3)
        ], dtype=np.float32)

    def _sync_sliders_to_ball(self, bp):
        """Push ball position into sliders without triggering callbacks."""
        self._updating_sliders = True
        labels = getattr(self.result.axis_info, 'labels', ['X', 'Y', 'Z'])
        for i in range(3):
            sv = self._raw_to_slider(bp[i], i)
            self._sliders[i].setValue(sv)
            self._slider_labels[i].setText(f"{labels[i]}: {sv / 10:.0f}%")
        self._updating_sliders = False

    def _update_info(self, pos):
        if not hasattr(self, '_info_overlay'):
            return
        try:
            info = probe_point(self.result, float(pos[0]), float(pos[1]), float(pos[2]))
            lines = []
            # Warn if probe is outside the semantic blob
            if self.flow is not None and self.flow.is_outside_blob(
                    float(pos[0]), float(pos[1]), float(pos[2])):
                lines.append("!! OUTSIDE SEMANTIC REGION !!")
                lines.append("(sparse data — flow may be unreliable)")
                lines.append("")
            lines.append("— Probe Location —")
            for name, pct in info['axis_percentages'].items():
                lines.append(f"  {name}: {pct:.1f}%")
            lines.append("")
            lines.append("— Distance to Clusters —")
            for name, d in info['cluster_distances'].items():
                lines.append(f"  {name}: {d:.1f}%")
            self._info_overlay.setText("\n".join(lines))
            self._info_overlay.adjustSize()
            self._reposition_overlays()
        except Exception:
            pass

    def _refresh_control_points(self):
        if not self._control_points:
            self.vis_markers.visible = False
            self.vis_marked_path.visible = False
            return
        pts = np.array(self._control_points, dtype=np.float32)
        white = np.ones((len(pts), 4), dtype=np.float32)
        self.vis_markers.set_data(pts, white)
        self.vis_markers.visible = True
        if len(pts) >= 2:
            spline = catmull_rom_spline(pts, 10).astype(np.float32)
            self.vis_marked_path.set_data(pos=spline, color=(1.0, 0.5, 0.2, 0.8))
            self.vis_marked_path.visible = True
        else:
            self.vis_marked_path.visible = False

    # ═══════════════════════════════════════════════════
    #  Animation loop
    # ═══════════════════════════════════════════════════

    def _on_timer(self, event):
        now = event.elapsed
        dt = now - self._last_time if self._last_time is not None else 1.0 / 60
        self._last_time = now
        self._frame_count += 1

        # Check for async explain result
        pending = getattr(self, '_explain_pending', None)
        if pending is not None:
            self._explain_pending = None
            self._show_explain(pending)

        # --- Auto-rotation disabled (kept for future use) ---
        # if self.auto_rotate:
        #     cam = self.view.camera
        #     cam.azimuth = (cam.azimuth or 0) + self.AUTO_ROTATE_SPEED * dt

        # Flow step (uses pre-allocated buffer for performance)
        if self.flow_active and self.flow is not None:
            pos, colors, alphas, speeds = self.flow.step()

            # Color mode selection
            color_mode = getattr(self, '_flow_color_mode', 'speed')
            if color_mode == 'entropy':
                from tracescope.visualization.flow_field import diverging_colormap
                max_speed = speeds.max() if speeds.max() > 0 else 1.0
                t = (speeds / max_speed) * 2.0 - 1.0
                colors = diverging_colormap(t).astype(np.float32)
            elif color_mode == 'cluster':
                # Color by inverse-distance-weighted blend of cluster colors
                centroids = self._cluster_centroids  # (K, 3)
                dists = np.linalg.norm(
                    pos[:, None, :] - centroids[None, :, :], axis=2
                )  # (N, K)
                # Inverse distance weights with softening to avoid div-by-zero
                # Power of 3 gives a smooth but visible gradient between clusters
                eps = 1e-6
                inv_dists = 1.0 / (dists + eps) ** 3  # (N, K)
                weights = inv_dists / inv_dists.sum(axis=1, keepdims=True)  # (N, K)
                # Weighted blend of cluster colors
                colors = (weights[:, :, None] * self._cluster_colors_arr[None, :, :]).sum(axis=1)  # (N, 3)

            rgba = self._flow_rgba
            if rgba is None or len(rgba) != len(pos):
                rgba = np.empty((len(pos), 4), dtype=np.float32)
                self._flow_rgba = rgba
            rgba[:, :3] = colors
            rgba[:, 3] = alphas
            opacity = getattr(self, '_flow_opacity', 1.0)
            if opacity < 1.0:
                rgba[:, 3] *= opacity
            self.vis_flow.set_data(pos, rgba)

        # Ball flow
        if self.ball_flowing and self.flow is not None:
            bp = self.flow.advance_ball()
            self.vis_ball.transform = scene.transforms.STTransform(translate=bp.tolist())
            self.vis_ball.visible = True

            trail = self.flow.ball_trail
            if len(trail) >= 2:
                self.vis_trail.set_data(
                    pos=np.array(trail, dtype=np.float32),
                    color=self.TRAIL_COLOR)
                self.vis_trail.visible = True

            # Update probe to follow ball
            self.vis_probe.set_data(bp.reshape(1, 3).astype(np.float32), self.PROBE_COLOR)

            # Sync sliders + info (throttled to every 6 frames, skip when user is dragging)
            if QtWidgets is not None and self._frame_count % 6 == 0 and not self._manual_slider_override:
                self._sync_sliders_to_ball(bp)
                self._update_info(bp)

        self.canvas.update()

    # ═══════════════════════════════════════════════════
    #  Keyboard (fallback when canvas has focus)
    # ═══════════════════════════════════════════════════

    def _on_key(self, event):
        key = event.key
        if key == 'Space' and self._has_flow:
            self.flow_active = not self.flow_active
            self.vis_flow.visible = self.flow_active
            self.vis_clusters.visible = self.show_points
            self._sync_path_visibility()
            if QtWidgets and hasattr(self, 'chk_flow'):
                self.chk_flow.setChecked(self.flow_active)
        elif key == 'B' and self._has_flow and self.flow:
            self.ball_flowing = not self.ball_flowing
            if self.ball_flowing:
                self.flow.start_ball_flow()
            else:
                self.flow.stop_ball_flow()
                self.vis_ball.visible = False
                self.vis_trail.visible = False
            if QtWidgets and hasattr(self, 'chk_ball'):
                self.chk_ball.setChecked(self.ball_flowing)
        elif key == 'P':
            self.show_points = not self.show_points
            self.vis_clusters.visible = self.show_points
            if QtWidgets and hasattr(self, 'chk_points'):
                self.chk_points.setChecked(self.show_points)
        elif key == 'L':
            self.show_path = not self.show_path
            self._sync_path_visibility()
            if QtWidgets and hasattr(self, 'chk_path'):
                self.chk_path.setChecked(self.show_path)
        # --- Auto-rotate key disabled (kept for future use) ---
        # elif key == 'A':
        #     self.auto_rotate = not self.auto_rotate
        #     if QtWidgets and hasattr(self, 'chk_rotate'):
        #         self.chk_rotate.setChecked(self.auto_rotate)
        elif key == 'R':
            self.view.camera.reset()
            self.view.camera.azimuth = 0
            self.view.camera.elevation = 20
            self.view.camera.distance = 5.0
            self.view.camera.center = tuple(self._data_center)
        elif key in ('+', '='):
            self.base_size *= 1.2
            self.vis_flow.set_base_size(self.base_size)
        elif key == '-':
            self.base_size /= 1.2
            self.vis_flow.set_base_size(self.base_size)
        elif key == 'D':
            # Toggle debug blob visualization
            self._debug_blob = not self._debug_blob
            self.vis_blob_debug.visible = self._debug_blob
        elif key == 'Escape':
            self.close()

    def _on_double_click(self, event):
        """Double-click on 3D canvas to jump sliders to nearest data point."""
        if event.button != 1:  # left button only
            return

        # Get the scene transform to project data points to screen coords
        try:
            tr = self.view.scene.transform
            pts = self.result.projected_3d
            # Project all data points to canvas (screen) coordinates
            pts_h = np.hstack([pts, np.ones((len(pts), 1))]).astype(np.float32)
            screen = tr.map(pts_h)[:, :2]

            # Find nearest to click position
            click_pos = np.array(event.pos[:2], dtype=np.float32)
            dists = np.linalg.norm(screen - click_pos, axis=1)
            nearest_idx = int(np.argmin(dists))

            # Only snap if click is reasonably close (within 50 pixels)
            if dists[nearest_idx] > 50:
                return

            # Jump sliders to that point's 3D position
            target = pts[nearest_idx].astype(np.float32)
            self._updating_sliders = True
            labels = getattr(self.result.axis_info, 'labels', ['X', 'Y', 'Z'])
            for i in range(3):
                sv = self._raw_to_slider(float(target[i]), i)
                self._sliders[i].setValue(sv)
                self._slider_labels[i].setText(f"{labels[i]}: {sv / 10:.0f}%")
            self._updating_sliders = False

            # Update probe + info
            self.vis_probe.set_data(target.reshape(1, 3), self.PROBE_COLOR)
            if not self.ball_flowing and self.flow is not None:
                self.flow.set_ball_position(*target)
            self._update_info(target)

            # Show point info overlay
            self._show_point_info(nearest_idx)
        except Exception:
            pass

    def _show_point_info(self, idx: int):
        """Show text description of a selected data point."""
        if not hasattr(self, '_point_overlay'):
            return
        entry = self.result.session.entries[idx]
        cluster_id = self.result.clusters.labels[idx]
        cluster_name = (self.result.cluster_labels[cluster_id]
                        if cluster_id < len(self.result.cluster_labels)
                        else f"Cluster {cluster_id}")

        self._point_header_label.setText(f"Point #{idx}  |  {cluster_name}")
        text = entry.text
        if len(text) > 200:
            text = text[:200] + "..."
        role = entry.role
        self._point_text_label.setText(f"[{role}] {text}")
        self._point_overlay.adjustSize()
        self._point_overlay.setVisible(True)
        self._reposition_overlays()

    # ═══════════════════════════════════════════════════
    #  Public API
    # ═══════════════════════════════════════════════════

    def show(self):
        # Initial flow step
        if self.flow_active and self.flow is not None:
            pos, colors, alphas, speeds = self.flow.step()
            rgba = np.empty((len(pos), 4), dtype=np.float32)
            rgba[:, :3] = colors.astype(np.float32)
            rgba[:, 3] = alphas.astype(np.float32)
            self.vis_flow.set_data(pos.astype(np.float32), rgba)

        # Start ball flow if enabled by default
        if self.ball_flowing and self.flow is not None:
            self.flow.start_ball_flow()

        if self.window is not None:
            self.window.show()
        else:
            self.canvas.show()
        self._timer.start()

    def close(self):
        self._timer.stop()
        if self.window is not None:
            self.window.close()
        else:
            self.canvas.close()

    def run(self):
        self.show()
        app.run()


# ═══════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════

def launch_renderer(result, particle_grid=40, base_size=40.0,
                    window_size=(1400, 800), explainer=None) -> FlowRenderer:
    """Launch the GPU-accelerated 3D flow renderer with interactive controls.

    Opens a native OpenGL window matching Android's My3DScatterRenderer quality,
    with a side panel providing all dashboard features: sliders, toggles,
    cluster legend, probe info, mark/clear buttons, and LLM explain.

    Falls back to software rendering automatically if no GPU is available.

    Args:
        result:        AnalysisResult from pipeline.analyze()
        particle_grid: 40 = 64k particles (Android default), 20 = 8k (lighter)
        base_size:     Particle size for perspective scaling (40.0 for desktop)
        window_size:   Window dimensions (width, height)
        explainer:     SemanticExplainer instance for LLM explanations (optional)
    """
    _ensure_viable_backend()
    renderer = FlowRenderer(result, particle_grid=particle_grid,
                             base_size=base_size, window_size=window_size,
                             explainer=explainer)
    renderer.run()
    return renderer
