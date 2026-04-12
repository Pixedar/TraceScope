"""
GPU-accelerated 3D renderer with interactive controls — faithful port of
Android's My3DScatterRenderer.java + FlowFieldSystem.java + DashboardFragment.

WARNING — Import order matters on Windows:
  Do NOT import this module before running pipeline.analyze() (which uses
  ChromaDB). Loading OpenGL libraries (via vispy/PyOpenGL) before the
  ChromaDB Rust backend can cause an access-violation segfault on Windows.
  Use the lazy launch_renderer() wrapper in tracescope/__init__.py instead
  of importing this module directly.

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
  Space  drag probe by flow   F  flow particles   B  ball
  M      mark probe point     E  explain attractor / marked path
  P      points   L  path     R  reset camera   Esc  quit
  +/-    particle size

Requires:  pip install vispy PyOpenGL PyQt5   (or PySide6)
"""

from __future__ import annotations

import threading
import warnings
import numpy as np

# Silence vispy's isosurface scalar overflow warnings (harmless — they fire
# during marching-cubes shift computation on narrow integer dtypes).
warnings.filterwarnings(
    "ignore",
    message="overflow encountered in scalar",
    category=RuntimeWarning,
    module=r"vispy\.geometry\.isosurface",
)

try:
    from vispy import app, scene, gloo
    from vispy.visuals import Visual
    from vispy.scene.visuals import create_visual_node
    from vispy.visuals.transforms import STTransform
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
        pos = positions if positions.flags['C_CONTIGUOUS'] and positions.dtype == np.float32 \
            else np.ascontiguousarray(positions, dtype=np.float32)
        col = colors if colors.flags['C_CONTIGUOUS'] and colors.dtype == np.float32 \
            else np.ascontiguousarray(colors, dtype=np.float32)
        self.shared_program['a_position'].set_data(pos)
        self.shared_program['a_color'].set_data(col)
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
    def __init__(self, point_size=9.0):
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

def _build_dark_qss(scale: float = 1.0) -> str:
    """Build Qt stylesheet with font sizes scaled for window size."""
    fs_12 = max(9, int(round(12 * scale)))
    fs_11 = max(9, int(round(11 * scale)))
    return f"""
QMainWindow, QWidget#central {{ background: #1E1E1E; }}
QGroupBox {{
    background: #2A2A2A; border: 1px solid #3A3A3A; border-radius: 6px;
    margin-top: 4px; padding: 24px 8px 8px 8px; font-size: {fs_12}px;
}}
QGroupBox::title {{
    subcontrol-origin: padding;
    subcontrol-position: top center;
    padding: 4px 8px;
    color: white; font-weight: bold;
}}
QCheckBox {{ color: #E0E0E0; spacing: 6px; font-size: {fs_12}px; }}
QCheckBox::indicator {{ width: 14px; height: 14px; }}
QLabel {{ color: #E0E0E0; font-size: {fs_12}px; }}
QSlider::groove:horizontal {{
    height: 4px; background: #3A3A3A; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 14px; height: 14px; margin: -5px 0;
    background: #FF6B35; border-radius: 7px;
}}
QPushButton {{
    background: #FF6B35; color: white; border: none;
    border-radius: 4px; padding: 5px 14px; font-size: {fs_12}px;
}}
QPushButton:hover {{ background: #FF8555; }}
QPushButton#btn-secondary {{ background: #555; }}
QPushButton#btn-secondary:hover {{ background: #777; }}
QTextEdit {{
    background: #1E1E1E; color: #E0E0E0; border: 1px solid #3A3A3A;
    font-size: {fs_11}px; font-family: Consolas, monospace;
}}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{
    width: 4px; background: transparent; margin: 0; padding: 0;
}}
QScrollBar::handle:vertical {{
    background: rgba(255, 255, 255, 30); border-radius: 2px; min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent; height: 0px;
}}
"""


DARK_QSS = _build_dark_qss(1.0)


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

    def __init__(self, result, particle_grid=40, base_size=28.0,
                 window_size=(1400, 800), title='TraceScope Flow Renderer',
                 explainer=None, dataset_name=None, initial_state=None):
        self.result = result
        self.base_size = base_size
        self._title = title
        self._explainer = explainer
        self._dataset_name = dataset_name
        self._initial_state = initial_state or {}

        # ── State ─────────────────────────────────────
        self._has_flow = result.velocity_grid is not None
        self.flow_active = self._has_flow
        self.show_points = False
        self.show_path = False
        self.ball_flowing = self._has_flow
        self.auto_rotate = False
        self._last_time = None
        self._frame_count = 0
        self._updating_sliders = False
        self._manual_slider_override = False
        self._control_points: list = []
        self._particle_grid = particle_grid
        self._spline_path = False  # default: straight lines; True = Catmull-Rom splines
        self._explain_pending = None       # async explain result (set by bg thread, consumed by timer)
        self._explain_pending_stage = None  # intermediate progress text
        self._debug_prompts = False         # set True to print all LLM prompts to stdout

        # Cluster colors (positions scaled below)
        n_cls = result.clusters.n_clusters
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

        # ── Scene Normalization ─────────────────────────
        # Apply a SCALE TRANSFORM to the scene so the data always appears
        # at a standard extent (target span magnitude = 6.0).  This way all
        # visual sizes (particles, fonts, camera distance, speed) can be
        # FIXED absolute values and look identical across datasets.
        _raw_span = self._axis_max - self._axis_min
        _raw_mag = float(np.linalg.norm(_raw_span))
        _TARGET_SPAN_MAG = 6.0
        self._scene_scale = _TARGET_SPAN_MAG / _raw_mag if _raw_mag > 0.01 else 1.0
        print(f"[SCALE] raw_span_mag={_raw_mag:.2f}, scene_scale={self._scene_scale:.3f} "
              f"(scene rescaled to standard size)")

        # Cluster centroids (in original data space)
        self._cluster_centroids = (
            result.cluster_centroids_3d.astype(np.float32)
            if result.cluster_centroids_3d is not None
            else np.zeros((n_cls, 3), dtype=np.float32)
        )

        # Use fixed absolute visual params — same for every dataset
        self.base_size = base_size  # Fixed (default 32)

        # ── Path sample points for flow blob (computed before flow init) ──
        # Build per-path splines to avoid spurious cross-path connections.
        # Always include the raw data points as blob seeds so every data
        # point is guaranteed to be inside the blob.
        self._path_sample_pts = None
        if len(result.projected_3d) >= 2:
            path_groups: dict[int | None, list[int]] = {}
            for idx, entry in enumerate(result.session.entries):
                pid = entry.path_id
                path_groups.setdefault(pid, []).append(idx)
            spline_parts = [result.projected_3d.copy()]  # raw data points first
            for pid, indices in path_groups.items():
                if len(indices) < 2:
                    continue
                path_pts = result.projected_3d[indices]
                sps = max(2, 20 // max(1, len(path_groups) // 10))
                spline_parts.append(catmull_rom_spline(path_pts, sps))
            self._path_sample_pts = np.concatenate(spline_parts, axis=0).astype(np.float32)

        # ── Flow system ──────────────────────────────
        self.flow: FlowFieldSystem | None = None
        if self._has_flow:
            self.flow = FlowFieldSystem(
                result.velocity_grid, self._axis_min, self._axis_max,
                particle_grid=particle_grid,
                path_points=self._path_sample_pts,
                confidence_grid=getattr(result, 'confidence_grid', None),
            )
            # Fixed speed — scene is already normalized via transform
            self.flow.speed_multiplier = 0.95

        # Pre-allocate flow RGBA buffer for performance
        if self.flow is not None:
            self._flow_rgba = np.empty((self.flow.particle_count, 4), dtype=np.float32)
        else:
            self._flow_rgba = None

        # ── Canvas + Scene ────────────────────────────
        self.canvas = scene.SceneCanvas(
            keys='interactive', bgcolor=(0.12, 0.12, 0.12, 1.0),
            size=window_size, show=False, title=title,
            vsync=False,  # Windows DWM otherwise throttles paints on
            # small/unfocused windows, making the animation look slow
            # until the window is resized larger.
        )
        self.view = self.canvas.central_widget.add_view()
        # Scaled center so camera sees the normalized scene
        _scaled_center = tuple((self._data_center * self._scene_scale).tolist())
        self.view.camera = scene.TurntableCamera(
            fov=60, distance=9.0,
            center=_scaled_center,
            up='+y', azimuth=0, elevation=20,
        )

        # Render root: intermediate Node with a scale transform so all
        # visuals render at a standard size regardless of raw data extent.
        # All 3D visuals are parented to this node instead of view.scene.
        self._render_root = scene.Node(parent=self.view.scene)
        self._render_root.transform = STTransform(
            scale=(self._scene_scale, self._scene_scale, self._scene_scale)
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

        # ── Ctrl+drag gizmo for probe positioning ─
        self.canvas.events.mouse_press.connect(self._on_click_track_press)
        self.canvas.events.mouse_move.connect(self._on_click_track_move)
        self.canvas.events.mouse_release.connect(self._on_click_track_release)
        self.canvas.events.mouse_press.connect(self._on_gizmo_press)
        self.canvas.events.mouse_press.connect(self._on_attractor_click)
        self.canvas.events.mouse_move.connect(self._on_gizmo_move)
        self.canvas.events.mouse_release.connect(self._on_gizmo_release)
        self.canvas.events.key_press.connect(self._on_gizmo_key)
        self.canvas.events.key_release.connect(self._on_gizmo_key_release)

        # ── Timer ─────────────────────────────────────
        self._timer = app.Timer(interval=1.0 / 60, connect=self._on_timer)
        # Force Qt PreciseTimer so the callback rate isn't throttled when the
        # window is small or unfocused (Windows DWM otherwise coalesces the
        # default CoarseTimer with paint events, slowing the animation until
        # the window is resized/maximized).
        try:
            self._timer._backend.setTimerType(QtCore.Qt.PreciseTimer)
        except Exception:
            pass

    # ═══════════════════════════════════════════════════
    #  3D Visual construction
    # ═══════════════════════════════════════════════════

    def _build_visuals(self):
        # Cluster points
        self.vis_clusters = ClusterPoint(point_size=7.0, parent=self._render_root)
        self.vis_clusters.order = 0
        self._setup_cluster_data()
        self.vis_clusters.visible = self.show_points

        # Flow halo particles
        self.vis_flow = FlowHalo(base_size=self.base_size, parent=self._render_root)
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
                parent=self._render_root,
            )
        except Exception:
            self.vis_path_tube = scene.visuals.Line(
                pos=spline_pts, color=spline_colors, width=4,
                parent=self._render_root, antialias=True,
            )
        self.vis_path_tube.order = 2
        self.vis_path_tube.visible = self.show_path and self._spline_path

        # Simple line path — straight lines between actual data points
        straight_pts, straight_colors = self._compute_straight_path()
        straight_colors[:, 3] = self._path_base_alpha
        self.vis_path_line = scene.visuals.Line(
            pos=straight_pts, color=straight_colors, width=3,
            parent=self._render_root, antialias=True,
        )
        self.vis_path_line.order = 2
        self.vis_path_line.visible = self.show_path and not self._spline_path

        # Probe marker (yellow) — render on top of everything (incl. attractor mesh)
        self.vis_probe = ClusterPoint(point_size=15.0, parent=self._render_root)
        self.vis_probe.order = 10
        self.vis_probe.set_gl_state(
            depth_test=False, cull_face=False, blend=True,
            blend_func=('src_alpha', 'one_minus_src_alpha'),
        )
        self.vis_probe.set_data(
            self._data_center.reshape(1, 3),
            self.PROBE_COLOR,
        )

        # ── Gizmo: 3 axis arrows for Ctrl+drag probe positioning ──
        pts = self.result.projected_3d
        self._gizmo_arrow_len = float(np.max(pts.max(0) - pts.min(0))) * 0.084
        self._gizmo_colors = [
            (1.0, 0.2, 0.2, 0.9),   # X = red
            (0.2, 1.0, 0.2, 0.9),   # Y = green
            (0.3, 0.5, 1.0, 0.9),   # Z = blue
        ]
        self._gizmo_axes = [
            np.array([1, 0, 0], dtype=np.float32),
            np.array([0, 1, 0], dtype=np.float32),
            np.array([0, 0, 1], dtype=np.float32),
        ]
        self._gizmo_lines = []    # shaft lines
        self._gizmo_heads = []    # arrowhead lines
        for i in range(3):
            line = scene.visuals.Line(
                pos=np.zeros((2, 3), dtype=np.float32),
                color=self._gizmo_colors[i], width=3,
                parent=self._render_root, antialias=True,
            )
            line.order = 10
            line.set_gl_state(depth_test=False, blend=True,
                              blend_func=('src_alpha', 'one_minus_src_alpha'))
            line.visible = False
            self._gizmo_lines.append(line)
            # Arrowhead (two short lines forming a V)
            head = scene.visuals.Line(
                pos=np.zeros((3, 3), dtype=np.float32),
                color=self._gizmo_colors[i], width=3,
                connect='segments',
                parent=self._render_root, antialias=True,
            )
            head.order = 10
            head.set_gl_state(depth_test=False, blend=True,
                              blend_func=('src_alpha', 'one_minus_src_alpha'))
            head.visible = False
            self._gizmo_heads.append(head)
        # Gizmo drag state
        self._gizmo_visible = False
        self._gizmo_dragging = None       # axis index (0/1/2) or None
        self._gizmo_drag_start_pos = None
        self._gizmo_drag_start_click = None
        self._gizmo_drag_axis_screen_n = None
        self._gizmo_drag_px_per_unit = 1.0
        self._gizmo_persistent = False
        # Plain-click tracking (for click-near-probe detection)
        self._click_track_pos = None
        self._click_track_moved = False
        # Start stuck so the initial hint invites the user to interact with the probe.
        # Flips to False once the ball has actually been moving for a while.
        self._ball_is_stuck = True
        self._ball_stuck_frames = 0
        self._ball_motion_frames = 0
        self._last_ball_pos = None

        # Control-point markers (white) + connecting spline
        self.vis_markers = ClusterPoint(point_size=10.0, parent=self._render_root)
        self.vis_markers.order = 3
        self.vis_markers.visible = False
        self.vis_marked_path = scene.visuals.Line(
            pos=np.zeros((2, 3), dtype=np.float32),
            color=(1.0, 0.5, 0.2, 0.8), width=3,
            parent=self._render_root, antialias=True,
        )
        self.vis_marked_path.order = 3
        self.vis_marked_path.visible = False

        # Attractor basin point clouds (like blob debug, but colored by score)
        self._attractor_clouds = []   # list of ClusterPoint visuals
        self._attractor_labels = []   # list of Text visuals
        self._attractor_data = []     # cached attractor dicts
        self._attractors_visible = False
        self._highlighted_attractor = None  # index into _attractor_data or None
        self._attractor_highlight_border = None  # ClusterPoint for selected border

        # Ball sphere
        self.vis_ball = scene.visuals.Sphere(
            radius=self.BALL_RADIUS, cols=24, rows=24,
            color=self.BALL_COLOR, parent=self._render_root, method='latitude',
        )
        self.vis_ball.order = 4
        self.vis_ball.visible = False

        # Ball trail
        self.vis_trail = scene.visuals.Line(
            pos=np.zeros((2, 3), dtype=np.float32),
            color=self.TRAIL_COLOR, width=4,
            parent=self._render_root, antialias=True,
        )
        self.vis_trail.order = 4
        self.vis_trail.visible = False

        # Debug: blob boundary visualization (toggle with D key)
        # Shows a point cloud of the blob-occupied cells as semi-transparent markers
        self._debug_blob = False
        self.vis_blob_debug = ClusterPoint(point_size=4.0, parent=self._render_root)
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

        # For each spline point, find nearest data point (KDTree avoids O(S*N) memory)
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        _, nearest_idx = tree.query(spline, k=1)
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
                parent=self._render_root, antialias=True,
            )
            line.order = 5
            self.vis_axes.append(line)

    # ── Axis label config ────────────────────────────────
    # Fixed absolute font sizes — scene is pre-normalized to standard extent.
    AXIS_LABEL_SIZE = 26
    ATTRACTOR_LABEL_SIZE = 15

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
                parent=self._render_root, anchor_x='left',
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

    _BASE_PANEL_WIDTH = 290

    def _build_controls(self):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(self._BASE_PANEL_WIDTH)
        self._controls_scroll = scroll

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

        self._gizmo_hint = QtWidgets.QLabel("Hold Ctrl + drag gizmo arrows to move probe")
        self._gizmo_hint.setStyleSheet("color: #88ccff; font-size: 11px; padding: 2px 0;")
        self._gizmo_hint.setVisible(not self._has_flow)  # show when ball flow is off
        gl.addWidget(self._gizmo_hint)

        grp.setLayout(gl)
        vbox.addWidget(grp)

        # ── 2. Display ───────────────────────────────────
        grp = QtWidgets.QGroupBox("Display")
        gl = QtWidgets.QVBoxLayout()

        self.chk_points = QtWidgets.QCheckBox("Show Data Points")
        self.chk_points.setChecked(False)
        self.chk_points.toggled.connect(self._qt_points_toggle)
        gl.addWidget(self.chk_points)

        self.chk_path = QtWidgets.QCheckBox("Show Path")
        self.chk_path.setChecked(False)
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

        self.chk_attractors = QtWidgets.QCheckBox("Show Attractors")
        self.chk_attractors.setChecked(False)
        self.chk_attractors.toggled.connect(self._qt_attractors_toggle)
        gl.addWidget(self.chk_attractors)

        # Sensitivity slider (hidden until attractors are shown)
        self._attractor_sensitivity = 0.7  # default
        self._sensitivity_row = QtWidgets.QWidget()
        sl_lay = QtWidgets.QHBoxLayout(self._sensitivity_row)
        sl_lay.setContentsMargins(0, 0, 0, 0)
        self._lbl_sensitivity = QtWidgets.QLabel("Sensitivity: 70.0%")
        sl_lay.addWidget(self._lbl_sensitivity)
        self._sl_sensitivity = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._sl_sensitivity.setRange(0, 1000)  # 0.0–100.0% with 0.1 steps
        self._sl_sensitivity.setValue(700)
        self._sl_sensitivity.setTracking(True)  # update label in real-time
        self._sl_sensitivity.valueChanged.connect(self._qt_sensitivity_label_update)
        self._sl_sensitivity.sliderReleased.connect(self._qt_sensitivity_released)
        sl_lay.addWidget(self._sl_sensitivity)
        self._sensitivity_row.setVisible(False)
        gl.addWidget(self._sensitivity_row)

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

        self._flow_opacity = 0.8
        lbl_op = QtWidgets.QLabel("Opacity: 80%")
        gl.addWidget(lbl_op)
        sl_opacity = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sl_opacity.setRange(0, 100)
        sl_opacity.setValue(80)
        sl_opacity.valueChanged.connect(lambda v, l=lbl_op: self._qt_flow_opacity(v, l))
        gl.addWidget(sl_opacity)

        self._lbl_speed = QtWidgets.QLabel("Speed: 0.95x")
        gl.addWidget(self._lbl_speed)
        self._sl_speed = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._sl_speed.setRange(10, 300)
        self._sl_speed.setValue(95)
        self._sl_speed.valueChanged.connect(lambda v, l=self._lbl_speed: self._qt_flow_speed(v, l))
        gl.addWidget(self._sl_speed)

        init_count = self._particle_grid ** 3
        lbl_pc = QtWidgets.QLabel(f"Particles: {init_count:,} ({self._particle_grid}\u00b3)")
        gl.addWidget(lbl_pc)
        sl_particles = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sl_particles.setRange(8, 50)
        sl_particles.setValue(self._particle_grid)
        sl_particles.valueChanged.connect(lambda v, l=lbl_pc: self._qt_particle_count(v, l))
        gl.addWidget(sl_particles)

        # Confidence opacity slider (MDN only)
        has_conf = (self.flow is not None and self.flow._confidence_grid is not None)
        self._confidence_strength = 0.0
        lbl_conf = QtWidgets.QLabel("Confidence fade: OFF")
        gl.addWidget(lbl_conf)
        sl_conf = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sl_conf.setRange(0, 100)
        sl_conf.setValue(0)
        sl_conf.setEnabled(has_conf)
        sl_conf.valueChanged.connect(lambda v, l=lbl_conf: self._qt_confidence(v, l))
        gl.addWidget(sl_conf)
        if not has_conf:
            lbl_conf.setText("Confidence fade: N/A (no density data)")

        # Constrain to path toggle
        self._chk_blob = QtWidgets.QCheckBox("Constrain to paths")
        self._chk_blob.setChecked(True)
        self._chk_blob.setEnabled(self.flow is not None)
        self._chk_blob.toggled.connect(self._qt_blob_toggle)
        gl.addWidget(self._chk_blob)

        self._flow_color_mode = "speed"
        self._chk_entropy = QtWidgets.QCheckBox("Entropy Colors")
        self._chk_entropy.setChecked(False)
        self._chk_entropy.toggled.connect(self._qt_entropy_colors)
        gl.addWidget(self._chk_entropy)

        self._chk_cluster_colors = QtWidgets.QCheckBox("Cluster Colors")
        self._chk_cluster_colors.setChecked(False)
        self._chk_cluster_colors.toggled.connect(self._qt_cluster_colors)
        gl.addWidget(self._chk_cluster_colors)

        # Score-based coloring (only shown when scores exist)
        self._score_color_checkboxes = []
        score_channels = self.result.score_channels
        if score_channels:
            for ch in score_channels:
                chk = QtWidgets.QCheckBox(f"Score: {ch}")
                chk.setChecked(False)
                chk.toggled.connect(lambda checked, c=ch: self._qt_score_colors(c, checked))
                gl.addWidget(chk)
                self._score_color_checkboxes.append((ch, chk))

        grp.setLayout(gl)
        grp.setEnabled(self._has_flow)
        vbox.addWidget(grp)

        # Score-based point coloring (separate from flow)
        if score_channels:
            grp_score = QtWidgets.QGroupBox("Point Coloring")
            gl_score = QtWidgets.QVBoxLayout()
            self._point_color_mode = "cluster"
            self._score_point_checkboxes = []
            for ch in score_channels:
                chk = QtWidgets.QCheckBox(f"Color points by: {ch}")
                chk.setChecked(False)
                chk.toggled.connect(lambda checked, c=ch: self._qt_score_point_coloring(c, checked))
                gl_score.addWidget(chk)
                self._score_point_checkboxes.append((ch, chk))
            grp_score.setLayout(gl_score)
            vbox.addWidget(grp_score)

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

        # Prevent sidebar widgets from stealing keyboard focus (Space, Enter, etc.)
        # so that all key presses go to the vispy canvas.
        for child in scroll.findChildren(QtWidgets.QWidget):
            child.setFocusPolicy(QtCore.Qt.NoFocus)
        scroll.setFocusPolicy(QtCore.Qt.NoFocus)

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

        # ── Hint overlay (center-bottom, orange text) ───────
        self._hint_overlay = QtWidgets.QLabel(canvas_widget)
        self._hint_overlay.setStyleSheet(
            "background: rgba(15, 15, 15, 200); color: #FF8A3D; "
            "font-family: Consolas, monospace; font-size: 12px; font-weight: bold; "
            "padding: 10px 14px; border-radius: 6px;"
        )
        self._hint_overlay.setWordWrap(True)
        self._hint_overlay.setTextFormat(QtCore.Qt.PlainText)
        self._hint_overlay.setMaximumWidth(520)
        self._hint_overlay.setText("")
        self._hint_overlay.setAlignment(QtCore.Qt.AlignCenter)
        self._hint_overlay.adjustSize()
        self._hint_overlay.setVisible(False)

        # ── Dataset name overlay (top-center, optional) ────
        self._name_overlay = QtWidgets.QLabel(canvas_widget)
        self._name_overlay.setStyleSheet(
            "background: transparent; color: rgba(255,255,255,180); "
            "font-family: Consolas, monospace; font-size: 16px; font-weight: bold;"
        )
        self._name_overlay.setAlignment(QtCore.Qt.AlignCenter)
        if self._dataset_name:
            self._name_overlay.setText(self._dataset_name)
            self._name_overlay.adjustSize()
            self._name_overlay.setVisible(True)
        else:
            self._name_overlay.setVisible(False)

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
        self._explain_full_text = ""  # backing text for the fullscreen dialog
        # Click to open fullscreen viewer
        self._explain_overlay.setCursor(QtCore.Qt.PointingHandCursor)
        self._explain_overlay.mousePressEvent = lambda e: self._open_explain_fullscreen()

        # Connect canvas resize to reposition overlays
        self.canvas.events.resize.connect(self._reposition_overlays)

    # Default window dimensions for font scaling reference
    _DEFAULT_WINDOW_W = 1400
    _DEFAULT_WINDOW_H = 800

    def _reposition_overlays(self, event=None):
        """Reposition overlays on canvas resize and scale fonts proportionally.

        Stylesheet updates (expensive) are only applied when the window scale
        actually changes.  Repositioning (cheap) runs every call.
        """
        if not hasattr(self, '_info_overlay'):
            return
        w = self.canvas.native.width()
        h = self.canvas.native.height()

        # Scale fonts relative to default window size (geometric mean of both axes)
        scale = max(0.7, min(1.5, (w * h) ** 0.5 / (self._DEFAULT_WINDOW_W * self._DEFAULT_WINDOW_H) ** 0.5))

        # Only rebuild stylesheets when scale actually changes (setStyleSheet
        # triggers expensive Qt style recalculation — must NOT run every frame).
        if getattr(self, '_last_qss_scale', None) != scale:
            if hasattr(self, 'window') and self.window is not None:
                self.window.setStyleSheet(_build_dark_qss(scale))
                if hasattr(self, '_controls_scroll'):
                    self._controls_scroll.setFixedWidth(
                        int(round(self._BASE_PANEL_WIDTH * scale))
                    )
            info_fs = max(9, int(11 * scale))
            point_hdr_fs = max(10, int(12 * scale))
            point_txt_fs = max(9, int(11 * scale))
            explain_fs = max(10, int(13 * scale))

            self._info_overlay.setStyleSheet(
                f"background: rgba(20, 20, 20, 180); color: white; "
                f"font-family: Consolas, monospace; font-size: {info_fs}px; "
                f"padding: 8px; border-radius: 4px;"
            )
            self._point_header_label.setStyleSheet(
                f"color: #FF6B35; font-weight: bold; font-size: {point_hdr_fs}px;"
            )
            self._point_text_label.setStyleSheet(
                f"color: #E0E0E0; font-size: {point_txt_fs}px;"
            )
            if hasattr(self, '_explain_overlay'):
                self._explain_overlay.setStyleSheet(
                    f"background: rgba(15, 15, 15, 200); color: white; "
                    f"font-family: Consolas, monospace; font-size: {explain_fs}px; font-weight: bold; "
                    f"padding: 12px 16px; border-radius: 6px;"
                )
            if hasattr(self, '_hint_overlay'):
                hint_fs = max(10, int(12 * scale))
                self._hint_overlay.setStyleSheet(
                    f"background: rgba(15, 15, 15, 200); color: #FF8A3D; "
                    f"font-family: Consolas, monospace; font-size: {hint_fs}px; font-weight: bold; "
                    f"padding: 10px 14px; border-radius: 6px;"
                )
            if hasattr(self, '_name_overlay') and self._name_overlay.isVisible():
                name_fs = max(12, int(16 * scale))
                self._name_overlay.setStyleSheet(
                    f"background: transparent; color: rgba(255,255,255,180); "
                    f"font-family: Consolas, monospace; font-size: {name_fs}px; font-weight: bold;"
                )
            self._last_qss_scale = scale

        # Info overlay: bottom-left
        self._info_overlay.adjustSize()
        self._info_overlay.move(10, h - self._info_overlay.height() - 10)

        # Point overlay: bottom-center
        self._point_overlay.adjustSize()
        pw = self._point_overlay.width()
        self._point_overlay.move((w - pw) // 2, h - self._point_overlay.height() - 10)

        # Explain overlay: bottom-center, above point overlay
        if hasattr(self, '_explain_overlay') and self._explain_overlay.isVisible():
            self._explain_overlay.adjustSize()
            ew = self._explain_overlay.width()
            self._explain_overlay.move(
                (w - ew) // 2,
                h - self._explain_overlay.height() - 20
            )

        # Hint overlay: bottom-center, just above point overlay
        if hasattr(self, '_hint_overlay') and self._hint_overlay.isVisible():
            self._hint_overlay.adjustSize()
            hw = self._hint_overlay.width()
            # Stack above point_overlay if that is showing, else sit near bottom
            y_offset = 20
            if self._point_overlay.isVisible():
                y_offset = self._point_overlay.height() + 24
            self._hint_overlay.move(
                (w - hw) // 2,
                h - self._hint_overlay.height() - y_offset
            )

        # Dataset name overlay: top-center
        if hasattr(self, '_name_overlay') and self._name_overlay.isVisible():
            self._name_overlay.adjustSize()
            nw = self._name_overlay.width()
            self._name_overlay.move((w - nw) // 2, 12)

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

    def _qt_confidence(self, value, label):
        self._confidence_strength = value / 100.0
        if self.flow is not None:
            self.flow.confidence_strength = self._confidence_strength
        if value == 0:
            label.setText("Confidence fade: OFF")
        else:
            label.setText(f"Confidence fade: {value}%")

    def _qt_attractors_toggle(self, checked):
        if checked:
            # Cap flow opacity at 50% so attractors stand out
            self._pre_attractor_opacity = self._flow_opacity
            if self._flow_opacity > 0.5:
                self._flow_opacity = 0.5
            self._compute_and_show_attractors()
        else:
            # Restore previous flow opacity
            if hasattr(self, '_pre_attractor_opacity'):
                self._flow_opacity = self._pre_attractor_opacity
            self._hide_attractors()
            self._update_contextual_hint()
        # Show/hide sensitivity slider
        if hasattr(self, '_sensitivity_row'):
            self._sensitivity_row.setVisible(checked)

    def _qt_sensitivity_label_update(self, value):
        """Update label in real-time while dragging (no recompute)."""
        pct = value / 10.0
        self._lbl_sensitivity.setText(f"Sensitivity: {pct:.1f}%")

    def _qt_sensitivity_released(self):
        """Recompute attractors when the user releases the slider."""
        value = self._sl_sensitivity.value()
        self._attractor_sensitivity = value / 1000.0
        if not self.chk_attractors.isChecked():
            return
        pct = value / 10.0
        self._sl_sensitivity.setEnabled(False)
        self._lbl_sensitivity.setText(f"Sensitivity: {pct:.1f}% (recomputing...)")
        QtCore.QTimer.singleShot(50, self._recompute_attractors_async)

    def _recompute_attractors_async(self):
        """Recompute attractors with current sensitivity, then re-enable slider."""
        try:
            self._compute_and_show_attractors()
        finally:
            self._sl_sensitivity.setEnabled(True)
            pct = self._attractor_sensitivity * 100
            self._lbl_sensitivity.setText(f"Sensitivity: {pct:.1f}%")

    def _compute_and_show_attractors(self):
        """Detect attractors and render each basin as a vibrant isosurface mesh.

        Each attractor's boolean basin_mask is upsampled 2× for smoother
        mesh quality, then gaussian-smoothed and rendered as a semi-transparent
        isosurface mesh via vispy.
        """
        from scipy.ndimage import gaussian_filter, zoom

        if self.flow is None:
            return

        score_grid = self.flow._score_grid if self.flow._score_grid is not None else None
        _cache = getattr(self.result, 'cache_path', None)
        _sens = getattr(self, '_attractor_sensitivity', 0.7)
        attractors = self.flow.find_attractors(
            score_grid=score_grid, cache_path=_cache, sensitivity=_sens,
        )
        self._attractor_data = attractors

        self._hide_attractors()

        if not attractors:
            print("[ATTRACTORS] No attractors found in flow field")
            return

        G = self.flow.grid_size
        att_colors = self._get_attractor_colors()

        # Upsample factor for smoother meshes (2× = 80³ from 40³)
        up = 2
        G_up = (G - 1) * up + 1

        # Transform: map UPSAMPLED grid indices → world coordinates
        scale = self.flow.span / (G_up - 1)
        translate = self.flow.axis_min.copy()

        for i, att in enumerate(attractors):
            basin_mask = att['basin_mask']
            if np.sum(basin_mask) == 0:
                continue

            base_rgb = att_colors[i] if i < len(att_colors) else np.array([0.5, 0.8, 1.0])

            # Upsample basin mask for smoother isosurface, then smooth.
            # sigma and iso level are tuned so the mesh tightly hugs the
            # actual basin_mask boundary without ballooning outward.
            volume_hi = zoom(basin_mask.astype(np.float32), up, order=1)
            volume = gaussian_filter(volume_hi, sigma=0.6)

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    iso = scene.visuals.Isosurface(
                        volume, level=0.45,
                        color=(*base_rgb.tolist(), 0.55),
                        shading='smooth',
                        parent=self._render_root,
                    )
                iso.transform = STTransform(
                    scale=scale.tolist(),
                    translate=translate.tolist(),
                )
                iso.set_gl_state(
                    depth_test=True, cull_face=False, blend=True,
                    blend_func=('src_alpha', 'one_minus_src_alpha'),
                )
                iso.order = 1
                self._attractor_clouds.append(iso)
            except Exception as e:
                # Fallback to point cloud if isosurface fails
                print(f"[ATTRACTORS] Isosurface failed for A{i+1}, "
                      f"falling back to points: {e}")
                indices = np.argwhere(basin_mask)
                pts = np.zeros((len(indices), 3), dtype=np.float32)
                for a in range(3):
                    pts[:, a] = (self.flow.axis_min[a]
                                 + indices[:, a] / (G - 1) * self.flow.span[a])
                colors = np.empty((len(pts), 4), dtype=np.float32)
                colors[:, :3] = base_rgb
                colors[:, 3] = 0.30
                cloud = ClusterPoint(point_size=6.0, parent=self._render_root)
                cloud.order = 1
                cloud.set_data(pts, colors)
                self._attractor_clouds.append(cloud)

            # Label — always show basin fraction
            label_text = f"A{i + 1} ({att['basin_fraction'] * 100:.1f}%)"
            center = att['position'].copy()
            center[2] += self.flow.span[2] * 0.06
            label = scene.visuals.Text(
                text=label_text, pos=center,
                color=(1, 1, 1, 0.95),
                font_size=self.ATTRACTOR_LABEL_SIZE,
                bold=True,
                parent=self._render_root,
            )
            label.order = 1
            self._attractor_labels.append(label)

        self._attractors_visible = True
        print(f"[ATTRACTORS] Found {len(attractors)} attractor(s):")
        for i, att in enumerate(attractors):
            score_str = (f", score={att['mean_score']:.2f}"
                         if att['mean_score'] is not None else "")
            print(f"  A{i + 1}: pos=({att['position'][0]:.2f}, "
                  f"{att['position'][1]:.2f}, {att['position'][2]:.2f}), "
                  f"strength={att['strength']:.2f}, "
                  f"basin={att['basin_fraction']:.1%}, "
                  f"div={att['divergence']:.4f}{score_str}")

    def _hide_attractors(self):
        """Remove attractor visuals from scene."""
        for cloud in self._attractor_clouds:
            cloud.parent = None
        for label in self._attractor_labels:
            label.parent = None
        if self._attractor_highlight_border is not None:
            self._attractor_highlight_border.parent = None
            self._attractor_highlight_border = None
        self._attractor_clouds.clear()
        self._attractor_labels.clear()
        self._attractors_visible = False
        self._highlighted_attractor = None

    def _on_attractor_click(self, event):
        """Click to highlight the nearest attractor (when attractors are shown)."""
        if event.button != 1 or not self._attractors_visible:
            return
        if not self._attractor_data:
            return
        # Don't interfere with active gizmo drag
        if self._gizmo_dragging is not None:
            return

        click = np.array([event.pos[0], event.pos[1]], dtype=np.float64)

        # Distance to probe in screen space
        probe_pos = self._probe_pos_from_sliders()
        probe_sc = self._vispy_project(probe_pos)
        probe_dist = float(np.linalg.norm(click - probe_sc))

        # Project each attractor center to 2D and find nearest
        best_idx = None
        best_dist = float('inf')
        for i, att in enumerate(self._attractor_data):
            sc = self._vispy_project(att['position'])
            dist = float(np.linalg.norm(click - sc))
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        # Probe wins ONLY when click is very close to probe (<30px)
        # AND the probe is meaningfully closer than the nearest attractor.
        # This lets users click on attractors even when the probe overlaps.
        if probe_dist < 30 and probe_dist < best_dist * 0.5:
            return

        # Any attractor click dismisses the explain overlay
        if hasattr(self, '_explain_overlay'):
            self._explain_overlay.setVisible(False)

        if best_idx is None or best_dist > 200:
            # Click too far from any attractor — deselect
            self._set_highlighted_attractor(None)
            self._update_contextual_hint()
            return

        self._set_highlighted_attractor(best_idx)

    def _set_highlighted_attractor(self, idx):
        """Highlight attractor with thick white contour lines around the mesh.

        Extracts a wireframe from the isosurface mesh edges and renders
        as thick white lines to clearly indicate selection.
        """
        from scipy.ndimage import gaussian_filter, zoom

        self._highlighted_attractor = idx

        # Remove previous highlight visuals
        if hasattr(self, '_attractor_highlight_border') and self._attractor_highlight_border is not None:
            self._attractor_highlight_border.parent = None
            self._attractor_highlight_border = None

        if idx is not None and idx < len(self._attractor_data):
            att = self._attractor_data[idx]
            basin_mask = att['basin_mask']
            G = self.flow.grid_size

            # Build the same upsampled isosurface as in _compute_and_show_attractors
            up = 2
            G_up = (G - 1) * up + 1
            scale = self.flow.span / (G_up - 1)
            translate = self.flow.axis_min.copy()

            volume_hi = zoom(basin_mask.astype(np.float32), up, order=1)
            volume = gaussian_filter(volume_hi, sigma=1.0)

            # Extract mesh at a slightly higher level for the contour outline
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                try:
                    from skimage.measure import marching_cubes
                    verts, faces, _, _ = marching_cubes(volume, level=0.25)
                except ImportError:
                    from vispy.geometry.isosurface import isosurface as _vispy_iso
                    verts, faces = _vispy_iso(volume, level=0.25)

            if len(verts) > 0 and len(faces) > 0:
                # Transform vertices from grid to world coordinates
                world_verts = verts * scale[np.newaxis, :] + translate[np.newaxis, :]

                # Extract unique edges from faces for wireframe
                edges = set()
                for f in faces:
                    for e in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
                        edges.add((min(e), max(e)))

                # Subsample edges for performance (keep ~2000 max)
                edge_list = list(edges)
                if len(edge_list) > 2000:
                    rng = np.random.default_rng(0)
                    indices = rng.choice(len(edge_list), 2000, replace=False)
                    edge_list = [edge_list[i] for i in indices]

                # Build line segments: pairs of vertices
                line_pts = np.zeros((len(edge_list) * 2, 3), dtype=np.float32)
                for j, (a, b) in enumerate(edge_list):
                    line_pts[j * 2] = world_verts[a]
                    line_pts[j * 2 + 1] = world_verts[b]

                connect = np.zeros(len(edge_list) * 2, dtype=bool)
                for j in range(len(edge_list)):
                    connect[j * 2] = True  # connect first to second in each pair
                    # connect[j*2+1] = False  # break between pairs (already False)

                border_vis = scene.visuals.Line(
                    pos=line_pts,
                    connect=connect,
                    color=(1.0, 1.0, 1.0, 0.85),
                    width=3.0,
                    parent=self._render_root,
                )
                border_vis.order = 0
                self._attractor_highlight_border = border_vis

            self._show_attractor_info(idx)
        else:
            # Clear attractor info — allow probe to update again
            if hasattr(self, '_info_overlay'):
                self._update_info(self._probe_pos_from_sliders())

    def _get_attractor_colors(self):
        """Return list of (R,G,B) arrays, one per attractor, matching the
        current flow color mode palette.  Colors are evenly spaced across
        the active colormap so attractors are always visually distinct even
        when their physical values are similar."""
        from tracescope.visualization.flow_field import turbo_colormap, diverging_colormap, score_colormap

        n = len(self._attractor_data)
        if n == 0:
            return []

        color_mode = getattr(self, '_flow_color_mode', 'speed')

        # Evenly spaced parameter values so each attractor gets a distinct hue
        t_vals = np.linspace(0.1, 0.9, n, dtype=np.float32) if n > 1 else np.array([0.5], dtype=np.float32)

        if color_mode == 'entropy':
            # Diverging colormap expects [-1, 1]
            t_div = t_vals * 2.0 - 1.0
            rgb = diverging_colormap(t_div).astype(np.float32)
        elif color_mode == 'cluster':
            # Assign each attractor the color of its nearest cluster centroid
            colors_list = []
            centroids = self._cluster_centroids
            for att in self._attractor_data:
                dists = np.linalg.norm(centroids - att['position'], axis=1)
                nearest_c = int(np.argmin(dists))
                colors_list.append(self._cluster_colors_arr[nearest_c].copy())
            # If multiple attractors map to same cluster, shift them slightly
            return colors_list
        elif color_mode.startswith('score:'):
            # Score colormap: red→green, use mean_score if available
            score_vals = np.array([
                att['mean_score'] if att['mean_score'] is not None else 0.5
                for att in self._attractor_data
            ], dtype=np.float32)
            # If scores are too similar, spread them for visual distinction
            if n > 1 and (score_vals.max() - score_vals.min()) < 0.15:
                score_vals = t_vals
            rgb = score_colormap(score_vals).astype(np.float32)
        else:
            # Default: turbo colormap (velocity-like), evenly spaced
            rgb = turbo_colormap(t_vals).astype(np.float32)

        return [rgb[i] for i in range(n)]

    def _refresh_attractor_colors(self):
        """Recolor existing attractor isosurfaces to match the current flow color mode.

        Each attractor has ONE isosurface mesh (1:1 mapping with _attractor_data).
        """
        if not self._attractors_visible or not self._attractor_clouds:
            return

        att_colors = self._get_attractor_colors()

        for ai in range(len(self._attractor_data)):
            if ai >= len(self._attractor_clouds):
                break
            base_rgb = att_colors[ai] if ai < len(att_colors) else np.array([0.5, 0.8, 1.0])
            cloud = self._attractor_clouds[ai]

            # Isosurface: update color via mesh_data
            try:
                cloud.color = (*base_rgb.tolist(), 0.45)
            except Exception:
                # Fallback for ClusterPoint (if isosurface failed)
                pass

        # Refresh highlight border if one is active
        if self._highlighted_attractor is not None:
            self._set_highlighted_attractor(self._highlighted_attractor)

    def _show_attractor_info(self, idx):
        """Show detailed info about the highlighted attractor."""
        if not hasattr(self, '_info_overlay'):
            return
        att = self._attractor_data[idx]
        pos = att['position']
        pts = self.result.projected_3d
        axis_labels = getattr(self.result.axis_info, 'labels', ['X', 'Y', 'Z'])

        lines = [f"— Attractor A{idx + 1} —"]

        # Location as axis percentages
        for i in range(3):
            mn, mx = pts[:, i].min(), pts[:, i].max()
            span = mx - mn
            pct = ((pos[i] - mn) / span * 100) if span > 0 else 50
            lines.append(f"  {axis_labels[i]}: {pct:.0f}%")

        # Strength and basin
        lines.append(f"  Strength: {att['strength']:.0%}")
        lines.append(f"  Basin: {att['basin_fraction']:.0%} of field")
        lines.append(f"  Divergence: {att['divergence']:.4f}")

        # Score
        if att['mean_score'] is not None:
            score = att['mean_score']
            interp = "positive" if score > 0.6 else "negative" if score < 0.4 else "neutral"
            lines.append(f"  Score: {score:.2f} ({interp})")

        # Nearest data point — only one, and only if short enough to fit
        _MAX_NEAREST_LEN = 70
        dists = np.linalg.norm(pts - pos, axis=1)
        for ni in np.argsort(dists)[:3]:
            entry = self.result.session.entries[int(ni)]
            if len(entry.text) <= _MAX_NEAREST_LEN:
                lines.append("")
                lines.append("— Nearest Point —")
                lines.append(f"  {entry.text}")
                break

        self._info_overlay.setText("\n".join(lines))
        self._info_overlay.adjustSize()
        self._update_contextual_hint()
        self._reposition_overlays()

    def _simulate_probe_to_attractor(self, start_pos, attractor_pos, n_marks=4,
                                      max_steps=500):
        """Simulate a probe flowing from start_pos toward attractor_pos.

        Returns a list of n_marks evenly-spaced waypoints along the trajectory,
        including the starting point and ending near the attractor.
        """
        if self.flow is None:
            return [start_pos.copy()]

        DT = self.flow.DT
        speed_mult = self.flow.speed_multiplier
        pos = start_pos.copy().astype(np.float32)
        trajectory = [pos.copy()]

        for _ in range(max_steps):
            vel = self.flow.sample_velocity(float(pos[0]), float(pos[1]), float(pos[2]))
            pos += vel * DT * speed_mult
            trajectory.append(pos.copy())
            # Stop if very close to attractor
            if np.linalg.norm(pos - attractor_pos) < 0.3:
                break

        # Sample n_marks evenly spaced waypoints
        trajectory = np.array(trajectory)
        total = len(trajectory)
        if total <= n_marks:
            return list(trajectory)
        indices = np.linspace(0, total - 1, n_marks, dtype=int)
        return [trajectory[i] for i in indices]

    def _find_distant_start_points(self, attractor_pos, basin_mask, n=3):
        """Find n starting positions far from the attractor that will converge to it.

        Strategy: pick points from the data cloud that are far away from the
        attractor center and spread in different angular directions.
        """
        pts = self.result.projected_3d
        center = attractor_pos
        dists = np.linalg.norm(pts - center, axis=1)

        # Get the far-away half of points
        median_dist = np.median(dists)
        far_mask = dists > median_dist
        far_pts = pts[far_mask]

        if len(far_pts) < n:
            # Fallback: use the farthest n points
            far_idx = np.argsort(-dists)[:n]
            return [pts[idx].copy() for idx in far_idx]

        # Spread angular directions: pick points far from each other
        # Start with the farthest point
        selected = []
        far_dists = dists[far_mask]
        candidates = far_pts.copy()
        cand_dists = far_dists.copy()

        # First: farthest from attractor
        best = np.argmax(cand_dists)
        selected.append(candidates[best].copy())

        for _ in range(n - 1):
            # Pick the candidate that maximizes minimum distance to all selected
            min_dists = np.full(len(candidates), np.inf)
            for sel in selected:
                d = np.linalg.norm(candidates - sel, axis=1)
                min_dists = np.minimum(min_dists, d)
            # Also weight by distance from attractor to avoid picking nearby points
            score = min_dists * (cand_dists / cand_dists.max())
            best = np.argmax(score)
            selected.append(candidates[best].copy())

        return selected

    def _explain_highlighted_attractor(self, deep=False):
        """Generate LLM explanation for the currently highlighted attractor.

        Args:
            deep: If True, runs the full multi-stage process:
                1. Find 3 distant starting points spread in different directions
                2. Simulate each probe flowing toward the attractor (3-4 waypoints)
                3. Use path explain logic to get an LLM explanation for each trajectory
                4. Feed all 3 trajectory explanations + attractor context into final prompt
                If False, does only the direct attractor explanation (faster).
        """
        if self._highlighted_attractor is None:
            return
        if self._explainer is None:
            self._show_explain("No explainer configured.\nPass explainer= to launch_renderer().")
            return

        idx = self._highlighted_attractor
        att = self._attractor_data[idx]
        pos = att['position']
        debug = getattr(self, '_debug_prompts', False)

        if deep:
            self._show_explain(
                f"Explaining attractor A{idx + 1}...\n\n"
                "Stage 1/3: Simulating convergence probes..."
            )
        else:
            self._show_explain(f"Explaining attractor A{idx + 1}...")

        # Pre-compute context on main thread (fast)
        pts = self.result.projected_3d
        axis_labels = getattr(self.result.axis_info, 'labels', ['X', 'Y', 'Z'])

        # Axis percentages
        axis_pcts = []
        for i in range(3):
            mn, mx = pts[:, i].min(), pts[:, i].max()
            span = mx - mn
            pct = int((pos[i] - mn) / span * 100) if span > 0 else 50
            axis_pcts.append(pct)

        # Cluster distances
        cluster_distances = []
        centroids = self.result.cluster_centroids_3d
        max_dist = self.result.max_cluster_distance
        if centroids is not None and max_dist > 0:
            for ci in range(len(centroids)):
                d = float(np.linalg.norm(pos - centroids[ci]))
                closeness = max(0, int((1.0 - d / max_dist) * 100))
                label = (self.result.cluster_labels[ci]
                         if ci < len(self.result.cluster_labels)
                         else f"Cluster {ci}")
                cluster_distances.append((label, closeness))

        # Nearest texts
        dists = np.linalg.norm(pts - pos, axis=1)
        nearest_idxs = np.argsort(dists)[:5]
        nearest_texts = [self.result.session.entries[ni].text for ni in nearest_idxs]

        # Score context
        score_info = None
        active_score = getattr(self, '_flow_color_mode', 'speed')
        if active_score.startswith('score:') and att['mean_score'] is not None:
            ch = active_score[len('score:'):]
            raw = att['mean_score']
            interp = "positive" if raw > 0.6 else "negative" if raw < 0.4 else "neutral"
            score_info = {"channel": ch, "mean_score_raw": raw, "interpretation": interp}

        strength = att['strength']
        basin_fraction = att['basin_fraction']
        divergence = att['divergence']

        # Deep mode: find 3 distant starting points + simulate trajectories
        trajectories = []
        if deep:
            start_points = self._find_distant_start_points(pos, att['basin_mask'], n=3)
            for sp in start_points:
                waypoints = self._simulate_probe_to_attractor(sp, pos, n_marks=4)
                trajectories.append(waypoints)

        def _run():
            try:
                trajectory_explanations = None

                if deep:
                    # Stage 2: Explain each convergence trajectory
                    self._explain_pending_stage = "Stage 2/3: Explaining convergence trajectories..."
                    trajectory_explanations = []
                    for ti, waypoints in enumerate(trajectories):
                        self._explain_pending_stage = (
                            f"Stage 2/3: Explaining trajectory {ti + 1}/3..."
                        )
                        # Build control points for path explain
                        cp_for_llm = []
                        for wp in waypoints:
                            info = probe_point(self.result, float(wp[0]), float(wp[1]), float(wp[2]))
                            pcts = info["axis_percentages"]
                            cdists = info["cluster_distances"]
                            cp_for_llm.append({
                                "axis_pcts": [int(v) for v in pcts.values()],
                                "cluster_distances": [(k, int(v)) for k, v in cdists.items()],
                            })
                        traj_text = self._explainer.explain_probe_multi(
                            axis_labels=axis_labels,
                            control_points=cp_for_llm,
                            debug=debug,
                        )
                        trajectory_explanations.append(traj_text)
                        if debug:
                            print(f"[DEBUG] Trajectory {ti+1} explanation: {traj_text}")

                # Final attractor explanation (with or without trajectory context)
                if deep:
                    self._explain_pending_stage = "Stage 3/3: Generating attractor explanation..."
                text = self._explainer.explain_attractor(
                    axis_labels=axis_labels,
                    axis_pcts=axis_pcts,
                    cluster_distances=cluster_distances,
                    nearest_texts=nearest_texts,
                    strength=strength,
                    basin_fraction=basin_fraction,
                    divergence=divergence,
                    score_info=score_info,
                    trajectory_explanations=trajectory_explanations,
                    debug=debug,
                )
                explanation = f"Attractor A{idx + 1} Explanation:\n\n{text}"
                if score_info:
                    explanation += (f"\n\nScore ({score_info['channel']}): "
                                    f"{score_info['mean_score_raw']:.2f} "
                                    f"({score_info['interpretation']})")
                self._explain_pending = explanation
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._explain_pending = f"Explain error: {e}"

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def _qt_blob_toggle(self, checked):
        if self.flow is not None:
            self.flow.blob_enabled = checked
            # Reinit particles when toggling blob
            if self.flow._blob_opacity is not None:
                self.flow.set_particle_grid(self.flow.particle_grid)

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
        self._refresh_attractor_colors()

    def _qt_cluster_colors(self, checked):
        if checked:
            self._flow_color_mode = 'cluster'
            self._chk_entropy.setChecked(False)
            for _, chk in getattr(self, '_score_color_checkboxes', []):
                chk.setChecked(False)
        elif self._flow_color_mode == 'cluster':
            self._flow_color_mode = 'speed'
        self._refresh_attractor_colors()

    def _qt_score_colors(self, channel, checked):
        if checked:
            self._flow_color_mode = f'score:{channel}'
            self._chk_entropy.setChecked(False)
            self._chk_cluster_colors.setChecked(False)
            for ch, chk in getattr(self, '_score_color_checkboxes', []):
                if ch != channel:
                    chk.setChecked(False)
            # Precompute score grid for fast per-frame sampling
            if self.flow is not None:
                self._precompute_score_grid(channel)
        elif self._flow_color_mode == f'score:{channel}':
            self._flow_color_mode = 'speed'
        self._refresh_attractor_colors()

    def _gather_scores_normalized(self, channel: str) -> np.ndarray:
        """Gather per-entry score values and normalize to [0, 1].

        Raw scores can have arbitrary ranges (e.g. reliability 1–5,
        error_rate 0–0.3).  We rescale so the colormap always spans
        the full red→green range across the actual data.
        """
        entry_scores = self.result.get_entry_scores(channel)
        path_score_map = self.result.get_path_scores(channel)
        raw = np.array([
            (entry_scores[i] if entry_scores[i] is not None
             else path_score_map.get(self.result.session.entries[i].path_id)
             if self.result.session.entries[i].path_id is not None else None)
            for i in range(len(self.result.projected_3d))
        ], dtype=object)

        # Convert to float, replacing None with NaN
        vals = np.array([float(v) if v is not None else np.nan for v in raw],
                        dtype=np.float32)

        # Normalize to [0, 1] using the actual data range
        valid = ~np.isnan(vals)
        if valid.any():
            lo = float(np.nanmin(vals))
            hi = float(np.nanmax(vals))
            if hi - lo > 1e-8:
                vals[valid] = (vals[valid] - lo) / (hi - lo)
            else:
                vals[valid] = 0.5  # all same value → neutral
        vals[~valid] = 0.5  # missing → neutral
        return vals

    def _precompute_score_grid(self, channel: str):
        """Build the score grid on the flow system for fast particle coloring."""
        data_pts = self.result.projected_3d
        data_scores = self._gather_scores_normalized(channel)
        self.flow.build_score_grid(data_pts, data_scores)

    def _qt_score_point_coloring(self, channel, checked):
        if checked:
            self._point_color_mode = f'score:{channel}'
            for ch, chk in getattr(self, '_score_point_checkboxes', []):
                if ch != channel:
                    chk.setChecked(False)
            self._apply_score_point_colors(channel)
        else:
            if getattr(self, '_point_color_mode', 'cluster') == f'score:{channel}':
                self._point_color_mode = 'cluster'
                self._setup_cluster_data()

    def _apply_score_point_colors(self, channel):
        from tracescope.visualization.flow_field import score_colormap
        pts = np.ascontiguousarray(self.result.projected_3d, dtype=np.float32)
        normalized = self._gather_scores_normalized(channel)
        rgb = score_colormap(normalized).astype(np.float32)
        colors = np.ones((len(pts), 4), dtype=np.float32)
        colors[:, :3] = rgb
        self.vis_clusters.set_data(pts, colors)

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
                self._auto_mark_start_point()
                self.flow.start_ball_flow()
                self._show_gizmo(False)
            else:
                self.flow.stop_ball_flow()
                self.vis_ball.visible = False
                self.vis_trail.visible = False
        if hasattr(self, '_gizmo_hint'):
            self._gizmo_hint.setVisible(not checked)
        self._update_contextual_hint()

    def _auto_mark_start_point(self):
        """Add the current probe position as a control point (skip if duplicate)."""
        pos = self._probe_pos_from_sliders()
        if self._control_points:
            last = self._control_points[-1]
            if float(np.linalg.norm(pos - last)) < 1e-4:
                return
        self._control_points.append(pos.copy())
        self._refresh_control_points()

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
        if self._gizmo_visible:
            self._update_gizmo(pos)

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

        # Stop ball flow so probe stays put during explanation
        if self.ball_flowing:
            self.ball_flowing = False
            if self.flow is not None:
                self.flow.stop_ball_flow()
                self.vis_ball.visible = False
                self.vis_trail.visible = False
            if hasattr(self, 'chk_ball'):
                self.chk_ball.setChecked(False)
            if hasattr(self, '_gizmo_hint'):
                self._gizmo_hint.setVisible(True)

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

    def _show_hint(self, text: str):
        """Show a centered orange hint on the canvas. Hide if text is empty."""
        if not hasattr(self, '_hint_overlay'):
            return
        if not text:
            self._hint_overlay.setVisible(False)
            return
        self._hint_overlay.setText(text)
        self._hint_overlay.adjustSize()
        self._hint_overlay.setVisible(True)
        self._reposition_overlays()

    def _update_contextual_hint(self):
        """Pick the right hint based on current probe / flow / attractor state."""
        if not hasattr(self, '_hint_overlay'):
            return
        # Attractor highlighted takes priority
        if self._attractors_visible and self._highlighted_attractor is not None:
            self._show_hint(
                "Press E to explain  ·  "
                "Shift+E for deeper explanation (slower)"
            )
            return
        # Probe being dragged by flow
        if self.ball_flowing and self._has_flow:
            if getattr(self, '_ball_is_stuck', False):
                self._show_hint("Click on the probe to move it")
            else:
                self._show_hint("Press M to mark point  ·  Press E to explain path")
            return
        # Gizmo visible → how to drive it + how to resume
        if self._gizmo_visible:
            self._show_hint(
                "Drag the arrows to move the probe  ·  "
                "Press Space or enable 'Drag probe by the flow' to resume"
            )
            return
        # Ball flow off, gizmo not visible → how to start editing
        if self._has_flow and not self.ball_flowing:
            self._show_hint(
                "Click on the probe to move it  ·  "
                "Press Space or enable 'Drag probe by the flow' to resume"
            )
            return
        self._hint_overlay.setVisible(False)

    def _show_explain(self, text: str):
        """Show text in both the sidebar panel and the canvas overlay."""
        if hasattr(self, '_explain_text'):
            self._explain_text.setPlainText(text)
            self._explain_group.setVisible(True)
        # Also show on canvas overlay (bold, bottom-center)
        if hasattr(self, '_explain_overlay'):
            self._explain_full_text = text
            # Truncate for overlay display (max ~300 chars)
            display = text
            if len(display) > 300:
                display = display[:300] + "…  (click to expand)"
            self._explain_overlay.setText(display)
            self._explain_overlay.adjustSize()
            self._explain_overlay.setVisible(True)
            self._reposition_overlays()

    def _open_explain_fullscreen(self):
        """Open the full explanation in a large modal dialog with scroll."""
        if not hasattr(self, '_explain_full_text') or not self._explain_full_text:
            self._explain_overlay.setVisible(False)
            return
        dlg = QtWidgets.QDialog(self.window)
        dlg.setWindowTitle("Explanation")

        # Size ~80% of screen so the dialog is large regardless of window size
        screen_geo = None
        try:
            screen_geo = QtWidgets.QApplication.primaryScreen().availableGeometry()
        except Exception:
            pass
        if screen_geo is not None:
            dlg_w = int(screen_geo.width() * 0.65)
            dlg_h = int(screen_geo.height() * 0.65)
        elif self.window is not None:
            ws = self.window.size()
            dlg_w = int(ws.width() * 0.7)
            dlg_h = int(ws.height() * 0.7)
        else:
            dlg_w, dlg_h = 900, 650
        dlg.resize(dlg_w, dlg_h)

        # Scale font with dialog height so text stays readable at any size
        base_font_px = max(20, min(28, int(dlg_h * 0.038)))
        btn_font_px = max(16, base_font_px - 2)
        dlg.setStyleSheet(
            "QDialog { background: #1E1E1E; } "
            "QLabel { background: transparent; color: #E0E0E0; "
            f"font-family: Consolas, monospace; font-size: {base_font_px}px; "
            "line-height: 1.5; } "
            "QScrollArea { background: #1E1E1E; border: none; } "
            "QPushButton { background: #FF6B35; color: white; border: none; "
            f"border-radius: 4px; padding: 10px 22px; font-size: {btn_font_px}px; "
            "} QPushButton:hover { background: #FF8555; }"
        )

        lay = QtWidgets.QVBoxLayout(dlg)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Scrollable text area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        text_label = QtWidgets.QLabel(self._explain_full_text)
        text_label.setWordWrap(True)
        text_label.setContentsMargins(28, 24, 28, 24)
        text_label.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse | QtCore.Qt.TextSelectableByKeyboard
        )
        scroll.setWidget(text_label)
        lay.addWidget(scroll, stretch=1)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setContentsMargins(12, 8, 12, 12)
        btn_row.addStretch()
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        btn_row.addWidget(btn_close)
        lay.addLayout(btn_row)
        dlg.exec_()

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

        # Gather score context at each control point (if scores exist)
        score_context = None
        channels = self.result.score_channels
        if channels:
            score_context = []
            pts = self.result.projected_3d
            for cp in self._control_points:
                probe = np.array([cp[0], cp[1], cp[2]])
                dists_to_data = np.linalg.norm(pts - probe, axis=1)
                # Weighted average of nearby scores (5 nearest points)
                nearest_5 = np.argsort(dists_to_data)[:5]
                cp_scores = {}
                for ch in channels:
                    vals = []
                    for idx in nearest_5:
                        entry = self.result.session.entries[idx]
                        s = entry.scores.get(ch)
                        if s is None and entry.path_id is not None:
                            s = self.result.session.path_scores.get(
                                entry.path_id, {}).get(ch)
                        if s is not None:
                            vals.append(s)
                    if vals:
                        cp_scores[ch] = round(sum(vals) / len(vals), 2)
                score_context.append(cp_scores)

        explanation = self._explainer.explain_probe_multi(
            axis_labels=axis_labels,
            control_points=control_points_for_llm,
            score_context=score_context,
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
        # Don't overwrite attractor info — it has priority
        if self._highlighted_attractor is not None:
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

        # Compensate for variable frame rate: if frames arrive slower than
        # 60 Hz, scale the physics step proportionally so animation speed
        # stays consistent regardless of window size or GPU/DWM throttling.
        dt_scale = min(4.0, dt * 60.0)  # cap at 4x to avoid huge jumps

        # Check for async explain stage update (intermediate progress)
        stage = getattr(self, '_explain_pending_stage', None)
        if stage is not None:
            self._explain_pending_stage = None
            self._show_explain(f"Explaining attractor...\n\n{stage}")

        # Check for async explain result (final)
        pending = getattr(self, '_explain_pending', None)
        if pending is not None:
            self._explain_pending = None
            self._explain_pending_stage = None
            self._show_explain(pending)

        # --- Auto-rotation disabled (kept for future use) ---
        # if self.auto_rotate:
        #     cam = self.view.camera
        #     cam.azimuth = (cam.azimuth or 0) + self.AUTO_ROTATE_SPEED * dt

        # Flow step (uses pre-allocated buffer for performance)
        if self.flow_active and self.flow is not None:
            pos, colors, alphas, speeds = self.flow.step(dt_scale=dt_scale)

            # Color mode selection
            color_mode = getattr(self, '_flow_color_mode', 'speed')
            if color_mode == 'entropy':
                from tracescope.visualization.flow_field import diverging_colormap
                max_speed = speeds.max() if speeds.max() > 0 else 1.0
                t = (speeds / max_speed) * 2.0 - 1.0
                colors = diverging_colormap(t).astype(np.float32)
            elif color_mode == 'cluster':
                # Color by inverse-distance-weighted blend of cluster colors.
                # Uses squared distances (avoids sqrt) and einsum to
                # reduce allocations on the hot path.
                centroids = self._cluster_centroids  # (K, 3)
                diff = pos[:, None, :] - centroids[None, :, :]  # (N, K, 3)
                dist_sq = np.einsum('nkd,nkd->nk', diff, diff)  # (N, K)
                eps = 1e-12
                inv_dists = 1.0 / (dist_sq + eps) ** 1.5  # power 3 on dist
                weights = inv_dists / inv_dists.sum(axis=1, keepdims=True)
                colors = (weights @ self._cluster_colors_arr).astype(np.float32)
            elif color_mode.startswith('score:'):
                # Color flow particles by precomputed score grid (fast)
                from tracescope.visualization.flow_field import score_colormap
                particle_scores = self.flow.sample_score_batch(pos)
                colors = score_colormap(particle_scores).astype(np.float32)

            rgba = self._flow_rgba
            if rgba is None or len(rgba) != len(pos):
                rgba = np.empty((len(pos), 4), dtype=np.float32)
                self._flow_rgba = rgba
            rgba[:, :3] = colors
            rgba[:, 3] = alphas
            opacity = getattr(self, '_flow_opacity', 1.0)
            if opacity < 1.0:
                rgba[:, 3] *= opacity

            # Confidence fading: slider controls how much the density
            # confidence affects particle opacity.
            #   fade = (1 - slider) + slider * conf^exponent
            # At slider=0%: fade=1 (no change).
            # At slider=1%: fade ≈ 0.99 + 0.01*conf (barely visible).
            # At slider=100%: fade = conf^exponent (full effect).
            # The power curve (exponent>1) makes high-conf particles stay
            # bright while low-conf ones drop faster.
            conf_str = getattr(self, '_confidence_strength', 0.0)
            if conf_str > 0 and self.flow is not None and self.flow._confidence_grid is not None:
                conf = self.flow.sample_confidence_batch(pos)
                exponent = 3.0  # nonlinear: conf=0.8→0.51, conf=0.2→0.008
                conf_powered = np.power(conf, exponent)
                fade = (1.0 - conf_str) + conf_str * conf_powered
                rgba[:, 3] *= fade
                # Skip rendering particles with near-zero alpha
                visible = rgba[:, 3] > 0.01
                if not np.all(visible):
                    pos = pos[visible]
                    rgba = rgba[visible]

            self.vis_flow.set_data(pos, rgba)

        # Ball flow
        if self.ball_flowing and self.flow is not None:
            prev_bp = self._last_ball_pos
            bp = self.flow.advance_ball(dt_scale=dt_scale)
            # Track whether the ball is actually moving (or stuck in a zero-flow region)
            if prev_bp is not None:
                step = float(np.linalg.norm(bp - prev_bp))
                if step < 1e-4:
                    self._ball_stuck_frames += 1
                    self._ball_motion_frames = 0
                else:
                    self._ball_stuck_frames = 0
                    self._ball_motion_frames = getattr(self, '_ball_motion_frames', 0) + 1
            self._last_ball_pos = bp.copy()
            # Refresh contextual hint on stuck/unstuck transitions (~every 30 frames)
            if self._frame_count % 30 == 0:
                stuck_frames = self._ball_stuck_frames
                motion_frames = getattr(self, '_ball_motion_frames', 0)
                if stuck_frames >= 30:
                    new_stuck = True
                elif motion_frames >= 60:
                    new_stuck = False
                else:
                    new_stuck = self._ball_is_stuck  # keep current (initial state)
                if new_stuck != self._ball_is_stuck:
                    self._ball_is_stuck = new_stuck
                    self._update_contextual_hint()
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
            if self._gizmo_visible:
                self._update_gizmo(bp.astype(np.float32))

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
        if key == 'Space' and self._has_flow and self.flow:
            # Space toggles "drag probe by the flow" (ball_flowing)
            self.ball_flowing = not self.ball_flowing
            if self.ball_flowing:
                self._auto_mark_start_point()
                self.flow.start_ball_flow()
                self._show_gizmo(False)
            else:
                self.flow.stop_ball_flow()
                self.vis_ball.visible = False
                self.vis_trail.visible = False
            if QtWidgets and hasattr(self, 'chk_ball'):
                self.chk_ball.blockSignals(True)
                self.chk_ball.setChecked(self.ball_flowing)
                self.chk_ball.blockSignals(False)
            if hasattr(self, '_gizmo_hint'):
                self._gizmo_hint.setVisible(not self.ball_flowing)
            self._update_contextual_hint()
        elif key == 'B' and self._has_flow and self.flow:
            self.ball_flowing = not self.ball_flowing
            if self.ball_flowing:
                self._auto_mark_start_point()
                self.flow.start_ball_flow()
                self._show_gizmo(False)
            else:
                self.flow.stop_ball_flow()
                self.vis_ball.visible = False
                self.vis_trail.visible = False
            if QtWidgets and hasattr(self, 'chk_ball'):
                self.chk_ball.blockSignals(True)
                self.chk_ball.setChecked(self.ball_flowing)
                self.chk_ball.blockSignals(False)
            if hasattr(self, '_gizmo_hint'):
                self._gizmo_hint.setVisible(not self.ball_flowing)
            self._update_contextual_hint()
        elif key == 'M':
            # Mark current probe position
            self._qt_mark()
        elif key == 'F' and self._has_flow:
            # F toggles the flow particles (moved from Space)
            self.flow_active = not self.flow_active
            self.vis_flow.visible = self.flow_active
            if QtWidgets and hasattr(self, 'chk_flow'):
                self.chk_flow.setChecked(self.flow_active)
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
        elif key == 'E':
            # Context-sensitive: attractor explain takes priority, else path explain
            if self._attractors_visible and self._highlighted_attractor is not None:
                shift = 'Shift' in event.modifiers if hasattr(event, 'modifiers') else False
                self._explain_highlighted_attractor(deep=shift)
            else:
                self._qt_explain()
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

        # Project data points to screen using vispy's own transform
        try:
            pts = self.result.projected_3d
            pts_h = np.hstack([pts, np.ones((len(pts), 1))]).astype(np.float32)
            mapped = self.view.scene.transform.map(pts_h)
            w_vals = mapped[:, 3:4]
            w_vals = np.where(np.abs(w_vals) < 1e-8, 1.0, w_vals)
            screen = mapped[:, :2] / w_vals  # perspective divide → canvas px

            click_pos = np.array([event.pos[0], event.pos[1]], dtype=np.float64)
            dists = np.linalg.norm(screen - click_pos, axis=1)
            nearest_idx = int(np.argmin(dists))

            # Only snap if click is within 50 canvas pixels
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
            if self._gizmo_visible:
                self._update_gizmo(target)
            if not self.ball_flowing and self.flow is not None:
                self.flow.set_ball_position(*target)
            self._update_info(target)

            # Show point info overlay
            self._show_point_info(nearest_idx)
        except Exception:
            pass

    # ═══════════════════════════════════════════════════
    #  Gizmo — Blender-style axis drag for probe
    # ═══════════════════════════════════════════════════

    def _update_gizmo(self, center):
        """Reposition the 3 axis arrows with arrowheads around *center*."""
        c = np.asarray(center, dtype=np.float32)
        head_frac = 0.075  # arrowhead length as fraction of shaft
        head_spread = 0.105  # arrowhead width as fraction of shaft length
        for i in range(3):
            tip = c.copy()
            tip[i] += self._gizmo_arrow_len
            self._gizmo_lines[i].set_data(
                pos=np.array([c, tip], dtype=np.float32),
                color=self._gizmo_colors[i],
            )
            # Arrowhead: two barbs from tip pointing back along shaft + sideways
            back = c.copy()
            back[i] = tip[i] - self._gizmo_arrow_len * head_frac
            perp1 = (i + 1) % 3
            perp2 = (i + 2) % 3
            barb_offset = self._gizmo_arrow_len * head_spread
            b1 = back.copy()
            b1[perp1] += barb_offset
            b2 = back.copy()
            b2[perp2] += barb_offset
            head_pts = np.array([tip, b1, tip, b2], dtype=np.float32)
            self._gizmo_heads[i].set_data(
                pos=head_pts,
                color=self._gizmo_colors[i],
            )

    def _show_gizmo(self, show=True, persistent=False):
        """Show/hide the gizmo. When persistent=True, Ctrl-release won't hide it."""
        self._gizmo_visible = show
        for line in self._gizmo_lines:
            line.visible = show
        for head in self._gizmo_heads:
            head.visible = show
        if show:
            if persistent:
                self._gizmo_persistent = True
            pos = self._probe_pos_from_sliders()
            self._update_gizmo(pos)
        else:
            self._gizmo_persistent = False
        if hasattr(self, '_hint_overlay'):
            self._update_contextual_hint()

    def _on_gizmo_key(self, event):
        """Show gizmo when Ctrl is pressed (not during ball flow)."""
        if event.key == 'Control' and not self.ball_flowing:
            if not self._gizmo_visible:
                self._show_gizmo(True)

    def _on_gizmo_key_release(self, event):
        """Hide gizmo when Ctrl is released, unless persistent or mid-drag."""
        if (event.key == 'Control'
                and self._gizmo_dragging is None
                and not getattr(self, '_gizmo_persistent', False)):
            self._show_gizmo(False)

    @staticmethod
    def _point_to_segment_dist_2d(p, a, b):
        """Shortest distance from point *p* to line segment *a*–*b* in 2D."""
        ab = b - a
        ab_sq = float(np.dot(ab, ab))
        if ab_sq < 1e-8:
            return float(np.linalg.norm(p - a))
        t = float(np.dot(p - a, ab)) / ab_sq
        t = max(0.0, min(1.0, t))
        closest = a + t * ab
        return float(np.linalg.norm(p - closest))

    def _vispy_project(self, point_3d):
        """Project a 3D point (in raw data coords) to 2D canvas pixel coords.

        Applies the render-root scale (normalizes data extent) then the
        vispy scene transform chain, giving canvas-pixel coordinates.
        """
        s = self._scene_scale
        pt = np.array([[point_3d[0] * s, point_3d[1] * s, point_3d[2] * s, 1.0]],
                      dtype=np.float32)
        mapped = self.view.scene.transform.map(pt)[0]
        w = mapped[3]
        if abs(w) < 1e-8:
            w = 1.0
        return np.array([mapped[0] / w, mapped[1] / w], dtype=np.float64)

    def _on_click_track_press(self, event):
        """Track press start to differentiate plain click from drag/orbit."""
        if event.button != 1:
            return
        self._click_track_pos = np.array([event.pos[0], event.pos[1]], dtype=np.float64)
        self._click_track_moved = False

    def _on_click_track_move(self, event):
        if self._click_track_pos is None:
            return
        cur = np.array([event.pos[0], event.pos[1]], dtype=np.float64)
        if float(np.linalg.norm(cur - self._click_track_pos)) > 4.0:
            self._click_track_moved = True

    def _on_click_track_release(self, event):
        """On plain click (no drag) near the probe, spawn the gizmo persistently
        and auto-disable ball flow."""
        if event.button != 1:
            return
        start = self._click_track_pos
        moved = self._click_track_moved
        self._click_track_pos = None
        self._click_track_moved = False
        if start is None or moved:
            return
        # Don't interfere with existing gizmo drag
        if self._gizmo_dragging is not None:
            return
        # If gizmo is already visible (transient, not persistent), skip
        if self._gizmo_visible and not getattr(self, '_gizmo_persistent', False):
            return
        # Check proximity to the probe in screen space
        probe_pos = self._probe_pos_from_sliders()
        probe_sc = self._vispy_project(probe_pos)
        click = np.array([event.pos[0], event.pos[1]], dtype=np.float64)
        probe_dist = float(np.linalg.norm(click - probe_sc))
        if probe_dist > 60.0:
            return
        # When attractors are visible, only grab the probe if the click is
        # really close (<30px) — otherwise let the attractor handler win.
        if self._attractors_visible and probe_dist > 30.0:
            return
        # Dismiss explain overlay when user starts editing the probe
        if hasattr(self, '_explain_overlay'):
            self._explain_overlay.setVisible(False)
        # Auto-disable ball flow and spawn persistent gizmo
        if self.ball_flowing:
            self.ball_flowing = False
            if self.flow is not None:
                self.flow.stop_ball_flow()
                self.vis_ball.visible = False
                self.vis_trail.visible = False
            if hasattr(self, 'chk_ball'):
                self.chk_ball.blockSignals(True)
                self.chk_ball.setChecked(False)
                self.chk_ball.blockSignals(False)
            if hasattr(self, '_gizmo_hint'):
                self._gizmo_hint.setVisible(True)
        self._show_gizmo(True, persistent=True)

    def _on_gizmo_press(self, event):
        """Ctrl+click: pick the axis whose screen-space line segment is
        closest to the click — pure 2D in canvas pixel coordinates."""
        if event.button != 1:
            return
        if not self._gizmo_visible:
            return

        probe_pos = self._probe_pos_from_sliders()
        click = np.array([event.pos[0], event.pos[1]], dtype=np.float64)

        # Project gizmo center and tips to canvas pixels (W-divided)
        center_sc = self._vispy_project(probe_pos)

        axis_names = ['X', 'Y', 'Z']
        best_axis = None
        best_dist = float('inf')
        debug_dists = []

        for i in range(3):
            tip_3d = probe_pos.copy().astype(np.float64)
            tip_3d[i] += self._gizmo_arrow_len
            tip_sc = self._vispy_project(tip_3d)

            dist = self._point_to_segment_dist_2d(click, center_sc, tip_sc)
            debug_dists.append((axis_names[i], dist, tip_sc))

            if dist < best_dist:
                best_dist = dist
                best_axis = i

        # Debug output — keep until gizmo confirmed working
        print(f"[GIZMO] click=({click[0]:.0f},{click[1]:.0f})  "
              f"center=({center_sc[0]:.0f},{center_sc[1]:.0f})")
        for name, d, tip in debug_dists:
            marker = " <<<" if name == axis_names[best_axis] else ""
            print(f"  {name}: tip=({tip[0]:.0f},{tip[1]:.0f})  dist={d:.1f}px{marker}")

        if best_axis is None or best_dist > 150:
            print(f"  → no axis picked (best_dist={best_dist:.0f} > 150px)")
            return

        print(f"  → picked {axis_names[best_axis]} axis (dist={best_dist:.1f}px)")

        # Disable camera so it doesn't rotate during drag
        self.view.camera.interactive = False

        # Screen-space drag setup: project +1 world unit along the
        # chosen axis to find the screen direction and scale factor.
        tip_3d = probe_pos.copy().astype(np.float64)
        tip_3d[best_axis] += 1.0
        tip_sc = self._vispy_project(tip_3d)
        axis_screen_dir = tip_sc - center_sc
        px_per_unit = float(np.linalg.norm(axis_screen_dir))
        if px_per_unit < 1e-3:
            px_per_unit = 1.0
        axis_screen_n = axis_screen_dir / px_per_unit

        self._gizmo_dragging = best_axis
        self._gizmo_drag_start_pos = probe_pos.copy()
        self._gizmo_drag_start_click = click.copy()
        self._gizmo_drag_axis_screen_n = axis_screen_n
        self._gizmo_drag_px_per_unit = px_per_unit

        # Highlight the active arrow, dim the others
        for j in range(3):
            c = list(self._gizmo_colors[j])
            c[3] = 1.0 if j == best_axis else 0.3
            self._gizmo_lines[j].set_data(color=c)
            self._gizmo_heads[j].set_data(color=c)

    def _on_gizmo_move(self, event):
        """Drag: project mouse movement along the axis in screen space,
        convert to world units using the pixels-per-unit ratio."""
        if self._gizmo_dragging is None:
            return

        axis = self._gizmo_dragging
        current = np.array([event.pos[0], event.pos[1]], dtype=np.float64)
        mouse_delta = current - self._gizmo_drag_start_click

        # Project mouse movement onto the screen-space axis direction
        along_px = float(np.dot(mouse_delta, self._gizmo_drag_axis_screen_n))
        world_delta = along_px / self._gizmo_drag_px_per_unit

        new_pos = self._gizmo_drag_start_pos.copy()
        new_pos[axis] += world_delta

        # Clamp to data range with margin
        pts = self.result.projected_3d
        for i in range(3):
            mn, mx = pts[:, i].min(), pts[:, i].max()
            margin = (mx - mn) * 0.15
            new_pos[i] = np.clip(new_pos[i], mn - margin, mx + margin)

        # Update sliders
        self._updating_sliders = True
        labels = getattr(self.result.axis_info, 'labels', ['X', 'Y', 'Z'])
        for i in range(3):
            sv = self._raw_to_slider(float(new_pos[i]), i)
            sv = int(np.clip(sv, 0, 1000))
            self._sliders[i].setValue(sv)
            self._slider_labels[i].setText(f"{labels[i]}: {sv / 10:.0f}%")
        self._updating_sliders = False

        # Update visuals
        self.vis_probe.set_data(new_pos.reshape(1, 3), self.PROBE_COLOR)
        self._update_gizmo(new_pos)
        if self.flow is not None and not self.ball_flowing:
            self.flow.set_ball_position(*new_pos)

    def _on_gizmo_release(self, event):
        """End axis drag, re-enable camera."""
        if self._gizmo_dragging is None:
            return

        # Re-enable camera interaction
        self.view.camera.interactive = True

        # Final position
        pos = self._probe_pos_from_sliders()
        self._update_info(pos)

        # Reset arrow colors
        for j in range(3):
            c = list(self._gizmo_colors[j])
            c[3] = 0.9
            self._gizmo_lines[j].set_data(color=c)
            self._gizmo_heads[j].set_data(color=c)

        self._gizmo_dragging = None
        self._gizmo_drag_start_pos = None
        self._gizmo_drag_start_click = None
        self._gizmo_drag_axis_screen_n = None

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

    def _apply_initial_state(self):
        """Apply optional initial UI state overrides from self._initial_state."""
        state = self._initial_state
        if not state:
            return

        # Speed multiplier (slider value = speed * 100)
        if 'speed' in state and hasattr(self, '_sl_speed'):
            speed_val = int(state['speed'] * 100)
            self._sl_speed.setValue(speed_val)

        # Attractor sensitivity (set BEFORE show_attractors so first
        # computation uses the correct value)
        if 'sensitivity' in state and hasattr(self, '_sl_sensitivity'):
            self._attractor_sensitivity = state['sensitivity']
            slider_val = int(max(0, min(1000, state['sensitivity'] * 1000)))
            self._sl_sensitivity.setValue(slider_val)
            pct = state['sensitivity'] * 100
            self._lbl_sensitivity.setText(f"Sensitivity: {pct:.1f}%")

        # Show attractors
        if state.get('show_attractors') and hasattr(self, 'chk_attractors'):
            self.chk_attractors.setChecked(True)

        # Flow color mode
        if 'flow_color' in state:
            self._flow_color_mode = state['flow_color']
            if state['flow_color'] == 'cluster':
                if hasattr(self, '_chk_cluster_colors'):
                    self._chk_cluster_colors.setChecked(True)
            elif state['flow_color'] == 'entropy':
                if hasattr(self, '_chk_entropy'):
                    self._chk_entropy.setChecked(True)
            elif state['flow_color'].startswith('score:'):
                channel = state['flow_color'].split(':', 1)[1]
                for ch, chk in getattr(self, '_score_color_checkboxes', []):
                    chk.setChecked(ch == channel)

        # Show data points
        if state.get('show_points') and hasattr(self, 'chk_points'):
            self.chk_points.setChecked(True)

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

        # Apply optional initial UI state overrides
        self._apply_initial_state()

        if self.window is not None:
            self.window.show()
            # Workaround: some GPU drivers render specific framebuffer
            # widths much slower than adjacent ones (observed: widths
            # where width%9==8 on some NVIDIA cards drop from 28 to 12
            # FPS).  A +2px jog after first paint nudges the canvas onto
            # a fast alignment without visible change to the layout.
            def _jog_resize():
                self.window.resize(
                    self.window.width() + 2,
                    self.window.height(),
                )
            QtCore.QTimer.singleShot(100, _jog_resize)
        else:
            self.canvas.show()
        self._timer.start()
        # Show initial contextual hint
        if hasattr(self, '_hint_overlay'):
            self._update_contextual_hint()

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

def launch_renderer(result, particle_grid=40, base_size=28.0,
                    window_size=(1400, 800), explainer=None,
                    dataset_name=None, initial_state=None) -> FlowRenderer:
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
        dataset_name:  Optional display name shown as title on the renderer
        initial_state: Optional dict to override default UI state on launch.
                       Supported keys (all optional):
                         speed (float): flow speed multiplier (default 1.0)
                         show_attractors (bool): show attractor meshes
                         flow_color (str): 'speed'|'cluster'|'entropy' or 'score:<channel>'
                         show_points (bool): show data point markers
    """
    from tracescope.models.analysis import AnalysisResult
    if not isinstance(result, AnalysisResult):
        raise TypeError(
            f"launch_renderer() requires an AnalysisResult from pipeline.analyze(), "
            f"got {type(result).__name__}."
        )
    _ensure_viable_backend()
    renderer = FlowRenderer(result, particle_grid=particle_grid,
                             base_size=base_size, window_size=window_size,
                             explainer=explainer, dataset_name=dataset_name,
                             initial_state=initial_state)
    renderer.run()
    return renderer
