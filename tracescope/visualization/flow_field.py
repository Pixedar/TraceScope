"""
Flow field system — faithful port from Android's FlowFieldSystem.java.

Features:
  - 40³ velocity grid with trilinear interpolation
  - Particle lifecycle (lattice init, pre-integration, lifespan, wrapping)
  - Turbo colormap (exact polynomial from Android)
  - Ball/probe following flow field
  - Pre-computed animation frames for dashboard

Constants matching Android:
  GRID = 40, LIFESPAN = 57, PRE_AGE = 34, DT = 0.02
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════
# TURBO COLORMAP (exact polynomial from FlowFieldSystem.java)
# ═══════════════════════════════════════════════════

def turbo_colormap(t: np.ndarray) -> np.ndarray:
    """Apply Turbo colormap to speed values.

    Exact polynomial coefficients from FlowFieldSystem.java.
    Works with scalar or array input.

    Args:
        t: Speed values normalized to [0, 1]. Can be scalar or (N,) array.

    Returns:
        (N, 3) array of RGB values in [0, 1].
    """
    t = np.clip(np.atleast_1d(t).astype(np.float64), 0.0, 1.0)

    r = (34.61 + t * (1172.33 + t * (-10793.56 + t * (33300.12
         + t * (-38345.17 + 14829.80 * t))))) / 255.0
    g = (23.31 + t * (557.33 + t * (1225.33 + t * (-3574.96
         + t * 2199.29)))) / 255.0
    b = (27.20 + t * (3211.10 + t * (-15327.97 + t * (34592.87
         + t * (-30538.66 + 9347.97 * t))))) / 255.0

    rgb = np.stack([np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1)], axis=-1)
    return rgb


# ═══════════════════════════════════════════════════
# DIVERGING COLORMAP (matching FlowFieldSystem.java)
# ═══════════════════════════════════════════════════

def diverging_colormap(t: np.ndarray) -> np.ndarray:
    """Red ↔ Blue ↔ Green diverging colormap.

    Args:
        t: Values in [-1, 1]. Negative → red, zero → blue, positive → green.

    Returns:
        (N, 3) array of RGB values in [0, 1].
    """
    t = np.clip(np.atleast_1d(t).astype(np.float64), -1.0, 1.0)

    RED = np.array([1.0, 0.1, 0.1])
    BLUE = np.array([0.18, 0.28, 1.0])
    GREEN = np.array([0.0, 0.78, 0.0])

    rgb = np.zeros((len(t), 3))
    neg_mask = t < 0
    pos_mask = ~neg_mask

    # Negative: blue → red
    if np.any(neg_mask):
        a = -t[neg_mask]
        rgb[neg_mask] = BLUE[None, :] * (1 - a[:, None]) + RED[None, :] * a[:, None]

    # Positive: blue → green
    if np.any(pos_mask):
        a = t[pos_mask]
        rgb[pos_mask] = BLUE[None, :] * (1 - a[:, None]) + GREEN[None, :] * a[:, None]

    return rgb


# ═══════════════════════════════════════════════════
# FLOW FIELD SYSTEM (matching FlowFieldSystem.java)
# ═══════════════════════════════════════════════════

class FlowFieldSystem:
    """Particle flow field system.

    Faithful port from Android's FlowFieldSystem.java.
    Uses a 40³ velocity grid with trilinear interpolation,
    particle lifecycle with wrapping, and Turbo speed coloring.

    Args:
        velocity_grid: (G, G, G, 3) velocity field array.
        axis_min: (3,) minimum bounds of the domain.
        axis_max: (3,) maximum bounds of the domain.
        particle_grid: Particle grid resolution (default 20 → 8000 particles).
            Use 40 for full Android fidelity (64,000 particles).
    """

    # Constants matching Android
    VELOCITY_GRID_SIZE = 40
    LIFESPAN = 57           # particle lifetime in frames
    PRE_AGE = 34            # pre-integration steps
    DT = 0.02               # world-units per step

    def __init__(
        self,
        velocity_grid: np.ndarray,
        axis_min: np.ndarray,
        axis_max: np.ndarray,
        particle_grid: int = 20,
    ):
        self.velocity_grid = velocity_grid.astype(np.float32)
        self.grid_size = velocity_grid.shape[0]  # typically 40
        self.axis_min = np.asarray(axis_min, dtype=np.float32)
        self.axis_max = np.asarray(axis_max, dtype=np.float32)
        self.span = self.axis_max - self.axis_min

        self.particle_grid = particle_grid
        self.particle_count = particle_grid ** 3
        self.speed_multiplier = 1.0

        # Particle state
        self.pos = np.zeros((self.particle_count, 3), dtype=np.float32)
        self.age = np.zeros(self.particle_count, dtype=np.int32)

        # Initialize on lattice
        self._init_lattice()
        # Pre-integrate to displace from regular lattice (matching Android)
        self._dry_integrate(self.PRE_AGE)
        # Save displaced positions as respawn points (Android: origPos)
        self.orig_pos = self.pos.copy()
        # Randomize ages to stagger respawns (matching Android initParticles)
        self.age = np.random.randint(0, self.LIFESPAN, self.particle_count, dtype=np.int32)

        # Ball state
        self.ball_pos = np.array([
            (axis_min[0] + axis_max[0]) / 2,
            (axis_min[1] + axis_max[1]) / 2,
            (axis_min[2] + axis_max[2]) / 2,
        ], dtype=np.float32)
        self.ball_trail: list = []
        self.ball_flowing = False

    def _init_lattice(self):
        """Initialize particles on a regular 3D lattice."""
        margin = 0.05
        idx = 0
        g = self.particle_grid
        for i in range(g):
            for j in range(g):
                for k in range(g):
                    fx = i / (g - 1) if g > 1 else 0.5
                    fy = j / (g - 1) if g > 1 else 0.5
                    fz = k / (g - 1) if g > 1 else 0.5
                    self.pos[idx, 0] = self.axis_min[0] + (margin + fx * (1 - 2 * margin)) * self.span[0]
                    self.pos[idx, 1] = self.axis_min[1] + (margin + fy * (1 - 2 * margin)) * self.span[1]
                    self.pos[idx, 2] = self.axis_min[2] + (margin + fz * (1 - 2 * margin)) * self.span[2]
                    self.age[idx] = 0
                    idx += 1

    def _dry_integrate(self, steps: int):
        """Pre-integrate without recording, to break lattice regularity.

        Matches Android's dryIntegrate(): advects and wraps only,
        does NOT increment age (ages are randomized after).
        """
        for _ in range(steps):
            velocities = self.sample_velocity_batch(self.pos)
            self.pos += velocities * self.DT
            self._wrap_all()

    def sample_velocity(self, x: float, y: float, z: float) -> np.ndarray:
        """Trilinear interpolation on the velocity grid.

        Ported from FlowFieldSystem.java VelocityField.sample().
        """
        G = self.grid_size
        # Normalize to grid coordinates [0, G-1]
        nx = (x - self.axis_min[0]) / self.span[0] * (G - 1) if self.span[0] > 0 else 0
        ny = (y - self.axis_min[1]) / self.span[1] * (G - 1) if self.span[1] > 0 else 0
        nz = (z - self.axis_min[2]) / self.span[2] * (G - 1) if self.span[2] > 0 else 0

        ix = int(np.floor(nx))
        iy = int(np.floor(ny))
        iz = int(np.floor(nz))
        tx = nx - ix
        ty = ny - iy
        tz = nz - iz

        ix = np.clip(ix, 0, G - 2)
        iy = np.clip(iy, 0, G - 2)
        iz = np.clip(iz, 0, G - 2)

        result = np.zeros(3, dtype=np.float32)
        for dx in range(2):
            for dy in range(2):
                for dz in range(2):
                    w = ((1 - tx) if dx == 0 else tx) * \
                        ((1 - ty) if dy == 0 else ty) * \
                        ((1 - tz) if dz == 0 else tz)
                    result += w * self.velocity_grid[ix + dx, iy + dy, iz + dz]

        return result

    def sample_velocity_batch(self, positions: np.ndarray) -> np.ndarray:
        """Vectorized trilinear interpolation for all particles.

        Args:
            positions: (N, 3) array of particle positions.

        Returns:
            (N, 3) array of velocities.
        """
        G = self.grid_size
        N = len(positions)

        # Normalize to grid coordinates
        normalized = np.zeros_like(positions)
        for a in range(3):
            if self.span[a] > 0:
                normalized[:, a] = (positions[:, a] - self.axis_min[a]) / self.span[a] * (G - 1)

        # Integer indices and fractional parts
        ix = np.floor(normalized).astype(np.int32)
        tx = normalized - ix
        ix = np.clip(ix, 0, G - 2)

        # 8-point trilinear interpolation (vectorized)
        result = np.zeros((N, 3), dtype=np.float32)
        for dx in range(2):
            for dy in range(2):
                for dz in range(2):
                    wx = np.where(dx == 0, 1 - tx[:, 0], tx[:, 0])
                    wy = np.where(dy == 0, 1 - tx[:, 1], tx[:, 1])
                    wz = np.where(dz == 0, 1 - tx[:, 2], tx[:, 2])
                    w = wx * wy * wz

                    gx = ix[:, 0] + dx
                    gy = ix[:, 1] + dy
                    gz = ix[:, 2] + dz

                    result += w[:, None] * self.velocity_grid[gx, gy, gz]

        return result

    def _wrap_all(self):
        """Apply periodic wrapping to all particles."""
        for a in range(3):
            if self.span[a] <= 0:
                continue
            dist = self.pos[:, a] - self.axis_min[a]
            dist = dist % self.span[a]
            self.pos[:, a] = self.axis_min[a] + dist

    def _respawn(self, indices: np.ndarray):
        """Respawn particles at their pre-integrated origin positions.

        Matches Android: age[i] = 0; pos[i] = origPos[i].
        """
        self.pos[indices] = self.orig_pos[indices]
        self.age[indices] = 0

    def step(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Advance all particles by one frame.

        Returns:
            (positions, colors, alphas, speeds) — all (N, ...) arrays.
        """
        # Compute velocities for all particles
        velocities = self.sample_velocity_batch(self.pos)

        # Compute speeds
        speeds = np.linalg.norm(velocities, axis=1)

        # Advect
        self.pos += velocities * self.DT * self.speed_multiplier
        self._wrap_all()

        # Age and respawn
        self.age += 1
        expired = self.age >= self.LIFESPAN
        if np.any(expired):
            self._respawn(np.where(expired)[0])

        # Speed-based colors (Turbo colormap)
        max_speed = speeds.max() if speeds.max() > 0 else 1.0
        normalized_speeds = speeds / max_speed
        colors = turbo_colormap(normalized_speeds)

        # Alpha based on age (fade in/out)
        age_frac = self.age.astype(np.float32) / self.LIFESPAN
        alphas = np.ones(self.particle_count, dtype=np.float32)
        # Fade in during first 10% of life
        fade_in = age_frac < 0.1
        alphas[fade_in] = age_frac[fade_in] / 0.1
        # Fade out during last 20% of life
        fade_out = age_frac > 0.8
        alphas[fade_out] = (1.0 - age_frac[fade_out]) / 0.2
        alphas = np.clip(alphas, 0, 1)

        return self.pos.copy(), colors, alphas, speeds

    def advance_ball(self) -> np.ndarray:
        """Advance the ball probe by one step along the flow field.

        Uses 0.8 * DT for slower movement (matching Android's dtSlow).

        Returns:
            New ball position (3,) array.
        """
        v = self.sample_velocity(*self.ball_pos)
        dt_slow = self.DT * 0.8 * self.speed_multiplier
        self.ball_pos = self.ball_pos + v * dt_slow

        # Clamp to domain
        self.ball_pos = np.clip(self.ball_pos, self.axis_min, self.axis_max)

        # Update trail (max 100 points, matching Android MAX_TRAIL)
        self.ball_trail.append(self.ball_pos.copy())
        if len(self.ball_trail) > 100:
            self.ball_trail.pop(0)

        return self.ball_pos.copy()

    def set_ball_position(self, x: float, y: float, z: float):
        """Set ball position manually (from slider interaction)."""
        self.ball_pos = np.array([x, y, z], dtype=np.float32)

    def set_particle_grid(self, new_grid: int):
        """Reinitialize particles with a new grid resolution."""
        self.particle_grid = new_grid
        self.particle_count = new_grid ** 3
        self.pos = np.zeros((self.particle_count, 3), dtype=np.float32)
        self.age = np.zeros(self.particle_count, dtype=np.int32)
        self._init_lattice()
        self._dry_integrate(self.PRE_AGE)
        self.orig_pos = self.pos.copy()
        self.age = np.random.randint(0, self.LIFESPAN, self.particle_count, dtype=np.int32)

    def start_ball_flow(self):
        """Start ball following the flow field."""
        self.ball_flowing = True
        self.ball_trail.clear()
        self.ball_trail.append(self.ball_pos.copy())

    def stop_ball_flow(self):
        """Stop ball following the flow field."""
        self.ball_flowing = False

    def precompute_frames(self, n_frames: int = 200) -> list:
        """Pre-compute N animation frames for efficient playback.

        Returns:
            List of (positions, colors, alphas) tuples.
        """
        frames = []
        for _ in range(n_frames):
            pos, colors, alphas, speeds = self.step()
            frames.append({
                "positions": pos,
                "colors": colors,
                "alphas": alphas,
                "speeds": speeds,
            })
        return frames


def build_flow_figure(
    result,
    flow_system: FlowFieldSystem,
    frame_data: dict,
    show_data_points: bool = True,
) -> dict:
    """Build Plotly trace data for one flow frame.

    Returns a dict of trace data that can be added to a figure.
    """
    from tracescope.visualization.scatter3d import CLUSTER_COLORS

    positions = frame_data["positions"]
    colors = frame_data["colors"]
    alphas = frame_data["alphas"]

    # Convert colors to Plotly format
    plotly_colors = [
        f"rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},{a:.2f})"
        for c, a in zip(colors, alphas)
    ]

    traces = []

    # Flow particles
    traces.append(dict(
        type="scatter3d",
        x=positions[:, 0].tolist(),
        y=positions[:, 1].tolist(),
        z=positions[:, 2].tolist(),
        mode="markers",
        marker=dict(
            size=3,
            color=plotly_colors,
            opacity=1.0,  # Per-particle alpha is in the color
        ),
        name="Flow particles",
        showlegend=False,
        hoverinfo="skip",
    ))

    return traces
