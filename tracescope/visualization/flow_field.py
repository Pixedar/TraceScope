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

def turbo_colormap(t: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
    """Apply Turbo colormap to speed values.

    Exact polynomial coefficients from FlowFieldSystem.java.
    Works with scalar or array input.

    Args:
        t: Speed values normalized to [0, 1]. Can be scalar or (N,) array.
        out: Optional pre-allocated (N, 3) output buffer to avoid allocation.

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

    if out is not None and out.shape == (len(t), 3):
        np.clip(r, 0, 1, out=out[:, 0])
        np.clip(g, 0, 1, out=out[:, 1])
        np.clip(b, 0, 1, out=out[:, 2])
        return out

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
# SCORE COLORMAP (red → yellow → green for 0 → 0.5 → 1)
# ═══════════════════════════════════════════════════

def score_colormap(t: np.ndarray) -> np.ndarray:
    """Map score values [0, 1] to red → yellow → green gradient.

    0.0 = red (bad/failure), 0.5 = yellow (neutral), 1.0 = green (success).

    Args:
        t: Score values in [0, 1]. Can be scalar or (N,) array.

    Returns:
        (N, 3) array of RGB values in [0, 1].
    """
    t = np.clip(np.atleast_1d(t).astype(np.float64), 0.0, 1.0)
    rgb = np.zeros((len(t), 3))

    # Red channel: 1.0 at t=0, 1.0 at t=0.5, 0.0 at t=1.0
    rgb[:, 0] = np.where(t <= 0.5, 1.0, 2.0 * (1.0 - t))
    # Green channel: 0.0 at t=0, 1.0 at t=0.5, 1.0 at t=1.0
    rgb[:, 1] = np.where(t <= 0.5, 2.0 * t, 1.0)
    # Blue stays near 0 for vivid colors
    rgb[:, 2] = 0.05

    return np.clip(rgb, 0.0, 1.0)


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
        path_points: Optional[np.ndarray] = None,
        confidence_grid: Optional[np.ndarray] = None,
    ):
        self.velocity_grid = velocity_grid.astype(np.float32)
        self.grid_size = velocity_grid.shape[0]  # typically 40
        self.axis_min = np.asarray(axis_min, dtype=np.float32)
        self.axis_max = np.asarray(axis_max, dtype=np.float32)
        self.span = self.axis_max - self.axis_min

        self.particle_grid = particle_grid
        self.particle_count = particle_grid ** 3
        self.speed_multiplier = 1.0

        # MDN confidence grid (None for RBF models)
        self._confidence_grid = confidence_grid  # (G, G, G) values in [0, 1]
        self.confidence_strength = 0.0  # 0 = ignore confidence, 1 = full effect

        # Build path occupancy mask for blob-based spawning
        self._path_mask = None
        self._blob_opacity = None
        self._score_grid = None
        self._mask_res = 0
        self.blob_enabled = True  # can be toggled off for full-cube spawning
        if path_points is not None and len(path_points) >= 2:
            self._build_path_mask(path_points)

        # Particle state + pre-allocated buffers (before init/pre-integration)
        N = self.particle_count
        self.pos = np.zeros((N, 3), dtype=np.float32)
        self.age = np.zeros(N, dtype=np.int32)
        self._buf_velocities = np.zeros((N, 3), dtype=np.float32)
        self._buf_speeds = np.zeros(N, dtype=np.float32)
        self._buf_colors = np.zeros((N, 3), dtype=np.float32)
        self._buf_alphas = np.ones(N, dtype=np.float32)
        self._buf_pos_out = np.zeros((N, 3), dtype=np.float32)
        self._buf_normalized = np.zeros((N, 3), dtype=np.float32)

        # Initialize on lattice (filtered by path mask if available)
        self._init_lattice()
        # Save blob-constrained positions as respawn points BEFORE
        # pre-integration so particles always respawn inside the blob
        self.orig_pos = self.pos.copy()
        # Pre-integrate to displace from regular lattice (matching Android)
        self._dry_integrate(self.PRE_AGE)
        # After pre-integration, snap any particles that drifted outside
        # the blob back to their blob-constrained origin
        if self._blob_opacity is not None and self.blob_enabled:
            for i in range(self.particle_count):
                if not self._point_in_blob_world(*self.pos[i]):
                    self.pos[i] = self.orig_pos[i]
        # Randomize ages to stagger respawns (matching Android initParticles)
        self.age = np.random.randint(0, self.LIFESPAN, N, dtype=np.int32)

        # Ball state
        self.ball_pos = np.array([
            (axis_min[0] + axis_max[0]) / 2,
            (axis_min[1] + axis_max[1]) / 2,
            (axis_min[2] + axis_max[2]) / 2,
        ], dtype=np.float32)
        self.ball_trail: list = []
        self.ball_flowing = False

    # ── Blob tuning constants ──────────────────────────────────
    # BLOB_RESOLUTION: occupancy grid resolution (higher = finer blob shape)
    BLOB_RESOLUTION = 42
    # BLOB_RADIUS: minimum influence radius around each path point, as
    #   fraction of the per-axis span.  The actual radius is computed
    #   adaptively from data density (median NN distance × 1.5) and
    #   floored at this value.  Increase for a looser blob.
    BLOB_RADIUS = 0.03
    # BLOB_MARGIN: additional cells beyond the core blob where opacity
    # fades from 1.0 to 0.0 (soft boundary falloff).
    BLOB_MARGIN_CELLS = 3

    def _build_path_mask(self, path_points: np.ndarray):
        """Build a 3D float opacity grid from path sample points.

        The goal is to cut out the empty void where there is absolutely
        no training data, while keeping everything that is even remotely
        near any data or path as one continuous cloud.  The blob is NOT
        meant to be a tight segmentation — it should be generous.

        Steps:
          1. Compute adaptive radius from data density (median NN × 1.5,
             floored at BLOB_RADIUS).
          2. Mark core cells around each path point.
          3. If the core has multiple disconnected components, dilate
             until they merge into one connected blob.
          4. Add soft opacity margin at the boundary.

        Args:
            path_points: (M, 3) densely sampled points along semantic paths.
        """
        from scipy.ndimage import distance_transform_edt, label, binary_dilation

        res = self.BLOB_RESOLUTION
        self._mask_res = res

        pp = np.asarray(path_points, dtype=np.float32)

        # Normalize path points to [0, 1] per axis
        norm = np.zeros_like(pp)
        for a in range(3):
            if self.span[a] > 0:
                norm[:, a] = (pp[:, a] - self.axis_min[a]) / self.span[a]

        # Adaptive radius: median nearest-neighbor distance × 1.5
        if len(norm) > 1:
            sample_idx = np.random.default_rng(0).choice(
                len(norm), min(500, len(norm)), replace=False
            )
            sample = norm[sample_idx]
            from scipy.spatial import cKDTree
            tree = cKDTree(sample)
            dists, _ = tree.query(sample, k=2)
            nn_dists = dists[:, 1]
            adaptive_r = float(np.median(nn_dists)) * 1.5
            radius_frac = max(adaptive_r, self.BLOB_RADIUS)
        else:
            radius_frac = self.BLOB_RADIUS

        # Core boolean mask
        core = np.zeros((res, res, res), dtype=bool)
        r_cells = int(np.ceil(radius_frac * (res - 1)))

        for p in range(len(norm)):
            ci = int(round(norm[p, 0] * (res - 1)))
            cj = int(round(norm[p, 1] * (res - 1)))
            ck = int(round(norm[p, 2] * (res - 1)))
            i0, i1 = max(0, ci - r_cells), min(res, ci + r_cells + 1)
            j0, j1 = max(0, cj - r_cells), min(res, cj + r_cells + 1)
            k0, k1 = max(0, ck - r_cells), min(res, ck + r_cells + 1)
            core[i0:i1, j0:j1, k0:k1] = True

        # Ensure one connected component — dilate until all islands merge
        labeled, n_components = label(core)
        max_dilations = res // 2  # safety cap
        dilations = 0
        while n_components > 1 and dilations < max_dilations:
            core = binary_dilation(core)
            labeled, n_components = label(core)
            dilations += 1

        # Build opacity grid: core=1.0, margin=falloff, outside=0.0
        margin = self.BLOB_MARGIN_CELLS
        opacity = np.zeros((res, res, res), dtype=np.float32)
        opacity[core] = 1.0

        if margin > 0:
            dist = distance_transform_edt(~core).astype(np.float32)
            margin_mask = (dist > 0) & (dist <= margin)
            opacity[margin_mask] = 1.0 - dist[margin_mask] / (margin + 1)

        self._path_mask = core
        self._blob_opacity = opacity
        self._path_mask_norm = norm

    def _sample_blob_opacity(self, x, y, z) -> float:
        """Sample the blob opacity at a world-space point."""
        if self._blob_opacity is None:
            return 1.0
        r = self._mask_res
        nx = (x - self.axis_min[0]) / self.span[0] if self.span[0] > 0 else 0.5
        ny = (y - self.axis_min[1]) / self.span[1] if self.span[1] > 0 else 0.5
        nz = (z - self.axis_min[2]) / self.span[2] if self.span[2] > 0 else 0.5
        ix = max(0, min(r - 1, int(round(nx * (r - 1)))))
        iy = max(0, min(r - 1, int(round(ny * (r - 1)))))
        iz = max(0, min(r - 1, int(round(nz * (r - 1)))))
        return float(self._blob_opacity[ix, iy, iz])

    def _point_in_blob_world(self, x, y, z):
        """Check if a world-space point is inside the path blob (core or margin)."""
        if self._blob_opacity is None:
            return True
        return self._sample_blob_opacity(x, y, z) > 0.0

    def _check_outside_blob_batch(self, positions: np.ndarray) -> np.ndarray:
        """Vectorized check: returns boolean mask of particles outside the blob."""
        r = self._mask_res
        norm = np.zeros_like(positions)
        for a in range(3):
            if self.span[a] > 0:
                norm[:, a] = (positions[:, a] - self.axis_min[a]) / self.span[a]
        # Convert to grid indices
        ix = np.clip(np.round(norm[:, 0] * (r - 1)).astype(np.int32), 0, r - 1)
        iy = np.clip(np.round(norm[:, 1] * (r - 1)).astype(np.int32), 0, r - 1)
        iz = np.clip(np.round(norm[:, 2] * (r - 1)).astype(np.int32), 0, r - 1)
        return self._blob_opacity[ix, iy, iz] <= 0.0

    def _init_lattice(self):
        """Initialize particles, constrained to path blob if available.

        Particles spawned in the margin zone get a baked blob_alpha < 1.0.
        """
        margin = 0.05
        g = self.particle_grid
        self.blob_alpha = np.ones(self.particle_count, dtype=np.float32)

        if self._blob_opacity is None:
            # Original bounding-box lattice
            idx = 0
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
        else:
            # Blob-constrained: uniform random sampling inside core + margin
            idx = 0
            rng = np.random.default_rng(42)
            batch = max(self.particle_count * 4, 10000)
            while idx < self.particle_count:
                candidates = np.empty((batch, 3), dtype=np.float32)
                for a in range(3):
                    lo = self.axis_min[a] + margin * self.span[a]
                    hi = self.axis_max[a] - margin * self.span[a]
                    candidates[:, a] = rng.uniform(lo, hi, batch).astype(np.float32)
                for c in range(len(candidates)):
                    if idx >= self.particle_count:
                        break
                    x, y, z = candidates[c]
                    opa = self._sample_blob_opacity(x, y, z)
                    if opa > 0.0:
                        self.pos[idx] = candidates[c]
                        self.blob_alpha[idx] = opa
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

    def sample_velocity_batch(self, positions: np.ndarray,
                              out: Optional[np.ndarray] = None) -> np.ndarray:
        """Trilinear interpolation for all particles using scipy's C backend.

        Args:
            positions: (N, 3) array of particle positions.
            out: Optional pre-allocated (N, 3) output buffer.

        Returns:
            (N, 3) array of velocities.
        """
        from scipy.ndimage import map_coordinates

        G = self.grid_size
        N = len(positions)

        if out is not None and out.shape == (N, 3):
            result = out
        else:
            result = np.zeros((N, 3), dtype=np.float32)

        # Normalize to grid coordinates (reuse buffer if possible)
        if N == self.particle_count:
            normalized = self._buf_normalized
        else:
            normalized = np.zeros_like(positions)
        for a in range(3):
            if self.span[a] > 0:
                normalized[:, a] = (positions[:, a] - self.axis_min[a]) / self.span[a] * (G - 1)

        # scipy's map_coordinates expects (ndim, N_points) coordinate array
        coords = normalized.T  # (3, N) — no copy, just transpose view

        # Interpolate each velocity component via C-implemented trilinear
        for c in range(3):
            result[:, c] = map_coordinates(
                self.velocity_grid[:, :, :, c], coords,
                order=1, mode='nearest',
            )

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

    def step(self, dt_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Advance all particles by one frame.

        Uses pre-allocated buffers to avoid per-frame memory allocation.

        Args:
            dt_scale: Multiplier for the time step (1.0 = normal 60 Hz frame).
                      Use >1 to compensate for lower frame rates.

        Returns:
            (positions, colors, alphas, speeds) — all (N, ...) arrays.
            positions is a copy; colors/alphas/speeds are internal buffers
            valid until the next step() call.
        """
        vel = self._buf_velocities
        speeds = self._buf_speeds
        colors = self._buf_colors
        alphas = self._buf_alphas

        # Compute velocities into pre-allocated buffer
        self.sample_velocity_batch(self.pos, out=vel)

        # Compute speeds without allocating new array
        np.einsum('ij,ij->i', vel, vel, out=speeds)
        np.sqrt(speeds, out=speeds)

        # Advect (dt_scale compensates for variable frame rate)
        self.pos += vel * (self.DT * self.speed_multiplier * dt_scale)
        self._wrap_all()

        # Kill particles that escaped the blob (vectorized check)
        if self._blob_opacity is not None and self.blob_enabled:
            escaped = self._check_outside_blob_batch(self.pos)
            if np.any(escaped):
                self.age[escaped] = self.LIFESPAN  # force respawn

        # Age and respawn
        self.age += 1
        expired = self.age >= self.LIFESPAN
        if np.any(expired):
            self._respawn(np.where(expired)[0])

        # Speed-based colors (Turbo colormap) into pre-allocated buffer
        max_speed = speeds.max() if speeds.max() > 0 else 1.0
        turbo_colormap(speeds / max_speed, out=colors)

        # Alpha based on age (fade in/out) into pre-allocated buffer
        age_frac = self.age.astype(np.float32) / self.LIFESPAN
        alphas[:] = 1.0
        fade_in = age_frac < 0.1
        alphas[fade_in] = age_frac[fade_in] / 0.1
        fade_out = age_frac > 0.8
        alphas[fade_out] = (1.0 - age_frac[fade_out]) / 0.2
        np.clip(alphas, 0, 1, out=alphas)
        alphas *= self.blob_alpha

        # Copy positions (renderer needs stable reference)
        np.copyto(self._buf_pos_out, self.pos)
        return self._buf_pos_out, colors, alphas, speeds

    def is_outside_blob(self, x, y, z) -> bool:
        """Check if a world-space point is outside the path blob."""
        return self._path_mask is not None and not self._point_in_blob_world(x, y, z)

    def get_blob_surface_points(self) -> Optional[np.ndarray]:
        """Return world-space points on the blob surface for debug viz.

        Samples all mask-True cells and returns their centres.
        """
        if self._path_mask is None:
            return None
        r = self._mask_res
        indices = np.argwhere(self._path_mask)  # (N, 3) of (i,j,k)
        if len(indices) == 0:
            return None
        # Convert grid indices to world coords
        pts = np.zeros((len(indices), 3), dtype=np.float32)
        for a in range(3):
            pts[:, a] = self.axis_min[a] + (indices[:, a] / (r - 1)) * self.span[a]
        return pts

    def advance_ball(self, dt_scale: float = 1.0) -> np.ndarray:
        """Advance the ball probe by one step along the flow field.

        Uses 0.8 * DT for slower movement (matching Android's dtSlow).
        Decelerates heavily when outside the path blob.

        Args:
            dt_scale: Multiplier for the time step (1.0 = normal 60 Hz frame).

        Returns:
            New ball position (3,) array.
        """
        v = self.sample_velocity(*self.ball_pos)
        dt_slow = self.DT * 0.8 * self.speed_multiplier * dt_scale

        # Slow down to 10% speed when outside blob
        if self.is_outside_blob(*self.ball_pos):
            dt_slow *= 0.1

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
        N = self.particle_count
        self.pos = np.zeros((N, 3), dtype=np.float32)
        self.age = np.zeros(N, dtype=np.int32)
        # Reallocate step() buffers for new particle count
        self._buf_velocities = np.zeros((N, 3), dtype=np.float32)
        self._buf_speeds = np.zeros(N, dtype=np.float32)
        self._buf_colors = np.zeros((N, 3), dtype=np.float32)
        self._buf_alphas = np.ones(N, dtype=np.float32)
        self._buf_pos_out = np.zeros((N, 3), dtype=np.float32)
        self._buf_normalized = np.zeros((N, 3), dtype=np.float32)
        self._init_lattice()
        self.orig_pos = self.pos.copy()  # save BEFORE pre-integration
        self._dry_integrate(self.PRE_AGE)
        if self._blob_opacity is not None and self.blob_enabled:
            for i in range(self.particle_count):
                if not self._point_in_blob_world(*self.pos[i]):
                    self.pos[i] = self.orig_pos[i]
        self.age = np.random.randint(0, self.LIFESPAN, self.particle_count, dtype=np.int32)

    def set_path_points(self, path_points: np.ndarray):
        """Update the path mask and reinitialize particles."""
        if path_points is not None and len(path_points) >= 2:
            self._build_path_mask(path_points)
        else:
            self._path_mask = None
            self._blob_opacity = None
        # Reinitialize with current grid size
        self.pos = np.zeros((self.particle_count, 3), dtype=np.float32)
        self.age = np.zeros(self.particle_count, dtype=np.int32)
        self._init_lattice()
        self.orig_pos = self.pos.copy()  # save BEFORE pre-integration
        self._dry_integrate(self.PRE_AGE)
        if self._blob_opacity is not None and self.blob_enabled:
            for i in range(self.particle_count):
                if not self._point_in_blob_world(*self.pos[i]):
                    self.pos[i] = self.orig_pos[i]
        self.age = np.random.randint(0, self.LIFESPAN, self.particle_count, dtype=np.int32)

    def start_ball_flow(self):
        """Start ball following the flow field."""
        self.ball_flowing = True
        self.ball_trail.clear()
        self.ball_trail.append(self.ball_pos.copy())

    def stop_ball_flow(self):
        """Stop ball following the flow field."""
        self.ball_flowing = False

    def build_score_grid(self, data_points: np.ndarray, data_scores: np.ndarray):
        """Precompute a 3D score grid for fast particle-score lookup.

        Each grid cell gets the score of the nearest data point.
        Uses the same grid resolution as the velocity field.

        Args:
            data_points: (M, 3) data point positions.
            data_scores: (M,) score values per data point.
        """
        G = self.grid_size
        # Build grid cell centers in world coords
        grid_coords = np.zeros((G, G, G, 3), dtype=np.float32)
        for a in range(3):
            linspace = np.linspace(self.axis_min[a], self.axis_max[a], G)
            if a == 0:
                grid_coords[:, :, :, 0] = linspace[:, None, None]
            elif a == 1:
                grid_coords[:, :, :, 1] = linspace[None, :, None]
            else:
                grid_coords[:, :, :, 2] = linspace[None, None, :]
        # Flatten to (G³, 3) and find nearest data point for each cell
        flat = grid_coords.reshape(-1, 3)
        # Chunked to avoid huge memory: process 1000 cells at a time
        score_flat = np.zeros(len(flat), dtype=np.float32)
        chunk = 2000
        for i in range(0, len(flat), chunk):
            batch = flat[i:i + chunk]
            dists = np.linalg.norm(batch[:, None, :] - data_points[None, :, :], axis=2)
            nearest = np.argmin(dists, axis=1)
            score_flat[i:i + chunk] = data_scores[nearest]
        self._score_grid = score_flat.reshape(G, G, G)

    def _sample_scalar_grid(self, grid: np.ndarray,
                            positions: np.ndarray) -> np.ndarray:
        """Trilinear interpolation on a scalar (G,G,G) grid using scipy C backend."""
        from scipy.ndimage import map_coordinates
        normalized = np.zeros_like(positions)
        for a in range(3):
            if self.span[a] > 0:
                normalized[:, a] = ((positions[:, a] - self.axis_min[a])
                                    / self.span[a] * (grid.shape[0] - 1))
        return map_coordinates(grid, normalized.T, order=1, mode='nearest'
                               ).astype(np.float32)

    def sample_score_batch(self, positions: np.ndarray) -> np.ndarray:
        """Sample precomputed score grid at particle positions.

        Returns (N,) interpolated score values, or 0.5 if no score grid.
        """
        if self._score_grid is None:
            return np.full(len(positions), 0.5, dtype=np.float32)
        return self._sample_scalar_grid(self._score_grid, positions)

    def sample_confidence_batch(self, positions: np.ndarray) -> np.ndarray:
        """Sample confidence grid at particle positions.

        Returns (N,) values in [0, 1] where 1 = high confidence.
        Returns all-ones if no confidence grid.
        """
        if self._confidence_grid is None:
            return np.ones(len(positions), dtype=np.float32)
        return self._sample_scalar_grid(self._confidence_grid, positions)

    # ═══════════════════════════════════════════════════
    #  Attractor detection
    # ═══════════════════════════════════════════════════

    # Bump this version when the attractor detection algorithm changes
    # to auto-invalidate cached results.
    _ATTRACTOR_CACHE_VERSION = 15

    @staticmethod
    def _sensitivity_params(s: float) -> dict:
        """Map sensitivity (0.0–1.0) to attractor detection parameters.

        At s=0.7 (default) the values match the hardcoded defaults.
        Lower sensitivity shrinks large basins first (via max_basin_frac
        and basin_thresh_pct), then removes weak attractors (peak_floor_pct).
        """
        s = max(0.0, min(1.0, s))
        # Interpolation helper: s=0 → val_lo, s=0.7 → val_mid, s=1.0 → val_hi
        def _lerp(val_lo, val_mid, val_hi):
            if s <= 0.7:
                t = s / 0.7
                return val_lo + t * (val_mid - val_lo)
            else:
                t = (s - 0.7) / 0.3
                return val_mid + t * (val_hi - val_mid)

        # max_basin_frac uses a power curve for steeper drop at low
        # sensitivity — large basins shrink first, small ones are spared.
        t_basin = s / 0.7 if s <= 0.7 else 1.0 + (s - 0.7) / 0.3
        if s <= 0.7:
            # Quadratic: drops fast at first, then flattens near default
            max_bf = 0.002 + (0.02 - 0.002) * (s / 0.7) ** 1.5
        else:
            max_bf = 0.02 + (0.05 - 0.02) * ((s - 0.7) / 0.3)

        return {
            'peak_floor_pct': _lerp(0.18, 0.05, 0.01),
            'basin_thresh_pct': _lerp(0.70, 0.35, 0.12),
            'max_basin_frac': max_bf,
            'speed_gate_mult': _lerp(1.2, 2.0, 4.0),
        }

    def _attractor_fingerprint(self, sensitivity: float = 0.7) -> str:
        """Build a fingerprint from the velocity grid + sensitivity."""
        import hashlib
        h = hashlib.sha256()
        h.update(f"v{self._ATTRACTOR_CACHE_VERSION}".encode())
        h.update(self.velocity_grid.tobytes()[:4096])  # first 4KB is enough
        h.update(f"|G={self.grid_size}|s={sensitivity:.3f}|".encode())
        return h.hexdigest()[:16]

    def save_attractors(self, path: str, attractors: list, sensitivity: float = 0.7):
        """Cache attractor results to disk as .npz for instant reload.

        Args:
            path: Base path (without extension).  Creates {path}_attractors.npz.
        """
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        npz_path = path + "_attractors.npz"
        arrays = {}
        meta_list = []
        for i, att in enumerate(attractors):
            arrays[f"basin_{i}"] = att['basin_mask']
            arrays[f"position_{i}"] = att['position']
            meta_list.append({
                'strength': att['strength'],
                'divergence': att['divergence'],
                'basin_size': att['basin_size'],
                'basin_fraction': att['basin_fraction'],
                'mean_score': att['mean_score'],
            })
        import json as _json
        meta_obj = {
            'attractors': meta_list,
            'fingerprint': self._attractor_fingerprint(sensitivity),
        }
        arrays['_meta'] = np.array([_json.dumps(meta_obj)], dtype=object)
        np.savez_compressed(npz_path, **arrays)
        print(f"[ATTRACTORS] Cached {len(attractors)} attractor(s) to {npz_path}")

    def load_attractors(self, path: str, sensitivity: float = 0.7) -> Optional[list]:
        """Load cached attractor results from disk.

        Validates the fingerprint against the current velocity grid so
        stale caches are automatically discarded when data changes.

        Args:
            path: Base path (without extension). Looks for {path}_attractors.npz.

        Returns:
            List of attractor dicts, or None if cache missing/stale.
        """
        npz_path = path + "_attractors.npz"
        import os
        if not os.path.exists(npz_path):
            return None
        try:
            data = np.load(npz_path, allow_pickle=True)
            import json as _json
            meta_obj = _json.loads(str(data['_meta'][0]))

            # Support both old format (list) and new format (dict with fingerprint)
            if isinstance(meta_obj, list):
                # Old cache without fingerprint — discard
                print("[ATTRACTORS] Old cache format without fingerprint — recomputing")
                return None
            meta_list = meta_obj['attractors']
            cached_fp = meta_obj.get('fingerprint', '')

            # Validate fingerprint
            current_fp = self._attractor_fingerprint(sensitivity)
            if cached_fp != current_fp:
                print(f"[ATTRACTORS] Cache fingerprint mismatch — recomputing")
                return None

            attractors = []
            for i, meta in enumerate(meta_list):
                attractors.append({
                    'basin_mask': data[f'basin_{i}'],
                    'position': data[f'position_{i}'],
                    'strength': meta['strength'],
                    'divergence': meta['divergence'],
                    'basin_size': meta['basin_size'],
                    'basin_fraction': meta['basin_fraction'],
                    'mean_score': meta['mean_score'],
                })
            print(f"[ATTRACTORS] Loaded {len(attractors)} cached attractor(s) from {npz_path}")
            return attractors
        except Exception as e:
            print(f"[ATTRACTORS] Cache load failed: {e}")
            return None

    def find_attractors(self, score_grid: Optional[np.ndarray] = None,
                        min_confidence: float = 0.3,
                        sim_steps: int = 800,
                        sample_interval: int = 8,
                        cache_path: Optional[str] = None,
                        sensitivity: float = 0.7) -> list:
        """Detect flow attractors using residence density + local peak finding.

        Three-stage approach:
        1. Simulate particles with renderer-compatible rules (wrapping,
           blob escape) but LONGER lifespan to reduce spawn bias.
           Accumulate occupancy = where particles spend time.
        2. Find LOCAL maxima in occupancy (not global threshold) so
           weaker attractors aren't erased by dominant ones.
        3. Validate each peak with negative divergence (field structure)
           so we're not just finding traffic jams.

        basin_mask = high-occupancy region around each peak (what gets drawn).

        Args:
            score_grid: Optional (G,G,G) score values for basin scoring.
            min_confidence: Minimum confidence to consider a cell.
            cache_path: If provided, try loading from / saving to this path.
            sim_steps: Total simulation steps.
            sample_interval: Sample occupancy every N steps.
            sensitivity: 0.0–1.0 controls attractor count & basin size.
                Default 0.6 matches hardcoded defaults.
                Lower = fewer, smaller basins (large ones shrink first).
                Higher = more, larger basins.
        """
        # ── Derive parameters from sensitivity ──
        sp = self._sensitivity_params(sensitivity)

        # ── Try loading from cache first ──
        if cache_path is not None:
            cached = self.load_attractors(cache_path, sensitivity)
            if cached is not None:
                return cached

        from scipy import ndimage
        from scipy.ndimage import maximum_filter, label

        G = self.grid_size
        vg = self.velocity_grid

        # ── Blob mask resampled to velocity grid ──
        # Use the CORE mask (not the soft-margin opacity) so attractor
        # basins don't expand into the fade-out margin zone.
        blob_valid = np.ones((G, G, G), dtype=bool)
        if self._path_mask is not None:
            res = self._path_mask.shape[0]
            bi = np.round(np.arange(G) / (G - 1) * (res - 1)).astype(int)
            bi = np.clip(bi, 0, res - 1)
            blob_valid = self._path_mask[np.ix_(bi, bi, bi)]

        conf = self._confidence_grid if self._confidence_grid is not None \
            else np.ones((G, G, G), dtype=np.float32)
        valid = (conf >= min_confidence) & blob_valid

        if np.sum(valid) < 4:
            return []

        # ═══════════════════════════════════════════════════
        # STAGE 1: Simulate particles, measure occupancy
        # Uses renderer rules but LONGER lifespan (5× normal)
        # to let particles reach distant attractors.
        # ═══════════════════════════════════════════════════

        LONG_LIFESPAN = self.LIFESPAN * 5  # 285 frames vs 57

        N_test = min(20000, max(5000, int(np.sum(valid)) * 8))
        rng = np.random.default_rng(42)

        # Seed inside blob
        positions = np.zeros((N_test, 3), dtype=np.float32)
        idx_p = 0
        batch = 2000
        while idx_p < N_test:
            candidates = np.zeros((batch, 3), dtype=np.float32)
            for a in range(3):
                candidates[:, a] = rng.uniform(
                    self.axis_min[a], self.axis_max[a], batch
                ).astype(np.float32)
            for c in range(len(candidates)):
                if idx_p >= N_test:
                    break
                if self._point_in_blob_world(*candidates[c]):
                    positions[idx_p] = candidates[c]
                    idx_p += 1

        # Pre-integrate
        for _ in range(self.PRE_AGE):
            vel = self.sample_velocity_batch(positions)
            positions += vel * self.DT
            for a in range(3):
                if self.span[a] > 0:
                    positions[:, a] = self.axis_min[a] + \
                        (positions[:, a] - self.axis_min[a]) % self.span[a]

        orig_pos = positions.copy()
        ages = rng.integers(0, LONG_LIFESPAN, N_test, dtype=np.int32)

        # Run simulation, accumulate occupancy
        occupancy = np.zeros((G, G, G), dtype=np.float32)
        n_samples = 0
        warmup = 150  # let particles spread out before counting

        for step_i in range(sim_steps):
            vel = self.sample_velocity_batch(positions)
            positions += vel * self.DT

            # Wrap at boundaries (same as MDN)
            for a in range(3):
                if self.span[a] > 0:
                    positions[:, a] = self.axis_min[a] + \
                        (positions[:, a] - self.axis_min[a]) % self.span[a]

            # Blob escape → force respawn (same as MDN)
            if self._blob_opacity is not None:
                escaped = self._check_outside_blob_batch(positions)
                if np.any(escaped):
                    ages[escaped] = LONG_LIFESPAN

            # Age and respawn with LONGER lifespan
            ages += 1
            expired = ages >= LONG_LIFESPAN
            if np.any(expired):
                exp_idx = np.where(expired)[0]
                positions[exp_idx] = orig_pos[exp_idx]
                ages[exp_idx] = 0

            # Sample occupancy after warmup
            if step_i >= warmup and step_i % sample_interval == 0:
                gi = np.zeros((N_test, 3), dtype=int)
                for a in range(3):
                    gi[:, a] = np.clip(
                        ((positions[:, a] - self.axis_min[a])
                         / self.span[a] * (G - 1)).astype(int),
                        0, G - 1)
                np.add.at(occupancy, (gi[:, 0], gi[:, 1], gi[:, 2]), 1.0)
                n_samples += 1

        if n_samples == 0:
            return []

        occupancy /= n_samples
        occupancy = ndimage.gaussian_filter(occupancy, sigma=0.8)
        occupancy[~valid] = 0.0

        # Speed grid
        speed = np.linalg.norm(vg, axis=3)
        speed_ref = max(float(np.percentile(speed[valid], 90)), 1e-8)
        speed_factor = 1.0 / (1.0 + speed / speed_ref)

        # Divergence (for validation + info)
        dvx_dx = np.gradient(vg[:, :, :, 0], axis=0)
        dvy_dy = np.gradient(vg[:, :, :, 1], axis=1)
        dvz_dz = np.gradient(vg[:, :, :, 2], axis=2)
        divergence = dvx_dx + dvy_dy + dvz_dz

        # Combined residence score
        residence_score = occupancy * speed_factor
        residence_score[~valid] = 0.0

        rs_max = float(np.max(residence_score))
        if rs_max < 1e-10:
            return []

        # ═══════════════════════════════════════════════════
        # STAGE 2: Find LOCAL maxima (not global threshold)
        # ═══════════════════════════════════════════════════

        neighborhood = 5
        local_max = maximum_filter(residence_score, size=neighborhood)
        is_peak = (residence_score == local_max) & (residence_score > 0)

        # Absolute floor: peak_floor_pct of max OR 80th percentile (whichever lower)
        abs_floor = min(
            rs_max * sp['peak_floor_pct'],
            float(np.percentile(residence_score[valid], 80))
        )
        is_peak = is_peak & (residence_score > abs_floor)

        peak_coords = np.argwhere(is_peak)
        if len(peak_coords) == 0:
            return []

        peak_scores = np.array([residence_score[tuple(p)] for p in peak_coords])
        order = np.argsort(-peak_scores)
        peak_coords = peak_coords[order]
        peak_scores = peak_scores[order]

        # Reject peaks where the local flow is not actually converging.
        # A real attractor must have negative divergence in its neighborhood
        # (not just at the exact peak cell, which can be noisy).
        # Smooth divergence and require neighborhood mean < 0.
        div_smooth = ndimage.gaussian_filter(divergence, sigma=1.5)
        real_peaks = []
        for pc in peak_coords:
            pi, pj, pk = int(pc[0]), int(pc[1]), int(pc[2])
            # 3×3×3 neighborhood mean divergence
            sl = tuple(slice(max(0, c - 1), min(G, c + 2)) for c in (pi, pj, pk))
            neigh_div = float(np.mean(div_smooth[sl]))
            if neigh_div < 0:
                real_peaks.append(pc)
        peak_coords = np.array(real_peaks) if real_peaks else np.empty((0, 3), dtype=int)
        peak_scores = np.array([residence_score[tuple(p)] for p in peak_coords]) \
            if len(peak_coords) > 0 else np.array([])

        if len(peak_coords) == 0:
            return []

        # Merge peaks within 3 cells
        merge_dist = 3.0
        kept = []
        for pc in peak_coords:
            too_close = False
            for kc in kept:
                if np.linalg.norm(pc.astype(float) - kc.astype(float)) < merge_dist:
                    too_close = True
                    break
            if not too_close:
                kept.append(pc)
        peak_coords = np.array(kept) if kept else np.empty((0, 3), dtype=int)

        if len(peak_coords) == 0:
            return []

        # ═══════════════════════════════════════════════════
        # STAGE 3: Build basins using divergence-modulated expansion
        #
        # Key insight: "parking lots" (real sinks) have negative
        # divergence, while "rivers" (slow flow) have ~zero divergence.
        # Instead of separate hacks (absolute floor, divergence penalty),
        # we build a SINGLE basin expansion score that naturally gives
        # generous basins to real sinks and chokes off rivers:
        #
        #   convergence = clamp(-div / div_ref, 0, 1)
        #   basin_score = residence_score * (0.1 + 0.9 * convergence)
        #
        # Near a sink:  convergence≈1 → basin_score ≈ residence_score
        # In a river:   convergence≈0 → basin_score ≈ 0.1 * residence
        # Near a source: convergence=0 → basin_score ≈ 0.1 * residence
        #
        # Then flood-fill each peak's basin in basin_score space.
        # Rivers self-limit because their basin_score is 10× lower.
        # ═══════════════════════════════════════════════════

        # Normalized divergence: scale so typical negative div → ~1.0
        div_ref = max(float(np.percentile(np.abs(divergence[valid]), 90)), 1e-8)
        convergence = np.clip(-divergence / div_ref, 0.0, 1.0)  # 1=sink, 0=source/neutral

        # Basin expansion score: occupancy weighted by convergence
        basin_score = residence_score * (0.1 + 0.9 * convergence)
        basin_score[~valid] = 0.0

        bs_max = float(np.max(basin_score))
        if bs_max < 1e-10:
            bs_max = rs_max  # fallback

        # Absolute floor: median of valid basin_scores (or 8% of max).
        # This prevents shallow hills from claiming huge territories.
        # The local percentage handles sharp peaks; the floor handles flat ones.
        valid_bs = basin_score[valid]
        abs_basin_floor = max(
            float(np.median(valid_bs)),
            bs_max * 0.08,
        )

        attractors_out = []
        n_valid = int(np.sum(valid))
        claimed = np.zeros((G, G, G), dtype=bool)

        for pc in peak_coords:
            pi, pj, pk = int(pc[0]), int(pc[1]), int(pc[2])
            peak_val = float(residence_score[pi, pj, pk])
            peak_bs = float(basin_score[pi, pj, pk])
            peak_div = float(divergence[pi, pj, pk])

            # ── Per-attractor speed gate ──
            # Basin mesh = where particles SETTLE. Only cells with speed
            # close to the peak's own (slow) speed belong in the basin.
            # Cells much faster are approach corridors, not settling zones.
            # Gate = 2× peak neighborhood speed, floored so we don't
            # choke basins where the entire field is uniformly slow.
            sl = tuple(slice(max(0, c - 1), min(G, c + 2)) for c in (pi, pj, pk))
            peak_speed = float(np.mean(speed[sl]))  # 3×3×3 neighborhood avg
            speed_gate = max(peak_speed * sp['speed_gate_mult'], speed_ref * 0.08)
            basin_slow = speed <= speed_gate

            # Basin threshold: HIGHER of local percentage and absolute floor.
            # - Sharp peaks: local 20% is high → reasonable basin size
            # - Shallow hills: local 20% is tiny → abs floor kicks in → tight basin
            local_thresh = max(peak_bs * sp['basin_thresh_pct'], abs_basin_floor)

            # Adaptive tightening: if basin exceeds 2% of valid cells,
            # raise threshold until it fits.  This prevents flat, uniformly
            # convergent flow fields from producing basins that swallow
            # half the grid.
            max_basin_cells = max(int(n_valid * sp['max_basin_frac']), 20)
            for _tighten in range(8):
                basin_candidates = (
                    (basin_score >= local_thresh) & valid
                    & (~claimed) & basin_slow
                )
                basin_labeled, _ = label(basin_candidates)
                peak_label = basin_labeled[pi, pj, pk]
                if peak_label == 0:
                    break
                basin_mask = basin_labeled == peak_label
                basin_size = int(np.sum(basin_mask))
                if basin_size <= max_basin_cells:
                    break
                # Tighten: raise threshold toward peak value
                local_thresh = local_thresh + (peak_bs - local_thresh) * 0.3

            if peak_label == 0:
                continue

            basin_size = int(np.sum(basin_mask))
            if basin_size < 5:
                continue

            claimed |= basin_mask

            position = np.array([
                self.axis_min[0] + pi / (G - 1) * self.span[0],
                self.axis_min[1] + pj / (G - 1) * self.span[1],
                self.axis_min[2] + pk / (G - 1) * self.span[2],
            ], dtype=np.float32)

            basin_fraction = basin_size / n_valid if n_valid > 0 else 0.0

            mean_score = None
            if score_grid is not None and score_grid.shape == (G, G, G):
                bs = score_grid[basin_mask]
                if len(bs) > 0:
                    mean_score = float(np.mean(bs))

            # Strength from basin_score (already incorporates divergence)
            attractors_out.append({
                'position': position,
                'strength': float(peak_bs / bs_max) if bs_max > 0 else 0.0,
                'divergence': peak_div,
                'basin_mask': basin_mask,
                'basin_size': basin_size,
                'basin_fraction': basin_fraction,
                'mean_score': mean_score,
            })

        # ── Merge encapsulated attractors ──
        # When one attractor's bounding box is contained inside another's,
        # they look like nested shells.  Merge by absorbing the weaker
        # one's basin into the stronger one.
        merged = True
        while merged:
            merged = False
            for i in range(len(attractors_out)):
                if attractors_out[i] is None:
                    continue
                bi = attractors_out[i]['basin_mask']
                ci = np.argwhere(bi)
                if len(ci) == 0:
                    continue
                mni, mxi = ci.min(0), ci.max(0)
                for j in range(len(attractors_out)):
                    if i == j or attractors_out[j] is None:
                        continue
                    bj = attractors_out[j]['basin_mask']
                    cj = np.argwhere(bj)
                    if len(cj) == 0:
                        continue
                    mnj, mxj = cj.min(0), cj.max(0)
                    # Check if j's bbox is inside i's bbox (or vice versa)
                    j_inside_i = np.all(mnj >= mni) and np.all(mxj <= mxi)
                    i_inside_j = np.all(mni >= mnj) and np.all(mxi <= mxj)
                    if j_inside_i or i_inside_j:
                        # Keep the stronger one, absorb the other's basin
                        si = attractors_out[i]['strength']
                        sj = attractors_out[j]['strength']
                        keep, drop = (i, j) if si >= sj else (j, i)
                        attractors_out[keep]['basin_mask'] = (
                            attractors_out[keep]['basin_mask'] |
                            attractors_out[drop]['basin_mask']
                        )
                        attractors_out[keep]['basin_size'] = int(
                            np.sum(attractors_out[keep]['basin_mask'])
                        )
                        attractors_out[keep]['basin_fraction'] = (
                            attractors_out[keep]['basin_size'] / n_valid
                            if n_valid > 0 else 0.0
                        )
                        attractors_out[drop] = None
                        merged = True
                        break
                if merged:
                    break
            attractors_out = [a for a in attractors_out if a is not None]

        attractors_out.sort(key=lambda a: a['strength'], reverse=True)

        # Cap at 8, but always keep at least the strongest one
        attractors_out = attractors_out[:8]

        # Guarantee at least 1 attractor: if all were filtered, take the
        # strongest peak and give it a minimal basin
        if not attractors_out and len(peak_coords) > 0:
            pc = peak_coords[0]
            pi, pj, pk = int(pc[0]), int(pc[1]), int(pc[2])
            basin_mask = np.zeros((G, G, G), dtype=bool)
            # 3×3×3 cube around peak
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    for dk in range(-1, 2):
                        ni, nj, nk = pi+di, pj+dj, pk+dk
                        if 0 <= ni < G and 0 <= nj < G and 0 <= nk < G:
                            if valid[ni, nj, nk]:
                                basin_mask[ni, nj, nk] = True
            position = np.array([
                self.axis_min[0] + pi / (G - 1) * self.span[0],
                self.axis_min[1] + pj / (G - 1) * self.span[1],
                self.axis_min[2] + pk / (G - 1) * self.span[2],
            ], dtype=np.float32)
            attractors_out.append({
                'position': position,
                'strength': 1.0,
                'divergence': float(divergence[pi, pj, pk]),
                'basin_mask': basin_mask,
                'basin_size': int(np.sum(basin_mask)),
                'basin_fraction': int(np.sum(basin_mask)) / n_valid if n_valid > 0 else 0.0,
                'mean_score': None,
            })

        print(f"[ATTRACTORS] {N_test} particles, {sim_steps} steps "
              f"(lifespan={LONG_LIFESPAN}), {n_samples} samples")
        print(f"[ATTRACTORS] Occupancy max={float(np.max(occupancy)):.1f}, "
              f"div_ref={div_ref:.4f}, abs_floor={abs_basin_floor:.3f}, "
              f"speed_ref={speed_ref:.4f}, "
              f"peaks={len(peak_coords)}, output={len(attractors_out)}")

        # ── Save to cache ──
        if cache_path is not None and attractors_out:
            try:
                self.save_attractors(cache_path, attractors_out, sensitivity)
            except Exception as e:
                print(f"[ATTRACTORS] Cache save failed: {e}")

        return attractors_out

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
