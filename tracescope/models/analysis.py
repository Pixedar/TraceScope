"""
Result models returned by the analysis pipeline.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np

from tracescope.models.trace import TraceSession


@dataclass
class ClusterResult:
    """Clustering output.

    Attributes:
        n_clusters: Number of clusters found.
        labels: Cluster label per entry (length N).
        clusters: Per-cluster info dicts with keys:
                  cluster_id, indices, centroid.
    """

    n_clusters: int
    labels: List[int]
    clusters: List[dict]


@dataclass
class AxisInfo:
    """PCA axis information for the 3D projection.

    Attributes:
        axes: (3, D) array – principal directions in high-dim space.
        lengths: Variance explained per axis (length 3).
        labels: LLM-generated semantic labels (length 3).
        min_point_idx: Index of the point at the negative extreme per axis.
        max_point_idx: Index of the point at the positive extreme per axis.
        min_cluster_idx: Cluster at negative extreme per axis.
        max_cluster_idx: Cluster at positive extreme per axis.
    """

    axes: np.ndarray
    lengths: List[float]
    labels: List[str] = field(default_factory=lambda: ["Axis 1", "Axis 2", "Axis 3"])
    min_point_idx: List[int] = field(default_factory=list)
    max_point_idx: List[int] = field(default_factory=list)
    min_cluster_idx: List[int] = field(default_factory=list)
    max_cluster_idx: List[int] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Full result of the analysis pipeline.

    Attributes:
        session: The analyzed TraceSession.
        embedding_model: Name of the embedding model used.
        embeddings: (N, D) high-dim embedding matrix.
        clusters: Clustering result.
        projected_3d: (N, 3) projected coordinates.
        axis_info: PCA axis information with semantic labels.
        cluster_labels: LLM-generated description per cluster.
        flow_model_trained: Whether the flow model was successfully trained.
        segments: Optional segmentation result.
    """

    session: TraceSession
    embedding_model: str
    embeddings: np.ndarray
    clusters: ClusterResult
    projected_3d: np.ndarray
    axis_info: AxisInfo
    cluster_labels: List[str] = field(default_factory=list)
    flow_model_trained: bool = False
    segments: Optional[List[dict]] = None
    # Flow field data (from MDN model)
    velocity_grid: Optional[np.ndarray] = None       # (40,40,40,3) velocity field
    confidence_grid: Optional[np.ndarray] = None     # (40,40,40) MDN confidence [0,1]
    axis_min: Optional[np.ndarray] = None             # bounding box min [3]
    axis_max: Optional[np.ndarray] = None             # bounding box max [3]
    mdn_simulate: Optional[Any] = None                # MDN simulate() callable
    fitted_reducer: Optional[Any] = None               # fitted UMAP/tSNE for .transform()
    # Cluster geometry in 3D projected space
    cluster_centroids_3d: Optional[np.ndarray] = None # (K,3) cluster centers
    max_cluster_distance: float = 1.0                 # max pairwise dist between centroids
    cache_path: Optional[str] = None                  # base cache path for attractor caching
    flow_mode: str = "mdn"                             # flow model type used ("mdn" or "rbf")
    seed: int = 42                                     # global seed used by every randomized stage
    deterministic: bool = True                         # whether RNGs were seeded this run

    @property
    def n_entries(self) -> int:
        return len(self.session)

    @property
    def score_channels(self) -> List[str]:
        """Return sorted list of all score channel names available in the data."""
        return self.session.score_channels

    def get_entry_scores(self, channel: str) -> List[Optional[float]]:
        """Return score values for a channel across all entries (None if missing)."""
        return [
            e.scores.get(channel) for e in self.session.entries
        ]

    def get_path_scores(self, channel: str) -> dict:
        """Return {path_id: score} for a channel from path_scores."""
        return {
            pid: scores[channel]
            for pid, scores in self.session.path_scores.items()
            if channel in scores
        }

    def find_attractors(self, score_channel: Optional[str] = None) -> list:
        """Find flow attractors in the velocity field.

        Detects regions where flow converges and particles accumulate,
        distinguishing real attractors from boundary effects and dead zones.
        Computed on demand from the velocity grid (fast, ~0.1s).

        Args:
            score_channel: Optional score channel name to compute mean
                score within each attractor's basin (e.g. "solved" to see
                if the attractor is positive or negative).

        Returns:
            List of attractor dicts with keys:
                position, strength, divergence, basin_mask,
                basin_size, basin_fraction, mean_score
        """
        if self.velocity_grid is None:
            return []
        from tracescope.visualization.flow_field import FlowFieldSystem
        flow = FlowFieldSystem(
            self.velocity_grid, self.axis_min, self.axis_max,
            particle_grid=2,  # minimal — we only need the grid, not particles
            confidence_grid=self.confidence_grid,
        )
        # Build score grid if requested
        score_grid = None
        if score_channel and self.score_channels:
            if score_channel in self.score_channels:
                from tracescope.visualization.gl_renderer import FlowRenderer
                # Get normalized scores per entry
                raw = self.get_entry_scores(score_channel)
                path_scores = self.get_path_scores(score_channel)
                vals = []
                for i, entry in enumerate(self.session.entries):
                    v = raw[i]
                    if v is None and entry.path_id is not None:
                        v = path_scores.get(entry.path_id)
                    vals.append(float(v) if v is not None else float('nan'))
                arr = np.array(vals, dtype=np.float32)
                valid = ~np.isnan(arr)
                if valid.any():
                    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
                    if hi - lo > 1e-8:
                        arr[valid] = (arr[valid] - lo) / (hi - lo)
                    else:
                        arr[valid] = 0.5
                    arr[~valid] = 0.5
                    flow.build_score_grid(self.projected_3d, arr)
                    score_grid = flow._score_grid
        return flow.find_attractors(score_grid=score_grid)

    def get_cluster_texts(self, cluster_id: int) -> List[str]:
        """Return texts belonging to a specific cluster."""
        return [
            self.session.entries[i].text
            for i, lbl in enumerate(self.clusters.labels)
            if lbl == cluster_id
        ]

    def get_entry_at_3d(self, idx: int) -> dict:
        """Return entry info with its 3D coordinates."""
        entry = self.session.entries[idx]
        coords = self.projected_3d[idx]
        return {
            "text": entry.text,
            "role": entry.role,
            "x": float(coords[0]),
            "y": float(coords[1]),
            "z": float(coords[2]),
            "cluster": self.clusters.labels[idx],
            "step_index": entry.step_index,
        }

    def fingerprint(self) -> str:
        """Compute a fingerprint from (texts + embedding_model + flow_mode
        + score channel names [+ seed]).

        Score channels are included so that changing the channel set
        (e.g. dropping a score or adding a new one) invalidates the
        cache — otherwise a cached result would be loaded with a stale
        score layout.

        Backward-compat: when run with the historical defaults
        (deterministic=True AND seed=42) the seed is *omitted* from
        the fingerprint, producing the exact same hash as pre-seed
        versions of TraceScope.  This keeps existing cached results
        valid bit-for-bit on disk, so users who never touch the seed
        don't need to recompute anything.  Non-default seeds and
        non-deterministic runs append a seed suffix, giving each
        alternate run its own distinct cached result.
        """
        texts = sorted(e.text for e in self.session.entries)
        score_channels = sorted(self.session.score_channels)
        is_default_seed = bool(self.deterministic) and int(self.seed) == 42
        if is_default_seed:
            blob = (
                json.dumps(texts, sort_keys=True) + "|"
                + self.embedding_model + "|"
                + self.flow_mode + "|"
                + json.dumps(score_channels, sort_keys=True)
            )
        else:
            seed_key = (
                f"{int(self.seed)}" if self.deterministic else "nondet"
            )
            blob = (
                json.dumps(texts, sort_keys=True) + "|"
                + self.embedding_model + "|"
                + self.flow_mode + "|"
                + json.dumps(score_channels, sort_keys=True) + "|"
                + seed_key
            )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def save_result(self, path: str):
        """Save the full AnalysisResult to disk.

        Creates two files:
          - {path}.npz  — numpy arrays (embeddings, projected_3d, velocity_grid, etc.)
          - {path}.json — metadata (session, cluster info, axis labels, etc.)

        Args:
            path: Base path (without extension). E.g. "my_result" creates
                  my_result.npz and my_result.json.
        """
        base = Path(path)

        # ── Numpy arrays ───────────────────────────────
        arrays = {
            "embeddings": self.embeddings,
            "projected_3d": self.projected_3d,
        }
        if self.velocity_grid is not None:
            arrays["velocity_grid"] = self.velocity_grid
        if self.confidence_grid is not None:
            arrays["confidence_grid"] = self.confidence_grid
        if self.axis_min is not None:
            arrays["axis_min"] = self.axis_min
        if self.axis_max is not None:
            arrays["axis_max"] = self.axis_max
        if self.axis_info.axes is not None:
            arrays["axis_info_axes"] = self.axis_info.axes
        if self.cluster_centroids_3d is not None:
            arrays["cluster_centroids_3d"] = self.cluster_centroids_3d

        base.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(base) + ".npz", **arrays)

        # ── JSON sidecar ───────────────────────────────
        meta = {
            "fingerprint": self.fingerprint(),
            "embedding_model": self.embedding_model,
            "session": self.session.to_dict(),
            "clusters": {
                "n_clusters": self.clusters.n_clusters,
                "labels": self.clusters.labels,
                "clusters": self.clusters.clusters,
            },
            "axis_info": {
                "lengths": self.axis_info.lengths,
                "labels": self.axis_info.labels,
                "min_point_idx": self.axis_info.min_point_idx,
                "max_point_idx": self.axis_info.max_point_idx,
                "min_cluster_idx": self.axis_info.min_cluster_idx,
                "max_cluster_idx": self.axis_info.max_cluster_idx,
            },
            "cluster_labels": self.cluster_labels,
            "flow_model_trained": self.flow_model_trained,
            "max_cluster_distance": self.max_cluster_distance,
            "flow_mode": self.flow_mode,
            "seed": int(self.seed),
            "deterministic": bool(self.deterministic),
        }

        with open(str(base) + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_result(cls, path: str) -> "AnalysisResult":
        """Load an AnalysisResult from disk.

        Args:
            path: Base path (without extension) matching a previous save_result() call.

        Returns:
            Reconstructed AnalysisResult (without mdn_simulate or fitted_reducer).
        """
        base = Path(path)

        # ── JSON sidecar ───────────────────────────────
        with open(str(base) + ".json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        # ── Numpy arrays ───────────────────────────────
        data = np.load(str(base) + ".npz", allow_pickle=False)

        session = TraceSession.from_dict(meta["session"])

        # Restore embeddings on entries
        embeddings = data["embeddings"]
        for i, entry in enumerate(session.entries):
            entry.embedding = embeddings[i]
            entry.model_name = meta["embedding_model"]

        clusters_meta = meta["clusters"]
        clusters = ClusterResult(
            n_clusters=clusters_meta["n_clusters"],
            labels=clusters_meta["labels"],
            clusters=clusters_meta["clusters"],
        )

        ai = meta["axis_info"]
        axis_info = AxisInfo(
            axes=data["axis_info_axes"] if "axis_info_axes" in data else np.eye(3),
            lengths=ai["lengths"],
            labels=ai["labels"],
            min_point_idx=ai.get("min_point_idx", []),
            max_point_idx=ai.get("max_point_idx", []),
            min_cluster_idx=ai.get("min_cluster_idx", []),
            max_cluster_idx=ai.get("max_cluster_idx", []),
        )

        import warnings
        warnings.warn(
            "AnalysisResult.load_result() does not restore the fitted UMAP/t-SNE "
            "reducer. TraceQuery projections (query_flow_at, explain_path, etc.) "
            "will use k-NN interpolation instead of the original transform. "
            "For highest accuracy, use pipeline.analyze() with cache_path.",
            UserWarning, stacklevel=2,
        )

        return cls(
            session=session,
            embedding_model=meta["embedding_model"],
            embeddings=embeddings,
            clusters=clusters,
            projected_3d=data["projected_3d"],
            axis_info=axis_info,
            cluster_labels=meta.get("cluster_labels", []),
            flow_model_trained=meta.get("flow_model_trained", False),
            velocity_grid=data["velocity_grid"] if "velocity_grid" in data else None,
            confidence_grid=data["confidence_grid"] if "confidence_grid" in data else None,
            axis_min=data["axis_min"] if "axis_min" in data else None,
            axis_max=data["axis_max"] if "axis_max" in data else None,
            cluster_centroids_3d=data["cluster_centroids_3d"] if "cluster_centroids_3d" in data else None,
            max_cluster_distance=meta.get("max_cluster_distance", 1.0),
            flow_mode=meta.get("flow_mode", "mdn"),
            seed=int(meta.get("seed", 42)),
            deterministic=bool(meta.get("deterministic", True)),
        )
