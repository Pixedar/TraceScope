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
    axis_min: Optional[np.ndarray] = None             # bounding box min [3]
    axis_max: Optional[np.ndarray] = None             # bounding box max [3]
    mdn_simulate: Optional[Any] = None                # MDN simulate() callable
    fitted_reducer: Optional[Any] = None               # fitted UMAP/tSNE for .transform()
    # Cluster geometry in 3D projected space
    cluster_centroids_3d: Optional[np.ndarray] = None # (K,3) cluster centers
    max_cluster_distance: float = 1.0                 # max pairwise dist between centroids

    @property
    def n_entries(self) -> int:
        return len(self.session)

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
        """Compute a fingerprint from (sorted texts + embedding_model).

        Used to detect when cache is stale.
        """
        texts = sorted(e.text for e in self.session.entries)
        blob = json.dumps(texts, sort_keys=True) + "|" + self.embedding_model
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
        if self.axis_min is not None:
            arrays["axis_min"] = self.axis_min
        if self.axis_max is not None:
            arrays["axis_max"] = self.axis_max
        if self.axis_info.axes is not None:
            arrays["axis_info_axes"] = self.axis_info.axes
        if self.cluster_centroids_3d is not None:
            arrays["cluster_centroids_3d"] = self.cluster_centroids_3d

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
            axis_min=data["axis_min"] if "axis_min" in data else None,
            axis_max=data["axis_max"] if "axis_max" in data else None,
            cluster_centroids_3d=data["cluster_centroids_3d"] if "cluster_centroids_3d" in data else None,
            max_cluster_distance=meta.get("max_cluster_distance", 1.0),
        )
