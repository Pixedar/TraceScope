"""
Interactive probe – explore 3D semantic space with sliders.
"""

from __future__ import annotations

from typing import List

import numpy as np

from tracescope.models.analysis import AnalysisResult
from tracescope.analysis.explainer import SemanticExplainer


def probe_point(
    result: AnalysisResult,
    x: float,
    y: float,
    z: float,
    k_nearest: int = 5,
) -> dict:
    """Given a 3D coordinate, return nearest texts, cluster info, and axis values.

    Args:
        result: AnalysisResult from the pipeline.
        x, y, z: 3D coordinates of the probe.
        k_nearest: Number of nearest points to return.

    Returns:
        dict with nearest_texts, nearest_indices, cluster_distances,
        axis_percentages, etc.
    """
    pts = result.projected_3d
    probe = np.array([x, y, z])

    # Distances to all points
    dists = np.linalg.norm(pts - probe, axis=1)
    nearest_indices = np.argsort(dists)[:k_nearest]

    nearest_texts = [
        {
            "index": int(idx),
            "text": result.session.entries[idx].text,
            "role": result.session.entries[idx].role,
            "distance": float(dists[idx]),
            "cluster": result.clusters.labels[idx],
        }
        for idx in nearest_indices
    ]

    # Axis percentages (0-100%)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    pct = ((probe - mins) / ranges) * 100

    # Cluster distances (closeness %)
    cluster_distances = {}
    max_dist = float(np.linalg.norm(maxs - mins))
    for c in range(result.clusters.n_clusters):
        centroid = result.cluster_centroids_3d[c]
        dist = float(np.linalg.norm(probe - centroid))
        closeness = max(0, (1 - dist / max_dist) * 100) if max_dist > 0 else 0
        label = (
            result.cluster_labels[c]
            if c < len(result.cluster_labels)
            else f"Cluster {c}"
        )
        cluster_distances[label] = round(closeness, 1)

    return {
        "probe": {"x": x, "y": y, "z": z},
        "axis_percentages": {
            result.axis_info.labels[i]: round(float(pct[i]), 1) for i in range(3)
        },
        "nearest_texts": nearest_texts,
        "cluster_distances": cluster_distances,
    }


def probe_with_explanation(
    result: AnalysisResult,
    explainer: SemanticExplainer,
    x: float,
    y: float,
    z: float,
    k_nearest: int = 5,
) -> dict:
    """Probe a point and get an LLM explanation of its meaning.

    Returns the same dict as probe_point plus an 'explanation' key.
    """
    info = probe_point(result, x, y, z, k_nearest)

    # Build cluster distances string
    dist_str = ", ".join(
        f"{name}: {pct}%" for name, pct in info["cluster_distances"].items()
    )

    nearest_texts = [item["text"] for item in info["nearest_texts"]]
    pcts = info["axis_percentages"]
    axis_labels = list(pcts.keys())
    axis_vals = list(pcts.values())

    explanation = explainer.explain_probe(
        axis_labels=axis_labels,
        x_val=axis_vals[0],
        y_val=axis_vals[1],
        z_val=axis_vals[2],
        nearest_texts=nearest_texts,
        cluster_distances=dist_str,
    )

    info["explanation"] = explanation
    return info
