"""
TraceQuery — Programmatic query API for the TraceScope semantic space.

After running the full pipeline once, wrap the AnalysisResult in a TraceQuery
to get fast, LLM-agent-friendly methods for querying the computed space:

    query = TraceQuery(result, embedding_provider)
    lookup = query.get_lookup()
    path_info = query.explain_path(["hello", "how are you", "goodbye"])
    flow_info = query.query_flow_at("tell me about work")
    dir_info  = query.query_direction_at(["start text", "end text"])
    sim       = query.path_similarity(["a","b"], ["x","y"])
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from tracescope.models.analysis import AnalysisResult
from tracescope.providers.embedding import EmbeddingProvider
from tracescope.analysis.explainer import compute_cosine_similarity

logger = logging.getLogger(__name__)


class TraceQuery:
    """Fast programmatic query interface over a computed TraceScope space.

    Args:
        result: AnalysisResult from AnalysisPipeline.analyze().
        embedding_provider: Same provider used during pipeline run.
        explainer: Optional SemanticExplainer for LLM-powered explanations.
    """

    def __init__(
        self,
        result: AnalysisResult,
        embedding_provider: EmbeddingProvider,
        explainer=None,
    ):
        self._result = result
        self._emb = embedding_provider
        self._explainer = explainer

        # Pre-compute lookup data
        self._lookup = self._build_lookup()

    # ─── Lookup table ─────────────────────────────────────────────────

    def _build_lookup(self) -> dict:
        r = self._result
        pts = r.projected_3d

        # Axis info
        axis_labels = list(r.axis_info.labels)
        axis_mins = pts.min(axis=0).tolist()
        axis_maxs = pts.max(axis=0).tolist()

        # Cluster info
        cluster_info = []
        for c in range(r.clusters.n_clusters):
            indices = [i for i, l in enumerate(r.clusters.labels) if l == c]
            label = r.cluster_labels[c] if c < len(r.cluster_labels) else f"Cluster {c}"
            centroid_3d = r.cluster_centroids_3d[c].tolist() if r.cluster_centroids_3d is not None else None
            sample_texts = [r.session.entries[i].text for i in indices[:5]]
            cluster_info.append({
                "id": c,
                "label": label,
                "centroid_3d": centroid_3d,
                "size": len(indices),
                "sample_texts": sample_texts,
            })

        lookup = {
            "axis_labels": axis_labels,
            "axis_ranges": [
                {"axis": axis_labels[i], "min": axis_mins[i], "max": axis_maxs[i]}
                for i in range(3)
            ],
            "clusters": cluster_info,
            "n_points": len(r.session),
            "embedding_model": r.embedding_model,
            "embedding_dim": r.embeddings.shape[1],
            "has_flow": r.velocity_grid is not None,
            "flow_bounds": {
                "axis_min": r.axis_min.tolist() if r.axis_min is not None else None,
                "axis_max": r.axis_max.tolist() if r.axis_max is not None else None,
            },
        }

        # Score channels (only if any exist)
        score_channels = r.score_channels
        if score_channels:
            score_info = {}
            for ch in score_channels:
                entry_vals = [v for v in r.get_entry_scores(ch) if v is not None]
                path_vals = list(r.get_path_scores(ch).values())
                all_vals = entry_vals + path_vals
                score_info[ch] = {
                    "entry_count": len(entry_vals),
                    "path_count": len(path_vals),
                    "min": round(min(all_vals), 4) if all_vals else None,
                    "max": round(max(all_vals), 4) if all_vals else None,
                    "mean": round(sum(all_vals) / len(all_vals), 4) if all_vals else None,
                }
            lookup["score_channels"] = score_info

        return lookup

    def get_lookup(self) -> dict:
        """Return the pre-computed lookup table with space metadata."""
        return self._lookup

    # ─── Internal helpers ─────────────────────────────────────────────

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed texts using the same provider as the pipeline."""
        return self._emb.embed_batch(texts)

    def _project_to_3d(self, embeddings: np.ndarray) -> np.ndarray:
        """Project high-dim embeddings to 3D using the stored fitted reducer.

        Falls back to nearest-neighbor interpolation if no reducer is stored.
        """
        reducer = self._result.fitted_reducer
        if reducer is not None and hasattr(reducer, "transform"):
            # L2-normalize to match pipeline preprocessing
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normed = embeddings / norms
            return reducer.transform(normed).astype(np.float32)

        # Fallback: weighted k-NN interpolation in high-dim space
        logger.warning("No fitted reducer available, using k-NN projection fallback")
        return self._knn_project(embeddings)

    def _knn_project(self, new_embeddings: np.ndarray, k: int = 5) -> np.ndarray:
        """Project new points via weighted k-NN in embedding space."""
        orig_embs = self._result.embeddings
        orig_3d = self._result.projected_3d

        # Normalize both
        orig_norms = np.linalg.norm(orig_embs, axis=1, keepdims=True)
        orig_norms[orig_norms == 0] = 1.0
        orig_normed = orig_embs / orig_norms

        new_norms = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        new_norms[new_norms == 0] = 1.0
        new_normed = new_embeddings / new_norms

        # Cosine similarity matrix: (M, N)
        sims = new_normed @ orig_normed.T

        projected = np.zeros((len(new_embeddings), 3), dtype=np.float32)
        for i in range(len(new_embeddings)):
            top_k = np.argsort(-sims[i])[:k]
            weights = np.maximum(sims[i, top_k], 0.0)
            w_sum = weights.sum()
            if w_sum > 0:
                weights /= w_sum
            else:
                weights = np.ones(k) / k
            projected[i] = (weights[:, None] * orig_3d[top_k]).sum(axis=0)

        return projected

    def _compute_axis_pcts(self, point_3d: np.ndarray) -> dict:
        """Compute axis percentages for a 3D point."""
        pts = self._result.projected_3d
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1.0
        pct = np.clip(((point_3d - mins) / ranges) * 100, 0, 100)
        labels = self._result.axis_info.labels
        return {labels[i]: round(float(pct[i]), 1) for i in range(3)}

    def _compute_cluster_distances(self, point_3d: np.ndarray) -> dict:
        """Compute closeness % to each cluster centroid."""
        r = self._result
        pts = r.projected_3d
        max_dist = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        if max_dist == 0:
            max_dist = 1.0

        distances = {}
        for c in range(r.clusters.n_clusters):
            centroid = r.cluster_centroids_3d[c]
            dist = float(np.linalg.norm(point_3d - centroid))
            closeness = max(0, (1 - dist / max_dist) * 100)
            label = r.cluster_labels[c] if c < len(r.cluster_labels) else f"Cluster {c}"
            distances[label] = round(closeness, 1)
        return distances

    def _find_nearest_points(self, point_3d: np.ndarray, k: int = 5) -> List[dict]:
        """Find k nearest original points to a 3D position."""
        pts = self._result.projected_3d
        dists = np.linalg.norm(pts - point_3d, axis=1)
        nearest = np.argsort(dists)[:k]
        results = []
        for idx in nearest:
            idx = int(idx)
            entry = self._result.session.entries[idx]
            d = {
                "index": idx,
                "text": entry.text,
                "distance": round(float(dists[idx]), 4),
                "cluster": self._result.clusters.labels[idx],
            }
            if entry.scores:
                d["scores"] = entry.scores
            return_path_id = entry.path_id
            if return_path_id is not None and return_path_id in self._result.session.path_scores:
                d["path_scores"] = self._result.session.path_scores[return_path_id]
            results.append(d)
        return results

    def _sample_velocity(self, point_3d: np.ndarray) -> Optional[np.ndarray]:
        """Sample velocity from the velocity grid at a 3D point."""
        r = self._result
        if r.velocity_grid is None or r.axis_min is None or r.axis_max is None:
            return None

        from tracescope.visualization.flow_field import FlowFieldSystem
        # Use the static sample_velocity method logic directly
        grid = r.velocity_grid
        G = grid.shape[0]
        span = r.axis_max - r.axis_min

        result_v = np.zeros(3, dtype=np.float32)
        nx = (point_3d[0] - r.axis_min[0]) / span[0] * (G - 1) if span[0] > 0 else 0
        ny = (point_3d[1] - r.axis_min[1]) / span[1] * (G - 1) if span[1] > 0 else 0
        nz = (point_3d[2] - r.axis_min[2]) / span[2] * (G - 1) if span[2] > 0 else 0

        ix = int(np.floor(nx))
        iy = int(np.floor(ny))
        iz = int(np.floor(nz))
        tx = nx - ix
        ty = ny - iy
        tz = nz - iz

        ix = np.clip(ix, 0, G - 2)
        iy = np.clip(iy, 0, G - 2)
        iz = np.clip(iz, 0, G - 2)

        for dx in range(2):
            for dy in range(2):
                for dz in range(2):
                    w = ((1 - tx) if dx == 0 else tx) * \
                        ((1 - ty) if dy == 0 else ty) * \
                        ((1 - tz) if dz == 0 else tz)
                    result_v += w * grid[ix + dx, iy + dy, iz + dz]

        return result_v

    def _estimate_settling_times(
        self,
        attractors: List[dict],
        n_samples: int = 500,
        max_steps: int = 400,
        eps: float = 1e-3,
        seed: int = 42,
    ) -> List[Optional[float]]:
        """For ``n_samples`` random points inside the bounding box, run
        a fixed-step forward integration and record how many steps it
        takes to either drop below ``eps`` velocity or exit the box.

        Returns a list of length ``n_samples`` where each entry is the
        settled step count (as a float), or ``None`` if the trajectory
        did not converge inside ``max_steps``.

        Used internally by ``topology_summary``; safe to call without a
        flow field (returns an empty list)."""
        r = self._result
        if r.velocity_grid is None or r.axis_min is None or r.axis_max is None:
            return []
        rng = np.random.default_rng(int(seed))
        axis_min = np.asarray(r.axis_min, dtype=np.float64)
        axis_max = np.asarray(r.axis_max, dtype=np.float64)
        span = axis_max - axis_min
        if not np.all(span > 0):
            return []

        G = r.velocity_grid.shape[0]
        # Coarse "in-basin" mask if attractors expose basin_mask;
        # used to early-exit a trajectory the moment it enters one.
        basin_union = None
        if attractors:
            for att in attractors:
                bm = att.get("basin_mask")
                if bm is None or bm.shape != (G, G, G):
                    continue
                basin_union = bm if basin_union is None else (basin_union | bm)

        dt = 0.05  # flow-time per step — matches scale of velocity grid
        out: List[Optional[float]] = []
        for _ in range(int(n_samples)):
            x = axis_min + rng.random(3) * span
            settled: Optional[float] = None
            for step in range(int(max_steps)):
                v = self._sample_velocity(x.astype(np.float32))
                if v is None:
                    break
                speed = float(np.linalg.norm(v))
                if speed < eps:
                    settled = float(step)
                    break
                if basin_union is not None:
                    gi = (
                        np.clip(((x - axis_min) / span * (G - 1)).astype(int),
                                0, G - 1)
                    )
                    if bool(basin_union[gi[0], gi[1], gi[2]]):
                        settled = float(step)
                        break
                x = x + np.asarray(v, dtype=np.float64) * dt
                if np.any(x < axis_min) or np.any(x > axis_max):
                    break
            out.append(settled)
        return out

    def _decompose_vector(
        self,
        point_3d: np.ndarray,
        direction: np.ndarray,
        include_nearby: bool = True,
    ) -> dict:
        """Decompose a direction vector into axis, cluster, and nearby-point components.

        Used by both query_flow_at (MDN velocity) and query_direction_at (path estimate).
        """
        r = self._result
        labels = r.axis_info.labels
        speed = float(np.linalg.norm(direction))

        # (a) Axis-relative decomposition
        axis_decomposition = []
        for i in range(3):
            component = float(direction[i])
            axis_decomposition.append({
                "axis_label": labels[i],
                "component": round(component, 6),
                "magnitude": round(abs(component), 6),
                "direction": "+" if component >= 0 else "-",
            })

        # (b) Cluster-relative decomposition
        cluster_pull = []
        for c in range(r.clusters.n_clusters):
            centroid = r.cluster_centroids_3d[c]
            to_cluster = centroid - point_3d
            dist = float(np.linalg.norm(to_cluster))
            if dist > 0 and speed > 0:
                to_cluster_normed = to_cluster / dist
                alignment = float(np.dot(direction, to_cluster_normed)) / speed
            else:
                alignment = 0.0

            label = r.cluster_labels[c] if c < len(r.cluster_labels) else f"Cluster {c}"
            interpretation = "toward" if alignment > 0.1 else ("away from" if alignment < -0.1 else "neutral")
            cluster_pull.append({
                "cluster_label": label,
                "alignment": round(alignment, 4),
                "distance": round(dist, 4),
                "interpretation": interpretation,
            })

        result = {
            "axis_decomposition": axis_decomposition,
            "cluster_pull": sorted(cluster_pull, key=lambda x: -abs(x["alignment"])),
        }

        # (c) Nearby points
        if include_nearby:
            pts = r.projected_3d
            dists = np.linalg.norm(pts - point_3d, axis=1)
            nearest = np.argsort(dists)[:5]
            nearby = []
            for idx in nearest:
                idx = int(idx)
                to_point = pts[idx] - point_3d
                dist = float(dists[idx])
                if dist > 0 and speed > 0:
                    to_normed = to_point / dist
                    vel_alignment = float(np.dot(direction, to_normed)) / speed
                else:
                    vel_alignment = 0.0

                # "would pass through" if alignment > 0.8 and point is close
                median_dist = float(np.median(dists))
                would_pass = vel_alignment > 0.8 and dist < median_dist * 0.5

                nearby.append({
                    "text": r.session.entries[idx].text,
                    "distance": round(dist, 4),
                    "velocity_alignment": round(vel_alignment, 4),
                    "would_pass_through": would_pass,
                })
            result["nearby_points"] = nearby

        return result

    # ─── Method 1: explain_path ───────────────────────────────────────

    def explain_path(self, texts: List[str]) -> dict:
        """Project a list of texts into the semantic space and explain the path.

        Args:
            texts: Ordered list of texts forming a semantic path.

        Returns:
            dict with path_3d, points (per-point info), and explanation.
        """
        if len(texts) < 2:
            raise ValueError("Need at least 2 texts to form a path")

        embeddings = self._embed_texts(texts)
        projected = self._project_to_3d(embeddings)

        points = []
        control_points = []
        for i, (text, pt) in enumerate(zip(texts, projected)):
            axis_pcts = self._compute_axis_pcts(pt)
            cluster_dists = self._compute_cluster_distances(pt)
            nearest = self._find_nearest_points(pt, k=3)

            points.append({
                "text": text,
                "position_3d": pt.tolist(),
                "axis_percentages": axis_pcts,
                "cluster_distances": cluster_dists,
                "nearest_texts": nearest,
            })

            # Build control_points for the explainer prompt
            control_points.append({
                "axis_pcts": [int(axis_pcts[l]) for l in self._result.axis_info.labels],
                "cluster_distances": [
                    (label, int(pct)) for label, pct in cluster_dists.items()
                ],
            })

        result = {
            "path_3d": projected.tolist(),
            "points": points,
        }

        # Generate LLM explanation if explainer available
        if self._explainer is not None:
            try:
                explanation = self._explainer.explain_probe_multi(
                    axis_labels=list(self._result.axis_info.labels),
                    control_points=control_points,
                )
                result["explanation"] = explanation
            except Exception as e:
                logger.warning(f"Path explanation failed: {e}")
                result["explanation"] = None
        else:
            result["explanation"] = None

        return result

    # ─── Method 2: query_flow_at ──────────────────────────────────────

    def query_flow_at(self, text: str) -> dict:
        """Query the MDN flow field at the position of a given text.

        Args:
            text: Input text to embed and query.

        Returns:
            dict with position, velocity, speed, axis/cluster/nearby decomposition.
        """
        embeddings = self._embed_texts([text])
        projected = self._project_to_3d(embeddings)
        pt = projected[0]

        velocity = self._sample_velocity(pt)
        if velocity is None:
            raise RuntimeError(
                "No velocity grid available. This can happen if: "
                "(1) train_flow=False was passed to pipeline.analyze(), "
                "(2) the flow model training failed silently (check for warnings about "
                "unsupported kernels or missing PyTorch), or "
                "(3) the result was loaded from a save that didn't include a velocity grid."
            )

        speed = float(np.linalg.norm(velocity))
        decomposition = self._decompose_vector(pt, velocity, include_nearby=True)

        return {
            "text": text,
            "position_3d": pt.tolist(),
            "velocity": velocity.tolist(),
            "speed": round(speed, 6),
            "source": "flow_field",
            "axis_percentages": self._compute_axis_pcts(pt),
            "cluster_distances": self._compute_cluster_distances(pt),
            **decomposition,
        }

    # ─── Method 3: query_direction_at ─────────────────────────────────

    def query_direction_at(self, texts: List[str]) -> dict:
        """Estimate movement direction from a sequence of texts (no flow field needed).

        Args:
            texts: Two or more texts defining a path.

        Returns:
            dict with estimated direction, decomposition (same format as query_flow_at).
        """
        if len(texts) < 2:
            raise ValueError("Need at least 2 texts to estimate direction")

        embeddings = self._embed_texts(texts)
        projected = self._project_to_3d(embeddings)

        # Estimate direction: average of consecutive differences
        diffs = np.diff(projected, axis=0)  # (N-1, 3)
        avg_direction = diffs.mean(axis=0)

        # Use last point as the reference position
        last_pt = projected[-1]
        magnitude = float(np.linalg.norm(avg_direction))
        decomposition = self._decompose_vector(last_pt, avg_direction, include_nearby=True)

        return {
            "texts": texts,
            "path_3d": projected.tolist(),
            "position_3d": last_pt.tolist(),
            "estimated_direction": avg_direction.tolist(),
            "estimated_magnitude": round(magnitude, 6),
            "source": "path_estimate",
            "axis_percentages": self._compute_axis_pcts(last_pt),
            "cluster_distances": self._compute_cluster_distances(last_pt),
            **decomposition,
        }

    # ─── Method 4: score_summary ─────────────────────────────────────

    def score_summary(self, channel: str) -> dict:
        """Get a statistical summary of a score channel across the space.

        Args:
            channel: Score channel name (e.g. "success", "error_rate").

        Returns:
            dict with per-cluster and per-path score breakdowns.
        """
        r = self._result
        if channel not in r.score_channels:
            raise ValueError(f"Score channel '{channel}' not found. "
                             f"Available: {r.score_channels}")

        entry_scores = r.get_entry_scores(channel)
        path_score_map = r.get_path_scores(channel)

        # Per-cluster breakdown
        cluster_stats = []
        for c in range(r.clusters.n_clusters):
            indices = [i for i, l in enumerate(r.clusters.labels) if l == c]
            vals = [entry_scores[i] for i in indices if entry_scores[i] is not None]
            label = r.cluster_labels[c] if c < len(r.cluster_labels) else f"Cluster {c}"
            stat = {
                "cluster": label,
                "count": len(vals),
                "mean": round(sum(vals) / len(vals), 4) if vals else None,
                "min": round(min(vals), 4) if vals else None,
                "max": round(max(vals), 4) if vals else None,
            }
            cluster_stats.append(stat)

        # Per-path breakdown
        path_stats = []
        path_ids = set(e.path_id for e in r.session.entries if e.path_id is not None)
        for pid in sorted(path_ids):
            indices = [i for i, e in enumerate(r.session.entries) if e.path_id == pid]
            entry_vals = [entry_scores[i] for i in indices if entry_scores[i] is not None]
            path_val = path_score_map.get(pid)
            path_label = None
            for i in indices:
                pl = r.session.entries[i].metadata.get("path_label")
                if pl:
                    path_label = pl
                    break
            stat = {
                "path_id": pid,
                "path_label": path_label or f"Path {pid}",
                "path_score": path_val,
                "entry_scores_count": len(entry_vals),
                "entry_mean": round(sum(entry_vals) / len(entry_vals), 4) if entry_vals else None,
            }
            path_stats.append(stat)

        all_vals = [v for v in entry_scores if v is not None]
        return {
            "channel": channel,
            "total_entries_with_score": len(all_vals),
            "overall_mean": round(sum(all_vals) / len(all_vals), 4) if all_vals else None,
            "total_paths_with_score": len(path_score_map),
            "cluster_breakdown": cluster_stats,
            "path_breakdown": path_stats,
        }

    # ─── Method 5: path_similarity ────────────────────────────────────

    def path_similarity(self, path_a: List[str], path_b: List[str]) -> dict:
        """Compare two semantic paths using high-dimensional embedding vectors.

        Pure vector computation — no 3D projection or flow field involved.

        Args:
            path_a: First ordered list of texts.
            path_b: Second ordered list of texts.

        Returns:
            dict with frechet_distance, mean_cosine_similarity,
            direction_similarity, start/end_similarity, overall_score.
        """
        if len(path_a) < 2 or len(path_b) < 2:
            raise ValueError("Both paths need at least 2 texts")

        emb_a = self._embed_texts(path_a)
        emb_b = self._embed_texts(path_b)

        # Normalize
        emb_a = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-12)
        emb_b = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-12)

        # 1. Discrete Frechet distance
        frechet = self._discrete_frechet(emb_a, emb_b)

        # 2. Mean pairwise cosine similarity (DTW-aligned)
        alignment = self._dtw_alignment(emb_a, emb_b)
        aligned_sims = [
            float(np.dot(emb_a[i], emb_b[j]))
            for i, j in alignment
        ]
        mean_cos = float(np.mean(aligned_sims)) if aligned_sims else 0.0

        # 3. Direction similarity (average direction vectors)
        dir_a = (emb_a[-1] - emb_a[0])
        dir_b = (emb_b[-1] - emb_b[0])
        norm_a = np.linalg.norm(dir_a)
        norm_b = np.linalg.norm(dir_b)
        if norm_a > 0 and norm_b > 0:
            direction_sim = float(np.dot(dir_a, dir_b) / (norm_a * norm_b))
        else:
            direction_sim = 0.0

        # 4. Endpoint similarities
        start_sim = float(np.dot(emb_a[0], emb_b[0]))
        end_sim = float(np.dot(emb_a[-1], emb_b[-1]))

        # 5. Overall score (weighted combination, normalized to [0, 1])
        # Frechet is a distance, convert to similarity: 1/(1+d)
        frechet_sim = 1.0 / (1.0 + frechet)
        overall = (
            0.3 * mean_cos +
            0.3 * direction_sim +
            0.15 * start_sim +
            0.15 * end_sim +
            0.1 * frechet_sim
        )
        # Clamp to [0, 1]
        overall = max(0.0, min(1.0, (overall + 1) / 2))  # shift from [-1,1] to [0,1]

        return {
            "frechet_distance": round(frechet, 6),
            "mean_cosine_similarity": round(mean_cos, 6),
            "direction_similarity": round(direction_sim, 6),
            "start_similarity": round(start_sim, 6),
            "end_similarity": round(end_sim, 6),
            "overall_score": round(overall, 6),
        }

    @staticmethod
    def _discrete_frechet(P: np.ndarray, Q: np.ndarray) -> float:
        """Compute discrete Frechet distance between two paths in embedding space.

        Uses cosine distance (1 - cosine_similarity) as the point distance.
        """
        n, m = len(P), len(Q)
        ca = np.full((n, m), -1.0)

        def _dist(i, j):
            return 1.0 - float(np.dot(P[i], Q[j]))

        def _c(i, j):
            if ca[i, j] > -0.5:
                return ca[i, j]
            d = _dist(i, j)
            if i == 0 and j == 0:
                ca[i, j] = d
            elif i == 0:
                ca[i, j] = max(_c(0, j - 1), d)
            elif j == 0:
                ca[i, j] = max(_c(i - 1, 0), d)
            else:
                ca[i, j] = max(min(_c(i - 1, j), _c(i - 1, j - 1), _c(i, j - 1)), d)
            return ca[i, j]

        return _c(n - 1, m - 1)

    @staticmethod
    def _dtw_alignment(P: np.ndarray, Q: np.ndarray) -> List[tuple]:
        """Simple DTW alignment returning index pairs."""
        n, m = len(P), len(Q)
        cost = np.full((n + 1, m + 1), np.inf)
        cost[0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                d = 1.0 - float(np.dot(P[i - 1], Q[j - 1]))
                cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

        # Backtrack
        path = []
        i, j = n, m
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            candidates = [
                (cost[i - 1, j - 1], i - 1, j - 1),
                (cost[i - 1, j], i - 1, j),
                (cost[i, j - 1], i, j - 1),
            ]
            _, i, j = min(candidates, key=lambda x: x[0])

        path.reverse()
        return path

    # ─── Method 6: topology_summary ───────────────────────────────────

    def topology_summary(
        self,
        n_settling_samples: int = 500,
        max_settling_steps: int = 400,
        settling_eps: float = 1e-3,
        score_channel: Optional[str] = None,
    ) -> dict:
        """Summarize the topology of the learned semantic flow field.

        Computes attractors, basin sizes, mean settling time, transition
        turbulence (curl / speed) and Jacobian-based stability over the
        velocity grid.  All metrics are derived from the existing
        ``velocity_grid`` — no new training is performed.

        These are *analysis-layer* topology metrics extracted from the flow
        field that TraceScope already learned.  They do not guarantee any
        property of the underlying language model itself.

        Args:
            n_settling_samples: Number of random initial points sampled
                inside the bounding box for the mean-settling-time
                estimate. Higher = more accurate but slower.
            max_settling_steps: Cap on integration steps per sample
                before declaring the trajectory non-converging.
            settling_eps: Velocity magnitude below which a trajectory
                is considered "settled" into an attractor.
            score_channel: Optional score channel for per-attractor
                ``mean_score`` (forwarded to ``find_attractors``).

        Returns:
            dict with keys:
              attractors (list of attractor dicts, ``basin_mask`` removed
                  for JSON-friendliness; original is preserved under
                  ``basin_mask`` if present),
              basin_sizes (list[int]),
              basin_fractions (list[float]),
              n_attractors (int),
              mean_settling_time (float | None),
              median_settling_time (float | None),
              fraction_converged (float),
              transition_turbulence (float | None),
              jacobian_stability (dict with spectral_radius_*/divergence_*),
              unstable_regions (list of {position_3d, spectral_radius}),
              has_flow (bool).
        """
        r = self._result
        out = {
            "has_flow": r.velocity_grid is not None,
            "attractors": [],
            "basin_sizes": [],
            "basin_fractions": [],
            "n_attractors": 0,
            "mean_settling_time": None,
            "median_settling_time": None,
            "fraction_converged": 0.0,
            "transition_turbulence": None,
            "jacobian_stability": None,
            "unstable_regions": [],
        }
        if r.velocity_grid is None or r.axis_min is None or r.axis_max is None:
            return out

        # ── Attractors + basin info ─────────────────────────────────
        try:
            attractors = r.find_attractors(score_channel=score_channel)
        except Exception as e:
            logger.warning(f"find_attractors failed in topology_summary: {e}")
            attractors = []

        basin_sizes = []
        basin_fractions = []
        att_serializable = []
        for i, att in enumerate(attractors):
            basin_sizes.append(int(att.get("basin_size", 0)))
            basin_fractions.append(float(att.get("basin_fraction", 0.0)))
            pos = att.get("position")
            att_serializable.append({
                "id": i,
                "position_3d": pos.tolist() if hasattr(pos, "tolist") else list(pos),
                "strength": float(att.get("strength", 0.0)),
                "divergence": float(att.get("divergence", 0.0)),
                "basin_size": int(att.get("basin_size", 0)),
                "basin_fraction": float(att.get("basin_fraction", 0.0)),
                "mean_score": (
                    float(att["mean_score"])
                    if att.get("mean_score") is not None else None
                ),
            })
        out["attractors"] = att_serializable
        out["basin_sizes"] = basin_sizes
        out["basin_fractions"] = basin_fractions
        out["n_attractors"] = len(attractors)

        # ── Mean settling time via short forward integration ────────
        try:
            settling_times = self._estimate_settling_times(
                attractors=attractors,
                n_samples=int(n_settling_samples),
                max_steps=int(max_settling_steps),
                eps=float(settling_eps),
            )
        except Exception as e:
            logger.warning(f"settling-time estimation failed: {e}")
            settling_times = []

        finite = [t for t in settling_times if t is not None]
        if settling_times:
            out["fraction_converged"] = float(len(finite) / len(settling_times))
        if finite:
            out["mean_settling_time"] = float(np.mean(finite))
            out["median_settling_time"] = float(np.median(finite))

        # ── Transition turbulence: mean curl / (speed + eps) ────────
        try:
            vg = r.velocity_grid
            speed = np.linalg.norm(vg, axis=3)
            dvz_dy = np.gradient(vg[:, :, :, 2], axis=1)
            dvy_dz = np.gradient(vg[:, :, :, 1], axis=2)
            dvx_dz = np.gradient(vg[:, :, :, 0], axis=2)
            dvz_dx = np.gradient(vg[:, :, :, 2], axis=0)
            dvy_dx = np.gradient(vg[:, :, :, 1], axis=0)
            dvx_dy = np.gradient(vg[:, :, :, 0], axis=1)
            curl_mag = np.sqrt(
                (dvz_dy - dvy_dz) ** 2
                + (dvx_dz - dvz_dx) ** 2
                + (dvy_dx - dvx_dy) ** 2
            )
            speed_ref = float(np.percentile(speed, 90)) if speed.size else 0.0
            if speed_ref > 1e-12:
                turbulence = float(
                    np.mean(curl_mag / (speed + speed_ref * 1e-3))
                )
                out["transition_turbulence"] = round(turbulence, 6)
        except Exception as e:
            logger.warning(f"turbulence computation failed: {e}")

        # ── Jacobian stability summary over the velocity grid ───────
        try:
            grid = self.stability_grid()
            sr = grid["spectral_radius"]
            dv = grid["divergence"]
            out["jacobian_stability"] = {
                "spectral_radius_mean": float(np.mean(sr)),
                "spectral_radius_median": float(np.median(sr)),
                "spectral_radius_p90": float(np.percentile(sr, 90)),
                "spectral_radius_max": float(np.max(sr)),
                "divergence_mean": float(np.mean(dv)),
                "divergence_negative_fraction": float(
                    np.mean(dv < 0.0)
                ),
            }
            # Top-K unstable regions by spectral radius
            G = sr.shape[0]
            flat = sr.reshape(-1)
            k = min(10, flat.size)
            if k > 0:
                top_idx = np.argpartition(-flat, k - 1)[:k]
                top_idx = top_idx[np.argsort(-flat[top_idx])]
                axis_min = r.axis_min
                axis_max = r.axis_max
                span = axis_max - axis_min
                unstable = []
                for fi in top_idx:
                    ix, iy, iz = np.unravel_index(int(fi), sr.shape)
                    pos = np.array([
                        axis_min[0] + ix / max(G - 1, 1) * span[0],
                        axis_min[1] + iy / max(G - 1, 1) * span[1],
                        axis_min[2] + iz / max(G - 1, 1) * span[2],
                    ], dtype=np.float32)
                    unstable.append({
                        "position_3d": pos.tolist(),
                        "spectral_radius": float(sr[ix, iy, iz]),
                        "divergence": float(dv[ix, iy, iz]),
                    })
                out["unstable_regions"] = unstable
        except Exception as e:
            logger.warning(f"jacobian stability computation failed: {e}")

        return out

    # ─── Method 7: integrate_flow ────────────────────────────────────

    def integrate_flow(
        self,
        text: str,
        method: str = "rk45",
        max_time: float = 25.0,
        convergence_eps: float = 1e-3,
        max_steps: int = 5000,
    ) -> dict:
        """ODE-style probing of the learned semantic flow field.

        Integrates ``dx/dt = velocity_field(x)`` from the embedding of
        ``text`` until either the speed drops below ``convergence_eps``
        (settled into an attractor), the trajectory leaves the bounding
        box, or ``max_time`` is reached.

        This is *analysis-layer* probing of the velocity field that
        TraceScope already learned.  It is **not** training a Neural
        ODE — TraceScope's flow model (MDN or RBF) is not parameterised
        as a continuous-depth neural net.  Neural ODE language only
        applies at the analysis layer (we use an ODE solver to walk
        the existing learned field).

        Args:
            text: Input text — embedded and projected into 3D.
            method: One of {"rk45", "euler"}. ``rk45`` uses
                ``scipy.integrate.solve_ivp`` (RK45 with adaptive step).
                ``euler`` is a lightweight fixed-step fallback.
            max_time: Integration horizon in flow-time units.
            convergence_eps: Speed threshold below which the trajectory
                is considered settled.
            max_steps: Hard cap on Euler steps (ignored for rk45).

        Returns:
            dict with keys:
                trajectory_3d (list[[x,y,z]]),
                times (list[float]),
                final_position (list[float]),
                final_speed (float),
                settling_time (float | None),
                attractor_id (int | None),
                converged (bool),
                method (str),
                escaped (bool),
                source ("integrate_flow").
        """
        r = self._result
        if r.velocity_grid is None or r.axis_min is None or r.axis_max is None:
            raise RuntimeError(
                "No velocity grid available. integrate_flow needs a flow "
                "field — run pipeline.analyze() with train_flow=True."
            )

        embeddings = self._embed_texts([text])
        projected = self._project_to_3d(embeddings)
        x0 = np.asarray(projected[0], dtype=np.float64)

        axis_min = np.asarray(r.axis_min, dtype=np.float64)
        axis_max = np.asarray(r.axis_max, dtype=np.float64)
        span = axis_max - axis_min
        # Pad bounding box very slightly so a probe starting on the
        # boundary doesn't immediately get flagged as "escaped".
        pad = np.where(span > 0, span * 0.05, 1e-6)
        bb_lo = axis_min - pad
        bb_hi = axis_max + pad

        method_lc = (method or "rk45").lower()

        def _vel(x: np.ndarray) -> np.ndarray:
            v = self._sample_velocity(np.asarray(x, dtype=np.float32))
            if v is None:
                return np.zeros(3, dtype=np.float64)
            return np.asarray(v, dtype=np.float64)

        traj = [x0.copy()]
        times_out = [0.0]
        settling_time: Optional[float] = None
        escaped = False
        converged = False

        if method_lc in ("rk45", "rk23", "dop853", "lsoda", "radau", "bdf"):
            try:
                from scipy.integrate import solve_ivp
            except ImportError as e:
                raise RuntimeError(
                    "scipy.integrate.solve_ivp is required for method='rk45'."
                ) from e

            def rhs(t, x):
                return _vel(x)

            # Event: speed below threshold (settled)
            def settled_event(t, x):
                v = _vel(x)
                return float(np.linalg.norm(v)) - convergence_eps
            settled_event.terminal = True
            settled_event.direction = -1

            # Event: left bounding box
            def escaped_event(t, x):
                d = np.concatenate([x - bb_lo, bb_hi - x])
                return float(np.min(d))
            escaped_event.terminal = True
            escaped_event.direction = -1

            method_map = {
                "rk45": "RK45", "rk23": "RK23", "dop853": "DOP853",
                "lsoda": "LSODA", "radau": "Radau", "bdf": "BDF",
            }
            sol = solve_ivp(
                rhs, (0.0, float(max_time)), x0,
                method=method_map[method_lc],
                events=[settled_event, escaped_event],
                max_step=float(max_time) / 50.0 if max_time > 0 else 0.1,
                dense_output=False,
                rtol=1e-4, atol=1e-6,
            )
            ys = sol.y.T  # (n_points, 3)
            ts = sol.t
            traj = [row.copy() for row in ys]
            times_out = ts.tolist()

            # Determine which event fired (if any)
            settled_events = sol.t_events[0] if len(sol.t_events) > 0 else np.array([])
            escape_events = sol.t_events[1] if len(sol.t_events) > 1 else np.array([])
            if settled_events.size > 0:
                converged = True
                settling_time = float(settled_events[0])
            if escape_events.size > 0:
                escaped = True

        else:
            # Fixed-step Euler fallback
            x = x0.copy()
            t = 0.0
            dt = float(max_time) / float(max_steps) if max_steps > 0 else 0.02
            for _step in range(int(max_steps)):
                v = _vel(x)
                speed = float(np.linalg.norm(v))
                if speed < convergence_eps:
                    converged = True
                    settling_time = t
                    break
                x = x + v * dt
                t += dt
                if np.any(x < bb_lo) or np.any(x > bb_hi):
                    escaped = True
                    break
                traj.append(x.copy())
                times_out.append(t)
                if t >= max_time:
                    break

        final_pos = np.asarray(traj[-1], dtype=np.float64)
        final_speed = float(np.linalg.norm(_vel(final_pos)))

        # Attractor matching: find the nearest attractor whose basin
        # contains the final position (if any).  If basin_mask is
        # available we use it; otherwise fall back to nearest-position.
        attractor_id: Optional[int] = None
        if converged:
            try:
                attractors = r.find_attractors()
            except Exception as e:
                logger.warning(f"attractor lookup failed: {e}")
                attractors = []
            if attractors:
                # Try basin-mask membership first
                G = r.velocity_grid.shape[0]
                gi = np.zeros(3, dtype=int)
                for a in range(3):
                    if span[a] > 0:
                        gi[a] = int(np.clip(
                            (final_pos[a] - axis_min[a]) / span[a] * (G - 1),
                            0, G - 1,
                        ))
                for ai, att in enumerate(attractors):
                    bm = att.get("basin_mask")
                    if bm is not None and bm.shape == (G, G, G):
                        if bool(bm[gi[0], gi[1], gi[2]]):
                            attractor_id = ai
                            break
                if attractor_id is None:
                    # Fallback: nearest attractor position
                    dists = [
                        float(np.linalg.norm(final_pos - np.asarray(a["position"])))
                        for a in attractors
                    ]
                    attractor_id = int(np.argmin(dists))

        return {
            "text": text,
            "method": method_lc,
            "trajectory_3d": [list(map(float, p)) for p in traj],
            "times": [float(t) for t in times_out],
            "final_position": final_pos.tolist(),
            "final_speed": round(final_speed, 8),
            "settling_time": settling_time,
            "attractor_id": attractor_id,
            "converged": bool(converged),
            "escaped": bool(escaped),
            "source": "integrate_flow",
        }

    # ─── Method 8: numerical Jacobian diagnostics ────────────────────

    def _numerical_jacobian(
        self,
        point_3d: np.ndarray,
        h: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """Estimate the local 3×3 Jacobian of the velocity field via
        central differences.  Returns None when no flow field is
        available."""
        r = self._result
        if r.velocity_grid is None or r.axis_min is None or r.axis_max is None:
            return None
        span = np.asarray(r.axis_max - r.axis_min, dtype=np.float64)
        # Default step = a small fraction of the grid cell so we sample
        # *across* a cell, not within numerical noise.
        if h is None:
            G = r.velocity_grid.shape[0]
            cell = span / max(G - 1, 1)
            h_vec = np.maximum(cell * 0.5, 1e-6)
        else:
            h_vec = np.full(3, float(h), dtype=np.float64)
        x = np.asarray(point_3d, dtype=np.float64)
        J = np.zeros((3, 3), dtype=np.float64)
        for j in range(3):
            ej = np.zeros(3, dtype=np.float64)
            ej[j] = h_vec[j]
            v_plus = self._sample_velocity((x + ej).astype(np.float32))
            v_minus = self._sample_velocity((x - ej).astype(np.float32))
            if v_plus is None or v_minus is None:
                return None
            J[:, j] = (np.asarray(v_plus, dtype=np.float64)
                       - np.asarray(v_minus, dtype=np.float64)) / (2.0 * h_vec[j])
        return J

    def local_stability_at(self, text: str) -> dict:
        """Estimate local stability of the flow field at the position
        of ``text`` via a numerical Jacobian.

        Returns spectral radius (max |eigenvalue|), divergence
        (trace of J), and the eigenvalues themselves.  Higher
        spectral radius → faster local divergence/convergence;
        negative trace → on-net contracting (attractor-like).
        """
        r = self._result
        if r.velocity_grid is None or r.axis_min is None or r.axis_max is None:
            raise RuntimeError(
                "No velocity grid available. local_stability_at needs a "
                "flow field — run pipeline.analyze() with train_flow=True."
            )
        embeddings = self._embed_texts([text])
        projected = self._project_to_3d(embeddings)
        pt = projected[0]
        J = self._numerical_jacobian(pt)
        if J is None:
            raise RuntimeError(
                "Failed to compute numerical Jacobian (no flow field)."
            )
        eigvals = np.linalg.eigvals(J)
        spectral_radius = float(np.max(np.abs(eigvals)))
        divergence = float(np.trace(J))
        return {
            "text": text,
            "position_3d": [float(x) for x in pt],
            "jacobian": J.tolist(),
            "eigenvalues_real": [float(v.real) for v in eigvals],
            "eigenvalues_imag": [float(v.imag) for v in eigvals],
            "spectral_radius": round(spectral_radius, 8),
            "divergence": round(divergence, 8),
            "is_locally_attracting": bool(divergence < 0),
            "source": "local_stability_at",
        }

    def stability_grid(self, resolution: Optional[int] = None) -> dict:
        """Return per-cell numerical stability statistics over the
        velocity grid.

        Uses ``np.gradient`` over the velocity grid to build the full
        spatial Jacobian per cell, then returns the spectral radius
        (max |eigenvalue|) and divergence (trace) at each cell.

        Args:
            resolution: Optional output resolution.  If smaller than
                the underlying grid, a stride is applied so callers can
                trade detail for speed.  Defaults to the grid resolution.

        Returns:
            dict with keys:
                spectral_radius (np.ndarray[G,G,G]),
                divergence (np.ndarray[G,G,G]),
                axis_min (list[3]), axis_max (list[3]),
                grid_size (int).
        """
        r = self._result
        if r.velocity_grid is None or r.axis_min is None or r.axis_max is None:
            raise RuntimeError(
                "No velocity grid available. stability_grid needs a "
                "flow field — run pipeline.analyze() with train_flow=True."
            )
        vg = r.velocity_grid.astype(np.float64)
        G = vg.shape[0]
        # Spacing = (axis_max - axis_min) / (G - 1) per axis
        span = np.asarray(r.axis_max - r.axis_min, dtype=np.float64)
        dx = float(span[0] / max(G - 1, 1))
        dy = float(span[1] / max(G - 1, 1))
        dz = float(span[2] / max(G - 1, 1))

        # ∂v_i / ∂x_j  for i,j in {0,1,2}
        dV0 = np.gradient(vg[..., 0], dx, dy, dz)
        dV1 = np.gradient(vg[..., 1], dx, dy, dz)
        dV2 = np.gradient(vg[..., 2], dx, dy, dz)
        # J[i, j] = dV_i / dx_j
        J = np.stack([
            np.stack([dV0[0], dV0[1], dV0[2]], axis=-1),  # i=0
            np.stack([dV1[0], dV1[1], dV1[2]], axis=-1),  # i=1
            np.stack([dV2[0], dV2[1], dV2[2]], axis=-1),  # i=2
        ], axis=-2)  # shape (G,G,G,3,3)

        # Vectorised eigenvalue computation
        eigvals = np.linalg.eigvals(J)        # (G,G,G,3) complex
        spectral_radius = np.max(np.abs(eigvals), axis=-1).astype(np.float32)
        divergence = (J[..., 0, 0] + J[..., 1, 1] + J[..., 2, 2]).astype(np.float32)

        if resolution is not None and 0 < int(resolution) < G:
            stride = max(1, G // int(resolution))
            spectral_radius = spectral_radius[::stride, ::stride, ::stride]
            divergence = divergence[::stride, ::stride, ::stride]

        return {
            "spectral_radius": spectral_radius,
            "divergence": divergence,
            "axis_min": [float(v) for v in r.axis_min],
            "axis_max": [float(v) for v in r.axis_max],
            "grid_size": int(spectral_radius.shape[0]),
        }
