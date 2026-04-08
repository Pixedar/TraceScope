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
        pct = ((point_3d - mins) / ranges) * 100
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
            raise RuntimeError("No velocity grid available. Run pipeline with train_flow=True.")

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
            stat = {"cluster": label, "count": len(vals)}
            if vals:
                stat.update({
                    "mean": round(sum(vals) / len(vals), 4),
                    "min": round(min(vals), 4),
                    "max": round(max(vals), 4),
                })
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
            }
            if entry_vals:
                stat["entry_mean"] = round(sum(entry_vals) / len(entry_vals), 4)
            path_stats.append(stat)

        return {
            "channel": channel,
            "total_entries_with_score": sum(1 for v in entry_scores if v is not None),
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
