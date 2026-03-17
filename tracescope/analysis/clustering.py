# clustering.py

import json
import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tracescope.analysis.metric_adapter import learn_metric, safe_transform


# ── Global logger setup ──
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s:%(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def cluster_embeddings(embedding_list, n_clusters=None, user_clusters=None, min_k=3):
    # build & normalize
    matrix = np.array(embedding_list, dtype=float)
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    n_samples = matrix.shape[0]
    logger.info(f"cluster_embeddings ▶ n_samples={n_samples}, requested n_clusters={n_clusters}")

    if n_samples < 2:
        raise ValueError("Need at least 2 samples for clustering")

    # auto‑select k by silhouette
    if n_clusters is None:
        max_k = min(10, n_samples - 1)

        if user_clusters:
            # 2-stage grid: α ∈ {0,.25,.5,.75,1} and k ∈ [2…max_k]
            matrix_orig = matrix.copy()
            L = learn_metric(matrix_orig, user_clusters)

            best_score = -1
            best_k = 2
            best_alpha = 0
            best_labels = None
            best_centroids = None

            for alpha in (0, 0.25, 0.5, 0.75, 1):
                Xα = safe_transform(matrix_orig, L, alpha)
                for k in range(2, max_k + 1):
                    km = KMeans(n_clusters=k, init='k-means++', random_state=42)
                    lbl = km.fit_predict(Xα)
                    sil = silhouette_score(Xα, lbl)
                    if sil > best_score:
                        best_score, best_k = sil, k
                        best_alpha = alpha
                        best_labels = lbl
                        best_centroids = km.cluster_centers_

            logger.info(
                f"user feedback → α={best_alpha}, k={best_k}, silhouette={best_score:.4f}"
            )
            n_clusters = best_k
            labels = best_labels
            centroids = best_centroids

        else:
            # no feedback → original silhouette-only loop
            best_score = -1
            best_k = min_k
            best_labels = None
            best_centroids = None

            for k in range(min_k, max_k + 1):
                logger.info(f"Trying k={k}")
                km = KMeans(n_clusters=k, init='k-means++', random_state=42)
                lbl = km.fit_predict(matrix)
                sil = silhouette_score(matrix, lbl)
                logger.info(f" k={k} silhouette={sil:.4f}")
                if sil > best_score:
                    best_score, best_k = sil, k
                    best_labels = lbl
                    best_centroids = km.cluster_centers_

            # unbalanced 2-cluster guard as before
            counts = np.bincount(best_labels)
            if best_k == 2 and (counts.min() < 2 or counts.max()/counts.min() > 10):
                logger.info("Detected unbalanced 2-cluster; searching k=3…max_k")
                alt_best = -1
                alt_labels = None
                alt_k = None
                for k in range(3, min(max_k + 1, 6)):
                    lbl = KMeans(n_clusters=k, random_state=42).fit_predict(matrix)
                    s = silhouette_score(matrix, lbl)
                    logger.info(f"  alt_k={k} silhouette={s:.4f}")
                    if s > alt_best:
                        alt_best, alt_k, alt_labels = s, k, lbl
                if alt_k:
                    logger.info(f"Switching to k={alt_k} with silhouette={alt_best:.4f}")
                    best_k, best_labels = alt_k, alt_labels
                    best_centroids = KMeans(n_clusters=alt_k, random_state=42).fit(matrix).cluster_centers_

            n_clusters = best_k
            labels = best_labels
            centroids = best_centroids

    else:
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        labels = kmeans.fit_predict(matrix)
        centroids = kmeans.cluster_centers_

    # build output
    clusters = []
    for i in range(n_clusters):
        idxs = [int(idx) for idx, lbl in enumerate(labels) if lbl == i]
        clusters.append({
            "cluster_id": i,
            "indices": idxs,
            "centroid": centroids[i].tolist()
        })

    return {
        "n_clusters": n_clusters,
        "clusters": clusters,
        "labels": labels.tolist(),
    }

def main(embeddings_json, n_clusters=None, aspect=None, user_clusters=None, compute_moving=False, min_k=3):
    # Load all embeddings directly from JSON
    embedding_list = json.loads(embeddings_json)

      # 2) if user_clusters came in as a JSON string, decode it,
      # and make sure every index is an int

    # ────────────────────────────────────────────────────────────────────────
    # sanitize user_clusters: if it’s a JSON string, parse it; then
    # coerce absolutely every index to a Python int, or error.
    if user_clusters:
        # 1) parse JSON if we were handed a string
        if isinstance(user_clusters, str):
            try:
                user_clusters = json.loads(user_clusters)
            except Exception as e:
                raise ValueError(f"Could not parse user_clusters JSON: {e}")

        # 2) coerce every element to int
        try:
            user_clusters = [
                [int(idx) for idx in cluster]
                for cluster in user_clusters
            ]
        except Exception as e:
            raise ValueError(
                f"Invalid user_clusters format (must be List[List[int]]): {user_clusters}"
            ) from e
    # ────────────────────────────────────────────────────────────────────────

    # Cluster them (no deduplication or pruning)
    result = cluster_embeddings(embedding_list, n_clusters, user_clusters=user_clusters, min_k=min_k)
    # print("RESULT:", result)

    # ── Optional moving‐average “closeness” to each cluster ──
    if compute_moving:
        try:
            # rebuild & normalize matrix
            matrix = np.array(embedding_list, dtype=float)
            matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
            n = matrix.shape[0]

            # weights decay with distance in list: w_{i,j} = 1 / (1 + |i-j|)
            idx = np.arange(n)
            W = 1 / (1 + np.abs(idx[:, None] - idx[None, :]))

            # compute weighted‐average embedding for each position
            weighted = (W[:, :, None] * matrix[None, :, :]).sum(axis=1) / W.sum(axis=1)[:, None]

            # extract centroids from the clustering result
            centroids = np.array([c["centroid"] for c in result["clusters"]])

            # for each position, compute similarity = 1/(1 + distance) to each centroid
            moving_sim = (
                1
                / (
                    1
                    + np.linalg.norm(
                        centroids[None, :, :] - weighted[:, None, :],
                        axis=2
                    )
                )
            ).tolist()

            result["moving_similarity"] = moving_sim

        except Exception as e:
            logger.error(f"Error computing moving‐average similarity: {e}")

    # Return the raw clustering result as JSON
    return json.dumps(result)


