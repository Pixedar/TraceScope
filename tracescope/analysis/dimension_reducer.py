# reduction.py

import logging
import json
import warnings
import numpy as np

# Suppress UMAP's noisy n_jobs warning when random_state is set
warnings.filterwarnings("ignore", message="n_jobs value.*overridden to 1 by setting random_state")

import umap
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from tracescope.analysis.metric_adapter import learn_metric_dim_red, safe_transform

_MIN_SAMPLES_FOR_ADVANCED = 4

# Stores the last fitted reducer for use with .transform() on new points
_last_fitted_reducer = None

# ── Global logger setup ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def _fallback_from_data(matrix: np.ndarray, target_dim: int) -> np.ndarray:
    """
    matrix      : (n_samples, d) float64
    target_dim  : 2 or 3
    Returns     : (n_samples, target_dim) float64
    Uses only the original data, never invents points.
    """
    n_samples, d = matrix.shape

    # Special‑case tiny n so the result is meaningful and deterministic
    if n_samples == 1:
        return np.zeros((1, target_dim), dtype=np.float64)

    if n_samples == 2:
        # Project onto the line connecting the two points,
        # centred at 0 and unit‑length – avoids “fake” coords.
        v = matrix[1] - matrix[0]
        v /= np.linalg.norm(v) + 1e-12
        coords = np.vstack([-0.5 * v, 0.5 * v])    # length = 1
        return np.hstack([coords, np.zeros((2, 3 - target_dim))])[:, :target_dim]

    # n_samples ≥ 3: use PCA on the *real* data
    pca = PCA(n_components=min(target_dim, d), svd_solver="full")
    pc  = pca.fit_transform(matrix)

    # If we asked for 3 D but got only 2 comps (d = 2, target_dim = 3) → pad zeros
    if pc.shape[1] < target_dim:
        pc = np.hstack([pc, np.zeros((n_samples, target_dim - pc.shape[1]))])

    return pc[:, :target_dim].astype(np.float64)

def reduce_embeddings(embedding_list, target_dim=2, n_neighbors=None, method=None):
#     matrix = np.array(embedding_list, dtype=float)



    embedding_list = [
        [float(v) for v in row]
        for row in embedding_list
    ]
    matrix = np.array(embedding_list, dtype=np.float64)
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    n_samples = matrix.shape[0]
    if n_samples < 2:
        raise ValueError("Need at least 2 samples for reduction.")

    if n_samples < _MIN_SAMPLES_FOR_ADVANCED:
        logger.warning("Only %d sample(s) – using trivial fallback", n_samples)
        vis_dims = _fallback_from_data(matrix, target_dim)
        return vis_dims.tolist()

    # defaults
    if n_neighbors is None:
        n_neighbors = min(50 if target_dim==2 else 15, n_samples-1)
    else:
        n_neighbors = min(n_neighbors, n_samples-1)

    logger.info(f"reduce_embeddings ▶ shape={matrix.shape}, dim={target_dim}, neighbors={n_neighbors}, method={method}")

    # choose method
    vis_dims = None
    if method == "tsne":
        perp = min(30, n_samples-1)
        logger.info(f"TSNE ▶ n_components={target_dim}, perplexity={perp}, init='pca'")
        clean_matrix = matrix.astype(np.float64)
        tsne = TSNE(
            n_components=target_dim,
            perplexity=perp,
            metric='cosine',
            init='pca',
            learning_rate='auto',
            random_state=42
        )
        vis_dims = tsne.fit_transform(clean_matrix)

    else:  # UMAP first, TSNE fallback
        try:
            logger.info(f"UMAP ▶ n_components={target_dim}, n_neighbors={n_neighbors}, metric='cosine'")
            reducer = umap.UMAP(
                n_components=target_dim,
                n_neighbors=n_neighbors,
                metric='cosine',
                random_state=42
            )
            vis_dims = reducer.fit_transform(matrix)
        except Exception as e:
            if target_dim == 3:
                perp = min(30, n_samples-1)
                logger.warning(f"UMAP failed ({e}), falling back to TSNE 3D")

                logger.info(f"original matrix dtype={matrix.dtype}, shape={matrix.shape}")

                clean_matrix = np.asarray(matrix, dtype=np.float64)
                logger.info(f"TSNE ▶ about to run on dtype={clean_matrix.dtype}, shape={clean_matrix.shape}")

                tsne = TSNE(
                    n_components=3,
                    perplexity=perp,
                    metric='cosine',
                    init='pca',
                    learning_rate='auto',
                    random_state=42
                )
                vis_dims = tsne.fit_transform(clean_matrix)
            else:
                raise

    # pad to 3 dims if needed
    if target_dim < 3:
        pad = np.zeros((vis_dims.shape[0], 3-target_dim))
        vis_dims = np.hstack([vis_dims, pad])

    logger.info(f"reduce_embeddings ▶ output shape={vis_dims.shape}, var={np.var(vis_dims,axis=0).round(4)}")
    return vis_dims.tolist()


# def optimize_reduction(embeddings_json, cluster_labels_json, mode="3D",user_clusters=None):

def optimize_reduction(embeddings_json,
                           cluster_labels_json,
                           mode="3D",
                           user_clusters=None,
                           init_coords_json=None,
                           n_epochs_local=200,
                           param_range=None):



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

    embedding_list = json.loads(embeddings_json)
    cluster_labels = json.loads(cluster_labels_json)
    # matrix = np.array(embedding_list, dtype=float)
    # load and coerce to a true floating‑point array
    embedding_list = [
        [float(v) for v in row]
        for row in embedding_list
    ]
    matrix = np.array(embedding_list, dtype=np.float64)

    if matrix.dtype.kind not in ('f', 'i'):
        matrix = matrix.astype(np.float64)


    n_samples = matrix.shape[0]
    if n_samples < 2:
        raise ValueError("Need at least 2 samples for reduction.")

    if n_samples < _MIN_SAMPLES_FOR_ADVANCED:
        logger.warning("Only %d sample(s) – using trivial fallback", n_samples)
        vis_dims = _fallback_from_data(matrix, target_dim)
        return vis_dims.tolist()


    if param_range is not None:
        candidate_params = [p for p in param_range if p < n_samples]
        if not candidate_params:
            candidate_params = [min(param_range)]
    else:
        candidate_params = list(range(5, min(200, max(n_samples//2,6)), 5))

    target_dim = 2 if mode=="2D" else 3
    best_score = -1
    best_param = None
    best_embedding = None
    best_method = None
    best_reducer_obj = None  # track fitted reducer for .transform() on new points

    logger.info(f"optimize_reduction ▶ n_samples={n_samples}, mode={mode}, params={candidate_params}")
    if user_clusters is None:
        for param in candidate_params:
            logger.info(f"Param={param}")
            # # UMAP
            # try:
            #     um = umap.UMAP(
            #         n_components=target_dim,
            #         n_neighbors=min(param, n_samples-1),
            #         metric='cosine',
            #         random_state=42
            #     ).fit_transform(matrix)
            # UMAP  (anchored if init_coords_json supplied)
            um_reducer = None
            try:
                kw_umap = dict(
                    n_components = target_dim,
                    n_neighbors  = min(param, n_samples-1),
                    metric       = 'cosine',
                    random_state = 42
                )
                if init_coords_json is not None:
                    kw_umap.update(
                        init      = np.array(json.loads(init_coords_json),
                                             dtype=float),
                        n_epochs  = n_epochs_local,
                        min_dist  = 0.0 )
                um_reducer = umap.UMAP(**kw_umap)
                um = um_reducer.fit_transform(matrix)


                score_umap = silhouette_score(um, cluster_labels)
            except Exception as e:
                logger.debug(f"UMAP failed for param={param}: {e}")
                um, score_umap, um_reducer = None, -1, None

            # TSNE
            try:
                clean_matrix = matrix.astype(np.float64)
                ts = TSNE(
                    n_components=target_dim,
                    perplexity=min(param, n_samples-1),
                    metric='cosine',
                    init='pca',
                    learning_rate='auto',
                    random_state=42
                ).fit_transform(clean_matrix)
                score_tsne = silhouette_score(ts, cluster_labels)
            except Exception as e:
                logger.debug(f"TSNE failed for param={param}: {e}")
                ts, score_tsne = None, -1

            logger.info(f" param={param} ▶ umap={score_umap:.4f}  tsne={score_tsne:.4f}")

            # pick better
            if score_umap > best_score:
                best_score, best_param, best_embedding, best_method = score_umap, param, um.tolist(), "umap"
                best_reducer_obj = um_reducer
            if score_tsne > best_score:
                best_score, best_param, best_embedding, best_method = score_tsne, param, ts.tolist(), "tsne"
                best_reducer_obj = None  # tSNE has no .transform()

    else:
        matrix_orig = matrix.copy()
        L = learn_metric_dim_red(matrix_orig, user_clusters)
        for alpha in (0, 0.25, 0.5, 0.75, 1):
            Xα = safe_transform(matrix_orig, L, alpha)
            # 2) ensure they’re float64, not strings
            Xα = np.array(Xα, dtype=np.float64)
            for param in candidate_params:
                logger.info(f"Param={param}")
                # UMAP
                um_reducer = None
                try:
                    um_reducer = umap.UMAP(
                        n_components=target_dim,
                        n_neighbors=min(param, n_samples - 1),
                        metric='cosine',
                        random_state=42
                    )
                    um = um_reducer.fit_transform(Xα)
                    score_umap = silhouette_score(um, cluster_labels)
                except Exception as e:
                    logger.debug(f"UMAP failed for param={param}, alpha={alpha}: {e}")
                    um, score_umap, um_reducer = None, -1, None

                # TSNE
                try:
                    ts = TSNE(
                        n_components=target_dim,
                        perplexity=min(param, n_samples - 1),
                        metric='cosine',
                        init='pca',
                        learning_rate='auto',
                        random_state=42
                    ).fit_transform(Xα)
                    score_tsne = silhouette_score(ts, cluster_labels)
                except Exception as e:
                    logger.debug(f"TSNE failed for param={param}, alpha={alpha}: {e}")
                    ts, score_tsne = None, -1

                logger.info(f" param={param} ▶ umap={score_umap:.4f}  tsne={score_tsne:.4f}")

                # pick better
                if score_umap > best_score:
                    best_score, best_param, best_embedding, best_method = score_umap, param, um.tolist(), "umap"
                    best_reducer_obj = um_reducer
                if score_tsne > best_score:
                    best_score, best_param, best_embedding, best_method = score_tsne, param, ts.tolist(), "tsne"
                    best_reducer_obj = None  # tSNE has no .transform()

    logger.info(f"Best reduction ▶ param={best_param}, method={best_method}, score={best_score:.4f}")
    # ensure all numpy types become native Python scalars
    # if we never found a valid scoring embedding, just do a plain 3D reduction
    if best_param is None or best_embedding is None:
        logger.warning("No valid optimized embedding found — falling back to default 3D reduction")
       # choose the same default n_neighbors UMAP would use
        default_nbrs = min(15, n_samples - 1)
        fallback = reduce_embeddings(embedding_list, target_dim=target_dim)
        best_param = default_nbrs
        best_method = "umap"
        best_score = None
        best_embedding = fallback

    logger.info(f"Best reduction ▶ param={best_param}, method={best_method}, score={best_score}")
    global _last_fitted_reducer
    _last_fitted_reducer = best_reducer_obj
    return json.dumps({
        "best_param": int(best_param),
        "best_score": float(best_score) if best_score is not None else None,
        "embedding": best_embedding,
        "embedding_method": best_method
    }, default=lambda o: float(o) if isinstance(o, np.generic) else o)


def main(embeddings_json, n_neighbors=None, method=None):
    embedding_list = json.loads(embeddings_json)
    projected = reduce_embeddings(embedding_list, target_dim=2, n_neighbors=n_neighbors, method=method)
    return json.dumps(projected, default=lambda o: float(o) if isinstance(o, np.generic) else o)


def main3D(embeddings_json, n_neighbors=None, method=None):
    embedding_list = json.loads(embeddings_json)
    projected = reduce_embeddings(embedding_list, target_dim=3, n_neighbors=n_neighbors, method=method)
    return json.dumps(projected, default=lambda o: float(o) if isinstance(o, np.generic) else o)



def compute_axes_info(embeddings_json: str, cluster_labels_json: str) -> str:
    """
    Now returns standard XYZ axes; computes lengths/extremes along first 3 dims.
    """
    embs   = np.array(json.loads(embeddings_json), dtype=float)  # (N,D)
    labels = np.array(json.loads(cluster_labels_json), dtype=int) # (N,)

    # axes = identity
    axes = [[1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]]

    # project onto first 3 coords
    proj = embs[:, :3]  # (N,3)
    min_pt = proj.argmin(axis=0).tolist()
    max_pt = proj.argmax(axis=0).tolist()
    lengths = (proj.max(axis=0) - proj.min(axis=0)).tolist()

    # cluster centroids in same space
    unique = np.unique(labels)
    centroids = np.vstack([embs[labels==l, :3].mean(axis=0) for l in unique])  # (C,3)
    cproj     = centroids  # already in 3D
    min_cl = cproj.argmin(axis=0).tolist()
    max_cl = cproj.argmax(axis=0).tolist()

    result = {
        "axes":            axes,
        "lengths":         lengths,
        "min_point_idx":   min_pt,
        "max_point_idx":   max_pt,
        "min_cluster_idx": min_cl,
        "max_cluster_idx": max_cl
    }
    return json.dumps(result)


def compute_axes(points):
    """
    points: List of [x, y, z]
    Returns:
      R: 3×3 rotation matrix (list of lists)
      lengths: list of 3 floats (axis extents)
    """
    P = np.array(points, dtype=float)
    centroid = P.mean(axis=0)
    Pc = P - centroid

    # scale if your coordinates are on different units:
    scaler = StandardScaler()
    Psc = scaler.fit_transform(Pc)

    pca = PCA(n_components=3, svd_solver='full')
    pca.fit(Psc)

    # columns of R are the principal axes in world-space
    R = pca.components_.T.tolist()

    # axis lengths in original units (after centering):
    mins = Pc.min(axis=0)
    maxs = Pc.max(axis=0)
    lengths = (maxs - mins).tolist()

    return R, lengths
