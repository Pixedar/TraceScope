import json
import numpy as np

def compute_axes_info(embeddings_json: str, cluster_labels_json: str) -> str:
    """
    Inputs:
      - embeddings_json: JSON list of shape [N][D]
      - cluster_labels_json: JSON list of shape [N]
    Output JSON string with keys:
      • axes: [[x1,y1,z1], [x2,y2,z2], [x3,y3,z3]] (unit‑length PCA components)
      • lengths: length along each axis (max−min projection)
      • min_point_idx / max_point_idx: indices of points at each axis extreme
      • min_cluster_idx / max_cluster_idx: cluster index with extreme centroid
    """
    # 1) load
    embs   = np.array(json.loads(embeddings_json), dtype=float)  # (N,D)
    labels = np.array(json.loads(cluster_labels_json), dtype=int) # (N,)

    # 2) center
    center = embs.mean(axis=0)
    embs_c = embs - center

    # 3) PCA: cov → eigenvectors
    cov    = np.cov(embs_c, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    axes3 = evecs[:, order[:3]].T    # (3,D)

    # 4) project all points
    proj = embs.dot(axes3.T)         # (N,3)
    min_pt = proj.argmin(axis=0).tolist()
    max_pt = proj.argmax(axis=0).tolist()
    lengths = (proj.max(axis=0) - proj.min(axis=0)).tolist()

    # 5) cluster centroids
    unique = np.unique(labels)
    centroids = np.vstack([embs[labels==l].mean(axis=0) for l in unique])  # (C,D)
    cproj     = centroids.dot(axes3.T)  # (C,3)
    min_cl = cproj.argmin(axis=0).tolist()
    max_cl = cproj.argmax(axis=0).tolist()

    # 6) pack & return
    result = {
        "axes":             axes3.tolist(),
        "lengths":          lengths,
        "min_point_idx":    min_pt,
        "max_point_idx":    max_pt,
        "min_cluster_idx":  min_cl,
        "max_cluster_idx":  max_cl
    }
    return json.dumps(result)
