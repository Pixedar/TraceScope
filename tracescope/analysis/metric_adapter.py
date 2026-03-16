# metric_adapter.py
import numpy as np
from itertools import combinations
from metric_learn import ITML
from sklearn.metrics.pairwise import cosine_distances

def learn_metric(X, user_clusters, gamma=0.05):
    """
    X: (n_samples, n_dims) L2-normalised embeddings
    user_clusters: list of lists of integer indices
    returns: L such that A = L.T @ L is the Mahalanobis matrix
    """
    pairs, y = [], []

    # must-link
    for cl in user_clusters:
        for i, j in combinations(cl, 2):
            pairs.append((i, j));  y.append(1)

    # cannot-link (only if >1 group)
    if len(user_clusters) > 1:
        for a, cl_a in enumerate(user_clusters):
            for cl_b in user_clusters[a+1:]:
                for i in cl_a:
                    for j in cl_b:
                        pairs.append((i, j));  y.append(-1)

    itml = ITML(gamma=gamma, prior='identity')
    itml.fit(X, np.array(pairs), np.array(y))

    # transformer_ is the lower-triangular L
    return itml.transformer_




def learn_metric_dim_red(X, user_clusters, gamma=0.05):
    """
    X: (n_samples, n_dims) L2-normalised embeddings
    user_clusters: list of lists of integer indices
    returns: L such that A = L.T @ L is the Mahalanobis matrix
    """
    pairs, y = [], []

    # must-link
    for cl in user_clusters:
        for i, j in combinations(cl, 2):
            pairs.append((i, j));  y.append(1)

    # cannot-link (only if >1 group)
    if len(user_clusters) > 1:
        for a, cl_a in enumerate(user_clusters):
            for cl_b in user_clusters[a+1:]:
                for i in cl_a:
                    for j in cl_b:
                        pairs.append((i, j));  y.append(-1)

    itml = ITML(gamma=gamma, prior='identity')
    pairs_arr = np.array(pairs)  # shape = (n_constraints,2)
    labels_arr = np.array(y)  # shape = (n_constraints,)
    itml.fit(pairs_arr, labels_arr)  # only two args
    L = itml.transformer_

    # transformer_ is the lower-triangular L
    return itml.transformer_


def safe_transform(X, L, alpha, corr_min=0.93):
    """
    Blend X_orig with X_metric = X @ L.T using alpha.
    If cosine-distance correlation drops below corr_min,
    return X_orig instead.
    """
    X_metric = X.dot(L.T)
    Xα = (1 - alpha) * X + alpha * X_metric

    corr = np.corrcoef(
        cosine_distances(X).ravel(),
        cosine_distances(Xα).ravel()
    )[0,1]

    return X if corr < corr_min else Xα
