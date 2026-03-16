"""SHA-256 helpers for cache keys."""

import hashlib

import numpy as np


def sha256_key(*parts: str) -> str:
    """Create a SHA-256 hex digest from concatenated parts."""
    combined = "||".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def embeddings_fingerprint(embeddings: np.ndarray) -> str:
    """SHA-256 of per-dimension mean vector.

    Matching Android's DimensionReducerAdapter fingerprint approach:
    computes the mean across all entries for each dimension,
    then hashes the resulting vector.

    Args:
        embeddings: (N, D) array of embedding vectors.

    Returns:
        64-char hex digest string.
    """
    mean_vec = embeddings.mean(axis=0).astype(np.float32)
    return hashlib.sha256(mean_vec.tobytes()).hexdigest()
