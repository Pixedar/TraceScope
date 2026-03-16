# #!/usr/bin/env python3
# """
# qualia_segmenter_highdim.py — adaptive‑CROPS (no duplicates!)
# -------------------------------------------------------------
#
# Detect the start of the *current* qualia episode in raw 3072‑D embeddings.
#
# Fixes compared with the previous build
# ======================================
# * **Adaptive penalty path**: starts at 0.5 and multiplies by (1+step)
#   until > pen_max (default 20).  The step is 2 % by default, but the code
#   skips any penalty that yields the *same* breakpoint list as the
#   previous — so you get **one line per distinct segmentation**.
# * **Elbow selection**: pick the index with the *largest* cohesion jump
#   (relative drop ≥ tol, default 5 %).  Falls back to the finest level
#   with > 1 segment if no clear elbow.
# """
# from __future__ import annotations
#
# import json
# from typing import List, Tuple
#
# import numpy as np
#
# try:
#     import ruptures as rpt
# except ModuleNotFoundError as e:
#     raise ModuleNotFoundError(
#         "qualia_segmenter_highdim requires the 'ruptures' package.\n"
#         "Install it with:  pip install ruptures"
#     ) from e
#
#
#
#
# from typing import List, Optional
#
#
#
# from sklearn.metrics import calinski_harabasz_score
#
#
# # --------------------------------------------------------------------------- #
# # Cosine‑velocity                                                             #
# # --------------------------------------------------------------------------- #
# def _cosine_velocity(x: np.ndarray) -> np.ndarray:
#     """Angular ‘speed’ between consecutive embeddings (verbose)."""
#     print(f"[cos-vel] embeddings.shape = {x.shape}")
#     eps = 1e-12
#     norms = np.linalg.norm(x, axis=1, keepdims=True) + eps
#     u = x / norms
#     cos_sim = np.sum(u[:-1] * u[1:], axis=1)
#     v = 1.0 - np.clip(cos_sim, -1.0, 1.0)
#     print(f"[cos-vel] velocity length={len(v)}, first 5: {v[:5]}")
#     return v
#
#
# # --------------------------------------------------------------------------- #
# # Adaptive CROPS: *only* distinct breakpoint sets                             #
# # --------------------------------------------------------------------------- #
# def crops_distinct(
#     velocity: np.ndarray,
#     *,
#     pen_min: float = 0.5,
#     pen_max: float = 20.0,
#     rel_step: float = 0.02,
#     min_size: int = 5,
#     model: str = "rbf",
#     max_levels: int = 200,
# ) -> List[Tuple[float, List[int]]]:
#     """
#     March through penalty space multiplicatively:
#         pen *= (1 + rel_step)
#     but **emit a row only when the breakpoint list changes**.
#     """
#     algo = rpt.Pelt(model=model, min_size=min_size).fit(velocity.reshape(-1, 1))
#
#     path: List[Tuple[float, List[int]]] = []
#     p = pen_min
#     last_bkps: List[int] | None = None
#     levels = 0
#
#     while p <= pen_max and levels < max_levels:
#         bkps = algo.predict(pen=p)
#         if last_bkps is None or bkps != last_bkps:
#             path.append((p, bkps))
#             last_bkps = bkps
#             levels += 1
#         p *= 1.0 + rel_step
#
#         # stop early once only one segment is left
#         if last_bkps == [len(velocity)]:
#             break
#
#     # ─── Debug dump ──────────────────────────────────────────────────────
#     print("\n[CROPS‑distinct] Hierarchy (unique levels):")
#     for p, b in path:
#         print(f"    pen={p:8.3f} → {b}")
#     print("[CROPS‑distinct] ────────────────\n")
#     # ────────────────────────────────────────────────────────────────────
#     return path
#
#
# # --------------------------------------------------------------------------- #
# # Cohesion score                                                              #
# # --------------------------------------------------------------------------- #
# def _cohesion(emb: np.ndarray, bkps: List[int]) -> float:
#     """Davies–Bouldin‑style cohesion (∞ if only one segment)."""
#     if len(bkps) <= 1:         # one segment
#         return float("inf")
#
#     start, W, M = 0, [], []
#     for b in bkps:
#         seg = emb[start:b]
#         W.append(seg.var(axis=0).mean())
#         M.append(seg.mean(axis=0))
#         start = b
#
#     intra = float(np.mean(W))
#     inter = float(
#         np.mean([np.linalg.norm(a - b)
#                  for i, a in enumerate(M) for j, b in enumerate(M) if j > i])
#     )
#     return intra / (inter + 1e-9)
#
#
# # --------------------------------------------------------------------------- #
# # Choose elbow by largest cohesion drop                                       #
# # --------------------------------------------------------------------------- #
# def pick_elbow(
#     levels: List[Tuple[float, List[int]]],
#     emb: np.ndarray,
#     tol: float = 0.05,
# ) -> Tuple[float, List[int]]:
#     """
#     Return the (penalty, bkps) pair with the biggest *relative* cohesion drop
#     (score[i] / score[i+1] ≥ 1 + tol).  If none qualify, pick the finest
#     level that has more than one segment, otherwise the coarsest.
#     """
#     scores = [(_cohesion(emb, b)) for _, b in levels]
#
#     print("[cohesion] level → score:")
#     for (p, _), s in zip(levels, scores):
#         print(f"    {p:8.3f} → {s if np.isfinite(s) else '∞'}")
#
#     best_idx = None
#     best_ratio = 1.0
#     for i in range(len(scores) - 1):
#         if not (np.isfinite(scores[i]) and np.isfinite(scores[i + 1])):
#             continue
#         ratio = scores[i] / scores[i + 1]
#         if ratio >= 1.0 + tol and ratio > best_ratio:
#             best_ratio = ratio
#             best_idx = i + 1      # keep the *lower* score side (i+1)
#
#     if best_idx is None:
#         # fall‑back: finest >1‑segment level, else last
#         for i in reversed(range(len(levels))):
#             if len(levels[i][1]) > 1:
#                 best_idx = i
#                 break
#         if best_idx is None:
#             best_idx = len(levels) - 1
#
#     print(f"[cohesion] picked elbow @ pen={levels[best_idx][0]:.3f}, "
#           f"ratio={best_ratio:.3f}\n")
#     return levels[best_idx]
#
# # ─── Vendored sliding_window_view ────────────────────────────────────────
# try:
#     # Use NumPy’s built-in if available
#     from numpy.lib.stride_tricks import sliding_window_view
# except ImportError:
#     # Otherwise, fall back to as_strided for 2-D arrays + full-width windows
#     from numpy.lib.stride_tricks import as_strided
#
#     def sliding_window_view(x: np.ndarray, window_shape):
#         """
#         Vendored replacement for numpy>=1.20:
#         x: array of shape (T, D)
#         window_shape: tuple (w, D)
#         returns view of shape (T-w+1, w, D)
#         """
#         w, D = window_shape
#         T, D0 = x.shape
#         if D0 != D:
#             raise ValueError(f"sliding_window_view: expected second dim={D}, got {D0}")
#         # If window longer than axis, return empty as numpy would
#         if T < w:
#             return np.empty((0, w, D), dtype=x.dtype)
#         shape = (T - w + 1, w, D)
#         strides = (x.strides[0], x.strides[0], x.strides[1])
#         return as_strided(x, shape=shape, strides=strides)
# # --------------------------------------------------------------------------- #
# # Recent‑segment extractor                                                    #
# # --------------------------------------------------------------------------- #
# def _recent_slice(
#     embeddings: np.ndarray,
#     *,
#     pen_min: float = 0.5,
#     pen_max: float = 20.0,
#     rel_step: float = 0.02,
#     min_size: int = 5,
#     model: str = "rbf",
#     tol: float = 0.05,
# ) -> Tuple[np.ndarray, int, List[int]]:
#     """
#     Returns (recent_slice, start_idx, chosen_breakpoints).
#     """
#     if embeddings.shape[0] < 2:
#         return np.empty((0, embeddings.shape[1])), 0, [0]
#
#     # ─── SMOOTH A 7-DAY WINDOW ─────────────────────────────────────────
# ###
#     win     = 57
#     pad     = win // 2
#     emb_pad = np.pad(embeddings, [(pad, pad), (0, 0)], mode="edge")
#     # exactly the same call signature as before:
#     embeddings_smooth = sliding_window_view(
#         emb_pad,
#         (win, embeddings.shape[1])
#     ).mean(axis=2)
#
#     velocity = _cosine_velocity(embeddings_smooth)
#     levels = crops_distinct(
#         velocity,
#         pen_min=pen_min,
#         pen_max=pen_max,
#         rel_step=rel_step,
#         min_size=min_size,
#         model=model,
#     )
#     pen_sel, bkps_sel = pick_elbow(levels, embeddings, tol=tol)
#
#     # --- Pretty print ------------------------------------------------------
#     print("[hierarchy] Nested (unique) tree; picked level marked:")
#     for p, b in levels:
#         prefix = "└─" if p == pen_sel else "├─"
#         seg_lens = list(np.diff([0] + b))
#         mark = "   ← picked" if p == pen_sel else ""
#         print(f"{prefix} pen={p:8.3f} → {b}\n    seg‑lens: {seg_lens}{mark}")
#     print("[hierarchy] ───────────────────────────\n")
#     # ----------------------------------------------------------------------
#
#     start_idx = bkps_sel[-2] + 1 if len(bkps_sel) >= 2 else 0
#     return embeddings[start_idx:].copy(), start_idx, bkps_sel
#
#
# # --------------------------------------------------------------------------- #
# # Public API                                                                  #
# # --------------------------------------------------------------------------- #
# def recent_qualia_from_json(
#     embeddings_json: str,
#     *,
#     pen_min: float = 0.5,
#     pen_max: float = 20.0,
#     rel_step: float = 0.02,
#     min_size: int = 5,
#     model: str = "rbf",
#     tol: float = 0.05,
# ) -> str:
#     """
#     Parse a JSON list‑of‑lists, extract the *current* qualia episode,
#     and return compact JSON with keys:
#         recent_embeddings, start_idx, breakpoints
#     """
#     print("[API] Parsing embeddings JSON …")
#     try:
#         arr = np.asarray(json.loads(embeddings_json), dtype=float)
#     except Exception as exc:
#         raise ValueError(
#             "Failed to parse embeddings_json — must be a JSON list of lists."
#         ) from exc
#     if arr.ndim != 2:
#         raise ValueError("Input embeddings must be 2‑D.")
#
#     recent, start_idx, bkps = _recent_slice(
#         arr,
#         pen_min=pen_min,
#         pen_max=pen_max,
#         rel_step=rel_step,
#         min_size=min_size,
#         model=model,
#         tol=tol,
#     )
#     print(f"[API] start_idx={start_idx}, bkps={bkps}, recent_len={len(recent)}")
#
#     result = {
#         "recent_embeddings": recent.tolist(),
#         "start_idx": int(start_idx),
#         "breakpoints": [int(i) for i in bkps],
#     }
#     print("[API] Done.\n")
#     return json.dumps(result, separators=(",", ":"))
#
#
#
# def compute_distances_from_centroids(
#     embeddings: np.ndarray, labels: np.ndarray
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Given embeddings of shape (N, D) and integer cluster labels of length N,
#     compute for each point the Euclidean distance to each cluster centroid.
#     Returns (distances of shape (N, C), unique_labels).
#     """
#     unique_labels = np.unique(labels)
#     centroids = np.vstack([
#         embeddings[labels == c].mean(axis=0)
#         for c in unique_labels
#     ])  # shape (C, D)
#     dists = np.linalg.norm(
#         embeddings[:, None, :] - centroids[None, :, :], axis=2
#     )
#     return dists, unique_labels
#
#
# def find_optimal_breakpoints(
#     features: np.ndarray,
#     target_segments: int,
#     min_size: int = 15
# ) -> List[int]:
#     """
#     Search breakpoints for K in {target-1, target, target+1} via Dynp,
#     enforcing minimum segment length and selecting by Calinski-Harabasz.
#     Returns list of end indices.
#     """
#     n, _ = features.shape
#     candidate_K = [k for k in (target_segments - 1, target_segments, target_segments + 1)
#                    if k >= 2 and k <= n // min_size]
#     best_score = -np.inf
#     best_bkps: List[int] = []
#
#     for K in candidate_K:
#         algo = rpt.Dynp(model="l2").fit(features)
#         bkps = algo.predict(n_bkps=K - 1)
#         seg_lens = np.diff([0] + bkps)
#         if seg_lens.min() < min_size:
#             continue
#         # assign segment ids
#         seg_labels = np.zeros(n, dtype=int)
#         start = 0
#         for idx, bp in enumerate(bkps):
#             seg_labels[start:bp] = idx
#             start = bp
#         score = calinski_harabasz_score(features, seg_labels)
#         if score > best_score:
#             best_score = score
#             best_bkps = bkps
#
#     if not best_bkps:
#         # fallback: target without size constraint
#         algo = rpt.Dynp(model="l2").fit(features)
#         best_bkps = algo.predict(n_bkps=target_segments - 1)
#
#     return best_bkps
#
#
# def all_segments_from_json(
#     embeddings_json: str,
#     labels_json: Optional[str] = None,
#     min_size: int = 15
# ) -> str:
#     """
#     If labels_json provided, reduce embeddings to centroid-distances,
#     segment via Dynp and select best K. Otherwise legacy CROPS pipeline.
#     Returns JSON list of {level, start, end}.
#     """
#     # Parse embeddings into array
#     data = np.asarray(json.loads(embeddings_json), dtype=float)
#     N = data.shape[0]
#     print(f"[DEBUG] Total points (N): {N}")
#
#     # Determine segmentation levels
#     if labels_json:
#         print("[DEBUG] Using label-based segmentation")
#         labels = np.asarray(json.loads(labels_json), dtype=int)
#         print(f"[DEBUG] Labels unique: {np.unique(labels)}")
#         features, unique_labels = compute_distances_from_centroids(data, labels)
#         print(f"[DEBUG] Distance features shape: {features.shape}")
#         target = len(unique_labels)
#         print(f"[DEBUG] Target segments (cluster count): {target}")
#         bkps = find_optimal_breakpoints(features, target, min_size=min_size)
#         print(f"[DEBUG] Raw breakpoints: {bkps}")
#         levels: List[Tuple[int, List[int]]] = [(target, bkps)]
#     else:
#         print("[DEBUG] Using legacy velocity-based CROPS segmentation")
#         from .qualia_segmenter_highdim import _cosine_velocity, crops_distinct
#         vel = _cosine_velocity(data)
#         print(f"[DEBUG] Velocity vector length: {len(vel)}")
#         levels = crops_distinct(vel, min_size=min_size)
#         print(f"[DEBUG] CROPS levels (penalty, breakpoints): {levels}")
#
#     # Clip breakpoints to valid range [1, N-1]
#     max_idx = N - 1
#     clipped_levels: List[Tuple[int, List[int]]] = []
#     for lvl, bps in levels:
#         clipped = []
#         for bp in bps:
#             bp_int = int(bp)
#             bp_clamped = max(1, min(bp_int, max_idx))
#             clipped.append(bp_clamped)
#         clipped_levels.append((lvl, clipped))
#     print(f"[DEBUG] Clipped levels: {clipped_levels}")
#     levels = clipped_levels
#
#     # Build segment dicts
#     segs: List[dict[str, int]] = []
#     for level_idx, (_, bkps) in enumerate(levels):
#         prev = 0
#         for bp in sorted(bkps):
#             seg = {"level": level_idx, "start": prev, "end": bp}
#             segs.append(seg)
#             print(f"[DEBUG] Added segment: {seg}")
#             prev = bp
#
#         result_json = json.dumps(segs, separators=(',', ':'))
#     print(f"[DEBUG] Final segments JSON: {result_json}")
#     return result_json
#

#!/usr/bin/env python3
"""
qualia_segmenter_highdim_hybrid.py — adaptive‑CROPS + hybrid centroid/velocity
==============================================================================

This file **extends** the original *qualia_segmenter_highdim.py* while **keeping
all earlier public APIs and helpers intact**.  Nothing was removed; new logic is
only *added* or *augments* existing functions.

New in this build
-----------------
1. **Hybrid representation & feature stream**
   • PCA (98 % variance) → optional LDA "emotion tilt" if cluster labels given.
   • Concatenate centroid‑distance matrix **and** Savitzky–Golay‑smoothed
     cosine‑velocity into one feature matrix.
2. **Windowed smoothing without phase‑lag**
   • Replaces the asymmetric 57‑point trailing mean by a **symmetric
     Savitzky–Golay filter** (window = 9 days, poly = 3).
3. **Post‑merge tiny fragments** (< `min_size` days) after the first cut to kill
   spurious micro‑segments.

Public API functions *recent_qualia_from_json* and *all_segments_from_json*
still take **exactly** the same arguments and return the same JSON payloads.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Standard & third‑party imports (original + new)                             #
# --------------------------------------------------------------------------- #
import json
from typing import List, Tuple, Optional

import numpy as np

try:
    import ruptures as rpt
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "qualia_segmenter_highdim_hybrid requires the 'ruptures' package.\n"
        "Install it with:  pip install ruptures"
    ) from e

from sklearn.metrics import calinski_harabasz_score

# Added for hybrid representation
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.signal import savgol_filter

# --------------------------------------------------------------------------- #
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ORIGINAL HELPERS (UNMODIFIED)                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# --------------------------------------------------------------------------- #
# Cosine‑velocity                                                             #
# --------------------------------------------------------------------------- #





def _semantic_cuts_within(
    emb_raw: np.ndarray,
    labels_arr: Optional[np.ndarray],
    start: int,
    end: int,
    *,
    min_size: int,
    tol: float = 0.05,
) -> list[int]:
    """
    Propose internal breakpoint indices (GLOBAL) inside [start, end]
    using the SAME semantics you use globally:
      • hybrid Dynp when labels are present (distance + vel features)
      • CROPS+elbow on velocity otherwise.
    Returns a (possibly empty) list of GLOBAL bp indices.
    """
    seg_len = end - start + 1
    if seg_len < 2 * min_size:
        return []

    emb_seg = emb_raw[start:end + 1]
    labels_seg = None if labels_arr is None else labels_arr[start:end + 1]

    # Representation + features
    emb_h, _pca, _lda = _pca_lda_transform(emb_seg, labels_seg)
    feats, uniq = _build_hybrid_features(emb_h, labels_seg)

    # Candidate breakpoints (LOCAL indices that are *ends* of segments)
    if uniq is not None and len(uniq) > 1:
        target = len(uniq)
        bkps_local = find_optimal_breakpoints(feats, target, min_size=min_size)
    else:
        vel = feats[:, -1]
        levels = crops_distinct(vel, min_size=min_size)
        _, bkps_local = pick_elbow(levels, emb_seg, tol=tol)

    # Map to GLOBAL indices and keep only internal breakpoints
    bkps_global = [start + int(b) for b in bkps_local if start < start + int(b) < end]

    # Enforce min_size within this segment
    if not bkps_global:
        return []

    # Drop any bkps that would create < min_size pieces
    candidate = sorted(bkps_global)
    pieces = [start] + candidate + [end + 1]
    if any((pieces[i+1] - pieces[i]) < min_size for i in range(len(pieces)-1)):
        # If proposed cut violates min_size, reject
        return []

    return candidate


def _refine_long_segments_semantic(
    emb_raw: np.ndarray,
    labels_arr: Optional[np.ndarray],
    segs: list[dict[str, int]],
    *,
    min_size: int,
    max_size: int,
    tol: float = 0.05,
) -> list[dict[str, int]]:
    """
    For each segment exceeding max_size, try semantic splits.
    If no elbow-based split is found, fall back to a single
    peak-velocity cut (semantic) provided it respects min_size.
    """
    changed = True
    while changed:
        changed = False
        next_segs: list[dict[str, int]] = []
        for seg in segs:
            s, e = seg["start"], seg["end"]
            length = e - s + 1
            if length <= max_size:
                next_segs.append(seg)
                continue

            # 1) Try semantic cuts (may return multiple splits)
            cuts = _semantic_cuts_within(emb_raw, labels_arr, s, e, min_size=min_size, tol=tol)

            # 2) If no semantic elbow, cut at peak velocity once (still semantic)
            if not cuts:
                emb_seg = emb_raw[s:e+1]
                emb_h, _pca, _lda = _pca_lda_transform(emb_seg, None if labels_arr is None else labels_arr[s:e+1])
                feats, _ = _build_hybrid_features(emb_h, None if labels_arr is None else labels_arr[s:e+1])
                vel = feats[:, -1].ravel()
                # avoid borders by min_size
                lo, hi = min_size, len(vel) - min_size
                if hi > lo:
                    cut_local = int(np.argmax(vel[lo:hi]) + lo)
                    cut_global = s + cut_local
                    # only accept if it helps satisfy the cap
                    if (cut_global - s + 1) >= min_size and (e - cut_global) + 1 >= min_size:
                        cuts = [cut_global]

            if not cuts:
                # nothing we can do; keep as-is
                next_segs.append(seg)
                continue

            # Split along accepted cuts
            parts = [s] + sorted(cuts) + [e]
            for i in range(len(parts) - 1):
                a, b = parts[i], parts[i + 1]
                next_segs.append({"level": seg["level"], "start": a, "end": b})

            changed = True

        segs = next_segs

        # If any piece STILL exceeds max_size, iterate (recursive refinement)
        if any(seg["end"] - seg["start"] + 1 > max_size for seg in segs):
            continue
    return segs


def _cosine_velocity(x: np.ndarray) -> np.ndarray:
    """Angular ‘speed’ between consecutive embeddings (verbose)."""
    print(f"[cos‑vel] embeddings.shape = {x.shape}")
    eps = 1e-12
    norms = np.linalg.norm(x, axis=1, keepdims=True) + eps
    u = x / norms
    cos_sim = np.sum(u[:-1] * u[1:], axis=1)
    v = 1.0 - np.clip(cos_sim, -1.0, 1.0)
    print(f"[cos‑vel] velocity length={len(v)}, first 5: {v[:5]}")
    return v

# --------------------------------------------------------------------------- #
# Adaptive CROPS: *only* distinct breakpoint sets                             #
# --------------------------------------------------------------------------- #

def crops_distinct(
    velocity: np.ndarray,
    *,
    pen_min: float = 0.5,
    pen_max: float = 20.0,
    rel_step: float = 0.02,
    min_size: int = 15,
    model: str = "rbf",
    max_levels: int = 200,
) -> List[Tuple[float, List[int]]]:
    """Multiplicative grid search through penalty space yielding *distinct* cuts."""
    algo = rpt.Pelt(model=model, min_size=min_size).fit(velocity.reshape(-1, 1))

    path: List[Tuple[float, List[int]]] = []
    p = pen_min
    last_bkps: List[int] | None = None
    levels = 0

    while p <= pen_max and levels < max_levels:
        bkps = algo.predict(pen=p)
        if last_bkps is None or bkps != last_bkps:
            path.append((p, bkps))
            last_bkps = bkps
            levels += 1
        p *= 1.0 + rel_step
        if last_bkps == [len(velocity)]:  # single segment left
            break

    # Debug
    print("\n[CROPS‑distinct] Hierarchy (unique levels):")
    for p, b in path:
        print(f"    pen={p:8.3f} → {b}")
    print("[CROPS‑distinct] ────────────────\n")
    return path

# --------------------------------------------------------------------------- #
# Cohesion score                                                              #
# --------------------------------------------------------------------------- #

def _cohesion(emb: np.ndarray, bkps: List[int]) -> float:
    """Davies–Bouldin‑style cohesion (∞ if only one segment)."""
    if len(bkps) <= 1:
        return float("inf")
    start, W, M = 0, [], []
    for b in bkps:
        seg = emb[start:b]
        W.append(seg.var(axis=0).mean())
        M.append(seg.mean(axis=0))
        start = b
    intra = float(np.mean(W))
    inter = float(np.mean([
        np.linalg.norm(a - b)
        for i, a in enumerate(M) for j, b in enumerate(M) if j > i]))
    return intra / (inter + 1e-9)


def _candidate_ks(n: int, min_size: int, target: int | None = None) -> list[int]:
    """
    Return a smart range of K to test.
       • always ≥ 2 slices
       • never smaller than min_size
       • capped at 10 to keep Dynp fast
       • if target is given (labels), bias towards ±3 around it
    """
    k_max = max(2, min(10, n // min_size))
    if target is None or target < 2:
        return list(range(2, k_max + 1))

    window = range(max(2, target - 3), min(k_max, target + 3) + 1)
    # add the extremes 2 and k_max so we don’t miss very coarse/fine cuts
    return sorted(set(window).union({2, k_max}))
# --------------------------------------------------------------------------- #
# Choose elbow by largest cohesion drop                                       #
# --------------------------------------------------------------------------- #

def pick_elbow(
    levels: List[Tuple[float, List[int]]],
    emb: np.ndarray,
    tol: float = 0.05,
) -> Tuple[float, List[int]]:
    scores = [(_cohesion(emb, b)) for _, b in levels]
    print("[cohesion] level → score:")
    for (p, _), s in zip(levels, scores):
        print(f"    {p:8.3f} → {s if np.isfinite(s) else '∞'}")

    best_idx, best_ratio = None, 1.0
    for i in range(len(scores) - 1):
        if not (np.isfinite(scores[i]) and np.isfinite(scores[i + 1])):
            continue
        ratio = scores[i] / scores[i + 1]
        if ratio >= 1.0 + tol and ratio > best_ratio:
            best_ratio, best_idx = ratio, i + 1  # keep lower‑score side

    if best_idx is None:
        for i in reversed(range(len(levels))):  # finest level with >1 segment
            if len(levels[i][1]) > 1:
                best_idx = i
                break
        if best_idx is None:
            best_idx = len(levels) - 1

    print(f"[cohesion] picked elbow @ pen={levels[best_idx][0]:.3f}, "
          f"ratio={best_ratio:.3f}\n")
    return levels[best_idx]

# --------------------------------------------------------------------------- #
# Vendored sliding_window_view (NumPy ≥1.20 ships it natively)                #
# --------------------------------------------------------------------------- #
try:
    from numpy.lib.stride_tricks import sliding_window_view  # type: ignore
except ImportError:  # <1.20 – vendored replacement for 2‑D arrays
    from numpy.lib.stride_tricks import as_strided  # type: ignore

    def sliding_window_view(x: np.ndarray, window_shape):
        """Simple 2‑D sliding window replacement."""
        w, D = window_shape
        T, D0 = x.shape
        if D0 != D:
            raise ValueError("sliding_window_view: dimension mismatch")
        if T < w:
            return np.empty((0, w, D), dtype=x.dtype)
        shape = (T - w + 1, w, D)
        strides = (x.strides[0], x.strides[0], x.strides[1])
        return as_strided(x, shape=shape, strides=strides)

# --------------------------------------------------------------------------- #
# Recent‑segment extractor                                                    #
# --------------------------------------------------------------------------- #

def _recent_slice(
    embeddings: np.ndarray,
    *,
    pen_min: float = 0.5,
    pen_max: float = 20.0,
    rel_step: float = 0.02,
    min_size: int = 5,
    model: str = "rbf",
    tol: float = 0.05,
) -> Tuple[np.ndarray, int, List[int]]:
    """Returns (recent_slice, start_idx, chosen_breakpoints) for *legacy* path."""
    if embeddings.shape[0] < 2:
        return np.empty((0, embeddings.shape[1])), 0, [0]

    # Symmetric 57‑day smoothing (unchanged)
    win = 57
    pad = win // 2
    emb_pad = np.pad(embeddings, [(pad, pad), (0, 0)], mode="edge")
    embeddings_smooth = sliding_window_view(emb_pad, (win, embeddings.shape[1])).mean(axis=2)

    velocity = _cosine_velocity(embeddings_smooth)
    levels = crops_distinct(
        velocity,
        pen_min=pen_min,
        pen_max=pen_max,
        rel_step=rel_step,
        min_size=min_size,
        model=model,
    )
    _pen_sel, bkps_sel = pick_elbow(levels, embeddings, tol=tol)

    start_idx = bkps_sel[-2] + 1 if len(bkps_sel) >= 2 else 0
    return embeddings[start_idx:].copy(), start_idx, bkps_sel

# --------------------------------------------------------------------------- #
# Public legacy API                                                           #
# --------------------------------------------------------------------------- #

def recent_qualia_from_json(
    embeddings_json: str,
    *,
    pen_min: float = 0.5,
    pen_max: float = 20.0,
    rel_step: float = 0.02,
    min_size: int = 5,
    model: str = "rbf",
    tol: float = 0.05,
) -> str:
    """Legacy API — unchanged."""
    print("[API] Parsing embeddings JSON …")
    try:
        arr = np.asarray(json.loads(embeddings_json), dtype=float)
    except Exception as exc:
        raise ValueError("Failed to parse embeddings_json — must be JSON list of lists.") from exc
    if arr.ndim != 2:
        raise ValueError("Input embeddings must be 2‑D.")

    recent, start_idx, bkps = _recent_slice(
        arr,
        pen_min=pen_min,
        pen_max=pen_max,
        rel_step=rel_step,
        min_size=min_size,
        model=model,
        tol=tol,
    )
    result = {
        "recent_embeddings": recent.tolist(),
        "start_idx": int(start_idx),
        "breakpoints": [int(i) for i in bkps],
    }
    return json.dumps(result, separators=(",", ":"))

# --------------------------------------------------------------------------- #
# Centroid distance helper                                                    #
# --------------------------------------------------------------------------- #

def compute_distances_from_centroids(
    embeddings: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    unique_labels = np.unique(labels)
    centroids = np.vstack([
        embeddings[labels == c].mean(axis=0) for c in unique_labels
    ])
    dists = np.linalg.norm(embeddings[:, None, :] - centroids[None, :, :], axis=2)
    return dists, unique_labels

# --------------------------------------------------------------------------- #
# Dynp breakpoint search (original)                                           #
# --------------------------------------------------------------------------- #

def find_optimal_breakpoints(
    features: np.ndarray,
    target_segments: int,
    *,
    min_size: int = 15,
) -> List[int]:
    n, _ = features.shape
    candidate_K = [k for k in (target_segments - 1, target_segments, target_segments + 1)
                   if k >= 2 and k <= n // min_size]
    best_score, best_bkps = -np.inf, []
    for K in candidate_K:
        algo = rpt.Dynp(model="l2").fit(features)
        bkps = algo.predict(n_bkps=K - 1)
        if np.min(np.diff([0] + bkps)) < min_size:
            continue
        seg_labels = np.zeros(n, dtype=int)
        start = 0
        for idx, bp in enumerate(bkps):
            seg_labels[start:bp] = idx
            start = bp
        score = calinski_harabasz_score(features, seg_labels)
        if score > best_score:
            best_score, best_bkps = score, bkps
    if not best_bkps:  # fallback no size constraint
        algo = rpt.Dynp(model="l2").fit(features)
        best_bkps = algo.predict(n_bkps=target_segments - 1)
    return best_bkps

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                ███   H Y B R I D   E X T E N S I O N   ███              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# --------------------------------------------------------------------------- #
# Hybrid representation (PCA → optional LDA)                                 #
# --------------------------------------------------------------------------- #

def _pca_lda_transform(
    emb: np.ndarray,
    labels: Optional[np.ndarray] = None,
    *,
    var_keep: float = 0.98,
) -> Tuple[np.ndarray, PCA, Optional[LDA]]:
    print(f"[hyb] PCA→LDA transform: input shape {emb.shape}")
    pca = PCA(n_components=var_keep, whiten=True, svd_solver="full")
    emb_pca = pca.fit_transform(emb)
    print(f"[hyb] PCA kept {emb_pca.shape[1]} dims (≈{var_keep:.0%} var)")
    lda_model: Optional[LDA] = None
    if labels is not None and len(np.unique(labels)) > 1:
        lda_model = LDA()
        emb_final = lda_model.fit_transform(emb_pca, labels).astype(float)
        print(f"[hyb] LDA projected to {emb_final.shape[1]} dims")
    else:
        emb_final = emb_pca.astype(float)
        print("[hyb] LDA skipped")
    return emb_final, pca, lda_model

# --------------------------------------------------------------------------- #
# Build hybrid feature matrix                                                 #
# --------------------------------------------------------------------------- #

def _build_hybrid_features(
    emb_h: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    # centroid distances (cluster aware)
    if labels is not None and len(np.unique(labels)) >= 2:
        dists, uniq = compute_distances_from_centroids(emb_h, labels)
        print(f"[hyb] dists shape {dists.shape}")
    else:
        dists = np.zeros((emb_h.shape[0], 1))
        uniq = None
        print("[hyb] centroid distances skipped")
    # cosine velocity with Savitzky‑Golay smoothing
    raw_vel = _cosine_velocity(emb_h)
    raw_vel = np.r_[raw_vel[0], raw_vel]  # pad front
    vel = savgol_filter(raw_vel, window_length=9, polyorder=3, mode="interp").reshape(-1, 1)
    features = np.hstack([dists, vel])
    print(f"[hyb] features matrix {features.shape}")
    return features, uniq

# --------------------------------------------------------------------------- #
# Post‑merge tiny segments                                                    #
# --------------------------------------------------------------------------- #

def _merge_small_segments(bkps: List[int], n_points: int, *, min_size: int) -> List[int]:
    if not bkps:
        return bkps
    seg_starts = [0] + bkps[:-1]
    seg_ends = bkps.copy()
    seg_sizes = [e - s for s, e in zip(seg_starts, seg_ends)]
    changed = True
    while changed and len(seg_sizes) > 1:
        changed = False
        for idx, size in enumerate(seg_sizes):
            # skip merging tail segment to preserve last small slice
            if idx == len(seg_sizes) - 1:
                continue
            if size >= min_size:
                continue
            # pick neighbour with larger size
            if idx == 0:
                merge_with = 1
            else:
                left = seg_sizes[idx - 1]
                right = seg_sizes[idx + 1]
                merge_with = idx - 1 if left >= right else idx + 1
                # avoid merging into tail
                if merge_with == len(seg_sizes) - 1:
                    merge_with = idx - 1
            # drop the breakpoint that separates the two
            del seg_ends[max(idx, merge_with)]
            seg_starts = [0] + seg_ends[:-1]
            seg_sizes = [e - s for s, e in zip(seg_starts, seg_ends)]
            changed = True
            print(f"[hyb-merge] merged seg {idx} into {merge_with}, new seg_sizes {seg_sizes}")
            break
    return seg_ends

# --------------------------------------------------------------------------- #
# ★★★  HYBRID VERSION OF all_segments_from_json  ★★★                          #
# --------------------------------------------------------------------------- #

def all_segments_from_json(
    embeddings_json: str,
    labels_json: Optional[str] = None,
    *,
    min_size: int = 15,
    max_size: int = 42
) -> str:
    """Drop‑in replacement that prefers the hybrid path when labels provided."""
    emb_raw = np.asarray(json.loads(embeddings_json), dtype=float)
    N = emb_raw.shape[0]
    if emb_raw.ndim != 2:
        raise ValueError("Input embeddings must be 2‑D.")
    labels_arr: Optional[np.ndarray] = None
    if labels_json is not None:
        labels_arr = np.asarray(json.loads(labels_json), dtype=int)
        if len(labels_arr) != N:
            raise ValueError("labels_json length does not match embeddings")
    # representation
    emb_h, _pca, _lda = _pca_lda_transform(emb_raw, labels_arr)
    # feature matrix
    features_h, uniq = _build_hybrid_features(emb_h, labels_arr)
    # segmentation
    if uniq is not None and len(uniq) > 1:
        target = len(uniq)
        print(f"[API] Dynp target={target}")
        bkps = find_optimal_breakpoints(features_h, target, min_size=min_size)
    else:
        print("[API] Legacy CROPS path")
        vel = features_h[:, -1]
        levels = crops_distinct(vel, min_size=min_size)
        _, bkps = pick_elbow(levels, emb_raw)
    bkps = sorted(int(b) for b in bkps if 0 < b < N)
    print(f"[API] initial bkps {bkps}")
    bkps = _merge_small_segments(bkps, N, min_size=min_size)
    print(f"[API] bkps after merge {bkps}")
    # build seg dict list
    segs: List[dict[str, int]] = []
    prev = 0
    for bp in bkps:
        segs.append({"level": 0, "start": prev, "end": bp})
        prev = bp
    # (tail segment omitted to match legacy behavior) or, if desired, include capped tail:
    if prev < N:
        segs.append({"level": 0, "start": prev, "end": N-1})

    segs = _refine_long_segments_semantic(
        emb_raw, labels_arr, segs, min_size=min_size, max_size=max_size, tol=0.05
    )

    return json.dumps(segs, separators=(",", ":"))


def all_segments_from_java(
    embeddings: list[list[float]],
    labels: list[int] | None = None,
    *,
    min_size: int = 14,
    max_size: int = 42
) -> str:
    """
    Like all_segments_from_json, but accepts native Python lists
    (passed by Chaquopy from Java) to avoid any JSON buffering
    on the Java heap.
    """
    # 1) convert to numpy arrays
    emb_raw = np.asarray(embeddings, dtype=float)
    N = emb_raw.shape[0]
    if emb_raw.ndim != 2:
        raise ValueError("Input embeddings must be 2‑D.")

    labels_arr = None
    if labels is not None:
        labels_arr = np.asarray(labels, dtype=int)
        if labels_arr.shape[0] != N:
            raise ValueError("labels length does not match embeddings")

    # 2) exactly the same hybrid logic you already have:
    emb_h, _pca, _lda = _pca_lda_transform(emb_raw, labels_arr)
    features_h, uniq = _build_hybrid_features(emb_h, labels_arr)

    if uniq is not None and len(uniq) > 1:
        target = len(uniq)
        bkps = find_optimal_breakpoints(features_h, target, min_size=min_size)
    else:
        vel = features_h[:, -1]
        levels = crops_distinct(vel, min_size=min_size)
        _, bkps = pick_elbow(levels, emb_raw)

    # 3) finalize & merge small segments
    bkps = sorted(int(b) for b in bkps if 0 < b < N)
    bkps = _merge_small_segments(bkps, N, min_size=min_size)

    # 4) build the JSON response
    segs: list[dict[str,int]] = []
    prev = 0
    for bp in bkps:
        segs.append({"level": 0, "start": prev, "end": bp})
        prev = bp
    if prev < N:
        segs.append({"level": 0, "start": prev, "end": N - 1})

    segs = _refine_long_segments_semantic(
        emb_raw, labels_arr, segs, min_size=min_size, max_size=max_size, tol=0.05
    )

    # return exactly the same JSON as before
    return json.dumps(segs, separators=(",", ":"))