"""
AnalysisPipeline – orchestrates the full analysis flow:
  embed → cluster → reduce to 3D → compute axes → label axes →
  label clusters → train MDN flow model → build velocity grid → return AnalysisResult

Faithful port of the Android EmotionApp pipeline with:
  - Multi-part axis labeling (TF-IDF keyword evolution, avoid mechanism)
  - Cluster labeling with avoidDesc + keyword differentiation
  - MDN flow model training + 40³ velocity grid building
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Optional, List

import numpy as np

from tracescope.config import TraceScopeConfig
from tracescope.models.trace import TraceSession
from tracescope.models.analysis import AnalysisResult, ClusterResult, AxisInfo
from tracescope.providers.embedding import EmbeddingProvider
from tracescope.providers.llm import LLMProvider
from tracescope.storage.vector_store import VectorStore
from tracescope.storage.cache import LLMResponseCache, ResultCache
from tracescope.analysis.explainer import SemanticExplainer
from tracescope.utils.hashing import embeddings_fingerprint

logger = logging.getLogger(__name__)


def _build_density_grid(grid_pts, data_pts, grid_size, path_ids=None):
    """Compute a data-density confidence grid from path coverage.

    For each grid cell, measures how close it is to real training paths.
    Grid cells near data get confidence ~1.0; empty regions far from any
    path get confidence ~0.0.

    The bandwidth is set to 3× the grid cell diagonal so that any cell
    within a few cells of real data stays bright, and only truly empty
    regions fade out.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(data_pts, dtype=np.float32)

    # Build dense samples along path segments (not just scatter points)
    if path_ids is not None:
        pid = np.asarray(path_ids)
        same_path = pid[:-1] == pid[1:]
    else:
        same_path = np.ones(len(pts) - 1, dtype=bool)

    samples = [pts]
    for i in np.where(same_path)[0]:
        seg_len = float(np.linalg.norm(pts[i + 1] - pts[i]))
        if seg_len < 1e-8:
            continue
        median_step = float(np.median(np.linalg.norm(
            np.diff(pts, axis=0), axis=1)) + 1e-8)
        n_interp = max(2, int(np.ceil(seg_len * 4 / median_step)))
        n_interp = min(n_interp, 10)
        ts = np.linspace(0, 1, n_interp + 2)[1:-1]
        for t in ts:
            samples.append((pts[i] * (1 - t) + pts[i + 1] * t).reshape(1, 3))

    all_samples = np.concatenate(samples, axis=0)

    tree = cKDTree(all_samples)
    dist_to_nearest, _ = tree.query(grid_pts, k=1)

    # Adaptive bandwidth: set so that the median-distance grid cell
    # gets confidence ≈ 0.5.  This guarantees meaningful contrast —
    # cells closer than median → conf > 0.5 (bright),
    # cells farther than median → conf < 0.5 (fading).
    bandwidth = float(np.median(dist_to_nearest))
    bandwidth = max(bandwidth, 1e-6)

    confidence = np.exp(-(dist_to_nearest / bandwidth) ** 2).astype(np.float32)
    return confidence.reshape(grid_size, grid_size, grid_size)


def _build_velocity_grid(axis_min, axis_max, grid_size=40,
                         data_points=None, path_ids=None):
    """Build velocity grid from a trained flow model (MDN or RBF).

    Samples the model's velocity at each point on a grid_size³ lattice.
    Computes a data-density confidence grid so that regions far from any
    real training data (where the model is extrapolating / "hallucinating")
    can be faded out.

    Args:
        data_points: (N, 3) array of projected_3d training points, used
            to compute the density-based confidence grid.
        path_ids: list of path IDs per point, so density accounts for
            interpolated path segments (not just scatter points).

    Returns:
        (velocity_grid, confidence_grid) tuple where:
        - velocity_grid: (G, G, G, 3) numpy array or None
        - confidence_grid: (G, G, G) numpy array in [0, 1] or None
    """
    from tracescope.analysis.axes_nn import _state

    a_min = np.asarray(axis_min, dtype=np.float32)
    a_max = np.asarray(axis_max, dtype=np.float32)

    kind = _state.get("kind")

    # Build the grid point coordinates (shared by all model types)
    lin = [np.linspace(a_min[d], a_max[d], grid_size) for d in range(3)]
    gx, gy, gz = np.meshgrid(lin[0], lin[1], lin[2], indexing="ij")
    grid_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)

    # ── RBF path: batch-evaluate predict_velocity ──────────────────
    if kind == "rbf" and "predict_velocity" in _state:
        predict_fn = _state["predict_velocity"]
        vel = predict_fn(grid_pts)  # (grid_size³, 3)
        grid = vel.reshape(grid_size, grid_size, grid_size, 3).astype(np.float32)
        conf_grid = None
        if data_points is not None and len(data_points) >= 5:
            conf_grid = _build_density_grid(grid_pts, data_points, grid_size, path_ids)
        logger.info(f"Built {grid_size}³ velocity grid from RBF"
                    f"{' + density confidence' if conf_grid is not None else ''}")
        return grid, conf_grid

    # ── MDN path: requires PyTorch ─────────────────────────────────
    if kind != "mdn" or "net" not in _state:
        logger.warning("No compatible flow model in _state, cannot build velocity grid")
        return None, None

    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not available, cannot build velocity grid")
        return None, None

    net = _state["net"]

    net.eval()
    with torch.no_grad():
        t = torch.from_numpy(grid_pts)
        mu1, mu2, sigma1, sigma2, logit_pi = net(t)
        pi = torch.sigmoid(logit_pi).unsqueeze(-1)
        v = (pi * mu1 + (1 - pi) * mu2).numpy()
        grid = v.reshape(grid_size, grid_size, grid_size, 3).astype(np.float32)

    # Data-density confidence (works for both MDN and RBF)
    conf_grid = None
    if data_points is not None and len(data_points) >= 5:
        conf_grid = _build_density_grid(grid_pts, data_points, grid_size, path_ids)

    logger.info(f"Built {grid_size}³ velocity + density-confidence grid from MDN")
    return grid, conf_grid


class AnalysisPipeline:
    """Main analysis pipeline.

    Args:
        config: TraceScopeConfig with API keys and settings.
    """

    def __init__(self, config: TraceScopeConfig):
        self.config = config
        self._embedding_provider: EmbeddingProvider = config.create_embedding_provider()
        self._llm_provider: LLMProvider = config.create_llm_provider()
        self._llm_provider_complex: LLMProvider = config.create_llm_provider_complex()
        self._vector_store = VectorStore(config.chromadb_dir)
        self._cache = LLMResponseCache(config.cache_db_path, config.cache_enabled)
        self._result_cache = ResultCache(config.cache_db_path, config.cache_enabled)
        # Simple model for labeling, complex model for explanations
        self._explainer = SemanticExplainer(self._llm_provider, self._cache)
        self._explainer_complex = SemanticExplainer(self._llm_provider_complex, self._cache)

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        return self._embedding_provider

    @embedding_provider.setter
    def embedding_provider(self, provider: EmbeddingProvider):
        self._embedding_provider = provider

    @property
    def llm_provider(self) -> LLMProvider:
        return self._llm_provider

    @llm_provider.setter
    def llm_provider(self, provider: LLMProvider):
        self._llm_provider = provider
        self._explainer = SemanticExplainer(provider, self._cache)

    @property
    def explainer(self) -> SemanticExplainer:
        """Explainer using the complex model (for rich explanations)."""
        return self._explainer_complex

    def analyze(
        self,
        session: TraceSession,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        train_flow: bool = True,
        flow_mode: Optional[str] = None,
        n_clusters: Optional[int] = None,
        mdn_hidden: Optional[int] = None,
        mdn_iters: Optional[int] = None,
        velocity_grid_size: Optional[int] = None,
        cache_path: Optional[str] = None,
        rbf_kernel: Optional[str] = None,
        rbf_smoothing: Optional[float] = None,
        min_k: int = 3,
        param_range: Optional[list] = None,
        deterministic: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> AnalysisResult:
        """Run the full analysis pipeline.

        Args:
            session: TraceSession to analyze.
            progress_callback: Optional callback(stage_name, progress_0_to_1).
            train_flow: Whether to train flow models (slower but enables flow viz).
            flow_mode: "mdn" (default, matching Android), "gpvf", or "rbf".
            n_clusters: Force a specific number of clusters (None = auto).
            mdn_hidden: Hidden layer size for MDN (default 100, matching Android).
                Higher = more expressive flow but slower training. Range: 50-300.
            mdn_iters: Maximum training iterations for MDN (default 8000).
                Higher = more refined flow but slower. Range: 2000-20000.
            velocity_grid_size: Resolution of the 3D velocity grid (default 40).
                Higher = smoother flow but uses more memory. Range: 20-60.
            cache_path: Optional base path (no extension) for full result caching.
                If provided and a valid cache exists with matching fingerprint,
                the entire pipeline is skipped and the cached result is returned
                instantly. On completion, the result is saved to this path.
            rbf_kernel: RBF kernel type (default "thin_plate_spline"). Options:
                "thin_plate_spline", "cubic", "linear", "quintic".
            rbf_smoothing: RBF regularization (default 0.1). 0 = exact interpolation.
            min_k: Minimum number of clusters for auto-selection (default 3).
                The silhouette search starts from this value. Range: 2-10.
            param_range: Optional list of n_neighbors values to search during
                dimension reduction (e.g. [45, 50, 55]). Default None = full
                search [5, 10, 15, ..., 195]. Use this to speed up re-runs
                when you already know the approximate best parameter.
            deterministic: If set, overrides `config.deterministic` for this
                run only. True seeds every stage; False lets each stage
                pick its own seed (different flow topology each run).
            seed: If set, overrides `config.seed` for this run only. The
                global integer seed used by every randomized stage when
                `deterministic=True` — MDN training (torch + numpy),
                KMeans clustering, and UMAP/t-SNE dimension reduction.
                Change this to experiment with alternative flow shapes,
                cluster layouts, and axis alignments without losing
                reproducibility. Included in the cache fingerprint, so a
                different seed re-runs the pipeline instead of loading a
                stale cached result.

        Returns:
            AnalysisResult with all computed data.
        """

        # ── Input validation ─────────────────────────────────────────
        if len(session) == 0:
            raise ValueError("Session has no entries. Provide at least 5 texts.")
        if len(session) < 5:
            raise ValueError(
                f"Need at least 5 entries for analysis, got {len(session)}. "
                f"Clustering, dimension reduction, and flow models require "
                f"a minimum sample size of 5."
            )

        # ── Fall back to config values for unset parameters ──────────
        if flow_mode is None:
            flow_mode = self.config.flow_mode
        if mdn_hidden is None:
            mdn_hidden = self.config.mdn_hidden
        if mdn_iters is None:
            mdn_iters = self.config.mdn_iters
        if velocity_grid_size is None:
            velocity_grid_size = self.config.velocity_grid_size
        if rbf_kernel is None:
            rbf_kernel = self.config.rbf_kernel
        if rbf_smoothing is None:
            rbf_smoothing = self.config.rbf_smoothing
        if deterministic is None:
            deterministic = self.config.deterministic
        if seed is None:
            seed = getattr(self.config, "seed", 42)

        # When deterministic is disabled, pass random_state=None to
        # sklearn/UMAP so every call picks a fresh entropy source.
        # `_effective_seed` is what we thread into stages that accept a
        # seed value (MDN always receives an int and consults
        # `deterministic` separately).
        _effective_random_state = seed if deterministic else None

        # ── Check full-result cache ───────────────────────────────────
        if cache_path is not None:
            import hashlib as _hl
            from pathlib import Path as _Path
            _Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            json_path = _Path(cache_path + ".json")
            npz_path = _Path(cache_path + ".npz")
            if json_path.exists() and npz_path.exists():
                try:
                    with open(str(json_path), "r", encoding="utf-8") as _f:
                        _meta = json.load(_f)
                    # Build fingerprint from current inputs.  Include
                    # score channel names so that changing the channel
                    # set (e.g. dropping found_error, adding path_length)
                    # invalidates the cache — otherwise the cached result
                    # would be loaded with stale scores missing.
                    #
                    # Backward-compat: when running with the historical
                    # defaults (deterministic=True AND seed=42) we emit
                    # the *exact* old fingerprint format so existing
                    # cached results stay valid bit-for-bit and do NOT
                    # need to be recomputed.  Any non-default seed (or
                    # deterministic=False) appends a seed suffix, giving
                    # alternate runs their own distinct cached results.
                    _texts = sorted(e.text for e in session.entries)
                    _score_channels = sorted(session.score_channels)
                    _is_default_seed = bool(deterministic) and int(seed) == 42
                    if _is_default_seed:
                        _blob = (
                            json.dumps(_texts, sort_keys=True) + "|"
                            + self._embedding_provider.model_name() + "|"
                            + flow_mode + "|"
                            + json.dumps(_score_channels, sort_keys=True)
                        )
                    else:
                        _seed_key = (
                            f"{int(seed)}" if deterministic else "nondet"
                        )
                        _blob = (
                            json.dumps(_texts, sort_keys=True) + "|"
                            + self._embedding_provider.model_name() + "|"
                            + flow_mode + "|"
                            + json.dumps(_score_channels, sort_keys=True) + "|"
                            + _seed_key
                        )
                    _fp = _hl.sha256(_blob.encode("utf-8")).hexdigest()
                    if _meta.get("fingerprint") == _fp:
                        logger.info("Full result cache hit — skipping pipeline")
                        if progress_callback:
                            progress_callback("cache_hit", 1.0)
                        _result = AnalysisResult.load_result(cache_path)
                        _result.cache_path = cache_path
                        return _result
                    else:
                        logger.info("Cache fingerprint mismatch — re-running pipeline")
                except Exception as e:
                    logger.warning(f"Cache load failed, re-running pipeline: {e}")

        def _progress(stage: str, pct: float):
            if progress_callback:
                progress_callback(stage, pct)
            logger.info(f"[{stage}] {pct*100:.0f}%")

        # ── Step 1: Embed ──────────────────────────────────────────────
        _progress("embedding", 0.0)
        model_name = self._embedding_provider.model_name()

        # Check vector store first
        stored = self._vector_store.load_embeddings(session.session_id, model_name)
        if stored is not None and len(stored) == len(session):
            logger.info("Using stored embeddings from ChromaDB")
            embeddings = stored
        else:
            texts = session.texts
            embeddings = self._embedding_provider.embed_batch(texts)
            # Store for future use
            self._vector_store.store_embeddings(
                session.session_id,
                model_name,
                texts,
                embeddings,
                metadatas=[
                    {"role": e.role, "step_index": e.step_index}
                    for e in session.entries
                ],
            )

        # Update entries with embeddings
        for i, entry in enumerate(session.entries):
            entry.embedding = embeddings[i]
            entry.model_name = model_name

        _progress("embedding", 1.0)

        # ── Step 2: Cluster (cached by embeddings fingerprint) ────────
        _progress("clustering", 0.0)
        from tracescope.analysis.clustering import main as cluster_main

        emb_fp = embeddings_fingerprint(embeddings)

        # Include seed in the per-stage fingerprint so flipping seeds
        # does not return a stale result from a previous seed.
        _cluster_fp = (
            emb_fp + f"|seed={seed if deterministic else 'nondet'}"
            + f"|mink={min_k}"
        )
        cached_clustering = self._result_cache.get_result(
            "clustering", _cluster_fp
        )

        if cached_clustering is not None:
            logger.info("Using cached clustering result")
            cluster_raw = json.loads(cached_clustering.decode("utf-8"))
        else:
            embeddings_json = json.dumps(embeddings.tolist())
            cluster_result_json = cluster_main(
                embeddings_json, n_clusters=n_clusters, min_k=min_k,
                random_state=_effective_random_state,
            )
            cluster_raw = json.loads(cluster_result_json)
            self._result_cache.put_result(
                "clustering", _cluster_fp, json.dumps(cluster_raw).encode("utf-8")
            )

        clusters = ClusterResult(
            n_clusters=cluster_raw["n_clusters"],
            labels=cluster_raw["labels"],
            clusters=cluster_raw["clusters"],
        )
        _progress("clustering", 1.0)

        # ── Step 3: Reduce to 3D (cached by fingerprint + labels) ─────
        _progress("dimension_reduction", 0.0)
        from tracescope.analysis.dimension_reducer import optimize_reduction, _last_fitted_reducer as _get_reducer
        import tracescope.analysis.dimension_reducer as _dr_mod
        import hashlib as _hashlib

        cluster_labels_json = json.dumps(clusters.labels)
        # dr_key includes labels (so it changes with clustering), but we
        # also mix in the seed so alternate seeds don't return a stale
        # projection from a previous run.
        _seed_tag = f"seed={seed if deterministic else 'nondet'}"
        dr_key = emb_fp + "_" + _hashlib.sha256(
            (cluster_labels_json + "|" + _seed_tag).encode("utf-8")
        ).hexdigest()[:16]

        cached_reduction = self._result_cache.get_result("dim_reduction", dr_key)

        if cached_reduction is not None:
            logger.info("Using cached dimension reduction result")
            n_entries = len(session)
            projected_3d = np.frombuffer(
                cached_reduction, dtype=np.float32
            ).reshape(n_entries, 3).copy()
        else:
            embeddings_json = json.dumps(embeddings.tolist())
            reduction_json = optimize_reduction(
                embeddings_json, cluster_labels_json,
                param_range=param_range,
                random_state=_effective_random_state,
            )
            reduction_raw = json.loads(reduction_json)
            projected_3d = np.array(reduction_raw["embedding"], dtype=np.float32)
            self._result_cache.put_result(
                "dim_reduction", dr_key, projected_3d.tobytes()
            )

        # Grab the fitted reducer (UMAP object) for projecting new points
        fitted_reducer = _dr_mod._last_fitted_reducer

        _progress("dimension_reduction", 1.0)

        # ── Step 4: Compute axes ──────────────────────────────────────
        _progress("axes", 0.0)
        from tracescope.analysis.compute_axes_info import compute_axes_info

        projected_json = json.dumps(projected_3d.tolist())
        axes_json = compute_axes_info(projected_json, cluster_labels_json)
        axes_raw = json.loads(axes_json)

        axis_info = AxisInfo(
            axes=np.array(axes_raw["axes"]),
            lengths=axes_raw["lengths"],
            min_point_idx=axes_raw["min_point_idx"],
            max_point_idx=axes_raw["max_point_idx"],
            min_cluster_idx=axes_raw.get("min_cluster_idx", []),
            max_cluster_idx=axes_raw.get("max_cluster_idx", []),
        )
        _progress("axes", 1.0)

        # ── Step 4.5: Compute cluster centroids in 3D ────────────────
        n_cls = clusters.n_clusters
        cluster_centroids_3d = np.zeros((n_cls, 3), dtype=np.float32)
        for c in range(n_cls):
            mask = [i for i, l in enumerate(clusters.labels) if l == c]
            if mask:
                cluster_centroids_3d[c] = projected_3d[mask].mean(axis=0)

        # Max pairwise distance between centroids
        max_cluster_distance = 0.0
        for i in range(n_cls):
            for j in range(i + 1, n_cls):
                d = float(np.linalg.norm(cluster_centroids_3d[i] - cluster_centroids_3d[j]))
                if d > max_cluster_distance:
                    max_cluster_distance = d
        if max_cluster_distance == 0:
            max_cluster_distance = 1.0

        # ── Step 5: Label axes via LLM (multi-part pipeline) ─────────
        _progress("axis_labeling", 0.0)
        axis_names = ["X", "Y", "Z"]
        axis_labels = []
        texts = session.texts

        for d in range(3):
            label = self._explainer.explain_axis(
                axis_dim=d,
                axis_name=axis_names[d],
                texts=texts,
                embeddings=embeddings,
                projected_3d=projected_3d,
                axis_info=axis_info,
                clusters=clusters,
                cluster_summaries=[],  # No cluster summaries yet during axis labeling
                previous_labels=axis_labels.copy(),  # Avoid mechanism
            )
            axis_labels.append(label)
            _progress("axis_labeling", (d + 1) / 3)

        axis_info.labels = axis_labels

        # ── Step 6: Label clusters via LLM (avoidDesc + TF-IDF) ──────
        _progress("cluster_labeling", 0.0)

        # Build all_cluster_texts for the differentiation pipeline
        all_cluster_texts = []
        for c in range(clusters.n_clusters):
            c_texts = [
                session.entries[i].text
                for i, lbl in enumerate(clusters.labels)
                if lbl == c
            ]
            all_cluster_texts.append(c_texts)

        cluster_labels_list: List[Optional[str]] = [None] * clusters.n_clusters
        for c in range(clusters.n_clusters):
            desc = self._explainer.label_cluster(
                cluster_index=c,
                all_cluster_texts=all_cluster_texts,
                embeddings=embeddings,
                cluster_labels=clusters.labels,
                prior_summaries=cluster_labels_list,  # Grows as we label
            )
            cluster_labels_list[c] = desc
            _progress("cluster_labeling", (c + 1) / clusters.n_clusters)

        # Replace None with fallback
        cluster_labels_final = [
            lbl if lbl is not None else f"Cluster {c}"
            for c, lbl in enumerate(cluster_labels_list)
        ]

        # ── Step 7: Train flow model (MDN or GPVF) ───────────────────
        flow_trained = False
        velocity_grid = None
        confidence_grid = None
        axis_min_arr = None
        axis_max_arr = None
        mdn_simulate = None

        if train_flow and len(session) >= 5:
            _progress("flow_model", 0.0)
            if flow_mode == "mdn":
                try:
                    import torch  # noqa: F401
                except ImportError:
                    import warnings
                    warnings.warn(
                        "PyTorch is not installed — MDN flow model cannot be trained. "
                        "Install PyTorch (`pip install torch`) or use flow_mode='rbf' "
                        "for a lightweight scipy-based alternative.",
                        UserWarning, stacklevel=2,
                    )
            try:
                from tracescope.analysis.axes_nn import train_models, _state

                cloud_json = json.dumps(projected_3d.tolist())

                # Extract path_ids for multi-path MDN training
                _path_ids = None
                if any(e.path_id is not None for e in session.entries):
                    _path_ids = [e.path_id for e in session.entries]

                # Forward model-specific params
                _flow_kwargs = {}
                if flow_mode == "mdn":
                    _flow_kwargs["hidden"] = mdn_hidden
                    _flow_kwargs["iters"] = mdn_iters
                    _flow_kwargs["deterministic"] = deterministic
                    _flow_kwargs["seed"] = int(seed)
                elif flow_mode == "rbf":
                    _flow_kwargs["kernel"] = rbf_kernel
                    _flow_kwargs["smoothing"] = rbf_smoothing

                result_json = train_models(
                    cloud_json, mode=flow_mode, path_ids=_path_ids,
                    **_flow_kwargs)
                result_data = json.loads(result_json)
                axis_min_arr = np.array(result_data["axis_min"], dtype=np.float32)
                axis_max_arr = np.array(result_data["axis_max"], dtype=np.float32)

                # Get simulate function from state
                if "simulate" in _state:
                    mdn_simulate = _state["simulate"]

                flow_trained = True
                _progress("flow_model", 0.5)

                # Build velocity grid (cached by projected_3d fingerprint).
                # Include (flow_mode, seed, deterministic) so two runs with
                # different seeds produce separately-cached grids.
                if flow_mode in ("mdn", "rbf") and flow_trained:
                    _progress("velocity_grid", 0.0)
                    _raw_grid_fp = embeddings_fingerprint(
                        projected_3d.reshape(len(projected_3d), -1)
                    )
                    grid_fp = (
                        _raw_grid_fp
                        + f"|{flow_mode}"
                        + f"|seed={seed if deterministic else 'nondet'}"
                    )
                    cached_grid = self._result_cache.get_result(
                        "velocity_grid", grid_fp
                    )

                    if cached_grid is not None:
                        logger.info("Using cached velocity grid")
                        velocity_grid = np.frombuffer(
                            cached_grid, dtype=np.float32
                        ).reshape(
                            velocity_grid_size, velocity_grid_size,
                            velocity_grid_size, 3
                        ).copy()
                        # Try loading cached confidence grid
                        cached_conf = self._result_cache.get_result(
                            "confidence_grid", grid_fp
                        )
                        if cached_conf is not None:
                            confidence_grid = np.frombuffer(
                                cached_conf, dtype=np.float32
                            ).reshape(
                                velocity_grid_size, velocity_grid_size,
                                velocity_grid_size
                            ).copy()
                    else:
                        velocity_grid, confidence_grid = _build_velocity_grid(
                            axis_min_arr, axis_max_arr, velocity_grid_size,
                            data_points=projected_3d,
                            path_ids=_path_ids,
                        )
                        if velocity_grid is not None:
                            self._result_cache.put_result(
                                "velocity_grid", grid_fp,
                                velocity_grid.tobytes()
                            )
                        if confidence_grid is not None:
                            self._result_cache.put_result(
                                "confidence_grid", grid_fp,
                                confidence_grid.tobytes()
                            )
                    _progress("velocity_grid", 1.0)

            except Exception as e:
                logger.warning(f"Flow model training failed: {e}")
            _progress("flow_model", 1.0)

        # ── Build result ──────────────────────────────────────────────
        result = AnalysisResult(
            session=session,
            embedding_model=model_name,
            embeddings=embeddings,
            clusters=clusters,
            projected_3d=projected_3d,
            axis_info=axis_info,
            cluster_labels=cluster_labels_final,
            flow_model_trained=flow_trained,
            velocity_grid=velocity_grid,
            confidence_grid=confidence_grid,
            axis_min=axis_min_arr,
            axis_max=axis_max_arr,
            mdn_simulate=mdn_simulate,
            cluster_centroids_3d=cluster_centroids_3d,
            max_cluster_distance=max_cluster_distance,
            fitted_reducer=fitted_reducer,
            cache_path=cache_path,
            flow_mode=flow_mode,
            seed=int(seed),
            deterministic=bool(deterministic),
        )

        # ── Save to full-result cache ─────────────────────────────────
        if cache_path is not None:
            try:
                from pathlib import Path as _Path
                _Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                result.save_result(cache_path)
                logger.info(f"Saved full result cache to {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save result cache: {e}")

        return result

    def get_vectors(
        self, session_id: str, model_name: str
    ) -> Optional[np.ndarray]:
        """Return stored vectors for a session+model pair."""
        return self._vector_store.load_embeddings(session_id, model_name)

    def compare_models(
        self, session_id: str, model_a: str, model_b: str
    ) -> dict:
        """Compare two embedding models' vectors for the same session."""
        vecs_a = self._vector_store.load_embeddings(session_id, model_a)
        vecs_b = self._vector_store.load_embeddings(session_id, model_b)

        if vecs_a is None or vecs_b is None:
            raise ValueError(
                f"Missing vectors: {model_a}={'found' if vecs_a is not None else 'missing'}, "
                f"{model_b}={'found' if vecs_b is not None else 'missing'}. "
                f"If you loaded results from cache, the vector store is not "
                f"repopulated — re-run pipeline.analyze() without cache_path "
                f"to populate the vector store before calling compare_models()."
            )

        if len(vecs_a) != len(vecs_b):
            raise ValueError(
                f"Vector count mismatch: {model_a} has {len(vecs_a)}, "
                f"{model_b} has {len(vecs_b)}"
            )

        if vecs_a.shape[1] != vecs_b.shape[1]:
            raise ValueError(
                f"Dimension mismatch: {model_a} is {vecs_a.shape[1]}D, "
                f"{model_b} is {vecs_b.shape[1]}D — cosine similarity is undefined "
                f"between different embedding spaces."
            )

        norms_a = np.linalg.norm(vecs_a, axis=1, keepdims=True)
        norms_b = np.linalg.norm(vecs_b, axis=1, keepdims=True)
        normed_a = vecs_a / norms_a
        normed_b = vecs_b / norms_b
        cosine_sims = np.sum(normed_a * normed_b, axis=1)

        return {
            "model_a": model_a,
            "model_b": model_b,
            "n_entries": len(vecs_a),
            "dim_a": vecs_a.shape[1],
            "dim_b": vecs_b.shape[1],
            "mean_cosine_similarity": float(np.mean(cosine_sims)),
            "min_cosine_similarity": float(np.min(cosine_sims)),
            "max_cosine_similarity": float(np.max(cosine_sims)),
            "per_entry_cosine_similarity": cosine_sims.tolist(),
        }

    def list_stored_sessions(self) -> list:
        """List all stored session/model combinations."""
        return self._vector_store.list_sessions()

    def clear_cache(self, what: str = "all"):
        """Clear cached data.

        Args:
            what: What to clear:
                - "all" — clear everything (embeddings, LLM responses, ML results)
                - "results" — clear clustering, dim reduction, velocity grid caches
                - "llm" — clear LLM response cache only
                - "embeddings" — clear stored embeddings only
        """
        if what in ("all", "results"):
            self._result_cache.clear()
            logger.info("Cleared result cache (clustering, dim reduction, velocity grid)")

        if what in ("all", "llm"):
            self._cache.clear()
            logger.info("Cleared LLM response cache")

        if what in ("all", "embeddings"):
            self._vector_store.clear_all()
            logger.info("Cleared embedding store")
