"""
Semantic explainer – faithful port from Android EmotionApp.

Implements the sophisticated pipelines from:
  - MoodProcessingWorker.java: computeKeywordSummary(), axis labeling
  - MoodAnalysis.java: avoidDesc, extractTopClusterKeywords(), keyword differentiation
  - DashboardFragment.java: probe explanation with cluster distances

Uses LLMProvider + cache + prompts to generate human-readable explanations.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from tracescope.providers.llm import LLMProvider
from tracescope.storage.cache import LLMResponseCache
from tracescope import prompts


# ═══════════════════════════════════════════════════
# STOP WORDS (matching Android's buildStopSet)
# ═══════════════════════════════════════════════════

STOP_WORDS = frozenset([
    # English
    "the", "and", "is", "a", "to", "of", "in", "that", "it", "with",
    "for", "on", "as", "at", "by", "from", "this", "be", "or", "an",
    "are", "was", "were", "has", "have", "but", "not", "will", "if",
    "they", "their", "them", "he", "she", "we", "you", "i", "my", "me",
    "can", "do", "does", "did", "would", "could", "should", "may",
    "about", "into", "than", "then", "so", "some", "these", "those",
    "its", "also", "just", "how", "what", "which", "who", "when",
    "where", "why", "been", "being", "had", "having", "here", "there",
    "each", "every", "all", "both", "few", "more", "most", "other",
    "very", "such", "only", "own", "same", "too", "any", "no",
])


def _tokenize(text: str) -> List[str]:
    """Tokenize text: lowercase, split on non-word chars, drop short + stop words."""
    words = re.sub(r"\W+", " ", text.lower()).split()
    return [w for w in words if len(w) >= 3 and w not in STOP_WORDS]


# ═══════════════════════════════════════════════════
# TF-IDF KEYWORD EVOLUTION (6 PERCENTILE BUCKETS)
# ═══════════════════════════════════════════════════
# Ported from MoodProcessingWorker.java:computeKeywordSummary() lines 886-997

def compute_keyword_summary(
    texts: List[str],
    projected_3d: np.ndarray,
    axis_dim: int,
) -> str:
    """Compute TF-IDF keyword evolution along an axis using 6 percentile buckets.

    Ported from Android's computeKeywordSummary():
    1. Sort entries by coordinate on axis d
    2. Split into 6 percentile buckets: [0,5%), [5,25%), [25,50%), [50,75%), [75,95%), [95,100%]
    3. Compute TF-IDF per bucket
    4. Fit linear slope across buckets
    5. Return top 5 words by absolute slope

    Args:
        texts: List of N text strings.
        projected_3d: (N, 3) array of 3D coordinates.
        axis_dim: 0 (X), 1 (Y), or 2 (Z).

    Returns:
        Formatted string like "Keyword evolution: 'word1' increases more at high end; ..."
    """
    N = len(texts)
    if N < 2:
        return ""

    # Sort indices by coordinate on axis d
    coords = projected_3d[:, axis_dim]
    sorted_indices = np.argsort(coords).tolist()

    # Define 6 percentile buckets (matching Android)
    p5 = max(1, round(N * 0.05))
    p25 = max(p5 + 1, round(N * 0.25))
    p50 = max(p25 + 1, round(N * 0.50))
    p75 = max(p50 + 1, round(N * 0.75))
    p95 = max(p75 + 1, round(N * 0.95))

    buckets = [
        sorted_indices[0:p5],       # Bucket 0: [0%, 5%)
        sorted_indices[p5:p25],     # Bucket 1: [5%, 25%)
        sorted_indices[p25:p50],    # Bucket 2: [25%, 50%)
        sorted_indices[p50:p75],    # Bucket 3: [50%, 75%)
        sorted_indices[p75:p95],    # Bucket 4: [75%, 95%)
        sorted_indices[p95:N],      # Bucket 5: [95%, 100%]
    ]

    M = len(buckets)  # 6

    # Tokenize all texts
    all_tokens = [_tokenize(t) for t in texts]

    # Compute document frequency
    doc_freq: Dict[str, int] = defaultdict(int)
    for tokens in all_tokens:
        for w in set(tokens):
            doc_freq[w] += 1

    # Compute term frequency per bucket
    term_freq: Dict[str, List[int]] = {}
    for b_idx, bucket in enumerate(buckets):
        for idx in bucket:
            for w in all_tokens[idx]:
                if w not in term_freq:
                    term_freq[w] = [0] * M
                term_freq[w][b_idx] += 1

    # Compute TF-IDF scores per bucket
    tfidf: Dict[str, List[float]] = {}
    for w, freqs in term_freq.items():
        df = doc_freq.get(w, 1)
        idf = math.log(N / (1 + df))
        tfidf[w] = [f * idf for f in freqs]

    # Fit linear slope across buckets
    mean_idx = (M - 1) / 2.0
    denom = sum((i - mean_idx) ** 2 for i in range(M))
    if denom == 0:
        return ""

    slopes: Dict[str, float] = {}
    for w, scores in tfidf.items():
        mean_score = sum(scores) / M
        num = sum((i - mean_idx) * (scores[i] - mean_score) for i in range(M))
        slopes[w] = num / denom

    # Select top 5 by absolute slope
    top_words = sorted(slopes.keys(), key=lambda w: abs(slopes[w]), reverse=True)[:5]

    if not top_words:
        return ""

    # Build descriptive summary
    parts = []
    for w in top_words:
        s = slopes[w]
        direction = "increases" if s > 0 else "decreases"
        end = "high" if s > 0 else "low"
        parts.append(f"'{w}' {direction} more at {end} end")

    return "Keyword evolution: " + "; ".join(parts)


# ═══════════════════════════════════════════════════
# COSINE SIMILARITY
# ═══════════════════════════════════════════════════

def compute_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Ported from MoodProcessingWorker.java:computeCosineSimilarity() lines 871-879.
    """
    dot = np.dot(v1, v2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(dot / (n1 * n2))


# ═══════════════════════════════════════════════════
# CLUSTER KEYWORD EXTRACTION (TF-IDF per cluster)
# ═══════════════════════════════════════════════════
# Ported from MoodAnalysis.java:extractTopClusterKeywords() lines 3313-3372

def extract_cluster_keywords(
    cluster_texts: List[str],
    all_cluster_texts: List[List[str]],
    max_df_ratio: float = 0.76,
    top_n: int = 5,
) -> List[str]:
    """Extract top TF-IDF keywords for a specific cluster.

    Ported from Android's extractTopClusterKeywords():
    1. Tokenize all clusters
    2. Compute document frequency (per cluster, not per text)
    3. Filter: drop words with df > max_df_ratio * num_clusters
    4. Compute TF for target cluster
    5. TF-IDF: score = tf * log(num_clusters / (1 + df))
    6. Return top-N

    Args:
        cluster_texts: Texts in the target cluster.
        all_cluster_texts: All cluster texts as list of lists.
        max_df_ratio: Maximum document frequency ratio (0.76 in Android).
        top_n: Number of top keywords to return.

    Returns:
        List of top-N keywords by TF-IDF score.
    """
    C = len(all_cluster_texts)
    if C == 0:
        return []

    # Tokenize every cluster and compute document frequency
    tokenized_clusters: List[List[str]] = []
    df: Dict[str, int] = defaultdict(int)

    for cluster in all_cluster_texts:
        seen = set()
        toks = []
        for text in cluster:
            for w in _tokenize(text):
                toks.append(w)
                seen.add(w)
        tokenized_clusters.append(toks)
        for w in seen:
            df[w] += 1

    # Filter vocabulary: drop terms with df > max_df_ratio * C
    max_df = int(math.ceil(max_df_ratio * C))
    vocab = {w for w, d in df.items() if d <= max_df}

    # Find target cluster index
    target_idx = None
    for i, ct in enumerate(all_cluster_texts):
        if ct is cluster_texts:
            target_idx = i
            break
    if target_idx is None:
        # Fallback: find by content match
        target_idx = 0

    # Compute TF for target cluster
    tf: Dict[str, int] = defaultdict(int)
    for w in tokenized_clusters[target_idx]:
        if w in vocab:
            tf[w] += 1

    # Compute TF-IDF
    scores: Dict[str, float] = {}
    for w, freq in tf.items():
        d = df.get(w, 1)
        idf = math.log(C / (1 + d))
        scores[w] = freq * idf

    # Return top-N
    return sorted(scores.keys(), key=lambda w: scores[w], reverse=True)[:top_n]


# ═══════════════════════════════════════════════════
# AVOID DESCRIPTIONS (cosine similarity between cluster centroids)
# ═══════════════════════════════════════════════════
# Ported from MoodAnalysis.java lines 3081-3178

def compute_avoid_descriptions(
    cluster_index: int,
    all_cluster_texts: List[List[str]],
    embeddings: np.ndarray,
    cluster_labels: List[int],
    prior_summaries: List[Optional[str]],
    llm: LLMProvider,
    cache: LLMResponseCache,
) -> List[str]:
    """Find the 2 most similar clusters and get/generate their summaries.

    Ported from Android's avoid description mechanism:
    1. Compute centroid (mean of embeddings) for each cluster
    2. Cosine similarity between current cluster centroid and all others
    3. Find top-2 most similar clusters
    4. Get or generate short summaries for those clusters (via LLM)

    Args:
        cluster_index: Index of the current cluster being labeled.
        all_cluster_texts: All cluster texts as list of lists.
        embeddings: (N, D) high-dim embedding matrix.
        cluster_labels: Cluster label per entry.
        prior_summaries: Previously generated summaries (may have None entries).
        llm: LLM provider for generating summaries if needed.
        cache: Cache for LLM responses.

    Returns:
        List of 1-2 avoid description strings.
    """
    K = len(all_cluster_texts)
    if K < 2:
        return []

    dim = embeddings.shape[1]

    # Compute centroids
    centroids = np.zeros((K, dim))
    for j in range(K):
        mask = [i for i, l in enumerate(cluster_labels) if l == j]
        if mask:
            centroids[j] = embeddings[mask].mean(axis=0)

    # Compute cosine similarities
    cur = centroids[cluster_index]
    cur_norm = np.linalg.norm(cur)
    sims = np.full(K, -1.0)
    for j in range(K):
        if j == cluster_index:
            continue
        other = centroids[j]
        other_norm = np.linalg.norm(other)
        if cur_norm > 0 and other_norm > 0:
            sims[j] = float(np.dot(cur, other) / (cur_norm * other_norm))

    # Find top-2 most similar
    sorted_idx = np.argsort(-sims)
    top2 = [int(idx) for idx in sorted_idx[:2] if sims[idx] > -1]

    # Get or generate summaries
    avoid_list = []
    for sim_idx in top2:
        if sim_idx < len(prior_summaries) and prior_summaries[sim_idx] is not None:
            avoid_list.append(prior_summaries[sim_idx])
        else:
            # Generate a quick summary via LLM
            cluster_text = "\n".join(
                f'sentence: "{t}"' for t in all_cluster_texts[sim_idx][:10]
            )
            quick_prompt = prompts.build_cluster_label_prompt(cluster_text)
            # Check cache
            cached = cache.get(llm.model_name(), "", quick_prompt)
            if cached is not None:
                avoid_list.append(cached.strip())
            else:
                resp = llm.complete("", quick_prompt)
                cache.put(llm.model_name(), "", quick_prompt, resp)
                avoid_list.append(resp.strip())

    return avoid_list


# ═══════════════════════════════════════════════════
# KEYWORD DIFFERENTIATION (set differences + frequency deltas)
# ═══════════════════════════════════════════════════
# Ported from MoodAnalysis.java lines 3183-3234

def compute_keyword_differentiation(
    cluster_index: int,
    all_cluster_texts: List[List[str]],
    top2_similar: List[int],
) -> dict:
    """Compute keyword set differences and frequency deltas.

    Ported from Android's TF-IDF keyword differentiation:
    1. Get top keywords for current cluster and 2 most similar
    2. onlyCur = unique to this cluster
    3. onlyOthers = unique to the similar clusters
    4. shared = in both
    5. For shared: delta = freq_cur(w) - max(freq_a(w), freq_b(w))

    Returns:
        dict with keys: unique_keywords, other_only_keywords, shared_deltas
    """
    if not top2_similar:
        return {"unique_keywords": [], "other_only_keywords": [], "shared_deltas": []}

    # Extract keywords for each
    top_cur = extract_cluster_keywords(
        all_cluster_texts[cluster_index], all_cluster_texts,
        max_df_ratio=0.76, top_n=5,
    )

    top_others = []
    for sim_idx in top2_similar:
        kws = extract_cluster_keywords(
            all_cluster_texts[sim_idx], all_cluster_texts,
            max_df_ratio=0.76, top_n=4,
        )
        top_others.extend(kws)
    set_others = set(top_others)

    # Compute set differences
    only_cur = [w for w in top_cur if w not in set_others]
    only_others = [w for w in set_others if w not in top_cur]
    shared = [w for w in top_cur if w in set_others]

    # Raw frequency counts for shared terms
    def raw_counts(texts: List[str]) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for t in texts:
            for w in _tokenize(t):
                counts[w] += 1
        return counts

    raw_cur = raw_counts(all_cluster_texts[cluster_index])
    raw_others_list = [raw_counts(all_cluster_texts[s]) for s in top2_similar]

    # Compute deltas for shared terms
    deltas = []
    for w in shared:
        c = raw_cur.get(w, 0)
        m = max(ro.get(w, 0) for ro in raw_others_list) if raw_others_list else 0
        deltas.append((w, c - m))

    # Sort by absolute delta, take top 2
    deltas.sort(key=lambda x: abs(x[1]), reverse=True)
    sig_deltas = deltas[:2]

    return {
        "unique_keywords": only_cur,
        "other_only_keywords": only_others,
        "shared_deltas": sig_deltas,
    }


# ═══════════════════════════════════════════════════
# SEMANTIC EXPLAINER (main class)
# ═══════════════════════════════════════════════════

class SemanticExplainer:
    """LLM-powered semantic explanation engine.

    Faithful port of the Android EmotionApp's explanation pipelines.
    """

    def __init__(self, llm: LLMProvider, cache: LLMResponseCache):
        self._llm = llm
        self._cache = cache

    def _cached_complete(
        self, system_prompt: str, user_prompt: str,
    ) -> str:
        """Complete with cache lookup."""
        cached = self._cache.get(self._llm.model_name(), system_prompt, user_prompt)
        if cached is not None:
            return cached
        response = self._llm.complete(system_prompt, user_prompt)
        self._cache.put(self._llm.model_name(), system_prompt, user_prompt, response)
        return response

    def explain_axis(
        self,
        axis_dim: int,
        axis_name: str,
        texts: List[str],
        embeddings: np.ndarray,
        projected_3d: np.ndarray,
        axis_info,
        clusters,
        cluster_summaries: List[str],
        previous_labels: List[str],
    ) -> str:
        """Label a PCA axis using the full multi-part pipeline.

        Ported from MoodProcessingWorker.java lines 665-827.
        """
        N = len(texts)
        coords = projected_3d[:, axis_dim]
        min_val = float(coords.min())
        max_val = float(coords.max())
        span = max_val - min_val if max_val != min_val else 1.0

        # Min/max point indices
        min_idx = int(axis_info.min_point_idx[axis_dim])
        max_idx = int(axis_info.max_point_idx[axis_dim])

        # Normalized positions (0-100)
        norm_min = round((coords[min_idx] - min_val) / span * 100)
        norm_max = round((coords[max_idx] - min_val) / span * 100)

        # Intermediate part (if N >= 4)
        intermediate_part = ""
        if N >= 4:
            sorted_by = np.argsort(coords).tolist()
            i1 = sorted_by[1]
            i2 = sorted_by[-2]
            v1 = round((coords[i1] - min_val) / span * 100)
            v2 = round((coords[i2] - min_val) / span * 100)
            intermediate_part = prompts.build_intermediate_part([
                (v1, texts[i1][:300]),
                (v2, texts[i2][:300]),
            ])

        # Cluster part
        cluster_part = ""
        if cluster_summaries and hasattr(axis_info, 'min_cluster_idx') and axis_info.min_cluster_idx:
            try:
                min_cl = axis_info.min_cluster_idx[axis_dim]
                max_cl = axis_info.max_cluster_idx[axis_dim]
                # Compute cluster centroids in projected space
                n_clusters = max(clusters.labels) + 1 if clusters.labels else 0
                centroids_3d = np.zeros((n_clusters, 3))
                for c in range(n_clusters):
                    mask = [i for i, l in enumerate(clusters.labels) if l == c]
                    if mask:
                        centroids_3d[c] = projected_3d[mask].mean(axis=0)

                min_summary = cluster_summaries[min_cl] if min_cl < len(cluster_summaries) else f"Cluster {min_cl}"
                max_summary = cluster_summaries[max_cl] if max_cl < len(cluster_summaries) else f"Cluster {max_cl}"
                cluster_part = prompts.build_cluster_part(
                    float(centroids_3d[min_cl, axis_dim]),
                    min_summary,
                    float(centroids_3d[max_cl, axis_dim]),
                    max_summary,
                )
            except (IndexError, ValueError):
                pass

        # Keyword summary (TF-IDF evolution)
        keyword_summary = compute_keyword_summary(texts, projected_3d, axis_dim)

        # Cosine similarity between extremes
        similarity_min = 0.0
        similarity_max = 0.0
        if embeddings is not None and N >= 4:
            sorted_by = np.argsort(coords).tolist()
            first_mid_idx = sorted_by[N // 4]
            second_mid_idx = sorted_by[3 * N // 4]
            similarity_min = compute_cosine_similarity(
                embeddings[min_idx], embeddings[first_mid_idx]
            )
            similarity_max = compute_cosine_similarity(
                embeddings[second_mid_idx], embeddings[max_idx]
            )

        # Build full prompt
        user_prompt = prompts.build_axis_label_prompt(
            axis_name=axis_name,
            min_text=texts[min_idx][:300],
            max_text=texts[max_idx][:300],
            norm_min=norm_min,
            norm_max=norm_max,
            intermediate_part=intermediate_part,
            cluster_part=cluster_part,
            keyword_summary=keyword_summary,
            similarity_min=similarity_min,
            similarity_max=similarity_max,
            previous_labels=previous_labels,
        )

        response = self._cached_complete("", user_prompt)
        # Extract just 2 words from the response
        words = response.strip().split()
        return " ".join(words[:2]) if len(words) >= 2 else response.strip()

    def label_cluster(
        self,
        cluster_index: int,
        all_cluster_texts: List[List[str]],
        embeddings: np.ndarray,
        cluster_labels: List[int],
        prior_summaries: List[Optional[str]],
    ) -> str:
        """Label a cluster using avoidDesc + TF-IDF keyword differentiation.

        Ported from MoodAnalysis.java lines 3064-3277.
        """
        # Get texts for this cluster
        cluster_texts = all_cluster_texts[cluster_index]
        cluster_text_formatted = "\n".join(
            f'sentence: "{t[:200]}"' for t in cluster_texts[:15]
        )

        # Compute avoid descriptions
        avoid_descs = compute_avoid_descriptions(
            cluster_index, all_cluster_texts, embeddings,
            cluster_labels, prior_summaries,
            self._llm, self._cache,
        )

        # Find top-2 most similar clusters for keyword differentiation
        K = len(all_cluster_texts)
        dim = embeddings.shape[1]
        centroids = np.zeros((K, dim))
        for j in range(K):
            mask = [i for i, l in enumerate(cluster_labels) if l == j]
            if mask:
                centroids[j] = embeddings[mask].mean(axis=0)

        cur = centroids[cluster_index]
        cur_norm = np.linalg.norm(cur)
        sims = np.full(K, -1.0)
        for j in range(K):
            if j == cluster_index:
                continue
            other_norm = np.linalg.norm(centroids[j])
            if cur_norm > 0 and other_norm > 0:
                sims[j] = float(np.dot(cur, centroids[j]) / (cur_norm * other_norm))

        sorted_idx = np.argsort(-sims)
        top2 = [int(idx) for idx in sorted_idx[:2] if sims[idx] > -1]

        # Keyword differentiation
        kw_diff = compute_keyword_differentiation(
            cluster_index, all_cluster_texts, top2,
        )

        # Build shared keyword deltas with cluster names
        shared_deltas = []
        for w, delta in kw_diff["shared_deltas"]:
            compared = prior_summaries[top2[0]] if top2 and top2[0] < len(prior_summaries) and prior_summaries[top2[0]] else f"Cluster {top2[0]}" if top2 else "other"
            shared_deltas.append((w, delta, compared))

        # Build prompt
        user_prompt = prompts.build_cluster_label_prompt(
            cluster_text=cluster_text_formatted,
            avoid_descriptions=avoid_descs if avoid_descs else None,
            unique_keywords=kw_diff["unique_keywords"] if kw_diff["unique_keywords"] else None,
            shared_keyword_deltas=shared_deltas if shared_deltas else None,
            other_only_keywords=kw_diff["other_only_keywords"] if kw_diff["other_only_keywords"] else None,
        )

        response = self._cached_complete("", user_prompt)
        # Extract 1-4 words
        words = response.strip().split()
        return " ".join(words[:4]) if words else response.strip()

    def explain_probe_single(
        self,
        axis_labels: List[str],
        slider_pcts: List[int],
        cluster_distances: List[Tuple[str, int]],
    ) -> str:
        """Single-point probe explanation."""
        user_prompt = prompts.build_probe_explain_prompt(
            axis_labels=axis_labels,
            slider_pcts=slider_pcts,
            cluster_distances=cluster_distances,
        )
        return self._cached_complete("", user_prompt).strip()

    def explain_probe_multi(
        self,
        axis_labels: List[str],
        control_points: List[dict],
    ) -> str:
        """Multi-point trajectory explanation."""
        user_prompt = prompts.build_path_explain_prompt(
            axis_labels=axis_labels,
            control_points=control_points,
        )
        return self._cached_complete("", user_prompt).strip()

    def explain_flow(
        self,
        axis_labels: List[str],
        x: float, y: float, z: float,
        vx: float, vy: float, vz: float,
        direction_description: str,
    ) -> str:
        """Explain the flow field direction at a point."""
        user_prompt = prompts.format_flow_explain(
            x_label=axis_labels[0], y_label=axis_labels[1], z_label=axis_labels[2],
            x=x, y=y, z=z, vx=vx, vy=vy, vz=vz,
            direction_description=direction_description,
        )
        return self._cached_complete(
            prompts.FLOW_EXPLAIN_SYSTEM, user_prompt,
        ).strip()

    def summarize(
        self,
        axis_labels: List[str],
        cluster_descriptions: List[str],
        n_entries: int,
        cluster_sequence: List[str],
    ) -> str:
        """Summarize the overall analysis result."""
        cluster_desc_str = "\n".join(
            f"  Cluster {i}: {d}" for i, d in enumerate(cluster_descriptions)
        )
        user_prompt = prompts.format_summary(
            x_label=axis_labels[0], y_label=axis_labels[1], z_label=axis_labels[2],
            cluster_descriptions=cluster_desc_str,
            n_entries=n_entries,
            cluster_sequence=" -> ".join(cluster_sequence),
        )
        return self._cached_complete(
            prompts.SUMMARY_SYSTEM, user_prompt,
        ).strip()
