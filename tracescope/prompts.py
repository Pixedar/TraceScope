"""
═══════════════════════════════════════════════════════════════════════════════
TraceScope Prompts — Faithful port from Android EmotionApp
═══════════════════════════════════════════════════════════════════════════════

ALL LLM prompts live here.  Each prompt mirrors the exact prompt structure
from the Android app (MoodProcessingWorker.java, MoodAnalysis.java,
DashboardFragment.java) adapted for LLM conversation analysis.

Source references:
  - Axis labeling:  MoodProcessingWorker.java lines 665-827
  - Cluster labels: MoodAnalysis.java lines 3064-3277
  - Probe explain:  DashboardFragment.java lines 646-731
═══════════════════════════════════════════════════════════════════════════════
"""

from typing import List, Optional, Tuple


# ═══════════════════════════════════════════════════
# AXIS LABELING — Multi-part prompt (9 parts)
# ═══════════════════════════════════════════════════
# Ported from MoodProcessingWorker.java lines 665-827.
# Builds the same composite prompt:
#   minPart + maxPart + intermediatePart + clusterPart
#   + keywordSummaryPart + similarityPart + attentionDirective
#   + contextSentence + axisEnding
#
# The LLM must respond with EXACTLY 2 words.

def build_axis_label_prompt(
    axis_name: str,
    min_text: str,
    max_text: str,
    norm_min: int,
    norm_max: int,
    intermediate_part: str = "",
    cluster_part: str = "",
    keyword_summary: str = "",
    similarity_min: float = 0.0,
    similarity_max: float = 0.0,
    previous_labels: Optional[List[str]] = None,
) -> str:
    """Build the full multi-part axis labeling prompt.

    Args:
        axis_name: "X", "Y", or "Z".
        min_text: Text of the entry at the axis minimum.
        max_text: Text of the entry at the axis maximum.
        norm_min: Normalized position (0-100) of the minimum point.
        norm_max: Normalized position (0-100) of the maximum point.
        intermediate_part: Pre-formatted string with intermediate point texts.
        cluster_part: Pre-formatted string with cluster centroid info.
        keyword_summary: TF-IDF keyword evolution string from explainer.
        similarity_min: Cosine similarity at the low end of the axis.
        similarity_max: Cosine similarity at the high end of the axis.
        previous_labels: Labels of previously-labeled axes (for avoid mechanism).

    Returns:
        Complete prompt string for the LLM.
    """
    dim_names = {"X": "X", "Y": "Y", "Z": "Z"}
    dim = dim_names.get(axis_name, axis_name)

    # Part 1: Minimum anchor
    min_part = f'Sentence at {dim} axis value {norm_min}: "{min_text}".'

    # Part 2: Maximum anchor
    max_part = f' Sentence at {dim} axis value {norm_max}: "{max_text}".'

    # Part 3: Intermediate points (pre-built by caller)
    # Already formatted as: " Intermediate points (...): sentence at value V1: "text1"; ..."

    # Part 4: Cluster info (pre-built by caller)
    # Already formatted as: " Lowest axis value cluster (centroid X.XXX): "summary"; ..."

    # Part 5: Keyword evolution (pre-built by caller)
    keyword_part = f" {keyword_summary}" if keyword_summary else ""

    # Part 6: Cosine similarity range
    similarity_part = ""
    if similarity_min > 0 or similarity_max > 0:
        similarity_part = (
            f" Sentences along the {dim} axis transition gradually in meaning "
            f"(cosine similarity from {similarity_min:.2f} to {similarity_max:.2f})."
        )

    # Part 7: Attention directive
    attention_directive = (
        " Pay special attention to how meaning subtly shifts from the "
        "lowest to highest points through intermediate sentences."
    )

    # Part 8: Context sentence (adapted from mood/dream to LLM conversations)
    context_sentence = (
        "All these sentences are LLM conversation messages so be much more "
        "specific than that and look what specifically is changing as we move "
        "along this axis!"
    )

    # Part 9: Avoid previous axes
    axis_ending = ""
    if previous_labels:
        prev = [l for l in previous_labels if l]
        if len(prev) == 1:
            axis_ending = (
                f" Your axis description must be different than '{prev[0]}' "
                f"because this was another axis description."
            )
        elif len(prev) >= 2:
            axis_ending = (
                f" Your axis description must be different than '{prev[0]}' "
                f"and '{prev[1]}' because these were the other axes descriptions."
            )

    # Final assembly (matching Android format)
    prompt = (
        f"I have a 3D semantic space, each point is a sentence embedding. "
        f"And along the {dim} axis. "
        f"{min_part}{max_part}"
        f"{intermediate_part}"
        f"{cluster_part}"
        f"{keyword_part}"
        f"{similarity_part}"
        f"{attention_directive}"
        f" What might this axis encode? What feature is changing along this axis. "
        f"Reply with axis meaning in exactly 2 words, without any additional explanation. "
        f"{context_sentence} "
        f"{axis_ending}"
        f"Remember: the axis must reflect gradual change (e.g., intensity, amount, degree), "
        f"not just name a type of thing. WARNING RESPOND WITH ONLY 2 WORDS!"
    )

    return prompt


def build_intermediate_part(
    texts_with_values: List[Tuple[int, str]],
) -> str:
    """Build the intermediate points string.

    Args:
        texts_with_values: List of (normalized_value_0_100, text) tuples.
            Typically the 2nd and (N-1)th entries when sorted by axis.

    Returns:
        Formatted string like: " Intermediate points (second highest axis distance):
        sentence at value 23: "text1"; sentence at value 78: "text2"."
    """
    if not texts_with_values:
        return ""
    parts = "; ".join(
        f'sentence at value {val}: "{text}"'
        for val, text in texts_with_values
    )
    return f" Intermediate points (second highest axis distance): {parts}."


def build_cluster_part(
    min_cluster_centroid: float,
    min_cluster_summary: str,
    max_cluster_centroid: float,
    max_cluster_summary: str,
) -> str:
    """Build the cluster context string for axis labeling.

    Args:
        min_cluster_centroid: Centroid value of the cluster at the axis minimum.
        min_cluster_summary: Summary label of that cluster.
        max_cluster_centroid: Centroid value of the cluster at the axis maximum.
        max_cluster_summary: Summary label of that cluster.

    Returns:
        Formatted cluster part string.
    """
    return (
        f' Lowest axis value cluster (centroid {min_cluster_centroid:.3f}): '
        f'"{min_cluster_summary}"; '
        f'highest axis value cluster (centroid {max_cluster_centroid:.3f}): '
        f'"{max_cluster_summary}".'
    )


# ═══════════════════════════════════════════════════
# CLUSTER LABELING — with avoidDesc + TF-IDF keywords
# ═══════════════════════════════════════════════════
# Ported from MoodAnalysis.java lines 3064-3277.
# Main prompt + avoidDesc appendix + keyword appendix.
# Response: 1-4 words.

def build_cluster_label_prompt(
    cluster_text: str,
    avoid_descriptions: Optional[List[str]] = None,
    unique_keywords: Optional[List[str]] = None,
    shared_keyword_deltas: Optional[List[Tuple[str, int, str]]] = None,
    other_only_keywords: Optional[List[str]] = None,
) -> str:
    """Build the cluster labeling prompt with avoidDesc and TF-IDF keywords.

    Args:
        cluster_text: Formatted text of all entries in this cluster.
        avoid_descriptions: Summaries of the 2 most similar clusters to avoid.
        unique_keywords: Keywords only found in this cluster.
        shared_keyword_deltas: List of (keyword, delta, compared_cluster_name) tuples.
        other_only_keywords: Keywords only found in other clusters.

    Returns:
        Complete prompt string.
    """
    prompt = (
        "Summarize the core shared essence of the following LLM conversation "
        "semantic cluster into a coherent phrase. "
        "Use as few words as possible. "
        "Avoid listing multiple separate ideas; instead, integrate concepts "
        "to reflect a unified meaning clearly. "
        "Do not provide generic summaries like 'mixed topics', do not use "
        "hyphen or ellipsis, always find specific commonality. "
        "The cluster description must be easily understandable by humans, "
        "avoid highly complex things but tend to produce something that "
        "everyone can easily understand and relate to. "
        "It must describe the data in the exact way but should be easy to "
        "relate to and understand for everyone. "
        "Return only the descriptive phrase in English, with no additional explanation.\n\n"
        f"Cluster text:\n{cluster_text}\n\n"
        "IMPORTANT: YOU MUST ONLY RESPOND WITH BETWEEN 1 TO 4 WORDS!"
    )

    # Append avoidDesc
    if avoid_descriptions:
        avoid_str = "', '".join(avoid_descriptions)
        prompt += (
            f"\n\nPay special attention to subtle thematic differences from "
            f"previous clusters, emphasizing a distinct semantic core not "
            f"previously captured so your summary must be very different than "
            f"previous clusters: '{avoid_str}'"
        )

    # Append keyword info
    has_keywords = unique_keywords or shared_keyword_deltas or other_only_keywords
    if has_keywords:
        prompt += "\n\nAlso below some less important auxiliary info: "

    if unique_keywords:
        prompt += (
            f"\n\nThese keywords were only specific to this cluster: "
            f"{', '.join(unique_keywords)}."
        )

    if shared_keyword_deltas:
        for keyword, delta, compared_cluster in shared_keyword_deltas:
            direction = "increase" if delta > 0 else "decrease"
            prompt += (
                f"\n\nThere was a significant {direction} of keyword "
                f"'{keyword}' compared to cluster '{compared_cluster}'."
            )

    if other_only_keywords:
        prompt += (
            f"\n\nThese keywords were only specific to the other clusters "
            f"but not this one: {', '.join(other_only_keywords)}."
        )

    return prompt


# ═══════════════════════════════════════════════════
# PROBE EXPLANATION — Single Point
# ═══════════════════════════════════════════════════
# Ported from DashboardFragment.java lines 652-682.
# Includes axis percentages + cluster distances.

def build_probe_explain_prompt(
    axis_labels: List[str],
    slider_pcts: List[int],
    cluster_distances: List[Tuple[str, int]],
) -> str:
    """Build single-point probe explanation prompt.

    Args:
        axis_labels: ["label_x", "label_y", "label_z"]
        slider_pcts: [x_pct, y_pct, z_pct] as integers 0-100
        cluster_distances: [(cluster_label, closeness_pct), ...] where
            0% = farthest, 100% = at center

    Returns:
        Complete prompt string.
    """
    prompt = "I am currently "

    # Axis percentages
    for i, (label, pct) in enumerate(zip(axis_labels, slider_pcts)):
        prompt += f'{pct}% at axis "{label}"'
        if i < len(axis_labels) - 1:
            prompt += " and "
        else:
            prompt += " "

    # Cluster distances
    prompt += "and my distances to clusters are "
    for i, (label, closeness) in enumerate(cluster_distances):
        prompt += (
            f'{closeness}% closeness to cluster "{label}" '
            f'(0% = farthest, 100% = at center)'
        )
        if i < len(cluster_distances) - 1:
            prompt += " and "
        else:
            prompt += " "

    prompt += (
        "Describe what this point in LLM conversation space corresponds to "
        "in exactly 30 words."
    )

    return prompt


# ═══════════════════════════════════════════════════
# MULTI-POINT PATH / TENDENCY EXPLANATION
# ═══════════════════════════════════════════════════
# Ported from DashboardFragment.java lines 684-722.
# Iterates through control points with axis % + cluster distances.

def build_path_explain_prompt(
    axis_labels: List[str],
    control_points: List[dict],
    score_context: list = None,
) -> str:
    """Build multi-point trajectory explanation prompt.

    Args:
        axis_labels: ["label_x", "label_y", "label_z"]
        control_points: List of dicts, each with:
            - "axis_pcts": [x%, y%, z%] as integers
            - "cluster_distances": [(label, closeness_pct), ...]
        score_context: Optional list parallel to control_points, each a dict
            mapping score channel name to its average value at that point.

    Returns:
        Complete prompt string.
    """
    prompt = "My conversation progressed like that:\n"

    for j, cp in enumerate(control_points):
        prompt += f"{j}: "

        # Axis percentages
        for i, (label, pct) in enumerate(zip(axis_labels, cp["axis_pcts"])):
            prompt += f'{pct}% at axis "{label}"; '

        # Cluster distances
        prompt += "distances "
        for k, (label, closeness) in enumerate(cp["cluster_distances"]):
            prompt += (
                f'{closeness}% closeness to cluster "{label}" '
                f'(0% = farthest, 100% = at center)"; '
            )

        # Score values at this control point
        if score_context and j < len(score_context) and score_context[j]:
            scores_str = "; ".join(
                f'{ch}={v:.2f}' for ch, v in score_context[j].items()
            )
            prompt += f"scores: {scores_str}; "

        prompt += "\n"

    # Score instruction
    if score_context and any(sc for sc in score_context if sc):
        prompt += (
            "Include how the score values changed along this path "
            "(e.g. moving toward higher/lower success, error rate, etc). "
        )

    prompt += (
        "Do not say how each component changed but fuse all components "
        "together to tell me what is the meaning of this transition as a whole. "
        "Describe how the conversation changed in exactly 45 words. In English."
    )

    return prompt


# ═══════════════════════════════════════════════════
# FLOW EXPLANATION
# ═══════════════════════════════════════════════════

FLOW_EXPLAIN_SYSTEM = (
    "You are an expert in LLM interpretability. "
    "You analyze flow fields in semantic embedding spaces."
)

FLOW_EXPLAIN = (
    "A flow model was trained on LLM conversation trajectories in 3D space.\n"
    "Axes: X={x_label}, Y={y_label}, Z={z_label}\n\n"
    "At point ({x:.2f}, {y:.2f}, {z:.2f}), the velocity vector points toward "
    "({vx:.2f}, {vy:.2f}, {vz:.2f}).\n\n"
    "This means conversations near this point tend to move toward "
    "{direction_description}.\n\n"
    "What does this flow direction mean semantically? "
    "What tendency does it reveal about how conversations evolve from this region?\n"
    "Reply in 1-2 sentences."
)


def format_flow_explain(
    x_label: str, y_label: str, z_label: str,
    x: float, y: float, z: float,
    vx: float, vy: float, vz: float,
    direction_description: str,
) -> str:
    return FLOW_EXPLAIN.format(
        x_label=x_label, y_label=y_label, z_label=z_label,
        x=x, y=y, z=z, vx=vx, vy=vy, vz=vz,
        direction_description=direction_description,
    )


# ═══════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════

SUMMARY_SYSTEM = (
    "You are an expert in LLM interpretability and conversation analysis."
)

SUMMARY = (
    "An LLM conversation was embedded and projected into 3D semantic space.\n\n"
    "Axes: X={x_label}, Y={y_label}, Z={z_label}\n"
    "Clusters found: {cluster_descriptions}\n"
    "Path length: {n_entries} messages\n"
    "Cluster transitions: {cluster_sequence}\n\n"
    "Provide a concise summary (80-120 words) of what this analysis reveals "
    "about the conversation's semantic structure. What patterns, transitions, "
    "or tendencies are notable?"
)


def format_summary(
    x_label: str, y_label: str, z_label: str,
    cluster_descriptions: str, n_entries: int, cluster_sequence: str,
) -> str:
    return SUMMARY.format(
        x_label=x_label, y_label=y_label, z_label=z_label,
        cluster_descriptions=cluster_descriptions,
        n_entries=n_entries,
        cluster_sequence=cluster_sequence,
    )


# ═══════════════════════════════════════════════════
# ATTRACTOR EXPLANATION
# ═══════════════════════════════════════════════════

ATTRACTOR_EXPLAIN_SYSTEM = (
    "You are an expert in LLM interpretability and dynamical systems. "
    "You analyze flow fields in semantic embedding spaces. "
    "An attractor is like a gravitational well in the flow field — "
    "a region that pulls nearby trajectories toward itself. "
    "Conversations, reasoning chains, and agent traces that enter its basin "
    "of attraction are drawn inexorably toward this point, like rivers "
    "flowing downhill into a lake. The attractor represents a stable "
    "semantic destination — an end-state that the system naturally evolves toward."
)


def build_attractor_explain_prompt(
    axis_labels: List[str],
    axis_pcts: List[int],
    cluster_distances: List[Tuple[str, int]],
    nearest_texts: List[str],
    strength: float,
    basin_fraction: float,
    divergence: float,
    score_info: Optional[dict] = None,
    trajectory_explanations: Optional[List[str]] = None,
) -> str:
    """Build prompt for explaining the semantic meaning of a flow attractor.

    Args:
        axis_labels: ["label_x", "label_y", "label_z"]
        axis_pcts: [x%, y%, z%] position of attractor center
        cluster_distances: [(cluster_label, closeness_pct), ...]
        nearest_texts: 3-5 nearest data point texts to the center
        strength: Attractor strength [0-1], 1 = strongest in the field
        basin_fraction: Fraction of flow field captured [0-1]
        divergence: Raw divergence value (more negative = stronger convergence)
        score_info: Optional dict with score channel context, e.g.
            {"channel": "solved", "mean_score_raw": 0.73, "interpretation": "positive"}
        trajectory_explanations: Optional list of 3 LLM-generated path
            explanations from probes that flowed from distant starting
            points and converged into this attractor basin.
    """
    prompt = "A flow attractor was detected in the semantic embedding space.\n"
    prompt += "Think of it like a gravitational well: traces that wander near "
    prompt += "this region get pulled in and settle here. It is NOT a path — "
    prompt += "it is a destination that the flow field funnels trajectories toward, "
    prompt += "like a river basin where multiple streams converge.\n\n"

    # Location
    prompt += "Attractor center location:\n"
    for label, pct in zip(axis_labels, axis_pcts):
        prompt += f'  {pct}% along axis "{label}"\n'

    # Cluster proximity
    prompt += "\nDistance to clusters:\n"
    for label, closeness in cluster_distances:
        prompt += (f'  {closeness}% closeness to cluster "{label}" '
                   f'(0%=farthest, 100%=at center)\n')

    # Strength context
    prompt += f"\nAttractor properties:\n"
    prompt += f"  Strength: {strength:.0%} (relative to strongest attractor)\n"
    prompt += f"  Basin size: captures {basin_fraction:.0%} of the flow field\n"
    prompt += f"  Convergence: divergence={divergence:.4f} "
    prompt += f"(more negative = stronger pull)\n"

    # Score context
    if score_info:
        ch = score_info.get("channel", "unknown")
        raw = score_info.get("mean_score_raw", 0.5)
        interp = score_info.get("interpretation", "neutral")
        prompt += (f'\nScore context (channel "{ch}"):\n'
                   f"  Mean score in basin: {raw:.2f} → this is a {interp} attractor\n"
                   f"  Meaning: traces that converge here tend toward {interp} outcomes "
                   f'for "{ch}"\n')

    # Nearby evidence
    if nearest_texts:
        prompt += "\nNearest data points to the attractor center:\n"
        for i, text in enumerate(nearest_texts[:5]):
            truncated = text[:200] + "..." if len(text) > 200 else text
            prompt += f'  {i + 1}. "{truncated}"\n'

    # Trajectory convergence evidence
    if trajectory_explanations:
        prompt += (
            "\nConvergence evidence — three probes were released from distant, "
            "different parts of the semantic space. All three were pulled by "
            "the flow field into this attractor. Here is what each journey "
            "looked like:\n"
        )
        for i, traj_text in enumerate(trajectory_explanations):
            prompt += f"\n  Trajectory {i + 1}: {traj_text}\n"
        prompt += (
            "\nNotice the COMMON DESTINATION these three very different "
            "starting points all converge to. This is the key insight.\n"
        )

    prompt += (
        "\nExplain what this attractor represents as a semantic destination. "
        "Why does the flow field funnel trajectories here — what outcome, "
        "end-state, or resolution does this basin represent? "
        "Think of it as: 'traces that reach this region have arrived at ___.' "
        "Be specific based on the axes, clusters, nearby texts, and "
        "the convergence trajectories above. "
        "Reply in exactly 50 words."
    )

    return prompt
